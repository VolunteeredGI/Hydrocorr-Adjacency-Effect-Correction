import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xml.dom.minidom

from raster import read_image, write_image
from ae_gain_fast import calculate_water_gain_fast
from plot_function import show_gain_image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ae_config = {
    "landuse_path": os.path.join(BASE_DIR, "China_LC_960m_mode.tif"),
    "ae_coff_path": os.path.join(BASE_DIR, "ae_corr_coff.xlsx"),
    "factor": 1,
    "cal_meter": 4000,
    "bands": ["485", "550", "660", "830"],  # Assume 4 spectral bands
    "Atmospheric": "MidlatitudeSummer",
    "Aerosol": "Continental",
    "sun_z": 30.0,
    "aot550": 0.12,
    "lc_distance_threshold": {
        10: 2850,
        11: 2650,
        12: 3050,
        13: 3250,
    },
}


def _get_xml_value(xml_files: str, tag_name: str, cast_type=float, default=None):
    """
    Safely read the value of a specified XML tag.

    Parameters
    ----------
    xml_files : str
        Path to the XML file.
    tag_name : str
        Name of the XML tag to retrieve.
    cast_type : type, optional
        Target type used to cast the extracted text value. Default is float.
    default : Any, optional
        Default value returned if the tag is missing.

    Returns
    -------
    Any
        Parsed and type-casted XML tag value.

    Raises
    ------
    KeyError
        If the required XML tag is missing and no default value is provided.
    ValueError
        If the tag value cannot be converted to the specified type.
    """
    dom = xml.dom.minidom.parse(xml_files)
    nodes = dom.getElementsByTagName(tag_name)

    if not nodes or nodes[0].firstChild is None:
        if default is not None:
            return default
        raise KeyError(f"Missing required XML tag: <{tag_name}>")

    text = nodes[0].firstChild.data.strip()

    try:
        return cast_type(text)
    except Exception as e:
        raise ValueError(
            f"Failed to convert the value of <{tag_name}> to "
            f"{cast_type.__name__}: '{text}'. Error: {e}"
        )


def ae_corr_main(img_path, xml_path, mask_path, out_path, out_gain_path=None, plot_fig=False):
    """
    Main function for atmospheric adjacency effect correction.

    Parameters
    ----------
    img_path : str
        Path to the input image to be corrected.
    xml_path : str
        Path to the XML metadata file corresponding to the input image.
    mask_path : str
        Path to the water mask file.
    out_path : str
        Output path for the corrected image.
    out_gain_path : str, optional
        Output path for the adjacency-effect gain image.
    plot_fig : bool, optional
        Whether to display the gain image. Default is False.
    """

    # Update atmospheric parameters in the configuration dictionary
    Atmospheric = "MidlatitudeSummer"
    ae_config["Atmospheric"] = Atmospheric

    Aerosol = "Continental"
    ae_config["Aerosol"] = Aerosol

    aot550 = 0.11
    ae_config["aot550"] = aot550

    # Read solar zenith angle from the XML metadata
    sun_z = _get_xml_value(xml_path, "SolarZenith", float)
    ae_config["sun_z"] = sun_z

    # Perform atmospheric adjacency effect correction
    image_gain, geo, crs = calculate_water_gain_fast(
        img_path,
        mask_path,
        ae_config,
        out_tif=out_gain_path,
        n_jobs=1,
    )
    
    # Read the original image
    image, geo1, proj1 = read_image(img_path)

    # Save the gain image if an output path is provided
    if out_gain_path is not None:
        write_image(out_gain_path, proj1, geo1, image_gain)

    # Apply correction band by band
    corrected_image = np.zeros_like(image, dtype=np.float32)
    for b in range(4):
        corrected_image[b, :, :] = image[b, :, :] * (1 - image_gain[b, :, :])

    im_array_int = corrected_image.astype(np.int32)

    # Save the corrected image
    write_image(out_path, proj1, geo1, im_array_int)

    # Display the gain image if requested
    if plot_fig:
        show_gain_image(image_gain)


if __name__ == "__main__":
    base_dir = r"./Raster_Data"

    image_path = os.path.join(
        base_dir,
        r"HJ2B_CCD3_E117.8_N31.5_20241106_L1A0001248841_SR.tif"
    )
    image_xml_path = os.path.join(
        base_dir,
        r"HJ2B_CCD3_E117.8_N31.5_20241106_L1A0001248841.xml"
    )
    image_mask_path = os.path.join(
        base_dir,
        r"HJ2B_CCD3_E117.8_N31.5_20241106_L1A0001248841_SR_water_mask.tif"
    )

    output_path = image_path.replace(".tif", "_ae_corr.tif")
    output_gain_path = image_path.replace(".tif", "_ae_gain.tif")

    ae_corr_main(
        image_path,
        image_xml_path,
        image_mask_path,
        output_path,
        output_gain_path,
        plot_fig=False,
    )