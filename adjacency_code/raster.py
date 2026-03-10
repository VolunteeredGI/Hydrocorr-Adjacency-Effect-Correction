from osgeo import gdal
from osgeo import osr

gdal.UseExceptions()


def write_image(filename, im_proj, im_geotrans, im_data, print_msg=True):
    """
    Save a NumPy array as a GeoTIFF raster file.

    Parameters
    ----------
    filename : str
        Output file path.
    im_proj : str
        Projection information in WKT format.
    im_geotrans : tuple
        GDAL geotransform tuple.
    im_data : np.ndarray
        Image array, either 2D (H, W) or 3D (B, H, W).
    print_msg : bool, optional
        Whether to print a success message after saving.
    """
    # Determine GDAL data type from NumPy dtype
    if "int8" in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif "int16" in im_data.dtype.name:
        datatype = gdal.GDT_Int16
    else:
        datatype = gdal.GDT_Float32

    # Normalize array dimensions
    if im_data.ndim == 3:
        im_bands, im_height, im_width = im_data.shape
    elif im_data.ndim == 2:
        im_bands = 1
        im_height, im_width = im_data.shape
        im_data = im_data.reshape(1, im_height, im_width)
    else:
        raise ValueError("im_data must be a 2D or 3D array.")

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

    if dataset is None:
        raise RuntimeError(f"Failed to create file: {filename}")

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)

    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])

    dataset.FlushCache()
    del dataset

    if print_msg:
        print(f"Image has been saved to: {filename}")


def read_image(src_ras_file, print_msg=True):
    """
    Read a raster image and return its array, geotransform, and projection.

    Parameters
    ----------
    src_ras_file : str
        Path to the raster file.
    print_msg : bool, optional
        Whether to print a success message after reading.

    Returns
    -------
    im_array : np.ndarray
        Raster data array. Can be single-band or multi-band.
    geotrans : tuple
        GDAL geotransform parameters.
    projection : str
        Projection information in WKT format.
    """
    dataset = gdal.Open(src_ras_file)
    if dataset is None:
        raise FileNotFoundError(f"Failed to open raster file: {src_ras_file}")

    projection = dataset.GetProjection()
    geotrans = dataset.GetGeoTransform()
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_array = dataset.ReadAsArray(0, 0, im_width, im_height)

    del dataset

    if print_msg:
        print(f"\nSuccessfully read: {src_ras_file}")

    # Ensure projection information exists
    if not projection:
        raise ValueError("The raster does not contain projection information. Please check the input file.")

    return im_array, geotrans, projection


if __name__ == "__main__":
    # Example usage
    # raster_path = "example.tif"
    # image, geo, proj = read_image(raster_path)
    # print("Image shape:", image.shape)
    # print("Projection:", proj)
    # print("GeoTransform:", geo)
    #
    # output_path = "output_example.tif"
    # write_image(output_path, proj, geo, image)

    raster_path = (
        r"E:\1_博士研究生\19 综合示范\Raster Data\Lake Bosten_2021-09-15"
        r"\HJ2A_CCD4_E87.8_N42.2_20210914_L1A0000265447\FLAASH\flaash.dat"
    )
    image, geo, proj = read_image(raster_path)
    print("Image shape:", image.shape)
    print("Projection:", proj)
    print("GeoTransform:", geo)