# Hydrocorr Adjacency Effect Correction

A lightweight Python implementation for adjacency-effect correction of inland water remote sensing imagery.

This repository provides a practical workflow for estimating water adjacency-effect gain from surrounding land pixels and applying correction to multi-band satellite images. The current implementation is designed for four-band optical imagery and supports CPU-accelerated gain computation.

---

## 1. Overview

The workflow includes:

1. Reading a surface reflectance image and its corresponding XML metadata.
2. Reading a water mask.
3. Cropping the land-cover raster around the valid water region.
4. Matching the image to the land-cover grid.
5. Estimating the land adjacency-effect gain for each water pixel.
6. Optionally visualizing the gain image.
7. Applying the gain to generate an adjacency-corrected image.

The implementation currently uses:

- `rasterio` / `pyproj` for most spatial operations
- `GDAL` for raster I/O compatibility
- precompiled coefficient lookup tables for efficient gain computation

---

## 2. Repository Structure

```text
.
├── __pycache__/
├── Cache/
├── test/
├── ae_corr_coff.xlsx
├── ae_corr_main.py
├── ae_gain_fast.py
├── China_LC_480m_mode_COG.tif
├── China_LC_960m_mode.tif
├── landuse_function.py
├── plot_function.py
├── raster.py
└── Raster_Data/
```

## 3. Main files

ae_corr_main.py
Main entry script for adjacency-effect correction.

ae_gain_fast.py
CPU-accelerated implementation for water gain calculation.

landuse_function.py
Utilities for:

cropping land-cover rasters

matching image grids

extracting neighboring land-use pixels

computing distance and azimuth bins

plot_function.py
Visualization of gain images.

raster.py
Basic raster read/write utilities based on GDAL.

ae_corr_coff.xlsx
Coefficient table used in the correction model.

China_LC_960m_mode.tif
Default land-cover raster used in the current configuration.

China_LC_480m_mode_COG.tif
Optional higher-resolution land-cover raster.

Cache/
Automatically stores cropped land-cover subsets for debugging and reuse.



Input requirements

1) Surface reflectance image
File format: GeoTIFF
Expected band order: 4 bands
Default wavelengths:
485 nm
550 nm
660 nm
830 nm

The current code assumes image values are scaled by 10000 and converts them to reflectance by:
image_data = image_data / 10000.0

2) XML metadata
The XML file must contain the tag:
<SolarZenith>...</SolarZenith>
This value is used as the solar zenith angle input for geometric correction.

3) Water mask
Must align with the target image
Non-zero values indicate water pixels
The current implementation assumes the mask CRS is EPSG:4326

4) Land-cover raster

The current implementation uses:
China_LC_960m_mode.tif

Requirements:

must have valid CRS information
current fast workflow assumes WGS84-compatible handling for the mask
land-cover classes are used to determine distance thresholds and adjacency coefficients

5) Coefficient table

The Excel file ae_corr_coff.xlsx should contain the following sheets:
Typical_Spectral_Line
Land_Cover_Coefficient
Atmospheric_Coefficient
Geometry_Coefficient

These tables are used to:
assign TSL levels
estimate land contribution decay
apply atmospheric scaling
apply solar geometry scaling

4. Core Workflow
Step 1: Read image, XML, and mask
ae_corr_main.py loads:
the input reflectance image
the XML metadata
the water mask

Step 2: Build adjacency-effect configuration

The default configuration includes:
land-cover raster path
coefficient file path
wavelength list
aerosol and atmosphere type
AOT at 550 nm
solar zenith angle
distance threshold for each land-cover class

Step 3: Compute water gain

calculate_water_gain_fast() performs:
land-cover cropping around the mask region
image reprojection to the land-cover grid
TSL classification
per-water-pixel neighborhood search
band-wise gain estimation

Step 4: Apply correction

The corrected image is computed band by band as:

corrected_image[b, :, :] = image[b, :, :] * (1 - image_gain[b, :, :])
Step 5: Optional visualization

show_gain_image() in plot_function.py can be used to display the gain image.

5. Example Usage
Run from ae_corr_main.py

6. Output
1) Gain image
When out_gain_path is provided, the adjacency-effect gain image is saved as:
*_ae_gain.tif


2) Corrected image
The corrected image is computed inside ae_corr_main.py.
To save the corrected raster, enable this line.

3) Cached land-cover subset
A cropped land-cover raster is automatically written to: ./Cache/
This is useful for debugging and performance optimization.

7. Visualization

You can visualize the gain raster with:
from plot_function import show_gain_image
show_gain_image(
    output_gain_path,
    title_prefix="Band",
    cbar_label="Gain",
    cmap_name="jet",
)

8. Main Functions
ae_corr_main.py
_get_xml_value()
ae_corr_main()
ae_gain_fast.py
read_raster()
write_raster()
extract_landuse()
match_image_to_landuse_grid()
calculate_tsl_fast()
build_coeff_tables()
_compute_gain_one_pixel()
calculate_water_gain_fast()
landuse_function.py
fast_mask_bbox_wgs84()
extract_landuse()
to_wkt()
match_image_to_landuse()
landuse_points_within_radius()
compute_dist_angle_bins_from_out()
plot_function.py
show_gain_image()
raster.py
read_image()
write_image()

9. Dependencies

Recommended environment:

Python >= 3.9
numpy
pandas
matplotlib
rasterio
pyproj
GDAL
openpyxl
tqdm

10. Notes and Limitations

The current workflow is designed for four-band optical imagery.
The current XML reader expects the metadata tag SolarZenith.
The mask is assumed to be in EPSG:4326.
The gain computation uses land-cover class thresholds defined in ae_config.
The corrected image writing step is currently disabled by default in ae_corr_main.py.
The current implementation mixes:
GDAL-based raster I/O in raster.py
rasterio/pyproj-based spatial processing in other modules
This is functional, but future refactoring toward a single raster backend would improve maintainability.


11. Acknowledgment

This implementation was developed for adjacency-effect correction experiments on inland water remote sensing imagery, 
with a focus on practical reproducibility, modularity, and compatibility with real satellite products.