"""
ae_gain_fast.py
================
Production-level, executable implementation for calculating water adjacency-effect gain
(CPU-accelerated refactored version).

Design goals
------------
- Use rasterio / pyproj consistently (without mixing GDAL Dataset API) to avoid
  Affine / WKT conversion issues.
- Move pandas conditional filtering out of inner loops by precompiling coefficient
  tables into NumPy lookup arrays.
- Compute neighborhood distance and azimuth in a vectorized way on the land-use grid
  window, avoiding per-point lon/lat -> row/col back-transformation.
- Support multiprocessing over water-pixel lists; serial execution is also supported.

Dependencies
------------
- numpy, pandas, rasterio, pyproj
- tqdm (optional)

Notes
-----
- This script assumes that both mask and landuse are in EPSG:4326 (WGS84), which is
  consistent with your current setup.
- The input image CRS can be arbitrary; it will be reprojected to the landuse grid
  using average resampling for downscaling.
"""

from __future__ import annotations

import os
import math
import warnings
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Any

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import Affine
from rasterio.enums import Resampling
from rasterio.warp import reproject
from pyproj import CRS, Transformer

EARTH_R = 6371000.0  # Earth radius in meters


# -------------------------
# I/O helpers (rasterio)
# -------------------------
def read_raster(path: str, as_float: bool = False) -> Tuple[np.ndarray, Affine, Any, Optional[float]]:
    """
    Read a raster using rasterio.

    Parameters
    ----------
    path : str
        Raster file path.
    as_float : bool, optional
        Whether to cast the array to float32.

    Returns
    -------
    tuple
        (arr, transform, crs, nodata)

        - arr: ndarray with shape (bands, H, W)
        - transform: Affine transform
        - crs: coordinate reference system
        - nodata: nodata value
    """
    with rasterio.open(path) as ds:
        arr = ds.read()  # shape: (bands, H, W)
        nodata = ds.nodata
        if as_float:
            arr = arr.astype(np.float32, copy=False)
        return arr, ds.transform, ds.crs, nodata


def write_raster(path: str, arr: np.ndarray, transform: Affine, crs: Any, nodata=None,
                 dtype=None, compress: str = "lzw"):
    """
    Write a raster using rasterio.

    Parameters
    ----------
    path : str
        Output raster path.
    arr : np.ndarray
        Input array with shape (bands, H, W) or (H, W).
    transform : Affine
        Output affine transform.
    crs : Any
        Output CRS.
    nodata : optional
        Output nodata value.
    dtype : optional
        Output data type. If None, uses the array dtype.
    compress : str, optional
        Compression method. Default is 'lzw'.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if arr.ndim == 2:
        arr3 = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        arr3 = arr
    else:
        raise ValueError("arr must be 2D or 3D")

    if dtype is None:
        dtype = arr3.dtype

    profile = {
        "driver": "GTiff",
        "height": arr3.shape[1],
        "width": arr3.shape[2],
        "count": arr3.shape[0],
        "dtype": dtype,
        "transform": transform,
        "crs": crs,
        "nodata": nodata,
        "tiled": True,
        "compress": compress,
        "bigtiff": "yes",
    }

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr3.astype(dtype, copy=False))


# -------------------------
# Geometry helpers
# -------------------------
def pixel_center_lonlat(transform: Affine, row: int, col: int) -> Tuple[float, float]:
    """
    Return pixel-center longitude and latitude for a WGS84 north-up grid.

    Parameters
    ----------
    transform : Affine
        Affine transform of the raster.
    row : int
        Row index.
    col : int
        Column index.

    Returns
    -------
    tuple
        (lon, lat) at the pixel center.
    """
    x, y = transform * (col + 0.5, row + 0.5)
    return float(x), float(y)


def rowcol_from_lonlat(transform: Affine, lon: float, lat: float) -> Tuple[int, int, float, float]:
    """
    Convert lon/lat to raster row/col using the inverse Affine transform.

    Parameters
    ----------
    transform : Affine
        Raster affine transform.
    lon : float
        Longitude.
    lat : float
        Latitude.

    Returns
    -------
    tuple
        (row_i, col_i, row_f, col_f)
        - row_i, col_i: integer indices obtained with floor()
        - row_f, col_f: floating-point coordinates
    """
    col_f, row_f = (~transform) * (lon, lat)
    return int(math.floor(row_f)), int(math.floor(col_f)), float(row_f), float(col_f)


def equirectangular_dist_and_bearing(c_lon, c_lat, lon_grid, lat_grid):
    """
    Compute approximate local distance and bearing using the equirectangular approximation.

    Bearing convention:
    - 0 degrees = north
    - clockwise positive

    Parameters
    ----------
    c_lon, c_lat : float
        Center longitude and latitude.
    lon_grid, lat_grid : np.ndarray
        Longitude and latitude grids.

    Returns
    -------
    tuple
        (dist_m, ang_deg, ang_bin_30)
        - dist_m: distance in meters
        - ang_deg: azimuth angle in degrees
        - ang_bin_30: angle binned to 30-degree intervals
    """
    lat0 = np.deg2rad(c_lat)
    dlon = np.deg2rad(lon_grid - c_lon)
    dlat = np.deg2rad(lat_grid - c_lat)

    x_east = EARTH_R * dlon * np.cos(lat0)
    y_north = EARTH_R * dlat
    dist = np.sqrt(x_east * x_east + y_north * y_north)

    ang = np.degrees(np.arctan2(x_east, y_north))
    ang = (ang + 360.0) % 360.0

    ang_bin = (np.round(ang / 30.0) * 30.0) % 360.0
    ang_bin = ang_bin.astype(np.int16)
    ang_bin[ang_bin == 360] = 0

    return dist, ang, ang_bin


# -------------------------
# Fast bbox from mask
# -------------------------
def fast_mask_bbox_wgs84(mask2d: np.ndarray, transform: Affine) -> Tuple[float, float, float, float]:
    """
    Compute the bounding box of valid pixels in a WGS84 mask.

    Pixels with mask != 0 are considered valid.

    Parameters
    ----------
    mask2d : np.ndarray
        2D mask array.
    transform : Affine
        Affine transform of the mask raster.

    Returns
    -------
    tuple
        (min_lon, max_lon, min_lat, max_lat)
    """
    ys, xs = np.where(mask2d != 0)
    if ys.size == 0:
        raise ValueError("mask has no valid pixels (mask != 0).")

    row_min, row_max = int(ys.min()), int(ys.max())
    col_min, col_max = int(xs.min()), int(xs.max())

    lon_min, lat_max = rasterio.transform.xy(transform, row_min, col_min, offset="center")
    lon_max, lat_min = rasterio.transform.xy(transform, row_max, col_max, offset="center")

    min_lon = float(min(lon_min, lon_max))
    max_lon = float(max(lon_min, lon_max))
    min_lat = float(min(lat_min, lat_max))
    max_lat = float(max(lat_min, lat_max))

    return min_lon, max_lon, min_lat, max_lat


def extract_landuse(mask_path: str, landuse_path: str, cal_meter: float,
                    cache_dir: Optional[str] = None, enlarge_factor: float = 1.2) -> Tuple[np.ndarray, Affine, Any, str]:
    """
    Crop the land-use raster using the buffered bounding box of the mask.

    The bounding box is computed in WGS84, buffered in meters using a local AEQD
    projection, then converted back and used to crop the land-use raster.

    Parameters
    ----------
    mask_path : str
        Path to the mask raster.
    landuse_path : str
        Path to the land-use raster.
    cal_meter : float
        Calculation radius in meters.
    cache_dir : str, optional
        Directory used to save the cropped land-use raster for debugging/cache.
    enlarge_factor : float, optional
        Buffer multiplier applied to cal_meter.

    Returns
    -------
    tuple
        (landuse2d, land_transform, land_crs, cache_tif_path)
    """
    if cache_dir is None:
        cache_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "Cache"))
    os.makedirs(cache_dir, exist_ok=True)

    landuse_base = os.path.splitext(os.path.basename(landuse_path))[0]
    mask_base = os.path.splitext(os.path.basename(mask_path))[0]
    out_tif = os.path.join(cache_dir, f"{landuse_base}_{mask_base}.tif")

    # Read mask
    mask_arr, m_transform, m_crs, _ = read_raster(mask_path, as_float=False)
    if m_crs is None:
        raise ValueError("mask has no CRS.")
    if CRS.from_user_input(m_crs).to_epsg() != 4326:
        raise ValueError(f"mask CRS must be EPSG:4326, got {m_crs}")
    mask2d = mask_arr[0]

    min_lon, max_lon, min_lat, max_lat = fast_mask_bbox_wgs84(mask2d, m_transform)

    # Buffer in meters using a local AEQD projection centered on the bbox center
    c_lon = (min_lon + max_lon) / 2.0
    c_lat = (min_lat + max_lat) / 2.0
    aeqd = CRS.from_proj4(f"+proj=aeqd +lat_0={c_lat} +lon_0={c_lon} +datum=WGS84 +units=m +no_defs")

    tf_w2a = Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)
    tf_a2w = Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True)

    corners = [(min_lon, min_lat), (min_lon, max_lat), (max_lon, min_lat), (max_lon, max_lat)]
    cx, cy = tf_w2a.transform([p[0] for p in corners], [p[1] for p in corners])
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)

    min_x, max_x = cx.min(), cx.max()
    min_y, max_y = cy.min(), cy.max()

    buf = cal_meter * float(enlarge_factor)
    min_x -= buf
    max_x += buf
    min_y -= buf
    max_y += buf

    buf_lon, buf_lat = tf_a2w.transform(
        [min_x, min_x, max_x, max_x],
        [min_y, max_y, min_y, max_y]
    )
    buf_lon = np.asarray(buf_lon, dtype=float)
    buf_lat = np.asarray(buf_lat, dtype=float)
    buf_min_lon, buf_max_lon = float(buf_lon.min()), float(buf_lon.max())
    buf_min_lat, buf_max_lat = float(buf_lat.min()), float(buf_lat.max())

    # Crop land-use raster
    with rasterio.open(landuse_path) as lds:
        if lds.crs is None:
            raise ValueError("landuse has no CRS. Please set CRS first.")
        land_crs = lds.crs

        tf_w2lu = Transformer.from_crs("EPSG:4326", land_crs, always_xy=True)
        lu_x, lu_y = tf_w2lu.transform(
            [buf_min_lon, buf_min_lon, buf_max_lon, buf_max_lon],
            [buf_min_lat, buf_max_lat, buf_min_lat, buf_max_lat]
        )
        lu_x = np.asarray(lu_x, dtype=float)
        lu_y = np.asarray(lu_y, dtype=float)

        left, right = float(lu_x.min()), float(lu_x.max())
        bottom, top = float(lu_y.min()), float(lu_y.max())

        win = from_bounds(left, bottom, right, top, transform=lds.transform)
        win = win.round_offsets().round_lengths()

        full_win = rasterio.windows.Window(0, 0, lds.width, lds.height)
        win = win.intersection(full_win)
        if win.width <= 0 or win.height <= 0:
            raise ValueError("Buffered bbox has no overlap with the landuse raster.")

        land_arr = lds.read(1, window=win)  # Assume single-band land-use raster
        land_transform = rasterio.windows.transform(win, lds.transform)

        # Save cache raster for debugging
        profile = lds.profile.copy()
        profile.update({
            "height": land_arr.shape[0],
            "width": land_arr.shape[1],
            "count": 1,
            "transform": land_transform,
            "crs": land_crs,
            "tiled": True,
            "compress": "lzw",
            "bigtiff": "yes",
        })
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(land_arr, 1)

    return land_arr, land_transform, land_crs, out_tif


# -------------------------
# Match image to landuse grid (rasterio reproject)
# -------------------------
def match_image_to_landuse_grid(image_arr: np.ndarray, img_transform: Affine, img_crs: Any,
                                land_shape: Tuple[int, int], land_transform: Affine, land_crs: Any,
                                resampling: str = "average", src_nodata=None,
                                dst_dtype=np.float32) -> np.ndarray:
    """
    Reproject and resample an image onto the land-use grid.

    Parameters
    ----------
    image_arr : np.ndarray
        Input image array with shape (bands, H, W) or (H, W).
    img_transform : Affine
        Input image transform.
    img_crs : Any
        Input image CRS.
    land_shape : tuple
        Target shape (land_H, land_W).
    land_transform : Affine
        Target grid transform.
    land_crs : Any
        Target grid CRS.
    resampling : str, optional
        Resampling method. Supported: 'nearest', 'bilinear', 'average', 'mode'.
    src_nodata : optional
        Source nodata value.
    dst_dtype : dtype, optional
        Output data type.

    Returns
    -------
    np.ndarray
        Reprojected array with shape (bands, land_H, land_W).
    """
    if image_arr.ndim == 2:
        src = image_arr[np.newaxis, ...]
    elif image_arr.ndim == 3:
        src = image_arr
    else:
        raise ValueError("image_arr must be 2D or 3D")

    land_h, land_w = land_shape
    out = np.zeros((src.shape[0], land_h, land_w), dtype=dst_dtype)

    rs = resampling.lower()
    rs_map = {
        "nearest": Resampling.nearest,
        "bilinear": Resampling.bilinear,
        "average": Resampling.average,
        "mode": Resampling.mode,
    }
    if rs not in rs_map:
        raise ValueError(f"Unsupported resampling={resampling}, choose {list(rs_map)}")

    for b in range(src.shape[0]):
        reproject(
            source=src[b].astype(np.float32, copy=False),
            destination=out[b],
            src_transform=img_transform,
            src_crs=img_crs,
            dst_transform=land_transform,
            dst_crs=land_crs,
            resampling=rs_map[rs],
            src_nodata=src_nodata,
            dst_nodata=np.nan if np.issubdtype(dst_dtype, np.floating) else 0,
            num_threads=2,
        )
    return out


# -------------------------
# Exponential decay function
# -------------------------
def exp_decay(x, A, B, C, alpha):
    """
    Exponential decay function.

    Model form:
        y = A * exp(-B * x^alpha) + C
    """
    return A * np.exp(-B * (x ** alpha)) + C


# -------------------------
# TSL calculation (vectorized per land-use code)
# -------------------------
def calculate_tsl_fast(landuse2d: np.ndarray, land_transform: Affine, land_crs: Any,
                       tsl_df: pd.DataFrame,
                       image_arr: np.ndarray, img_transform: Affine, img_crs: Any,
                       bands: List[str]) -> np.ndarray:
    """
    Compute the TSL level map aligned to the land-use grid.

    Workflow
    --------
    - Reproject the image to the land-use grid using average resampling.
    - For each land-use code, compute the MSE between pixel spectra and all
      candidate TSL spectra.
    - Assign argmin(MSE) + 1 as the TSL level.

    Parameters
    ----------
    landuse2d : np.ndarray
        Land-use raster.
    land_transform : Affine
        Land-use raster transform.
    land_crs : Any
        Land-use raster CRS.
    tsl_df : pd.DataFrame
        TSL lookup table.
    image_arr : np.ndarray
        Input image array.
    img_transform : Affine
        Input image transform.
    img_crs : Any
        Input image CRS.
    bands : list of str
        Band names used for TSL matching.

    Returns
    -------
    np.ndarray
        TSL level array with dtype int8 and shape matching landuse2d.
    """
    land_h, land_w = landuse2d.shape
    img_match = match_image_to_landuse_grid(
        image_arr, img_transform, img_crs,
        (land_h, land_w), land_transform, land_crs,
        resampling="average",
        src_nodata=None,
        dst_dtype=np.float32,
    )

    band_cols = [str(b) for b in bands]

    # Ensure required columns exist
    for c in band_cols + ["Code"]:
        if c not in tsl_df.columns:
            raise ValueError(f"TSL sheet missing column '{c}'. Available: {list(tsl_df.columns)}")

    tsl_num = np.zeros((land_h, land_w), dtype=np.int8)
    code_kind = tsl_df["Code"].unique().tolist()

    # Process by land-use code to avoid per-pixel loops
    for code in code_kind:
        m = (landuse2d == code)
        if not np.any(m):
            continue

        # Pixel spectra: (N, nbands)
        pix = np.stack([img_match[i][m] for i in range(len(bands))], axis=1).astype(np.float32)

        # TSL spectra for the current code: (M, nbands)
        spec = tsl_df.loc[tsl_df["Code"] == code, band_cols].values.astype(np.float32)
        if spec.size == 0:
            continue

        # Compute MSE: shape (N, M)
        diff = pix[:, None, :] - spec[None, :, :]
        mse = np.mean(diff * diff, axis=2)

        # Use 1-based indexing for TSL levels
        best = np.argmin(mse, axis=1).astype(np.int16) + 1
        tsl_num[m] = best.astype(np.int8)

    return tsl_num


# -------------------------
# Coefficient tables (precompiled)
# -------------------------
@dataclass
class CoeffTables:
    """
    Precompiled coefficient tables for fast lookup.
    """
    bands: List[str]
    band_to_idx: Dict[int, int]
    lcc_params: np.ndarray       # [code, level, ang_idx, band, 4] -> a,b,c,alpha
    atc_poly: np.ndarray         # [band, 4] -> polynomial coefficients
    gmc_poly: np.ndarray         # [band, 2] -> linear coefficients
    thr_map: np.ndarray          # [code] -> max distance in meters; 0 means skip


def _build_lcc_table(lcc: pd.DataFrame, bands: List[str]) -> Tuple[np.ndarray, Dict[int, int]]:
    """
    Build the land-cover coefficient lookup table.
    """
    band_ints = [int(b) for b in bands]
    band_to_idx = {int(b): i for i, b in enumerate(band_ints)}

    max_code = int(lcc["Code"].max())
    max_level = int(lcc["Level"].max())
    n_ang = 12
    n_band = len(band_ints)

    params = np.full((max_code + 1, max_level + 1, n_ang, n_band, 4),
                     np.nan, dtype=np.float32)

    required = ["Code", "Level", "Azimuth_angle", "Wavelength", "a", "b", "c", "alpha"]
    miss = [c for c in required if c not in lcc.columns]
    if miss:
        raise ValueError(f"LCC sheet missing columns: {miss}")

    for _, r in lcc.iterrows():
        code = int(r["Code"])
        level = int(r["Level"])
        ang = int(r["Azimuth_angle"]) % 360
        wl = int(r["Wavelength"])

        if wl not in band_to_idx:
            continue

        ang_idx = ang // 30
        bidx = band_to_idx[wl]
        params[code, level, ang_idx, bidx, :] = np.array(
            [r["a"], r["b"], r["c"], r["alpha"]], dtype=np.float32
        )

    return params, band_to_idx


def _build_atc_table(atc: pd.DataFrame, bands: List[str], Atmospheric: str, Aerosol: str,
                     band_to_idx: Dict[int, int]) -> np.ndarray:
    """
    Build the atmospheric coefficient lookup table.
    """
    out = np.full((len(bands), 4), np.nan, dtype=np.float32)

    required = ["Atmospheric", "Aerosol", "Wavelength", "a", "b", "c", "d"]
    miss = [c for c in required if c not in atc.columns]
    if miss:
        raise ValueError(f"ATC sheet missing columns: {miss}")

    sub = atc[(atc["Atmospheric"] == Atmospheric) & (atc["Aerosol"] == Aerosol)]
    for _, r in sub.iterrows():
        wl = int(r["Wavelength"])
        if wl in band_to_idx:
            out[band_to_idx[wl], :] = np.array([r["a"], r["b"], r["c"], r["d"]], dtype=np.float32)

    return out


def _build_gmc_table(gmc: pd.DataFrame, bands: List[str], band_to_idx: Dict[int, int]) -> np.ndarray:
    """
    Build the geometry coefficient lookup table.
    """
    out = np.full((len(bands), 2), np.nan, dtype=np.float32)

    required = ["Wavelength", "a", "b"]
    miss = [c for c in required if c not in gmc.columns]
    if miss:
        raise ValueError(f"GMC sheet missing columns: {miss}")

    # If multiple rows exist for the same wavelength, the last one is used
    for _, r in gmc.iterrows():
        wl = int(r["Wavelength"])
        if wl in band_to_idx:
            out[band_to_idx[wl], :] = np.array([r["a"], r["b"]], dtype=np.float32)

    return out


def build_coeff_tables(excel_path: str, ae_config: Dict[str, Any]) -> Tuple[pd.DataFrame, CoeffTables]:
    """
    Load the coefficient Excel file and precompile all lookup tables.

    Parameters
    ----------
    excel_path : str
        Path to the coefficient Excel file.
    ae_config : dict
        Adjacency-effect configuration.

    Returns
    -------
    tuple
        (tsl_df, tables)
    """
    xls = pd.ExcelFile(excel_path)
    sheets = set(xls.sheet_names)

    def _pick(name_candidates: List[str]) -> str:
        for n in name_candidates:
            if n in sheets:
                return n
        raise ValueError(f"Cannot find any of sheets {name_candidates}. Available: {xls.sheet_names}")

    # Robust sheet-name matching
    tsl_name = _pick([
        "Typical_Spectral_Line",
        "Typical_Spectral_Lines",
        "Typical_Spectral_Line ",
        "Typical_Spectral_Lines ",
    ])
    lcc_name = _pick([
        "Land_Cover_Coefficient",
        "LandCover_Coefficient",
        "Land_Cover_Coefficients",
    ])
    atc_name = _pick([
        "Atmospheric_Coefficient",
        "Atmospheric_Coefficients",
    ])
    gmc_name = _pick([
        "Geometry_Coefficient",
        "Geometry_Coefficients",
    ])

    tsl = pd.read_excel(excel_path, sheet_name=tsl_name)
    lcc = pd.read_excel(excel_path, sheet_name=lcc_name)
    atc = pd.read_excel(excel_path, sheet_name=atc_name)
    gmc = pd.read_excel(excel_path, sheet_name=gmc_name)

    bands = [str(b) for b in ae_config["bands"]]
    lcc_params, band_to_idx = _build_lcc_table(lcc, bands)
    atc_poly = _build_atc_table(
        atc, bands, ae_config["Atmospheric"], ae_config["Aerosol"], band_to_idx
    )
    gmc_poly = _build_gmc_table(gmc, bands, band_to_idx)

    # Distance thresholds -> dense lookup array
    lc_thr = ae_config.get("lc_distance_threshold", {})
    max_code = max(int(k) for k in lc_thr.keys()) if lc_thr else 255
    thr_map = np.zeros((max(max_code, lcc_params.shape[0] - 1) + 1,), dtype=np.float32)
    for k, v in lc_thr.items():
        if int(k) < len(thr_map):
            thr_map[int(k)] = float(v)

    tables = CoeffTables(
        bands=bands,
        band_to_idx=band_to_idx,
        lcc_params=lcc_params,
        atc_poly=atc_poly,
        gmc_poly=gmc_poly,
        thr_map=thr_map,
    )

    return tsl, tables


# -------------------------
# Per-pixel computation (vectorized on land-use window)
# -------------------------
def _compute_gain_one_pixel(i: int, j: int,
                            mask_transform: Affine,
                            landuse2d: np.ndarray,
                            land_transform: Affine,
                            tsl_num: np.ndarray,
                            tables: CoeffTables,
                            factor: float,
                            radius_m: float,
                            aot550: float,
                            sun_z: float,
                            nodata_val: int = 15) -> np.ndarray:
    """
    Compute the gain vector for a single water pixel.

    Parameters
    ----------
    i, j : int
        Row and column index of the target water pixel in the mask raster.
    mask_transform : Affine
        Mask raster transform.
    landuse2d : np.ndarray
        Cropped land-use raster.
    land_transform : Affine
        Land-use raster transform.
    tsl_num : np.ndarray
        TSL level raster aligned to the land-use grid.
    tables : CoeffTables
        Precompiled coefficient tables.
    factor : float
        Scaling factor.
    radius_m : float
        Search radius in meters.
    aot550 : float
        Aerosol optical thickness at 550 nm.
    sun_z : float
        Solar zenith angle.
    nodata_val : int, optional
        Land-use nodata value.

    Returns
    -------
    np.ndarray
        Gain vector with shape (nbands,).
    """
    c_lon, c_lat = pixel_center_lonlat(mask_transform, i, j)

    # Convert radius from meters to an approximate geographic bounding box
    lat_buf = radius_m / 111320.0
    coslat = float(np.cos(np.deg2rad(c_lat)))
    coslat = max(coslat, 1e-6)
    lon_buf = radius_m / (111320.0 * coslat)

    min_lon, max_lon = c_lon - lon_buf, c_lon + lon_buf
    min_lat, max_lat = c_lat - lat_buf, c_lat + lat_buf

    # Convert bbox to a land-use subwindow
    r0, c0, _, _ = rowcol_from_lonlat(land_transform, min_lon, max_lat)  # upper-left
    r1, c1, _, _ = rowcol_from_lonlat(land_transform, max_lon, min_lat)  # lower-right

    rmin, rmax = (r0, r1) if r0 <= r1 else (r1, r0)
    cmin, cmax = (c0, c1) if c0 <= c1 else (c1, c0)

    H, W = landuse2d.shape
    rmin = max(rmin, 0)
    cmin = max(cmin, 0)
    rmax = min(rmax, H - 1)
    cmax = min(cmax, W - 1)

    if rmin > rmax or cmin > cmax:
        return np.zeros((len(tables.bands),), dtype=np.float32)

    sub_code = landuse2d[rmin:rmax + 1, cmin:cmax + 1]
    sub_tsl = tsl_num[rmin:rmax + 1, cmin:cmax + 1]

    # Build lon/lat grids for pixel centers in the subwindow
    rows = np.arange(rmin, rmax + 1, dtype=np.int32)
    cols = np.arange(cmin, cmax + 1, dtype=np.int32)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")

    lon_grid, lat_grid = land_transform * (cc + 0.5, rr + 0.5)
    lon_grid = lon_grid.astype(np.float64, copy=False)
    lat_grid = lat_grid.astype(np.float64, copy=False)

    dist, ang, ang_bin = equirectangular_dist_and_bearing(c_lon, c_lat, lon_grid, lat_grid)

    # Keep only neighbors within radius and exclude nodata / invalid classes
    m = dist <= radius_m
    m &= (sub_code != nodata_val)
    m &= (sub_code != 14) & (sub_code != 15)

    if not np.any(m):
        return np.zeros((len(tables.bands),), dtype=np.float32)

    # Flatten selected neighbors
    code = sub_code[m].astype(np.int32, copy=False)
    lvl = sub_tsl[m].astype(np.int32, copy=False)
    d = dist[m].astype(np.float32, copy=False)
    aidx = (ang_bin[m] // 30).astype(np.int32, copy=False)

    # Apply per-code maximum distance threshold
    thr_map = tables.thr_map
    ok_thr = np.zeros_like(d, dtype=bool)
    code_in = code < len(thr_map)
    ok_thr[code_in] = d[code_in] <= thr_map[code[code_in]]
    ok = ok_thr & (lvl > 0)  # lvl == 0 indicates unknown TSL

    if not np.any(ok):
        return np.zeros((len(tables.bands),), dtype=np.float32)

    code = code[ok]
    lvl = lvl[ok]
    d = d[ok]
    aidx = aidx[ok]

    nb = len(tables.bands)
    gain = np.zeros((nb,), dtype=np.float32)

    # Precompute atmospheric and solar-geometry coefficients per band
    atc = tables.atc_poly
    gmc = tables.gmc_poly
    atc_coff = np.ones((nb,), dtype=np.float32)
    sun_coff = np.ones((nb,), dtype=np.float32)

    for b in range(nb):
        if np.all(np.isfinite(atc[b])):
            atc_coff[b] = float(np.polyval(atc[b], aot550))
        if np.all(np.isfinite(gmc[b])):
            sun_coff[b] = float(np.polyval(gmc[b], sun_z))

    lcc_params = tables.lcc_params

    # Check lookup-table bounds
    code_ok = code < lcc_params.shape[0]
    lvl_ok = lvl < lcc_params.shape[1]
    a_ok = aidx < lcc_params.shape[2]
    ok2 = code_ok & lvl_ok & a_ok

    if not np.any(ok2):
        return np.zeros((nb,), dtype=np.float32)

    code = code[ok2]
    lvl = lvl[ok2]
    d = d[ok2]
    aidx = aidx[ok2]

    # Avoid zero distance in exponential operations
    d_safe = np.maximum(d, 1e-3).astype(np.float32, copy=False)
    exp_fac = np.exp(np.float32(factor) / d_safe).astype(np.float32, copy=False)

    for b in range(nb):
        params = lcc_params[code, lvl, aidx, b, :]  # shape: (N, 4)

        # Remove missing rows
        good = np.isfinite(params).all(axis=1)
        if not np.any(good):
            continue

        p = params[good]
        dd = d_safe[good]
        ef = exp_fac[good]

        A = p[:, 0]
        B = p[:, 1]
        C = p[:, 2]
        alpha = p[:, 3]

        land_rate = (A * np.exp(-B * (dd ** alpha)) + C) * factor
        rate_value = np.sum(land_rate, dtype=np.float64)  # use float64 for accumulation stability

        rate = np.float32(rate_value) * atc_coff[b] * sun_coff[b]
        gain[b] = rate

    return gain


# -------------------------
# Parallel driver
# -------------------------
_GLOBALS = {}


def _init_worker(globals_dict):
    """
    Initializer for multiprocessing workers.
    """
    global _GLOBALS
    _GLOBALS = globals_dict


def _worker_task(idx):
    """
    Worker task for computing one pixel gain.
    """
    i, j = idx
    return (
        i,
        j,
        _compute_gain_one_pixel(
            i, j,
            _GLOBALS["mask_transform"],
            _GLOBALS["landuse2d"],
            _GLOBALS["land_transform"],
            _GLOBALS["tsl_num"],
            _GLOBALS["tables"],
            _GLOBALS["factor"],
            _GLOBALS["radius_m"],
            _GLOBALS["aot550"],
            _GLOBALS["sun_z"],
            _GLOBALS["nodata_val"],
        )
    )


def calculate_water_gain_fast(image_path: str, mask_path: str, ae_config: Dict[str, Any],
                              out_tif: Optional[str] = None,
                              n_jobs: int = 1,
                              radius_m: float = 4000.0,
                              verbose: bool = True) -> Tuple[np.ndarray, Affine, Any]:
    """
    Main entry point for computing water adjacency-effect gain.

    Workflow
    --------
    - Read image and mask using rasterio
    - Crop the land-use raster around the mask bounding box
    - Build coefficient lookup tables from the Excel file
    - Compute TSL levels aligned to the land-use grid
    - For each water pixel, accumulate neighbor contributions band by band
    - Optionally parallelize across water pixels

    Parameters
    ----------
    image_path : str
        Path to the input image.
    mask_path : str
        Path to the water mask.
    ae_config : dict
        Configuration dictionary.
    out_tif : str, optional
        Output path for the gain raster.
    n_jobs : int, optional
        Number of worker processes. Use 1 for serial execution.
    radius_m : float, optional
        Search radius in meters.
    verbose : bool, optional
        Whether to print progress information.

    Returns
    -------
    tuple
        (gain_image, mask_transform, mask_crs)
    """
    # Read image and mask
    img, img_tr, img_crs, _ = read_raster(image_path, as_float=True)
    img = img / 10000.0  # Normalize reflectance to [0, 1]

    msk, m_tr, m_crs, _ = read_raster(mask_path, as_float=False)
    if m_crs is None or CRS.from_user_input(m_crs).to_epsg() != 4326:
        raise ValueError(f"mask must be EPSG:4326, got {m_crs}")
    mask2d = msk[0]

    # Initialize output gain array
    nb = len(ae_config["bands"])
    gain_image = np.zeros((nb, mask2d.shape[0], mask2d.shape[1]), dtype=np.float32)

    # Extract cropped land-use raster
    landuse_path = ae_config["landuse_path"]
    cal_meter = float(ae_config.get("cal_meter", 4000))
    landuse2d, land_tr, land_crs, cache_tif = extract_landuse(mask_path, landuse_path, cal_meter)

    # Load and compile coefficient tables
    ae_coff_path = ae_config["ae_coff_path"]
    tsl_df, tables = build_coeff_tables(ae_coff_path, ae_config)

    # Compute TSL levels on the land-use grid
    tsl_num = calculate_tsl_fast(
        landuse2d, land_tr, land_crs,
        tsl_df, img, img_tr, img_crs,
        [str(b) for b in ae_config["bands"]]
    )

    # List of water pixels
    water_idx = np.argwhere(mask2d != 0)
    if verbose:
        print(f"[INFO] water pixels: {len(water_idx)}")
        print(f"[INFO] landuse crop cached at: {cache_tif}")

    # Optional progress bar
    try:
        from tqdm import tqdm
        it = tqdm(water_idx, desc="Calculating water gain (fast)")
    except Exception:
        it = water_idx

    factor = float(ae_config["factor"])
    aot550 = float(ae_config["aot550"])
    sun_z = float(ae_config["sun_z"])
    nodata_val = int(ae_config.get("landuse_nodata", 15))

    if n_jobs is None or n_jobs <= 1:
        for ii, jj in it:
            ii = int(ii)
            jj = int(jj)
            gain_image[:, ii, jj] = _compute_gain_one_pixel(
                ii, jj, m_tr,
                landuse2d, land_tr, tsl_num,
                tables,
                factor=factor,
                radius_m=radius_m,
                aot550=aot550,
                sun_z=sun_z,
                nodata_val=nodata_val,
            )
    else:
        # Use multiprocessing and initialize workers with shared globals
        from concurrent.futures import ProcessPoolExecutor

        globals_dict = {
            "mask_transform": m_tr,
            "landuse2d": landuse2d,
            "land_transform": land_tr,
            "tsl_num": tsl_num,
            "tables": tables,
            "factor": factor,
            "radius_m": radius_m,
            "aot550": aot550,
            "sun_z": sun_z,
            "nodata_val": nodata_val,
        }

        with ProcessPoolExecutor(
            max_workers=int(n_jobs),
            initializer=_init_worker,
            initargs=(globals_dict,)
        ) as ex:
            for i, j, gv in ex.map(_worker_task, map(tuple, water_idx), chunksize=256):
                gain_image[:, int(i), int(j)] = gv

    # Optionally save the gain raster
    if out_tif is not None:
        write_raster(out_tif, gain_image, m_tr, m_crs, nodata=np.nan, dtype="float32")
    
    gain_image[gain_image > 0.8] = 0.0
    gain_image[gain_image < 0.0] = 0.0

    return gain_image, m_tr, m_crs


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    # Example ae_config (modify the paths before running)
    BASE_DIR = os.path.dirname(__file__)
    ae_config = {
        "landuse_path": os.path.join(BASE_DIR, "China_LC_960m_mode.tif"),
        "ae_coff_path": os.path.join(BASE_DIR, "ae_corr_coff.xlsx"),
        "factor": 4500,
        "cal_meter": 4000,
        "bands": ["485", "550", "660", "830"],
        "Atmospheric": "MidlatitudeSummer",
        "Aerosol": "Continental",
        "sun_z": 30.0,
        "aot550": 0.12,
        "lc_distance_threshold": {10: 2850, 11: 2650, 12: 3050, 13: 3250},
        "landuse_nodata": 15,
    }

    # Update these paths before running
    image_path = r"YOUR_IMAGE.tif"
    mask_path = r"YOUR_MASK.tif"
    out_gain = os.path.join(BASE_DIR, "gain_image.tif")

    if os.path.exists(image_path) and os.path.exists(mask_path):
        gain, geo, crs = calculate_water_gain_fast(
            image_path,
            mask_path,
            ae_config,
            out_tif=out_gain,
            n_jobs=1
        )
        print("Done. Gain raster saved to:", out_gain)
    else:
        print("Please set image_path and mask_path to existing files before running.")