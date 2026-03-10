import os
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.transform import Affine
from rasterio.enums import Resampling
from pyproj import CRS, Transformer
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from osgeo import gdal


def fast_mask_bbox_wgs84(mask, transform):
    """
    从 mask!=0 的像元中，快速提取 WGS84 bbox（min_lon, max_lon, min_lat, max_lat）
    仅转换 4 个极值像元，避免全量 xy 转换
    """
    ys, xs = np.where(mask != 0)
    if ys.size == 0:
        raise ValueError("mask 中没有有效像元（mask!=0）")

    # 像元索引极值
    row_min = ys.min()
    row_max = ys.max()
    col_min = xs.min()
    col_max = xs.max()

    # 只转换 4 个角点（像元中心）
    # 注意：row 是 y，col 是 x
    lon_min, lat_max = rasterio.transform.xy(
        transform, row_min, col_min, offset="center"
    )
    lon_max, lat_min = rasterio.transform.xy(
        transform, row_max, col_max, offset="center"
    )

    min_lon = float(min(lon_min, lon_max))
    max_lon = float(max(lon_min, lon_max))
    min_lat = float(min(lat_min, lat_max))
    max_lat = float(max(lat_min, lat_max))

    return min_lon, max_lon, min_lat, max_lat


def extract_landuse(mask_path: str, landuse_path: str, cal_meter: float):
    """
    根据 WGS84 掩膜(mask)的有效区域边界，向四周扩展 cal_meter（米），
    在 landuse 栅格中裁剪对应区域，返回 landuse, geo3, pro3，并把裁剪结果写入缓存 tif。

    输出文件路径：
      ../hydrocorr/Cache/{basename(landuse_path)}_{basename(mask_path)}.tif

    返回：
      landuse: np.ndarray，形状为 (bands, rows, cols) 或 (rows, cols) 取决于源数据波段数
      geo3:    rasterio transform (Affine)
      pro3:    CRS（rasterio.crs.CRS）
    """

    # ---------------------------
    # 0) Cache path
    # ---------------------------
    cache_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "Cache"))
    os.makedirs(cache_dir, exist_ok=True)

    landuse_base = os.path.splitext(os.path.basename(landuse_path))[0]
    mask_base = os.path.splitext(os.path.basename(mask_path))[0]
    out_tif = os.path.join(cache_dir, f"{landuse_base}_{mask_base}.tif")

    # ---------------------------
    # 1) Read mask (WGS84)
    # ---------------------------
    with rasterio.open(mask_path) as mds:
        mask_crs = mds.crs
        if mask_crs is None:
            raise ValueError("mask_path 没有 CRS 信息。请确保掩膜是 WGS84(EPSG:4326) 且写入了 CRS。")

        # 你说明 geo 是 WGS84，这里做一个硬检查（更稳健）
        if CRS.from_user_input(mask_crs).to_epsg() != 4326:
            raise ValueError(f"mask CRS 不是 EPSG:4326（WGS84），而是: {mask_crs}")

        mask = mds.read(1)  # 掩膜通常单波段
        m_transform = mds.transform

    # 有效区域：非 0 像元（如你的 mask 是 0/1 或 0/255 均可）
    ys, xs = np.where(mask != 0)
    if ys.size == 0:
        raise ValueError("掩膜中没有发现有效像元（mask!=0）。")

    # ---------------------------
    # 2) Mask bbox in WGS84 (lon/lat)
    # ---------------------------

    # 像元中心坐标
    # lon, lat = rasterio.transform.xy(m_transform, ys, xs, offset="center")
    # lon = np.asarray(lon, dtype=float)
    # lat = np.asarray(lat, dtype=float)
    #
    # min_lon, max_lon = float(lon.min()), float(lon.max())
    # min_lat, max_lat = float(lat.min()), float(lat.max())
    min_lon, max_lon, min_lat, max_lat = fast_mask_bbox_wgs84(mask, m_transform)

    # ---------------------------
    # 3) Buffer bbox by cal_meter (meters) in a local meter-based projection
    #    用局部 AEQD（方位等距）以 bbox 中心为投影中心，便于“按米扩展”
    # ---------------------------
    c_lon = (min_lon + max_lon) / 2.0
    c_lat = (min_lat + max_lat) / 2.0

    aeqd = CRS.from_proj4(
        f"+proj=aeqd +lat_0={c_lat} +lon_0={c_lon} +datum=WGS84 +units=m +no_defs"
    )

    tf_wgs84_to_aeqd = Transformer.from_crs("EPSG:4326", aeqd, always_xy=True)
    tf_aeqd_to_wgs84 = Transformer.from_crs(aeqd, "EPSG:4326", always_xy=True)

    # bbox 四角转到米制坐标
    corners_lonlat = [
        (min_lon, min_lat),
        (min_lon, max_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
    ]

    cx, cy = tf_wgs84_to_aeqd.transform(
        [p[0] for p in corners_lonlat],
        [p[1] for p in corners_lonlat]
    )
    cx = np.asarray(cx, dtype=float)
    cy = np.asarray(cy, dtype=float)

    min_x, max_x = cx.min(), cx.max()
    min_y, max_y = cy.min(), cy.max()

    # 按米扩展
    cal_meter = cal_meter * 1.2
    min_x -= cal_meter
    max_x += cal_meter
    min_y -= cal_meter
    max_y += cal_meter

    # 扩展后的 bbox（在 WGS84 中）
    buf_lon, buf_lat = tf_aeqd_to_wgs84.transform(
        [min_x, min_x, max_x, max_x],
        [min_y, max_y, min_y, max_y]
    )
    buf_lon = np.asarray(buf_lon, dtype=float)
    buf_lat = np.asarray(buf_lat, dtype=float)

    buf_min_lon, buf_max_lon = float(buf_lon.min()), float(buf_lon.max())
    buf_min_lat, buf_max_lat = float(buf_lat.min()), float(buf_lat.max())

    # ---------------------------
    # 4) Crop landuse by buffered bbox
    #    做法：将 buffered WGS84 bbox -> landuse CRS，然后 window 裁剪读取
    # ---------------------------
    with rasterio.open(landuse_path) as lds:
        landuse_crs = lds.crs
        if landuse_crs is None:
            raise ValueError("landuse_path 没有 CRS 信息。请先为 landuse 写入正确 CRS。")

        tf_wgs84_to_lu = Transformer.from_crs("EPSG:4326", landuse_crs, always_xy=True)

        # bbox 四角转到 landuse CRS（投影/经纬度都可）
        lu_x, lu_y = tf_wgs84_to_lu.transform(
            [buf_min_lon, buf_min_lon, buf_max_lon, buf_max_lon],
            [buf_min_lat, buf_max_lat, buf_min_lat, buf_max_lat]
        )
        lu_x = np.asarray(lu_x, dtype=float)
        lu_y = np.asarray(lu_y, dtype=float)

        crop_left, crop_right = float(lu_x.min()), float(lu_x.max())
        crop_bottom, crop_top = float(lu_y.min()), float(lu_y.max())

        # 生成 window，并与影像范围相交（避免越界）
        win = from_bounds(crop_left, crop_bottom, crop_right, crop_top, transform=lds.transform)
        win = win.round_offsets().round_lengths()

        # 与影像范围相交（clip）
        full_win = rasterio.windows.Window(0, 0, lds.width, lds.height)
        win = win.intersection(full_win)

        if win.width <= 0 or win.height <= 0:
            raise ValueError("扩展后的 bbox 与 landuse 影像无交集，无法裁剪。")

        # 读取裁剪区域（保持原值，不重采样）
        landuse = lds.read(window=win)  # shape: (bands, h, w)
        geo3 = rasterio.windows.transform(win, lds.transform)
        pro3 = lds.crs

        # 写入缓存 tif（用于调试）
        profile = lds.profile.copy()
        profile.update({
            "height": landuse.shape[1],
            "width": landuse.shape[2],
            "transform": geo3,
            "crs": pro3,
            "tiled": True,
            "compress": "lzw",
            "bigtiff": "yes",
        })

        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(landuse)

    # 你若希望单波段直接返回 (h,w)，可在这里 squeeze
    if landuse.shape[0] == 1:
        landuse = landuse[0]

    return landuse, geo3, pro3


def to_wkt(proj):
    """
    将不同形式的 CRS / projection 转成 GDAL 可用的 WKT 字符串
    """
    if proj is None:
        return None

    # 已经是 WKT 字符串
    if isinstance(proj, str):
        return proj

    # rasterio.crs.CRS
    if hasattr(proj, "to_wkt"):
        return proj.to_wkt()

    # pyproj CRS
    if hasattr(proj, "srs"):
        return proj.srs

    raise TypeError(f"Unsupported projection type: {type(proj)}")


def match_image_to_landuse(
    image_data: np.ndarray,
    img_geo: tuple,
    img_pro: str,
    landuse: np.ndarray,
    land_geo: tuple,
    land_pro: str,
    out_tif: str = None,
    resampling: str = "average",   # 连续变量：average/bilinear；分类：nearest/mode
    src_nodata=None,
    dst_dtype=np.float32,
):
    """
    将 image_data 重投影/重采样到 landuse(WGS84经纬度) 的网格上（同 CRS + 同 transform + 同 shape）。

    返回：
      img_match, dst_geo, dst_pro
    """

    gdal.UseExceptions()
    gdal.SetConfigOption("NUM_THREADS", "ALL_CPUS")

    # 1) 目标网格（来自 landuse 数组与其 geo/pro）
    if landuse.ndim == 2:
        dst_h, dst_w = landuse.shape
    elif landuse.ndim == 3:
        dst_h, dst_w = landuse.shape[-2], landuse.shape[-1]
    else:
        raise ValueError("landuse must be 2D (H,W) or 3D (B,H,W).")

    dst_geo = land_geo
    if hasattr(dst_geo, "a") and hasattr(dst_geo, "e") and hasattr(dst_geo, "c") and hasattr(dst_geo, "f"):
        # rasterio Affine -> GDAL geotransform tuple
        dst_geo = (dst_geo.c, dst_geo.a, dst_geo.b, dst_geo.f, dst_geo.d, dst_geo.e)
    dst_pro = land_pro
    dst_pro = to_wkt(dst_pro)

    # 2) 规范化 image_data 维度
    if image_data.ndim == 2:
        src_arr = image_data[np.newaxis, ...]
    elif image_data.ndim == 3:
        src_arr = image_data
    else:
        raise ValueError("image_data must be (H,W) or (B,H,W).")

    bands, src_h, src_w = src_arr.shape

    # 3) 选择重采样算法
    resampling = resampling.lower()
    rs_map = {
        "nearest": gdal.GRA_NearestNeighbour,
        "bilinear": gdal.GRA_Bilinear,
        "average": gdal.GRA_Average,
        "mode": gdal.GRA_Mode,
    }
    if resampling not in rs_map:
        raise ValueError(f"Unsupported resampling={resampling}. Choose from {list(rs_map)}")
    gra = rs_map[resampling]

    # 4) 用 MEM 构建源/目标数据集
    mem = gdal.GetDriverByName("MEM")

    # 源 ds（用 float32 写入更稳：average/bilinear 需要）
    src_ds = mem.Create("", src_w, src_h, bands, gdal.GDT_Float32)
    src_ds.SetGeoTransform(img_geo)
    src_ds.SetProjection(img_pro)

    for b in range(bands):
        sb = src_ds.GetRasterBand(b + 1)
        if src_nodata is not None:
            sb.SetNoDataValue(float(src_nodata))
        sb.WriteArray(src_arr[b].astype(np.float32, copy=False))

    # 目标 ds（严格按 landuse 网格）
    dst_ds = mem.Create("", dst_w, dst_h, bands, gdal.GDT_Float32)
    dst_ds.SetGeoTransform(dst_geo)
    dst_ds.SetProjection(dst_pro)

    # 5) 重投影/重采样到 landuse 网格
    gdal.ReprojectImage(src_ds, dst_ds, img_pro, dst_pro, gra)

    # 6) 读回 numpy
    out = np.stack([dst_ds.GetRasterBand(i + 1).ReadAsArray() for i in range(bands)], axis=0)
    out = out.astype(dst_dtype, copy=False)

    if image_data.ndim == 2:
        out = out[0]

    # 7) 可选写出
    if out_tif is not None:
        os.makedirs(os.path.dirname(out_tif), exist_ok=True)
        gtiff = gdal.GetDriverByName("GTiff")
        out_ds = gtiff.Create(
            out_tif, dst_w, dst_h, bands, gdal.GDT_Float32,
            options=["TILED=YES", "COMPRESS=LZW", "BIGTIFF=YES"]
        )
        out_ds.SetGeoTransform(dst_geo)
        out_ds.SetProjection(dst_pro)
        for b in range(bands):
            ob = out_ds.GetRasterBand(b + 1)
            if src_nodata is not None:
                ob.SetNoDataValue(float(src_nodata))
            ob.WriteArray(out[b] if image_data.ndim == 3 else out)
        out_ds.FlushCache()
        out_ds = None

    src_ds = None
    dst_ds = None

    return out, dst_geo, dst_pro


EARTH_R = 6371000.0  # meters


def _pixel_center_lonlat(geo, row, col):
    """
    GDAL geotransform -> pixel center lon/lat for (row, col)
    geo = (x0, px_w, rot1, y0, rot2, px_h)
    """

    x0, px_w, rot1, y0, rot2, px_h = geo
    lon = x0 + (col + 0.5) * px_w + (row + 0.5) * rot1
    lat = y0 + (col + 0.5) * rot2 + (row + 0.5) * px_h
    return float(lon), float(lat)


def _rowcol_from_lonlat_northup(geo, lon, lat):
    """
    lon/lat -> (row, col) for north-up rasters (rot terms ~ 0).
    """
    if hasattr(geo, "a") and hasattr(geo, "e") and hasattr(geo, "c") and hasattr(geo, "f"):
        # rasterio Affine -> GDAL geotransform tuple
        geo = (geo.c, geo.a, geo.b, geo.f, geo.d, geo.e)
    x0, px_w, rot1, y0, rot2, px_h = geo
    if abs(rot1) > 1e-12 or abs(rot2) > 1e-12:
        raise ValueError("landuse geotransform has rotation; use a general affine inverse if needed.")

    col = (lon - x0) / px_w
    row = (lat - y0) / px_h  # px_h usually negative
    return int(np.floor(row)), int(np.floor(col))


def landuse_points_within_radius(
    mask_geo,
    i, j,
    landuse, landuse_geo,
    radius_m=4000.0,
    nodata=None
):
    """
    从 mask 像元(i,j)获取中心经纬度，提取 landuse 中以该点为中心、半径 radius_m 的所有像元中心点，
    返回 Nx3 数组: [lon, lat, value].

    参数
    - mask_geo: mask 的 geotransform (WGS84)
    - i, j: mask 像元行列
    - landuse: 2D ndarray (H,W) 或 3D (B,H,W)（若 3D 取 band0）
    - landuse_geo: landuse 的 geotransform (WGS84)
    - radius_m: 半径（米），默认 4000
    - nodata: landuse 的 nodata（可选）

    返回
    - out: np.ndarray shape (N, 3), columns = lon, lat, value
    """
    # 将Affine转为tuple
    if hasattr(landuse_geo, "a") and hasattr(landuse_geo, "e") and hasattr(landuse_geo, "c") and hasattr(landuse_geo, "f"):
        # rasterio Affine -> GDAL geotransform tuple
        landuse_geo = (landuse_geo.c, landuse_geo.a, landuse_geo.b, landuse_geo.f, landuse_geo.d, landuse_geo.e)


    # 1) mask 像元中心经纬度
    c_lon, c_lat = _pixel_center_lonlat(mask_geo, i, j)

    # 2) 用近似把 4km 转成经纬度包络（先裁剪 landuse 窗口以提速）
    #    纬度方向：~111.32 km per degree
    lat_buf = radius_m / 111320.0
    #  经度方向：随纬度缩放
    coslat = np.cos(np.deg2rad(c_lat))
    coslat = max(coslat, 1e-6)
    lon_buf = radius_m / (111320.0 * coslat)

    min_lon, max_lon = c_lon - lon_buf, c_lon + lon_buf
    min_lat, max_lat = c_lat - lat_buf, c_lat + lat_buf

    # 3) landuse 数组取二维
    if landuse.ndim == 3:
        lu = landuse[0]
    else:
        lu = landuse
    H, W = lu.shape

    # 4) bbox -> landuse row/col window（假设 north-up, 无旋转）
    r0, c0 = _rowcol_from_lonlat_northup(landuse_geo, min_lon, max_lat)  # 左上
    r1, c1 = _rowcol_from_lonlat_northup(landuse_geo, max_lon, min_lat)  # 右下

    rmin, rmax = sorted([r0, r1])
    cmin, cmax = sorted([c0, c1])

    # clip to raster bounds
    rmin = max(rmin, 0)
    cmin = max(cmin, 0)
    rmax = min(rmax, H - 1)
    cmax = min(cmax, W - 1)

    if rmin > rmax or cmin > cmax:
        return np.zeros((0, 3), dtype=float)

    sub = lu[rmin:rmax + 1, cmin:cmax + 1]

    # 5) 生成该窗口内所有像元中心经纬度（向量化）
    rows = np.arange(rmin, rmax + 1)
    cols = np.arange(cmin, cmax + 1)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")

    x0, px_w, rot1, y0, rot2, px_h = landuse_geo
    if abs(rot1) > 1e-12 or abs(rot2) > 1e-12:
        raise ValueError("landuse geotransform has rotation; add general affine support if you need it.")

    lon_grid = x0 + (cc + 0.5) * px_w
    lat_grid = y0 + (rr + 0.5) * px_h  # px_h negative is fine

    # 6) 计算距离并筛选圆内点（快速：局部等矩形近似 / equirectangular）
    dlon = np.deg2rad(lon_grid - c_lon)
    dlat = np.deg2rad(lat_grid - c_lat)
    lat0 = np.deg2rad(c_lat)
    x = dlon * np.cos(lat0)
    y = dlat
    dist = EARTH_R * np.sqrt(x * x + y * y)

    m = dist <= radius_m

    # nodata 过滤（可选）
    if nodata is not None:
        m &= (sub != nodata)

    if not np.any(m):
        return np.zeros((0, 3), dtype=float)

    # 该 mask 中心点所在的 landuse 像元 row/col
    r0, c0 = _rowcol_from_lonlat_northup(landuse_geo, c_lon, c_lat)

    # 边界保护
    if 0 <= r0 < H and 0 <= c0 < W:
        # landuse 像元中心
        x0, px_w, _, y0, _, px_h = landuse_geo
        lu_lon0 = x0 + (c0 + 0.5) * px_w
        lu_lat0 = y0 + (r0 + 0.5) * px_h
        lu_val0 = lu[r0, c0]

        center_point = np.array([[lu_lon0, lu_lat0, lu_val0]], dtype=float)
    else:
        center_point = None

    # 7) 组装 Nx3
    out_lon = lon_grid[m].astype(np.float64)
    out_lat = lat_grid[m].astype(np.float64)
    out_val = sub[m]
    out = np.column_stack([out_lon, out_lat, out_val])

    if center_point is not None:
        # 移除与中心点重复的项（避免重复）
        same_center = (
                np.isclose(out[:, 0], center_point[0, 0]) &
                np.isclose(out[:, 1], center_point[0, 1])
        )
        out = out[~same_center]

        # 将中心点插到第一行
        out = np.vstack([center_point, out])

    return out


def compute_dist_angle_bins_from_out(out, skip_vals=(14, 15)):
    """
    out: np.ndarray (N,3), columns = [lon, lat, val]
         out[0] must be the center point.

    处理：
    - 跳过 out[0]
    - 若 val in skip_vals(14/15) 则跳过
    - 计算每点到中心点的距离(米)与方位角(北为0顺时针)
    - 角度归一化到30度倍数（0..330）

    返回：
      res: np.ndarray (M,6)
           [lon, lat, val, dist_m, angle_deg, angle_bin]
    """
    out = np.asarray(out)
    if out.ndim != 2 or out.shape[1] != 3:
        raise ValueError("out must have shape (N, 3) with columns [lon, lat, val].")
    if out.shape[0] < 2:
        return np.zeros((0, 6), dtype=float)

    c_lon, c_lat, _ = out[0]

    pts = out[1:]
    lon = pts[:, 0].astype(np.float64)
    lat = pts[:, 1].astype(np.float64)
    val = pts[:, 2]

    # 过滤 14/15
    mask = np.ones(len(pts), dtype=bool)
    for sv in skip_vals:
        mask &= (val != sv)

    if not np.any(mask):
        return np.zeros((0, 6), dtype=float)

    lon = lon[mask]
    lat = lat[mask]
    val = val[mask].astype(np.float64)

    # 距离：局部等矩形近似（4km量级误差可忽略）
    lat0 = np.deg2rad(c_lat)
    dlon = np.deg2rad(lon - c_lon)
    dlat = np.deg2rad(lat - c_lat)

    x_east = EARTH_R * dlon * np.cos(lat0)  # 东向米
    y_north = EARTH_R * dlat               # 北向米
    dist_m = np.sqrt(x_east**2 + y_north**2)

    # 方位角：北为0，顺时针为正
    # atan2(east, north) -> 0=北, 90=东
    ang = np.degrees(np.arctan2(x_east, y_north))
    ang = (ang + 360.0) % 360.0

    # 归一化到 30 的倍数：0..330
    # 采用四舍五入到最近的30度扇区
    ang_bin = (np.round(ang / 30.0) * 30.0) % 360.0
    ang_bin = ang_bin.astype(np.int32)
    ang_bin[ang_bin == 360] = 0  # 理论上不会出现，但保险

    # 你要求范围【0-330】，因此把 360 映射到 0 已满足；并确保不会输出 360
    # 组装输出
    res = np.column_stack([lon, lat, val, dist_m, ang, ang_bin]).astype(np.float64)
    return res


# 示例调用：
# landuse, geo3, pro3 = extract_landuse(mask_path, landuse_path, cal_meter)
