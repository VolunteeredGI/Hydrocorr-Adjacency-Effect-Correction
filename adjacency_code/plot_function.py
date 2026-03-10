import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from hydrocorr.core.raster import read_image


def show_gain_image(image_gain, title_prefix="Band", cbar_label="Gain", cmap_name="jet"):
    """
    可视化 gain 图像（多波段）：0 显示为白色，其余按 cmap 显示。

    Parameters
    ----------
    image_gain : np.ndarray or str
        形状支持 (B, H, W) 或 (H, W)。
    title_prefix : str
        子图标题前缀，如 "Band"。
    cbar_label : str
        colorbar 标签。
    cmap_name : str
        基础 colormap 名称，默认 'jet'。
    """
    # 如果image_gain为str,则读取图像数据
    if not isinstance(image_gain, np.ndarray):
        arr, _, _ = read_image(image_gain)
    else:
        arr = np.asarray(image_gain)

    # 统一成 (B, H, W)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim != 3:
        raise ValueError(f"image_gain must be 2D or 3D, got shape={arr.shape}")

    B, H, W = arr.shape

    # ---- 自定义 colormap：最小值(0)为白色 ----
    base_cmap = plt.get_cmap(cmap_name)
    new_cmap = base_cmap(np.arange(base_cmap.N))
    new_cmap[0] = [1, 1, 1, 1]  # RGBA 白色
    custom_cmap = mcolors.ListedColormap(new_cmap)

    # ---- 子图布局：优先 2x2，其他情况自适应 ----
    if B == 4:
        nrows, ncols = 2, 2
    else:
        ncols = int(np.ceil(np.sqrt(B)))
        nrows = int(np.ceil(B / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10))
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        if i >= B:
            ax.axis("off")
            continue

        rate = arr[i, :, :]

        # vmin 固定 0；vmax 取当前波段最大值并保留 1 位小数
        vmin = 0
        vmax = np.round(np.nanmax(rate), 1)
        if not np.isfinite(vmax) or vmax <= vmin:
            vmax = vmin + 1e-6  # 防止全 0 或异常导致 imshow 报错

        cax = ax.imshow(rate, cmap=custom_cmap, vmin=vmin, vmax=vmax)

        ax.set_title(f"{title_prefix} {i + 1}")
        ax.axis("off")

        cbar = fig.colorbar(
            cax, ax=ax, orientation="vertical",
            fraction=0.1, pad=-0.05, shrink=0.5, aspect=25
        )
        cbar.set_label(cbar_label)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = (r"E:\2_Code\total_code\hydrocorr\test\20241107-巢湖"
                  r"\RasterData\HJ2B_CCD3_E117.8_N31.5_20241106_L1A0001248841_SR.tif")
    image_xml_path = (r"E:\2_Code\total_code\hydrocorr\test\20241107-巢湖"
                      r"\RasterData\HJ2B_CCD3_E117.8_N31.5_20241106_L1A0001248841.xml")
    image_mask_path = (r"E:\2_Code\total_code\hydrocorr\test\20241107-巢湖"
                       r"\RasterData\HJ2B_CCD3_E117.8_N31.5_20241106_L1A0001248841_SR_water_mask_final.tif")
    output_path = image_path.replace('.tif', '_ae_corr.tif')
    output_gain_path = image_path.replace('.tif', '_ae_gain.tif')
    show_gain_image(output_gain_path, title_prefix="Band", cbar_label="Gain", cmap_name="jet")

