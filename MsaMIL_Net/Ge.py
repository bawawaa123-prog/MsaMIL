import os
import gc
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
try:
    from concurrent.futures.process import BrokenProcessPool  # Python 3.x location
except ImportError:
    class BrokenProcessPool(Exception):
        pass
from pathlib import Path
from PIL import Image, ImageFile

# 允许处理超大图（禁用 PIL 像素数限制）
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 日志：控制台 + 文件 logs/Ge_internal.log
LOG_DIR = Path(__file__).parent / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / 'Ge_internal.log'

logger = logging.getLogger("Ge")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    fh = logging.FileHandler(LOG_FILE, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)

# 内存信息（psutil 优先，回退 resource）
try:
    import psutil  # type: ignore
    def _mem_info():
        p = psutil.Process()
        rss = p.memory_info().rss
        return f"RSS={rss/1024/1024:.1f}MB"
except Exception:
    try:
        import resource  # type: ignore
        def _mem_info():
            kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # Linux ru_maxrss 单位KB，macOS 单位Bytes；统一转MB近似显示
            mb = kb/1024 if kb > 10_000 else kb/1024/1024
            return f"ru_maxrss≈{mb:.1f}MB"
    except Exception:
        def _mem_info():
            return "mem=n/a"

# 仅保留三个尺度裁剪与带倍率命名，其他一概不需要
PATCH_SIZES = {
    '20x': 512,
    '10x': 1024,
    '5x': 2048,
}

# 每写入多少个 patch 主动触发一次 GC，以更快释放内存
GC_EVERY_PATCHES = 200

# 背景过滤配置：使用灰度直方图，统计中间亮度比例，过滤极暗/极亮为主的背景块
FILTER_BACKGROUND = True
GRAY_LOW = 10        # 低于此视为“接近黑”
GRAY_HIGH = 245      # 高于此视为“接近白”
MIN_TISSUE_FRAC = 0.20  # 中间亮度占比阈值，低于此认为是背景，跳过

def process_image(img_path: Path, output_dir: Path) -> None:
    stem = img_path.stem
    t0 = time.perf_counter()
    img = None
    try:
        with Image.open(img_path) as im_src:
            # 不对整图做 convert('RGB')，避免整幅拷贝占用大量内存
            W, H = im_src.size

            wsi_dir = output_dir / stem
            wsi_dir.mkdir(parents=True, exist_ok=True)

            total_patches = 0

            for scale, dk in PATCH_SIZES.items():
                if W < dk or H < dk:
                    continue
                scale_dir = wsi_dir / scale
                scale_dir.mkdir(parents=True, exist_ok=True)

                idx = 0  # 仅对保留的 patch 递增
                ts = time.perf_counter()
                local_gc_cnt = 0
                skipped_bg = 0
                filtered_detail = []  # 记录每个被过滤patch的坐标

                for top in range(0, H - dk + 1, dk):
                    for left in range(0, W - dk + 1, dk):
                        cx = left + dk // 2
                        cy = top + dk // 2
                        region = im_src.crop((left, top, left + dk, top + dk))

                        # 背景过滤
                        filtered = False
                        if FILTER_BACKGROUND:
                            gray = region.convert('L')
                            hist = gray.histogram()
                            total_px = dk * dk
                            mid = sum(hist[GRAY_LOW:GRAY_HIGH + 1])
                            try:
                                gray.close()
                            except Exception:
                                pass
                            del gray

                            if total_px > 0 and (mid / total_px) < MIN_TISSUE_FRAC:
                                try:
                                    region.close()
                                except Exception:
                                    pass
                                del region
                                skipped_bg += 1
                                filtered_detail.append((cx, cy))
                                local_gc_cnt += 1
                                if local_gc_cnt % GC_EVERY_PATCHES == 0:
                                    gc.collect()
                                filtered = True
                        if filtered:
                            continue

                        # 保存 patch
                        patch = region.convert('RGB') if region.mode != 'RGB' else region
                        fname = f"{stem}_{scale}_{cx}_{cy}_{idx:04d}.png"
                        patch.save(scale_dir / fname, format="PNG", compress_level=0)

                        # 释放对象
                        try:
                            patch.close()
                        except Exception:
                            pass
                        try:
                            region.close()
                        except Exception:
                            pass
                        del patch
                        del region

                        local_gc_cnt += 1
                        if local_gc_cnt % GC_EVERY_PATCHES == 0:
                            gc.collect()
                        idx += 1

                total_patches += idx
                logger.info(f"{stem}: {scale} 完成，保留 {idx} 块，跳过背景 {skipped_bg} 块 | {time.perf_counter()-ts:.2f}s | {_mem_info()}")
                logger.info(f"{stem}: {scale} 被过滤掉的patch数量: {skipped_bg}")
                gc.collect()

            logger.info(f"{stem}: 全部尺度切割完成，总计 {total_patches} 块 | {time.perf_counter()-t0:.2f}s | {_mem_info()}\n")
    except MemoryError as e:
        logger.error(f"{stem}: MemoryError during processing | {_mem_info()} | {e}")
        raise
    except Exception as e:
        logger.error(f"{stem}: Failed with error: {e}")
        raise
    finally:
        if img is not None:
            try:
                img.close()
            except Exception:
                pass
        gc.collect()

def _collect_images(input_dir: Path):
    patterns = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
    files = []
    for pat in patterns:
        files.extend(input_dir.glob(pat))
    return sorted(files)


def _expected_patches(W: int, H: int, dk: int) -> int:
    if W < dk or H < dk:
        return 0
    # 步长=dk 的不重叠网格，计数为整除数
    return (W // dk) * (H // dk)


def _is_already_processed(img_path: Path, output_dir: Path) -> bool:
    """判断该图像是否已按三尺度切割完成（逐尺度核对期望块数）。"""
    try:
        with Image.open(img_path) as im:
            W, H = im.size
    except Exception:
        return False

    stem = img_path.stem
    for scale, dk in PATCH_SIZES.items():
        exp = _expected_patches(W, H, dk)
        if exp == 0:
            continue
        scale_dir = output_dir / stem / scale
        if not scale_dir.exists():
            return False
        # 仅统计本脚本命名规范下的 PNG 文件
        actual = len(list(scale_dir.glob(f"{stem}_{scale}_*.png")))
        if actual < exp:
            return False
    return True


def _process_parallel(todo_files, output_dir: Path) -> None:
    # 起始并行数：CPU 核心数-1，不超过待处理数量
    cpu_cnt = os.cpu_count() or 1
    workers = 4

    while workers > 1:
        try:
            logger.info(f"Parallel processing with {workers} workers… | {_mem_info()}")
            with ProcessPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(process_image, img_path, output_dir) for img_path in todo_files]
                for _ in as_completed(futures):
                    pass
            return
        except (BrokenProcessPool, MemoryError, OSError) as e:
            logger.warning(f"Pool failed at workers={workers} ({e}). Reducing and retrying… | {_mem_info()}")
            workers = max(1, workers // 2)

    # 回退到串行
    logger.info("Falling back to sequential processing…")
    for img_path in todo_files:
        process_image(img_path, output_dir)


def batch_process_images(input_dir: str, output_dir: str) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    files = _collect_images(input_dir)
    logger.info(f"Found {len(files)} images in {input_dir}")

    if not files:
        return

    # 先跳过已完成的图像，仅对未完成的进行处理
    todo = []
    skipped = 0
    for img_path in files:
        if _is_already_processed(img_path, output_dir):
            skipped += 1
        else:
            todo.append(img_path)

    logger.info(f"Skip {skipped} already done | Todo {len(todo)}")

    if not todo:
        logger.info("All images already processed.")
        return

    if len(todo) == 1:
        process_image(todo[0], output_dir)
        return

    _process_parallel(todo, output_dir)

if __name__ == "__main__":
    # 仅三个尺度裁剪，PNG 无压缩保存，并行固定 13 进程
    batch_process_images(
        input_dir="/mnt/nas/ljh/MsaMIL_Net_Data/images",
        output_dir="/mnt/nas/ljh/MsaMIL_Net_Data/patches_grid_xin",
    )