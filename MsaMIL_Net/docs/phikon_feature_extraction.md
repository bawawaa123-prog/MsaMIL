# Phikon-v2 特征批量提取脚本

`tools/extract_phikon_features.py` 会读取 `data/all_data.csv` 中的 `slide_id`，遍历每个倍率的 Patch 文件夹（命名格式 `slide_x_y.png`），并完成以下步骤：

- 聚合 20x/10x/5x（默认）等所有倍率的图像块。
- 依据 `x + patch_size` 和 `y + patch_size` 推断整幅 WSI 的宽高，并将每个 patch 的中心点归一化到 `[0, 1]`。
- 按 `(x, y, scale)` 排序后批量送入 `owkin/phikon-v2`，提取 CLS (1024-D) 特征。
- 为每个 WSI 生成 `data/features/<slide_id>.pt`、`data/features/<slide_id>_coords.npy` 与 `data/features/<slide_id>_scales.npy`：
  - `.pt`：shape `[N, 1024]`，CLS 特征。
  - `_coords.npy`：shape `[N, 2]`，patch 中心坐标归一化到 `[0, 1]`。
  - `_scales.npy`：shape `[N]`，记录每个 patch 的倍率，编码为 `5x→0`、`10x→1`、`20x→2`（若配置了其他倍率，自动回退到配置顺序）。

## 依赖

将下列依赖加入环境（`requirements_phikon.txt` 已列出版本）：

- `torch`
- `transformers`
- `pandas`
- `numpy`
- `Pillow`
- `tqdm`

```bash
pip install -r requirements_phikon.txt
```

## 使用示例

```bash
python tools/extract_phikon_features.py \
  --csv data/all_data.csv \
  --scale 20x:512:/mnt/nas/ljh/data/results/patches_20x \
  --scale 10x:1024:/mnt/nas/ljh/data/results/patches_10x \
  --scale 5x:2048:/mnt/nas/ljh/data/results/patches_5x \
  --output-dir data/features \
  --batch-size 32 \
  --skip-existing
```

### 常用参数

- `--manual-resize / --no-manual-resize`：是否在送入模型前统一 resize 到 processor 建议大小（默认开启）。
- `--limit N`：仅处理前 N 个 slide，方便冒烟测试。
- `--device`：显式选择 `cpu`、`cuda` 或 `cuda:1` 等。

脚本会输出进度条，若某个 slide 缺失任一倍率的 patch，则只处理存在的部分并给出警告。`_coords.npy` 中坐标来自 patch 中心点，相对于每个 slide 的推断宽/高做归一化；`_scales.npy` 则保证与 `.pt` 顺序完全匹配，可在自定义 Dataset 中区分不同倍率。