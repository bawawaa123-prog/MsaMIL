# NMFEM+IAAM 端到端训练流程与推荐配置

> 本文档只用于解释当前代码实现，不参与运行。

## 1. 整体训练流程

1. **读取标签与划分 WSI**  (`load_labels`, `stratified_split`)
   - 从 `TrainConfig.label_file` 读入 CSV（`all_data.csv`）。
   - 构建 `labels_map: {wsi_id -> label_idx}` 和 `class_names`。
   - 按类别分层划分为 `train_ids` 和 `val_ids`，保证类比例接近。

2. **按 WSI 建立 patch 索引** (`WSIPatchDataset`)
   - 对每个 `wsi_id`，在 `patch_root/patches_{20x,10x,5x}/wsi_id` 下扫描 `*.png`：
     - 解析文件名中的 `x, y, patch_size, scale` 信息。
     - 记录为字典：`{"path", "scale", "x", "y", "patch_size"}`。
   - 对每个 WSI 的补丁列表按 `(x, y, scale)` 排序，保证空间顺序一致（`sort_order = xy`）。
   - 同时统计 `min_x, max_x, min_y, max_y` 作为坐标归一化的范围。
   - `__getitem__` 返回：
     ```python
     {
       'wsi_id': wid,
       'label': label_idx,
       'patches': List[patch_meta_dict],
       'wsi_info': {min_x, max_x, min_y, max_y},
     }
     ```

3. **DDP 采样器：按 WSI 级 shuffle** (`ChunkedWSISampler`)
   - 每个 index 对应一个 WSI；多卡场景下按 `idx % world_size` 分配到各个 rank。
   - 每个 epoch 调用 `set_epoch(epoch)` 后，在 `__iter__` 中按 `seed + epoch` 用 `torch.randperm` 对本 rank 的 index 做随机排列。
   - 这样每个 step = 一个 WSI bag，各 GPU 看到的 WSI 集合不重叠。

4. **collate_fn：先按 max_patches_per_wsi 抽样，再读取 PNG** (`create_collate_fn`)
   - `DataLoader` 的 `batch_size = 1`（MIL：一张 WSI 一个 bag）。
   - 对 batch 中唯一的 sample：
     1. 取出 `patches = sample['patches']`（已按 `x,y,scale` 排序）。
     2. 若 `len(patches) > cfg.max_patches_per_wsi`：
        - 使用 `torch.randperm(n_total)[:max_patches_per_wsi]` 生成随机索引子集；
        - 仅保留这些 index 对应的补丁元信息；
        - **未被选中的 patch 不会被读取 PNG**，从 IO 侧直接丢弃。
     3. 遍历剩余补丁，调用 `load_patch_img(path, transform)`：
        - 应用与离线 EfficientNet 特征提取一致的 resize + normalization；
        - 生成 `images [P, 3, H, W]`、`scales [P]`、`coords_raw [P,2]`。
     4. 用 `wsi_info` 做坐标归一化：
        ```python
        coords_norm = (coords_raw - [min_x, min_y]) / ([max_x, max_y] - [min_x, min_y])
        ```
     5. 返回：
        ```python
        {
          'wsi_id': wid,
          'label': label_idx,
          'images': images,      # P <= max_patches_per_wsi
          'scales': scales,
          'coords': coords_norm,
        }
        ```

5. **NMFEMDualPath：多尺度 patch → 1024 维特征** (`init_NMFEM`, `NMFEMDualPath`)
   - 主体结构：
     - EfficientNet-B3 backbone（ImageNet 预训练）。
     - Transformer 分支（flatten + learnable position embedding + TransformerEncoder + CLS）。
     - GAP 分支（对 feature map 做全局平均池化）。
     - `DualPathFusion`：对两路特征做可学习加权融合。
     - `FeatureAligner`（可选）：用 LayerNorm + MLP + residual + optional mean/std scaling，将在线特征对齐离线 EfficientNet 统计分布。
   - 输出：给定 `[B, 3, H, W]` patch batch，输出 `[B, 1024]` patch embedding。

6. **每个 WSI bag 的前向：microbatch 分块过 NMFEM** (`run_epoch`)
   - 从 DataLoader 取出一个 batch：
     ```python
     all_patches = batch['images']    # [P, 3, H, W]
     all_scales  = batch['scales']    # [P]
     all_coords  = batch['coords']    # [P, 2]
     ```
   - 若 `P == 0`，跳过该 WSI（打印 warning）。
   - 将三者搬到 GPU：
     ```python
     patches = all_patches.to(device)
     scales_tensor = all_scales.to(device)
     coords_tensor = all_coords.to(device)
     ```
   - 设定 `microbatch = cfg.NMFEM_forward_microbatch`，然后：
     - 当 `use_gradient_checkpoint=True` 时：
       ```python
       features = []
       for i in range(0, len(patches), microbatch):
           mini = patches[i:i+microbatch]
           features.append(checkpoint(NMFEM, mini, use_reentrant=False))
       feats = torch.cat(features, dim=0)
       ```
     - 否则：
       ```python
       if len(patches) <= microbatch:
           feats = NMFEM(patches)
       else:
           chunks = []
           for i in range(0, len(patches), microbatch):
               mini = patches[i:i+microbatch]
               chunk_feats = NMFEM(mini)
               chunks.append(chunk_feats)
           feats = torch.cat(chunks, dim=0)
       ```
     - **最后不足 microbatch 的那一小段同样会被独立前向并拼接，不会丢失。**
   - 得到 `feats [P, 1024]` 后，转换为 float32（AMP 下常见），准备给 IAAM 使用。

7. **IAAM：patch 特征 → WSI bag logits**
   - 输入：
     - `patch_features = feats_fp32 [P, 1024]`
     - `scale_info = scales_tensor [P]`
     - `spatial_info = coords_tensor [P, 2]`
   - 内部：
     - `input_proj` 将 1024 映射到 d_model=512；
     - MHE：将 `[x,y,scale]` 编码并叠加到 patch 特征上，多层低秩自注意力；
     - DMQ：用若干 learnable queries 和 MHE 输出做跨注意力，提取固定大小的 bag 表达；
     - GatedAttention + classifier：输出最终 bag-level logits `[num_classes]`。
   - 训练早期 IAAM 按策略部分/全部冻结，仅 NMFEM + FeatureAligner 在学习；到指定 epoch 之后，再按 scope 解冻 IAAM 的一部分或全部模块。

8. **损失与反向传播**
   - 对每个 WSI bag：
     ```python
     logits, _ = iaam(feats_fp32, scales_tensor, coords_tensor)  # [C]
     loss = criterion(logits.unsqueeze(0), label.unsqueeze(0))   # label: [1]
     ```
   - 若使用梯度累积：
     ```python
     loss = loss / cfg.grad_accum_steps
     loss.backward()
     if (step+1) % cfg.grad_accum_steps == 0:
         clip_grad_norm(...)
         optimizer.step()
         scheduler.step()
         optimizer.zero_grad(set_to_none=True)
     ```
   - 梯度从 IAAM（已解冻部分）一路反传到 NMFEM + FeatureAligner，再到 EfficientNet-B3（已解冻的 blocks）。

9. **指标与日志**
   - `chunk_acc`：当前实现中，一个 step = 一个 WSI bag，因此 `chunk_acc` 实际上就是 **WSI 级 step 准确率**（本 epoch 累积的正确 WSI 数 / 总 WSI 数），名称沿用历史代码。
   - `wsi_acc`：按 `update_wsi_store` 将每个 WSI 的 logits 累积，`compute_wsi_metrics` 在 epoch 末计算全局 WSI acc / AUC / per-class acc。
   - 训练和验证阶段均返回：`{'loss', 'chunk_acc', 'wsi_acc', 'wsi_auc', 'wsi_records'}`。

10. **IAAM 冻结/解冻策略**
    - 初始（`apply_initial_iaam_freeze_policy`）：
      - 默认将 IAAM 所有参数 `requires_grad=False`。
      - 若 `cfg.train_iaam_classifier=True`：
        - 立即放开 `classifier` 与 `gated_attention` 的参数；
        - MHE、DMQ 等 patch-level 模块仍然冻结，等待后续 epoch 再解冻。
    - 训练主循环中（每个 epoch 开头）：
      - 当 `epoch >= cfg.unfreeze_iaam_epoch` 且尚未执行过解冻：
        - 调用 `unfreeze_iaam_modules(iaam, cfg.iaam_unfreeze_scope)`：
          - `scope='classifier'`：仅解冻分类头；
          - `scope='attn'` 或 `'attention'`：解冻 MHE+DMQ+GatedAttention；
          - `scope='all'`：解冻 IAAM 全部参数。

---

## 2. train_iaam_classifier 与 unfreeze_iaam_epoch 的含义与推荐设置

- `train_iaam_classifier`：是否 **在训练一开始就让 IAAM 的 bag-level 分类头参与训练**。
  - True：
    - 初始冻结全部 IAAM 参数 → 立即放开 `classifier` 和 `gated_attention`；
    - 随着 NMFEM 特征逐步对齐，bag-level 头可以更早适配新的特征分布；
    - 其余模块（MHE/DMQ）仍由 `unfreeze_iaam_epoch` + `iaam_unfreeze_scope` 控制。
  - False：
    - 初始阶段 IAAM 全部冻结，只做“离线特征对齐”；
    - 到 `epoch >= unfreeze_iaam_epoch` 之后，才按 scope 解冻部分或全部模块。

- `unfreeze_iaam_epoch`：**从第几个 epoch 开始解冻 IAAM 的 patch-level / 全部模块**。
  - 例如设为 5：
    - epoch 1-4：只更新 NMFEM (+ 可选 classifier)，MHE/DMQ 保持离线权重；
    - epoch 5 起：根据 `iaam_unfreeze_scope` 解冻对应模块，让端到端整体微调。

> 推荐策略（你目前场景）：
>
> - 前期希望 **只先让 NMFEM+FeatureAligner 学会模仿离线特征**，不要过早动 IAAM：
>   - `train_iaam_classifier = False`
>   - `unfreeze_iaam_epoch = 5`（或更保守一些如 8-10）
>   - `iaam_unfreeze_scope = 'classifier'` 或 `'attn'`（先解冻 bag 头或者注意力，再视情况扩到 `all`）。
>
> 这样就不会出现“一开始就解冻”的问题：在 epoch < 5 时，IAAM 完全冻结；到达第 5 个 epoch 以后，才按 scope 解冻。

---

## 3. 推荐配置汇总（可直接应用于 TrainConfig）

以下是一套相对稳健、适合当前双 A10 端到端训练的默认配置（仅列出关键项，路径请按你本地实际修改）：

```python
class TrainConfig:
    # 路径相关
    patch_root: str = '/mnt/nas/ljh/MsaMIL_Net_Data/results'
    label_file: str = '/home/bawa/xiangmu/MsaMIL/MsaMIL_Net/data/all_data.csv'
    iaam_checkpoint: str = '.../checkpoints/iaam_efficientnet_xxx/best_model.pth'
    save_dir: str = 'checkpoints/NMFEM_refit_efficientnet_xxx'

    # 训练基本设置
    epochs: int = 60
    batch_size: int = 1                  # MIL: 1 WSI per batch
    max_patches_per_wsi: int = 512       # 与 IAAM 预训练保持一致
    input_patch_size: int = 512

    # NMFEM 前向微批
    NMFEM_forward_microbatch: int = 24   # 比 32 略小，减轻显存与单 step 时长

    # Dataloader
    num_workers: int = 6
    prefetch_factor: int = 4
    persistent_workers: bool = True
    pin_memory: bool = True
    auto_tune_dataloader: bool = True
    loader_memory_budget_gb: float = 12.0
    dataloader_use_amp_dtype: bool = True

    # 优化器 & 调度
    lr: float = 3e-5
    weight_decay: float = 3e-5
    warmup_epochs: int = 0
    warmup_steps: int = 300
    grad_accum_steps: int = 1
    max_grad_norm: float = 10.0

    # AMP & checkpoint
    amp: bool = True
    use_gradient_checkpoint: bool = True
    amp_dtype: torch.dtype = torch.bfloat16

    # 划分与随机性
    val_ratio: float = 0.2
    seed: int = 42

    # 数据增强
    use_augmentation: bool = False       # 先观察收敛，再考虑打开

    # 旧 chunk 字段保留但不再使用
    chunk_size: int = 256
    pad_chunk_to_full: bool = False
    drop_last_incomplete: bool = True

    # 损坏补丁处理
    skip_broken_patches: bool = True

    # FeatureAligner / 分布对齐
    feature_stats_path: Optional[str] = 'data/features_efficientnet_stats.json'
    enable_feature_aligner: bool = True
    feature_aligner_hidden: int = 2048
    feature_aligner_dropout: float = 0.1
    feature_align_scale_to_target: bool = True

    # IAAM 冻结 / 解冻
    label_smoothing: float = 0.0
    unfreeze_iaam_epoch: int = 5         # 更保守的解冻时机
    train_iaam_classifier: bool = False  # 前期完全冻结 IAAM
    iaam_classifier_lr_scale: float = 1.0
    reset_iaam_classifier: bool = False
    iaam_sort_order: str = 'xy'
    iaam_unfreeze_scope: str = 'classifier'   # 先只解冻 bag-level 头

    # EfficientNet backbone 微调
    freeze_backbone: bool = True
    unfreeze_backbone_blocks: int = 2

    # 日志与调试
    log_interval: int = 5
    ddp_debug: bool = False
    save_every_epoch: bool = True
    tqdm_dynamic_ncols: bool = True
    tqdm_ncols: Optional[int] = None

    # 不平衡与 FocalLoss
    use_class_weights: bool = True
    use_focal_loss: bool = False         # 先用加权 CE，后续视情况再开
    focal_gamma: float = 2.0

    # 断点恢复
    resume_checkpoint: Optional[str] = None
```

若你后续想更激进地微调 IAAM，可以从这套配置出发，逐步调整：

- 先把 `train_iaam_classifier` 改成 `True`，观察前 5 个 epoch 是否稳定；
- 若稳定，再将 `iaam_unfreeze_scope` 从 `'classifier'` 扩到 `'attn'` 或 `'all'`，并根据 val AUC 与 NonAdenocarcinoma 的 recall 来微调 `unfreeze_iaam_epoch`。
