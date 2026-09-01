# 特征提取器替代方案调研（NMFEM → 新骨干）

## 1. 背景与现状
- 当前项目任务：基于冰冻切片的腺癌 / 非腺癌二分类，采用多实例学习（MIL）框架，NMFEM 负责将 512×512 patch 转换为 1024 维特征，再交由 IAAM 聚合分类。
- 现用骨干：EfficientNet-B3 + Transformer Encoder。优点是参数量适中，但在 512 输入、分布式 AMP 训练下，梯度常出现震荡；需要较长 warmup 且对 chunk_size 较敏感，导致“难训、慢收敛”。
- 需求：找到更稳定、易调优且对冰冻切片纹理更友好的特征提取器，并能无缝替换到 `NMFEM.py`（或以模块化方式调用）。

## 2. 选型评估指标
1. **优化难度**：是否对 lr、AMP、chunk_size 过度敏感；能否少量 epoch 即达到可用效果。
2. **多尺度/局部表现**：Frozen section 的纹理噪声较大，模型需具备较强的局部与全局表征能力。
3. **兼容性**：能否轻松输出 1024 维特征（可通过线性层适配 IAAM）。
4. **预训练可得性**：是否有公开 ImageNet / 自监督 / 病理学预训练权重，便于快速尝试。
5. **算力占用**：单次前向能否在 chunk_size≥128 的设置下稳定运行，避免显存爆炸。

## 3. 候选方案概览
| 方案 | 类型 | 主要优点 | 可能风险 | 推荐用途 |
| --- | --- | --- | --- | --- |
| ResNet-50 / 101 / RegNetY-8GF | 经典 CNN | 收敛稳定、实现简单、可渐进解冻 | 表达力略低，需要更强数据增强 | 作为"稳健 baseline"，验证数据/训练流程 |
| ConvNeXt-Tiny / Base (含 V2) | 现代 CNN | 对大尺寸 patch 友好，梯度平滑，官方权重齐全 | 相比 ResNet 更耗显存 | 建议首选，兼顾稳定与性能 |
| Swin Transformer (Tiny/Base) | 分层 ViT | 原生处理大 patch，多尺度 self-attention | 训练速度慢、需良好正则 | 当想保留 Transformer 风格骨干时使用 |
| ViT-B/16 (DeiT / DINOv2 / CLIP) | 纯 ViT | 预训练丰富，可直接输出 token | 需更多算力，微调难度高 | 以“冻结特征 + 线性适配”方式尝试 |
| 领域特化模型（CTransPath、UNI、HIPT） | 病理自监督 | 直接面向病理 patch，自带归一化策略 | 部分模型较大，需转换权重格式 | 当追求 SOTA，或已有充足算力时 |

## 4. 方案细节与落地建议
### 4.1 ResNet / RegNet 家族
- **优势**：BatchNorm 行为成熟，warmup 需求低；梯度波动远小于 EfficientNet；TorchVision 提供 `resnet50`, `resnet101`, `regnet_y_8gf` 等现成权重。
- **实现要点**：
  1. 使用 `torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)`；去掉最终 `fc` 并提取 `avgpool` 输出（2048 维）。
  2. 在 `NMFEM` 末尾保留 `self.final_proj = nn.Linear(2048, 1024)`，即可与 IAAM 对齐。
  3. 若想模拟多尺度，可保留中间层 (`layer2/3`) 做特征金字塔，再用 `nn.Conv2d` 降维。
- **训练策略**：前 3~5 epoch 冻结 backbone，先稳定 IAAM；之后按 block 逐级解冻，LR 设为主干 LR × 0.1。

### 4.2 ConvNeXt / ConvNeXtV2
- **优势**：面向大分辨率设计的现代 CNN，宽核 + 深度卷积使梯度更平滑，对 AMP 友好；官方提供 Tiny / Small / Base 多种规模。
- **实现要点**：
  1. 通过 `torchvision.models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)` 获取 backbone；使用 `.features` 作为主干。
  2. ConvNeXt 输出 channel=768（Tiny），可直接接 `nn.LayerNorm` + `nn.Linear(768, 1024)`；若需要 Transformer encoder，可保留 `NMFEM` 的 encoder 部分。
  3. 若关注训练速度，可关闭 `NMFEM` 内部 Transformer，直接把 CLS token 换成 GAP + 线性层，作为“ConvNeXt-only”版本。
- **训练策略**：
  - 建议使用较高基础 LR（2e-4 ~ 3e-4）并配合 Cosine + warmup=200 steps。
  - chunk_size=256 场景下，如显存紧张，可将 `NMFEM_forward_microbatch` 调至 16，同时保持梯度检查点。

### 4.3 Swin Transformer（Tiny/Base）
- **优势**：分层窗口注意力天然支持多尺度上下文，对组织学纹理表现好；ImageNet 预训练权重稳定。
- **实现要点**：
  1. TorchVision 提供 `swin_t`、`swin_b`。调用后可获取 `features` 序列，每个 stage 最后一层输出 `[B, C, H/32, W/32]`。
  2. 使用 `AdaptiveAvgPool2d` 压成 1024 / 1536 维，再接 `nn.Linear` 对齐 IAAM。
  3. 若想保留 token 序列，可直接取 `swin_t.forward_features` 返回 `[B, L, C]`，替换 NMFEM 的位置编码与 Transformer。
- **训练策略**：
  - 需更强 regularization：RandAugment + Mixup/CutMix + Label Smoothing 0.1。
  - 建议使用 `AdamW`，权重衰减 0.05。

### 4.4 ViT/DeiT/DINOv2/CLIP 等通用 ViT
- **优势**：利用更大规模数据的自监督或文本对齐信息，能在小样本下提供强特征。
- **落地方式**：
  1. 从 `torchvision.models.vit_b_16` 或 HuggingFace (`facebook/dino-vits16`, `openai/clip-vit-large-patch14`) 加载模型。
  2. 直接获取 CLS token（768/1024/1536 维）作为 patch 特征。
  3. 大模型通常冻结前半部分，只微调最后的 LayerNorm/MLP，以防过拟合。
- **注意事项**：显存占用较高，chunk_size 建议回落到 128；必要时配合梯度累积。

### 4.5 领域专用自监督模型
- **可选模型**：HIPT、UNI、CTransPath、RetCCL 等。这些模型针对病理全景图自监督预训练，在组织结构辨识上更有优势。
- **集成思路**：
  - 通过官方仓库导出 `state_dict`，只取 patch encoder 部分。
  - 若输出维度已为 1024，可直接对接 IAAM；否则添加线性层。
  - 通常建议 **冻结 encoder**，只训练后端，以免小批量导致灾难性遗忘。
- **优先级**：当已有稳定 pipeline 后，再尝试此类模型作为性能冲刺方案。

## 5. 实现改造建议
1. **模块化骨干工厂**：将 `NMFEM` 中的 EfficientNet 构建逻辑提取为 `build_backbone(backbone_name, pretrained)`，支持 `efficientnet_b3`, `resnet50`, `convnext_tiny`, `swin_t`, `custom_path`。可通过 `TrainConfig.backbone` 来选择。
2. **可配置输出维度**：新增 `backbone_out_dim`，用于自动设定 `self.d_model` 与 `self.final_proj`。
3. **Transformer 可选**：提供 `use_transformer_head` 开关，允许“纯 CNN”直连 IAAM，以减少层数。
4. **脚本联动**：
   - `train_NMFEM_end2end.py`：新增 CLI `--backbone`、`--backbone-pretrained`、`--backbone-freeze-epochs`。
   - `train_iaam_from_features.py`：在特征预提取场景中亦可指定 backbone，保持特征兼容性。
5. **推理/断点兼容**：checkpoint 中记录 `cfg.backbone` 和 `cfg.backbone_out_dim`，方便 resume 时构建相同结构。

## 6. 推荐路线（按易 → 难）
1. **ResNet-50 Baseline**：快速验证“换骨干”是否能稳定收敛；期望在 5~8 epoch 内跑出 >0.6 val AUC。
2. **ConvNeXt-Tiny 主力方案**：在 baseline 稳定后切换，保留现有 Transformer 头，获得更强特征表征。
3. **Swin-Tiny 多尺度方案**：当需要保留 Transformer 架构时替换；可与现有 `position_embedding + transformer_encoder` 融合。
4. **冻结 ViT / Foundation 模型**：作为长期优化方向，结合少量学习率微调。

## 7. 下一步计划
1. 在 `NMFEM` 中实现 backbone 工厂与配置参数；先跑 ResNet 版本验证训练曲线。
2. 结合 `train_iaam_from_features.py`，对 ConvNeXt 特征做一次单独 IAAM 训练，评估特征质量。
3. 根据结果决定是否引入 Swin/ViT 或病理自监督模型。
4. 最终在 `history.json` 中比较各 backbone 的前三轮指标，选取最稳方案写入主 README/训练总结。
