import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
import numpy as np
from typing import Optional, Dict, Any


class FeatureAligner(nn.Module):

    def __init__(self, dim: int, hidden_dim: int = 2048, dropout: float = 0.1,
                 target_mean: Optional[torch.Tensor] = None,
                 target_std: Optional[torch.Tensor] = None,
                 scale_to_target: bool = False):
        super().__init__()
        self.layer_norm = nn.LayerNorm(dim)
        self.adapter = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        mean_buf = target_mean.clone().detach().float() if target_mean is not None else torch.empty(0)
        std_buf = target_std.clone().detach().float() if target_std is not None else torch.empty(0)
        self.register_buffer('target_mean', mean_buf, persistent=False)
        self.register_buffer('target_std', std_buf, persistent=False)
        self._has_target_stats = mean_buf.numel() > 0 and std_buf.numel() > 0
        self.scale_to_target = scale_to_target and self._has_target_stats
        print(f"✓ FeatureAligner initialized. Scale to target: {self.scale_to_target}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        z = self.layer_norm(x)
        z = self.adapter(z) + x
        if self.scale_to_target and self._has_target_stats:
            z = z * self.target_std + self.target_mean
        return z

class NMFEM(nn.Module):
    
    def __init__(self,
                 output_dim: int = 1024,
                 num_heads: int = 8,
                 num_layers: int = 2,
                 use_checkpoint: bool = False,
                 checkpoint_segments: int = 2,

                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 unfreeze_backbone_blocks: int = 0,
                input_patch_size: int = 512,
                feature_aligner_cfg: Optional[Dict[str, Any]] = None):
        super(NMFEM, self).__init__()
        
        self.output_dim = output_dim
        self.use_checkpoint = use_checkpoint
        self.checkpoint_segments = max(1, int(checkpoint_segments))
        self._pretrained_loaded = None
        self._pretrained_source = None
        self.input_patch_size = int(input_patch_size)
        if self.input_patch_size <= 0:
            raise ValueError("input_patch_size must be positive")
        if self.input_patch_size % 32 != 0:
            raise ValueError("input_patch_size must be divisible by 32 to keep integer feature grids")
        self._feature_grid = self.input_patch_size // 32
        self._seq_len = self._feature_grid * self._feature_grid
        

        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
        try:
            if pretrained:
                backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
                self._pretrained_loaded = True
                self._pretrained_source = 'ImageNet(EfficientNet_B3_Weights.IMAGENET1K_V1)'
                print("✓ Loaded EfficientNet-B3 pretrained on ImageNet")
            else:
                backbone = efficientnet_b3(weights=None)
                self._pretrained_loaded = False
                self._pretrained_source = 'random'
                print("✓ Initialized EfficientNet-B3 with random weights")
        except Exception:
            # fallback to random init (offline env)
            backbone = efficientnet_b3(weights=None)
            self._pretrained_loaded = False
            self._pretrained_source = 'random(fallback)'
            print("⚠ Failed to load pretrained weights, initialized EfficientNet-B3 with random weights")



        self.backbone = backbone.features


        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        unfreeze_n = max(0, int(unfreeze_backbone_blocks))
        if unfreeze_n > 0:

            blocks = list(self.backbone.children()) if isinstance(self.backbone, nn.Sequential) else []
            if len(blocks) > 0:
                for m in blocks[-unfreeze_n:]:
                    for p in m.parameters():
                        p.requires_grad = True
        

        with torch.no_grad():
            test_input = torch.randn(1, 3, self.input_patch_size, self.input_patch_size)
            test_output = self.backbone(test_input)
            expected_hw = self._feature_grid
            assert test_output.shape == (1, 1536, expected_hw, expected_hw), (
                f"Expected [1, 1536, {expected_hw}, {expected_hw}], got {test_output.shape}"
            )
        

        self.d_model = 1536
        

        self.position_embedding = nn.Parameter(
            torch.randn(1, self._seq_len, self.d_model) * 0.02
        )
        

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=num_heads,
            dim_feedforward=2048,
            dropout=0.1,
            activation='relu',
            batch_first=True
        )
        self.feature_aligner = None
        if feature_aligner_cfg:
            self.feature_aligner = FeatureAligner(output_dim, **feature_aligner_cfg)
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        


        self.final_proj = nn.Linear(self.d_model, output_dim)

        # concise init prints
        print(
            f"✓ NMFEM初始化: EfficientNet-B3, 输出维度={self.output_dim}, Transformer层数={num_layers}, 输入尺寸={self.input_patch_size}"
        )
    
    def forward(self, patch_batch: torch.Tensor) -> torch.Tensor:
        batch_size = patch_batch.size(0)
        

        if patch_batch.size(-1) != self.input_patch_size or patch_batch.size(-2) != self.input_patch_size:
            patch_batch = F.interpolate(
                patch_batch,
                size=(self.input_patch_size, self.input_patch_size),
                mode='bilinear',
                align_corners=False,
            )
        

        if self.use_checkpoint and self.training:

            from torch.utils.checkpoint import checkpoint_sequential
            modules = list(self.backbone.children()) if isinstance(self.backbone, nn.Sequential) else [self.backbone]

            segments = max(1, min(self.checkpoint_segments, len(modules)))

            features = checkpoint_sequential(self.backbone, segments, patch_batch, use_reentrant=False)
        else:
            features = self.backbone(patch_batch)  # [B, 1536, 16, 16]
        

        B, C, H, W = features.shape
        seq_features = features.view(B, C, H * W).transpose(1, 2)  # [B, seq_len, 1536]
        

        pos_embed = self.position_embedding[:, :seq_features.size(1), :]  # [1, seq_len, 1536]
        seq_features_with_pos = seq_features + pos_embed  # S + E
        

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, 1536]
        seq_with_cls = torch.cat([cls_tokens, seq_features_with_pos], dim=1)  # [B, 1+N_tokens, 1536]
        

        if self.use_checkpoint and self.training:

            from torch.utils.checkpoint import checkpoint
            x = seq_with_cls
            for layer in self.transformer_encoder.layers:

                x = checkpoint(layer, x, use_reentrant=False)
            transformer_output = x
        else:
            transformer_output = self.transformer_encoder(seq_with_cls)  # [B, 1+N_tokens, 1536]
        

        cls_output = transformer_output[:, 0, :]  # [B, 1536]
        


        patch_features = self.final_proj(cls_output)  # [B, 1024]
        if self.feature_aligner is not None:
            patch_features = self.feature_aligner(patch_features)
        
        return patch_features
    
    def process_multiscale_patches(self, filtered_patches: dict, patch_coords: dict) -> tuple:
        all_features = []
        scale_info = []
        spatial_info = []
        
        scale_mapping = {'20x': 0, '10x': 1, '5x': 2}
        
        with torch.no_grad():
            for scale, patches in filtered_patches.items():
                coords = patch_coords[scale]
                scale_id = scale_mapping[scale]

                if len(patches) == 0:
                    continue


                batch_size = 16
                for i in range(0, len(patches), batch_size):
                    batch_patches = patches[i:i+batch_size]
                    batch_coords = coords[i:i+batch_size]


                    patch_tensors = []
                    for patch in batch_patches:
                        if isinstance(patch, np.ndarray):
                            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
                        else:
                            from torchvision import transforms
                            patch_tensor = transforms.ToTensor()(patch)
                        patch_tensors.append(patch_tensor)

                    patch_batch = torch.stack(patch_tensors).to(next(self.parameters()).device)


                    features = self.forward(patch_batch)  # [batch_size, output_dim]

                    all_features.append(features.cpu())
                    scale_info.extend([scale_id] * len(batch_patches))
                    spatial_info.extend(batch_coords)

        if len(all_features) == 0:

            device = next(self.parameters()).device
            return (torch.empty(0, self.output_dim).to(device),
                    torch.empty(0, dtype=torch.long).to(device),
                    torch.empty(0, 2).to(device))


        all_features = torch.cat(all_features, dim=0)  # [N, output_dim]
        scale_info = torch.tensor(scale_info, dtype=torch.long)  # [N,]
        spatial_info = torch.tensor(spatial_info, dtype=torch.float32)  # [N, 2]


        device = next(self.parameters()).device
        all_features = all_features.to(device)
        scale_info = scale_info.to(device)
        spatial_info = spatial_info.to(device)

        print(f"✓ Processed {all_features.shape[0]} patches")
        print(f"   20x: {torch.sum(scale_info == 0).item()} patches")
        print(f"   10x: {torch.sum(scale_info == 1).item()} patches")
        print(f"   5x: {torch.sum(scale_info == 2).item()} patches")

        return all_features, scale_info, spatial_info
    
    def get_feature_dim(self):
        return self.output_dim


if __name__ == "__main__":
    import numpy as np
    

    NMFEM = NMFEM(output_dim=1024, input_patch_size=224)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NMFEM.to(device)
    
    print(f"NMFEM model loaded on {device}")
    print(f"Model parameters: {sum(p.numel() for p in NMFEM.parameters()):,}")
    

    test_patch = torch.randn(2, 3, NMFEM.input_patch_size, NMFEM.input_patch_size).to(device)
    with torch.no_grad():
        features = NMFEM(test_patch)
        print(f"Single batch test - Input shape: {test_patch.shape}")
        print(f"Output features shape: {features.shape}")
    

    mock_filtered_patches = {
        '20x': [np.random.randint(0, 255, (NMFEM.input_patch_size, NMFEM.input_patch_size, 3), dtype=np.uint8) for _ in range(5)],
        '10x': [np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8) for _ in range(3)],
        '5x': [np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8) for _ in range(2)]
    }
    
    mock_patch_coords = {
        '20x': [(100*i, 200*i) for i in range(5)],
        '10x': [(1000*i, 2000*i) for i in range(3)],
        '5x': [(5000*i, 10000*i) for i in range(2)]
    }
    
    with torch.no_grad():
        all_features, scale_info, spatial_info = NMFEM.process_multiscale_patches(
            mock_filtered_patches, mock_patch_coords
        )
        print(f"\nMultiscale test:")
        print(f"Features shape: {all_features.shape}")
        print(f"Scale info shape: {scale_info.shape}")
        print(f"Spatial info shape: {spatial_info.shape}")