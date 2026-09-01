import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, List
import numpy as np


from .SFFM import SFFM
from .NMFEM import NMFEM  
from .IAAM import IAAM

class MsaMILNet(nn.Module):
    
    def __init__(self,
                 unet_model_path: str = None,
                 num_classes: int = 2,
                 NMFEM_output_dim: int = 512,
                 iaam_mhe_layers: int = 4,
                 iaam_num_heads: int = 8,
                 iaam_low_rank: int = 64,
                 iaam_num_queries: int = 10,
                 device: str = 'cuda'):
        super(MsaMILNet, self).__init__()
        
        self.num_classes = num_classes
        self.device = device
        

        self.sffm = SFFM(
            unet_model_path=unet_model_path,
            low_res_size=1024,
            lesion_threshold=0.7,
            device=device
        )
        

        self.NMFEM = NMFEM(
            output_dim=NMFEM_output_dim
        )
        

        self.iaam = IAAM(
            d_model=NMFEM_output_dim,
            input_dim=NMFEM_output_dim,
            mhe_layers=iaam_mhe_layers,
            num_heads=iaam_num_heads,
            low_rank=iaam_low_rank,
            num_queries=iaam_num_queries,
            num_classes=num_classes
        )
        
        print("=" * 60)

        print("DgMsa-MIL 初始化完成")
        print("=" * 60)
        print(f"设备: {device}")
        print(f"分类类别数: {num_classes}")
        print(f"NMFEM输出维度: {NMFEM_output_dim}")
        print(f"IAAM MHE层数: {iaam_mhe_layers}")
        print(f"IAAM查询数: {iaam_num_queries}")
        print("=" * 60)

    def forward_complete_pipeline(self, wsi_path: str) -> Tuple[torch.Tensor, torch.Tensor]:


        filtered_patches, patch_coords, _ = self.sffm.process_wsi(wsi_path, save_patches=False)
        

        total_patches = sum(len(patches) for patches in filtered_patches.values())
        if total_patches == 0:
            print("⚠️ No valid patches found, returning zero prediction")
            device = next(self.parameters()).device
            return torch.zeros(self.num_classes).to(device), torch.zeros(512).to(device)
        

        all_features, scale_info, spatial_info = self.NMFEM.process_multiscale_patches(
            filtered_patches, patch_coords
        )
        

        if spatial_info.numel() > 0:
            spatial_info = spatial_info / 50000.0
        

        logits, bag_features = self.iaam(all_features, scale_info, spatial_info)
        
        return logits, bag_features
    
    def forward_training(self, 
                        patch_batch: torch.Tensor,
                        scale_info: torch.Tensor,
                        spatial_info: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        patch_features = self.NMFEM(patch_batch)  # [B, 512]
        

        logits, bag_features = self.iaam(patch_features, scale_info, spatial_info)
        
        return logits, bag_features
    
    def forward(self, *args, **kwargs):
        if len(args) == 1 and isinstance(args[0], str):

            return self.forward_complete_pipeline(args[0])
        elif len(args) == 3:

            return self.forward_training(*args)
        else:
            raise ValueError("Invalid input format. Expected WSI path or (patch_batch, scale_info, spatial_info)")
    
    def extract_features_for_mil_training(self, wsi_paths: List[str], save_dir: str = None):
        if save_dir:
            import os
            os.makedirs(save_dir, exist_ok=True)
        
        all_features = []
        all_labels = []
        
        self.eval()
        with torch.no_grad():
            for i, wsi_path in enumerate(wsi_paths):
                print(f"Processing {i+1}/{len(wsi_paths)}: {wsi_path}")
                

                filtered_patches, patch_coords, _ = self.sffm.process_wsi(wsi_path, save_patches=False)
                
                if sum(len(patches) for patches in filtered_patches.values()) == 0:
                    print(f"⚠️ No patches for {wsi_path}")
                    continue
                
                features, scale_info, spatial_info = self.NMFEM.process_multiscale_patches(
                    filtered_patches, patch_coords
                )
                

                if save_dir:
                    wsi_name = os.path.basename(wsi_path).split('.')[0]
                    feature_data = {
                        'features': features.cpu(),
                        'scale_info': scale_info.cpu(),
                        'spatial_info': spatial_info.cpu(),
                        'wsi_path': wsi_path
                    }
                    torch.save(feature_data, os.path.join(save_dir, f"{wsi_name}_features.pt"))
                
                all_features.append(features.cpu())
        
        print(f"✓ 特征提取完成，共处理 {len(all_features)} 个WSI")
        return all_features
    
    def get_model_summary(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        

        sffm_params = sum(p.numel() for p in self.sffm.unet.parameters())
        NMFEM_params = sum(p.numel() for p in self.NMFEM.parameters())
        iaam_params = sum(p.numel() for p in self.iaam.parameters())
        
        summary = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'sffm_params': sffm_params,
            'NMFEM_params': NMFEM_params,
            'iaam_params': iaam_params
        }
        
        print("模型参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  SFFM参数: {sffm_params:,}")
        print(f"  NMFEM参数: {NMFEM_params:,}")
        print(f"  IAAM参数: {iaam_params:,}")
        
        return summary


class DgMsaMILNet(MsaMILNet):
    """DgMsa-MIL 的代码别名。

    为保持现有导入路径与历史 checkpoint 兼容，保留 `MsaMILNet` 类名；
    论文/发布版本对外推荐使用 `DgMsaMILNet` 表述。
    """

    pass


if __name__ == "__main__":

    model = DgMsaMILNet(
        unet_model_path="checkpoints/unet_pretrain/best_unet_iou.pth",
        num_classes=2,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    

    model.get_model_summary()
    

    print("\n测试训练模式:")
    batch_size = 200
    patch_batch = torch.randn(batch_size, 3, 512, 512).to(device)
    scale_info = torch.randint(0, 3, (batch_size,)).to(device)
    spatial_info = torch.randn(batch_size, 2).to(device)
    
    with torch.no_grad():
        logits, bag_features = model.forward_training(patch_batch, scale_info, spatial_info)
        print(f"输入: {batch_size} patches")
        print(f"输出 logits: {logits.shape}")
        print(f"输出 bag features: {bag_features.shape}")
        print(f"预测类别: {torch.argmax(logits).item()}")

    print("\n✓ DgMsa-MIL 测试完成!")