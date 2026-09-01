import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# �����Զ���ģ��
from models.SFFM import SFFM
from models.NMFEM import NMFEM
from models.IAAM import IAAM
from models.MsaMILNet import MsaMILNet
from utils.helpers import load_checkpoint, setup_logging, normalize_coordinates

class MsaMILInference:
    """
    DgMsa-MIL 推理模块。
    """
    
    def __init__(self, 
                 model_path: str,
                 unet_path: str,
                 device: str = 'auto',
                 num_classes: int = 2,
                 class_bias_json: Optional[str] = None):
        """
        Args:
            model_path: 训练好的 DgMsa-MIL 模型路径
            unet_path: UNet++�ָ�ģ��·��
            device: �����豸
            num_classes: ���������
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
        self.num_classes = num_classes
        self.class_bias: Optional[torch.Tensor] = None
        
        print(f"Using device: {self.device}")
        
        # ����ģ��
        self.model = self._load_model(model_path)
        self.unet_path = unet_path
        
        # ��ʼ��SFFM
        self.sffm = SFFM(unet_model_path=unet_path, device=self.device)
        
        # 可选加载类别偏置（来自训练阶段的校准/调优结果）
        if class_bias_json and os.path.isfile(class_bias_json):
            try:
                with open(class_bias_json, 'r') as f:
                    obj = json.load(f)
                bias = obj.get('class_bias', None)
                if isinstance(bias, list) and len(bias) == self.num_classes:
                    self.class_bias = torch.tensor(bias, dtype=torch.float32, device=self.device)
                    print(f"✓ Loaded class bias from {class_bias_json}: {self.class_bias.tolist()}")
                else:
                    print(f"[Warn] class_bias_json 不包含有效的 'class_bias' 列表，已忽略。")
            except Exception as e:
                print(f"[Warn] 读取 class_bias_json 失败: {e}")

        print("✓ DgMsa-MIL inference model loaded successfully")
    
    def _load_model(self, model_path: str) -> MsaMILNet:
        """����ѵ���õ�ģ��"""
        # ����ģ��
        model = MsaMILNet(
            num_classes=self.num_classes,
            d_model=512,
            num_heads=8,
            num_layers=3,
            low_rank=64
        )
        
        # ���ؼ���
        checkpoint = load_checkpoint(model, model_path, self.device)
        model.eval()
        
        print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best metric: {checkpoint.get('best_metric', 'unknown')}")
        
        return model
    
    def predict_wsi(self, 
                   wsi_path: str,
                   return_attention: bool = False,
                   save_visualization: bool = False,
                   output_dir: Optional[str] = None) -> Dict:
        """
        �Ե���WSI����Ԥ��
        
        Args:
            wsi_path: WSI�ļ�·��
            return_attention: �Ƿ񷵻�ע����Ȩ��
            save_visualization: �Ƿ񱣴���ӻ����
            output_dir: ���Ŀ¼
        
        Returns:
            prediction_results: Ԥ�����ֵ�
        """
        print(f"\n? Processing WSI: {os.path.basename(wsi_path)}")
        
        try:
            # 1. SFFM���� - ��ȡ��߶�patches
            print("Step 1: Extracting multi-scale patches with SFFM...")
            filtered_patches, patch_coords = self.sffm.process_wsi(wsi_path)
            
            # ͳ��patches����
            total_patches = sum(len(patches) for patches in filtered_patches.values())
            print(f"Extracted patches: 20x={len(filtered_patches.get('20x', []))}, "
                  f"10x={len(filtered_patches.get('10x', []))}, "
                  f"5x={len(filtered_patches.get('5x', []))}, Total={total_patches}")
            
            if total_patches == 0:
                print("? No valid patches extracted from WSI")
                return {
                    'predicted_class': -1,
                    'confidence': 0.0,
                    'class_probabilities': [0.0] * self.num_classes,
                    'error': 'No valid patches extracted'
                }
            
            # 2. ׼����������
            all_patches = []
            all_coords = []
            all_scales = []
            
            scale_mapping = {'20x': 0, '10x': 1, '5x': 2}
            
            for scale, patches in filtered_patches.items():
                coords = patch_coords[scale]
                scale_id = scale_mapping[scale]
                
                all_patches.extend(patches)
                all_coords.extend(coords)
                all_scales.extend([scale_id] * len(patches))
            
            # ����patches����(�����ڴ����)
            max_patches = 3000
            if len(all_patches) > max_patches:
                indices = np.random.choice(len(all_patches), max_patches, replace=False)
                all_patches = [all_patches[i] for i in indices]
                all_coords = [all_coords[i] for i in indices]
                all_scales = [all_scales[i] for i in indices]
                print(f"Randomly sampled {max_patches} patches for inference")
            
            # 3. 计算原始WSI尺寸并对坐标做[0,1]归一化（与论文一致）
            try:
                import openslide
                suffix = str(wsi_path).lower()
                if any(suffix.endswith(ext) for ext in ['.svs', '.ndpi', '.tif', '.tiff', '.mrxs']):
                    slide = openslide.OpenSlide(wsi_path)
                    W, H = slide.dimensions
                    slide.close()
                else:
                    with Image.open(wsi_path) as im:
                        W, H = im.size
            except Exception:
                # 回退：以坐标最大值近似，防止流程中断
                if len(all_coords) > 0:
                    W = max(1.0, float(max(c[0] for c in all_coords)))
                    H = max(1.0, float(max(c[1] for c in all_coords)))
                else:
                    W, H = 1.0, 1.0
            if len(all_coords) > 0:
                all_coords = [(c[0] / max(W, 1.0), c[1] / max(H, 1.0)) for c in all_coords]

            # 4. 模型推理
            print("Step 2: Running DgMsa-MIL inference...")
            with torch.no_grad():
                # ת��Ϊtensor
                patch_tensors = []
                for patch in all_patches:
                    if isinstance(patch, np.ndarray):
                        patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
                    else:
                        patch_tensor = transforms.ToTensor()(patch)
                    patch_tensors.append(patch_tensor)
                
                patch_batch = torch.stack(patch_tensors).to(self.device)
                scale_info = torch.tensor(all_scales, device=self.device)
                spatial_info = torch.tensor(all_coords, dtype=torch.float32, device=self.device)
                
                # ǰ向推理
                logits, attention_weights = self.model(
                    patch_batch, scale_info, spatial_info, return_attention=return_attention
                )

                # 应用类别偏置（仅改变决策，不改变排序/AUC）
                if self.class_bias is not None:
                    try:
                        if self.class_bias.numel() == logits.shape[-1]:
                            logits = logits + self.class_bias
                    except Exception:
                        pass
                
                # �������
                probabilities = F.softmax(logits, dim=-1).cpu().numpy()
                predicted_class = np.argmax(probabilities)
                confidence = float(np.max(probabilities))
            
            # 5. 汇总结果
            results = {
                'wsi_path': wsi_path,
                'predicted_class': int(predicted_class),
                'confidence': confidence,
                'class_probabilities': probabilities.tolist(),
                'total_patches': total_patches,
                'patches_used': len(all_patches),
                'patch_distribution': {
                    '20x': len(filtered_patches.get('20x', [])),
                    '10x': len(filtered_patches.get('10x', [])),
                    '5x': len(filtered_patches.get('5x', []))
                }
            }
            
            if return_attention and attention_weights is not None:
                results['attention_weights'] = attention_weights.cpu().numpy().tolist()
            
            # 6. 可视化（可选）
            if save_visualization and output_dir:
                self._save_visualization(
                    results, all_patches, all_coords, all_scales,
                    attention_weights if return_attention else None,
                    output_dir
                )
            
            print(f"? Prediction completed - Class: {predicted_class}, Confidence: {confidence:.3f}")
            return results
            
        except Exception as e:
            print(f"? Error during inference: {e}")
            return {
                'predicted_class': -1,
                'confidence': 0.0,
                'class_probabilities': [0.0] * self.num_classes,
                'error': str(e)
            }
    
    def predict_batch(self, 
                     wsi_paths: List[str],
                     output_file: Optional[str] = None) -> List[Dict]:
        """
        ����Ԥ����WSI
        """
        results = []
        
        print(f"\n? Starting batch inference for {len(wsi_paths)} WSIs")
        
        for i, wsi_path in enumerate(wsi_paths):
            print(f"\nProgress: {i+1}/{len(wsi_paths)}")
            
            result = self.predict_wsi(wsi_path)
            result['wsi_index'] = i
            results.append(result)
        
        # ������
        if output_file:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"? Batch results saved to: {output_file}")
        
        # ͳ�ƽ��
        self._print_batch_summary(results)
        
        return results
    
    def _save_visualization(self, 
                           results: Dict,
                           patches: List[np.ndarray],
                           coords: List[Tuple[int, int]],
                           scales: List[int],
                           attention_weights: Optional[torch.Tensor],
                           output_dir: str):
        """������ӻ����"""
        os.makedirs(output_dir, exist_ok=True)
        
        wsi_name = os.path.splitext(os.path.basename(results['wsi_path']))[0]
        
        # 1. ����Ԥ�������״ͼ
        plt.figure(figsize=(8, 6))
        class_names = [f'Class {i}' for i in range(len(results['class_probabilities']))]
        probabilities = results['class_probabilities']
        
        bars = plt.bar(class_names, probabilities, 
                      color=['red' if i == results['predicted_class'] else 'lightblue' 
                            for i in range(len(probabilities))])
        
        plt.title(f'WSI Classification Results\n{wsi_name}')
        plt.ylabel('Probability')
        plt.ylim(0, 1)
        
        # ������ֵ��ǩ
        for bar, prob in zip(bars, probabilities):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{wsi_name}_probabilities.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ����patch�ֲ�ͼ
        if len(patches) > 0:
            plt.figure(figsize=(10, 8))
            
            # ���߶ȷ�����ʾpatches
            scale_names = ['20x', '10x', '5x']
            scale_colors = ['red', 'green', 'blue']
            
            coords_array = np.array(coords)
            scales_array = np.array(scales)
            
            for scale_id in range(3):
                mask = scales_array == scale_id
                if np.any(mask):
                    plt.scatter(coords_array[mask, 0], coords_array[mask, 1], 
                              c=scale_colors[scale_id], label=scale_names[scale_id], 
                              alpha=0.6, s=10)
            
            plt.title(f'Patch Distribution - {wsi_name}')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{wsi_name}_patch_distribution.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. ����ע������ͼ(�����)
        if attention_weights is not None:
            plt.figure(figsize=(10, 8))
            
            # ��һ��ע����Ȩ��
            attention = np.array(attention_weights)
            if attention.ndim > 1:
                attention = attention.mean(axis=0)  # ƽ�����ע����ͷ
            
            # ��������������в�ֵ
            coords_array = np.array(coords)
            if len(coords_array) > 0:
                from scipy.interpolate import griddata
                
                # ������������
                x_min, x_max = coords_array[:, 0].min(), coords_array[:, 0].max()
                y_min, y_max = coords_array[:, 1].min(), coords_array[:, 1].max()
                
                grid_x, grid_y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
                
                # ��ֵע����Ȩ��
                grid_attention = griddata(coords_array, attention[:len(coords_array)], 
                                        (grid_x, grid_y), method='linear')
                
                # ������ͼ
                plt.imshow(grid_attention.T, extent=[x_min, x_max, y_min, y_max], 
                          origin='lower', cmap='hot', alpha=0.7)
                plt.colorbar(label='Attention Weight')
                
                # ����patchλ��
                plt.scatter(coords_array[:, 0], coords_array[:, 1], 
                           c='white', s=1, alpha=0.5)
                
                plt.title(f'Attention Heatmap - {wsi_name}')
                plt.xlabel('X Coordinate')
                plt.ylabel('Y Coordinate')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{wsi_name}_attention_heatmap.png'), 
                           dpi=300, bbox_inches='tight')
                plt.close()
        
        print(f"? Visualizations saved to: {output_dir}")
    
    def _print_batch_summary(self, results: List[Dict]):
        """��ӡ��������ͳ��"""
        print("\n" + "="*60)
        print("BATCH INFERENCE SUMMARY")
        print("="*60)
        
        total_wsis = len(results)
        successful = sum(1 for r in results if 'error' not in r)
        failed = total_wsis - successful
        
        print(f"Total WSIs: {total_wsis}")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        
        if successful > 0:
            # ���ֲ�
            class_counts = {}
            confidences = []
            
            for result in results:
                if 'error' not in result:
                    pred_class = result['predicted_class']
                    class_counts[pred_class] = class_counts.get(pred_class, 0) + 1
                    confidences.append(result['confidence'])
            
            print(f"\nClass Distribution:")
            for class_id, count in sorted(class_counts.items()):
                print(f"  Class {class_id}: {count} samples ({count/successful*100:.1f}%)")
            
            print(f"\nConfidence Statistics:")
            print(f"  Mean: {np.mean(confidences):.3f}")
            print(f"  Std:  {np.std(confidences):.3f}")
            print(f"  Min:  {np.min(confidences):.3f}")
            print(f"  Max:  {np.max(confidences):.3f}")
        
        print("="*60)


def main():
    """������"""
    parser = argparse.ArgumentParser(description='DgMsa-MIL Model Inference')
    parser.add_argument('--wsi_path', type=str, required=True,
                       help='Path to WSI file or directory of WSI files')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained DgMsa-MIL model')
    parser.add_argument('--unet_path', type=str, required=True,
                       help='Path to trained UNet++ segmentation model')
    parser.add_argument('--output_dir', type=str, default='inference_results',
                       help='Output directory for results and visualizations')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classification classes')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for inference')
    parser.add_argument('--return_attention', action='store_true',
                       help='Return attention weights')
    parser.add_argument('--save_visualization', action='store_true',
                       help='Save visualization results')
    parser.add_argument('--batch_mode', action='store_true',
                       help='Process all WSI files in directory')
    parser.add_argument('--class_bias_json', type=str, default=None,
                       help='Path to class bias JSON saved during training for threshold/bias calibration')
    
    args = parser.parse_args()
    
    # �������Ŀ¼
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ������־
    logger = setup_logging(args.output_dir, "inference")
    
    # ��ʼ��������
    try:
        inferencer = MsaMILInference(
            model_path=args.model_path,
            unet_path=args.unet_path,
            device=args.device,
            num_classes=args.num_classes,
            class_bias_json=args.class_bias_json
        )
    except Exception as e:
        print(f"? Failed to initialize inference model: {e}")
        return
    
    # ִ������
    if args.batch_mode:
        # ����ģʽ
        if os.path.isdir(args.wsi_path):
            wsi_files = [f for f in os.listdir(args.wsi_path)
                        if f.lower().endswith(('.svs', '.tif', '.tiff', '.ndpi', '.mrxs'))]
            wsi_paths = [os.path.join(args.wsi_path, f) for f in wsi_files]
        else:
            print("? Batch mode requires a directory path")
            return
        
        if not wsi_paths:
            print("? No WSI files found in directory")
            return
        
        # ��������
        results = inferencer.predict_batch(
            wsi_paths=wsi_paths,
            output_file=os.path.join(args.output_dir, 'batch_results.json')
        )
        
    else:
        # ���ļ�ģʽ
        if not os.path.isfile(args.wsi_path):
            print(f"? WSI file not found: {args.wsi_path}")
            return
        
        result = inferencer.predict_wsi(
            wsi_path=args.wsi_path,
            return_attention=args.return_attention,
            save_visualization=args.save_visualization,
            output_dir=args.output_dir if args.save_visualization else None
        )
        
        # ���浥�����
        result_file = os.path.join(args.output_dir, 'inference_result.json')
        with open(result_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        # ��ӡ���
        print(f"\n{'='*60}")
        print("INFERENCE RESULT")
        print(f"{'='*60}")
        print(f"WSI: {os.path.basename(args.wsi_path)}")
        print(f"Predicted Class: {result.get('predicted_class', 'Unknown')}")
        print(f"Confidence: {result.get('confidence', 0.0):.3f}")
        
        if 'class_probabilities' in result:
            print("Class Probabilities:")
            for i, prob in enumerate(result['class_probabilities']):
                print(f"  Class {i}: {prob:.3f}")
        
        if 'error' in result:
            print(f"Error: {result['error']}")
        
        print(f"Results saved to: {result_file}")
        print(f"{'='*60}")

if __name__ == "__main__":
    main()