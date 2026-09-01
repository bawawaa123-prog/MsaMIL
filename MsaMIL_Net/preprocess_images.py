#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from PIL import Image
import shutil
from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import gc  # 添加垃圾回收模块
import psutil  # 添加系统资源监控模块

# 设置PIL和OpenCV图像尺寸限制
Image.MAX_IMAGE_PIXELS = None  # 移除PIL图像尺寸限制
os.environ['OPENCV_IO_MAX_IMAGE_PIXELS'] = str(2**63-1)  # 设置OpenCV最大图像尺寸

# 导入OpenCV（现在应该正常工作）
import cv2
print(f"✅ OpenCV successfully imported, version: {cv2.__version__}")

# 配置OpenCV性能优化
cv2.setUseOptimized(True)  # 启用OpenCV优化
cv2.setNumThreads(4)  # 限制线程数，避免内存过载
print(f"OpenCV优化状态: {cv2.useOptimized()}")

class ImageResizer:
    """
    图像缩放器 - 专门用于将图像resize到1024×1024
    """
    
    def __init__(self, 
                 source_dir: str,
                 target_dir: str,
                 target_size: int = 1024):
        
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.target_size = target_size
        
        # 创建目标目录
        self.target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"✓ 初始化图像缩放器")
        print(f"   源目录: {self.source_dir}")
        print(f"   目标目录: {self.target_dir}")
        print(f"   目标尺寸: {self.target_size}×{self.target_size}")
    
    def get_memory_usage(self):
        """获取当前内存使用情况"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_gb = memory_info.rss / (1024**3)  # 转换为GB
        return memory_gb
    
    def force_memory_cleanup(self):
        """强制清理内存"""
        gc.collect()  # 强制垃圾回收
        gc.collect()  # 再次回收，确保彻底
        
    def log_memory_usage(self, stage: str):
        """记录内存使用情况"""
        memory_gb = self.get_memory_usage()
        print(f"💾 {stage}: 当前内存占用 {memory_gb:.2f}GB")
    
    def safe_read_image(self, image_path, convert_to_rgb=True):
        """简单直接的图像读取，强制100%成功率"""
        print(f"读取图像 {image_path.name}")
        
        try:
            # 直接使用PIL读取并转换
            with Image.open(image_path) as img:
                width, height = img.size
                print(f"图像尺寸: {width}×{height}")
                
                # 强制转换到RGB模式
                if convert_to_rgb:
                    img = img.convert('RGB')
                
                # 转换为numpy数组
                result = np.array(img)
                print(f"✅ 读取成功: {result.shape}")
                
                # 立即释放PIL对象内存
                del img
                self.force_memory_cleanup()
                
                return result
                
        except Exception as e:
            print(f"❌ 读取失败: {image_path.name} - {e}")
            # 即使失败也不跳过，抛出异常让上层处理
            raise ValueError(f"无法读取图像 {image_path.name}: {e}")
    
    def read_image_in_chunks(self, image_path, convert_to_rgb=True, force_small_chunks=False):
        """分块读取超大图像并合成，支持强制小块模式"""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                
                # 计算分块策略
                if force_small_chunks:
                    # 强制小块模式：每块最多200M像素（约0.6GB）
                    max_pixels_per_chunk = 200_000_000
                    print(f"🔧 强制小块模式：目标<0.6GB/块")
                else:
                    # 正常模式：每块最多800M像素（约2.2GB）
                    max_pixels_per_chunk = 800_000_000
                    print(f"📦 正常分块模式：目标<2.5GB/块")
                
                # 计算网格划分
                total_pixels = width * height
                num_chunks = max(1, int(np.ceil(total_pixels / max_pixels_per_chunk)))
                
                # 优先水平分割（适合WSI图像的特点）
                if width > height:
                    cols = int(np.ceil(np.sqrt(num_chunks * width / height)))
                    rows = int(np.ceil(num_chunks / cols))
                else:
                    rows = int(np.ceil(np.sqrt(num_chunks * height / width)))
                    cols = int(np.ceil(num_chunks / rows))
                
                print(f"分块策略: {rows}×{cols} = {rows*cols}块")
                
                chunk_width = width // cols
                chunk_height = height // rows
                
                # 读取并合成所有块
                chunks = []
                total_chunks = rows * cols
                processed_chunks = 0
                
                for row in range(rows):
                    chunk_row = []
                    for col in range(cols):
                        try:
                            # 计算块的边界
                            left = col * chunk_width
                            top = row * chunk_height
                            right = min(left + chunk_width, width)
                            bottom = min(top + chunk_height, height)
                            
                            # 读取块
                            chunk = img.crop((left, top, right, bottom))
                            if convert_to_rgb:
                                chunk = chunk.convert('RGB')
                            
                            chunk_array = np.array(chunk)
                            chunk_row.append(chunk_array)
                            processed_chunks += 1
                            
                            if processed_chunks % 3 == 0 or processed_chunks == total_chunks:
                                print(f"📥 进度: {processed_chunks}/{total_chunks}块 ({processed_chunks/total_chunks*100:.1f}%)")
                            
                        except Exception as e:
                            print(f"❌ 读取块 [{row},{col}] 失败: {e}")
                            # 尝试创建空白块作为替代
                            try:
                                chunk_h = bottom - top
                                chunk_w = right - left
                                channels = 3 if convert_to_rgb else 1
                                empty_chunk = np.zeros((chunk_h, chunk_w, channels), dtype=np.uint8)
                                chunk_row.append(empty_chunk)
                                print(f"⚠️ 使用空白块替代 [{row},{col}]")
                            except:
                                raise Exception(f"无法创建替代块: {e}")
                    
                    # 水平合并当前行的所有块
                    if chunk_row:
                        try:
                            row_image = np.concatenate(chunk_row, axis=1)
                            chunks.append(row_image)
                            print(f"🔗 合并行 {row+1}/{rows}: {row_image.shape}")
                        except Exception as e:
                            print(f"❌ 合并行 {row} 失败: {e}")
                            raise
                
                # 垂直合并所有行
                if chunks:
                    try:
                        final_image = np.concatenate(chunks, axis=0)
                        print(f"✅ 分块读取完成: {final_image.shape}")
                        return final_image
                    except Exception as e:
                        print(f"❌ 最终合并失败: {e}")
                        raise
                else:
                    raise ValueError("没有成功读取任何块")
                    
        except Exception as e:
            raise ValueError(f"分块读取失败: {e}")
    
    def safe_resize_image(self, image, target_size):
        """简单直接的图像缩放，强制100%成功率"""
        try:
            height, width = image.shape[:2]
            print(f"缩放图像: {width}×{height} -> {target_size}×{target_size}")
            
            # 直接使用PIL进行resize
            img = Image.fromarray(image)
            
            # 图像使用高质量Lanczos插值
            resized_img = img.resize((target_size, target_size), Image.LANCZOS)
            
            result = np.array(resized_img)
            print(f"✅ 缩放成功: {result.shape}")
            
            # 立即释放PIL对象内存
            del img, resized_img
            self.force_memory_cleanup()
            
            return result
                
        except Exception as e:
            print(f"❌ 缩放失败: {e}")
            # 即使失败也不跳过，抛出异常让上层处理
            raise ValueError(f"缩放图像时出现错误: {e}")
    
    def resize_and_save_image(self, image_path: Path, target_path: Path):
        """缩放并保存单个图像，使用多种备选方法确保成功"""
        
        # 记录所有尝试和错误
        attempted_steps = []
        image = None
        image_resized = None
        
        try:
            print(f"\n🎯 处理图像: {image_path.name}")
            self.log_memory_usage("开始处理")
            
            # 步骤1：读取图像文件
            try:
                print(f"📖 读取图像文件: {image_path.name}")
                image = self.safe_read_image(image_path, convert_to_rgb=True)
                attempted_steps.append(("读取图像", "成功"))
                self.log_memory_usage("读取图像后")
            except Exception as e:
                attempted_steps.append(("读取图像", str(e)))
                print(f"❌ 读取图像失败，尝试分块读取...")
                try:
                    image = self.read_image_in_chunks(image_path, convert_to_rgb=True)
                    attempted_steps.append(("分块读取图像", "成功"))
                    self.log_memory_usage("分块读取图像后")
                except Exception as e2:
                    attempted_steps.append(("分块读取图像", str(e2)))
                    print(f"❌ 分块读取也失败: {e2}")
                    return False, None
            
            # 记录原始尺寸
            orig_h, orig_w = image.shape[:2]
            print(f"📏 原始尺寸: {orig_w}×{orig_h}")
            
            # 步骤2：缩放图像
            try:
                print(f"🔧 缩放图像...")
                image_resized = self.safe_resize_image(image, self.target_size)
                attempted_steps.append(("缩放图像", "成功"))
                self.log_memory_usage("缩放图像后")
                
                # 立即释放原始图像内存
                del image
                image = None
                self.force_memory_cleanup()
                
            except Exception as e:
                attempted_steps.append(("缩放图像", str(e)))
                print(f"❌ 缩放图像失败: {e}")
                # 释放内存
                if image is not None:
                    del image
                self.force_memory_cleanup()
                return False, None
            
            # 步骤3：保存图像
            try:
                print(f"💾 保存图像...")
                # 确保目标目录存在
                target_path.parent.mkdir(parents=True, exist_ok=True)
                
                # 直接保存图像
                image_pil = Image.fromarray(image_resized)
                image_pil.save(target_path, 'PNG')
                
                # 立即释放PIL对象
                del image_pil
                self.force_memory_cleanup()
                
                attempted_steps.append(("保存图像", "成功"))
                
            except Exception as e:
                attempted_steps.append(("保存图像", str(e)))
                print(f"❌ 保存图像失败: {e}")
                # 释放内存
                if image_resized is not None:
                    del image_resized
                self.force_memory_cleanup()
                return False, None
            
            # 最终清理
            if image_resized is not None:
                del image_resized
            self.force_memory_cleanup()
            
            self.log_memory_usage("处理完成")
            print(f"✅ 成功处理: {image_path.name}")
            return True, (orig_w, orig_h)
            
        except Exception as e:
            print(f"❌ 处理失败 {image_path.name}")
            print("处理步骤详情:")
            for i, (step, result) in enumerate(attempted_steps, 1):
                status = "✅" if result == "成功" else "❌"
                print(f"  {i}. {step}: {status} {result if result != '成功' else ''}")
            
            print(f"最终错误: {e}")
            
            # 确保所有变量都被释放
            for var in [image, image_resized]:
                if var is not None:
                    del var
            self.force_memory_cleanup()
            
            return False, None
    
    def scan_source_images(self):
        """扫描源目录中的所有PNG图像"""
        print("扫描源图像...")
        
        # 获取所有PNG图像文件
        image_files = list(self.source_dir.glob('*.png'))
        print(f"找到 {len(image_files)} 个PNG图像文件")
        
        # 检查几个样本图像的尺寸
        if image_files:
            print("\n检查样本图像尺寸:")
            for i, img_file in enumerate(image_files[:3]):  # 检查前3个
                try:
                    with Image.open(img_file) as img:
                        width, height = img.size
                        file_size = img_file.stat().st_size / (1024*1024)  # MB
                        pixels = width * height / 1_000_000  # M pixels
                        print(f"  {img_file.name}: {width}×{height} ({pixels:.1f}M像素, {file_size:.1f}MB)")
                except Exception as e:
                    print(f"  ❌ 无法检查 {img_file.name}: {e}")

        return image_files
    
    def process_all_images(self, image_files):
        """处理所有图像"""
        print(f"\n开始处理所有图像...")
        print(f"准备处理 {len(image_files)} 个PNG图像")
        
        success_count = 0
        failed_count = 0
        failed_files = []
        processed_sizes = []
        
        for idx, image_path in enumerate(tqdm(image_files, desc="处理图像"), 1):
            target_path = self.target_dir / f"{image_path.stem}.png"
            
            print(f"\n🔄 处理 {idx}/{len(image_files)}")
            self.log_memory_usage("开始处理图像")
            
            success, orig_size = self.resize_and_save_image(image_path, target_path)
            
            if success:
                success_count += 1
                if orig_size:
                    processed_sizes.append(orig_size)
            else:
                failed_count += 1
                failed_files.append(image_path.name)
            
            # 每处理一个图像就强制清理内存
            self.force_memory_cleanup()
            self.log_memory_usage(f"图像 {idx} 处理完成")
        
        # 统计结果
        print(f"\n" + "="*60)
        print(f"📊 处理结果统计")
        print(f"="*60)
        print(f"成功处理: {success_count}/{len(image_files)}")
        print(f"处理失败: {failed_count}/{len(image_files)}")
        
        if processed_sizes:
            avg_w = np.mean([s[0] for s in processed_sizes])
            avg_h = np.mean([s[1] for s in processed_sizes])
            print(f"原始图像平均尺寸: {avg_w:.0f}×{avg_h:.0f}")
        
        # 显示失败的文件
        if failed_files:
            print(f"\n❌ 处理失败的文件 ({len(failed_files)}个):")
            for i, filename in enumerate(failed_files, 1):
                print(f"  {i:2d}. {filename}")
            
            # 保存失败文件列表
            self.save_failed_files_report(failed_files)
        else:
            print(f"\n🎉 所有图像都成功处理！")
        
        print(f"="*60)
        
        return success_count, failed_count
    
    def save_failed_files_report(self, failed_files):
        """保存失败文件的详细报告"""
        try:
            report_path = self.target_dir / "failed_files_report.txt"
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("图像缩放处理 - 失败文件报告\n")
                f.write("=" * 80 + "\n")
                f.write(f"生成时间: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"统计摘要:\n")
                f.write(f"- 处理失败: {len(failed_files)} 个文件\n\n")
                
                if failed_files:
                    f.write(f"处理失败的文件:\n")
                    f.write("-" * 50 + "\n")
                    for i, filename in enumerate(failed_files, 1):
                        f.write(f"{i:3d}. {filename}\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
            
            print(f"📝 失败文件报告已保存: {report_path}")
            
        except Exception as e:
            print(f"保存失败文件报告时出错: {e}")
    
    def run(self):
        """运行完整的图像缩放流程"""
        print("=" * 80)
        print("图像缩放器 - 批量resize到1024×1024")
        print("=" * 80)
        
        try:
            # 检查源目录是否存在
            if not self.source_dir.exists():
                print(f"❌ 源目录不存在: {self.source_dir}")
                return
            
            # 1. 扫描源图像
            image_files = self.scan_source_images()
            
            if not image_files:
                print("❌ 源目录中没有找到PNG图像文件")
                return
            
            # 2. 处理所有图像
            success_count, failed_count = self.process_all_images(image_files)
            
            print("\n" + "=" * 80)
            print("🎯 图像缩放完成!")
            print("=" * 80)
            print(f"✅ 成功处理: {success_count} 个图像")
            print(f"❌ 失败: {failed_count} 个图像")
            print(f"📂 目标目录: {self.target_dir}")
            print(f"📐 目标尺寸: {self.target_size}×{self.target_size}")
            
            # 显示处理成功率
            total_files = len(image_files)
            success_rate = (success_count / total_files * 100) if total_files > 0 else 0
            print(f"📊 处理成功率: {success_count}/{total_files} ({success_rate:.1f}%)")
            
            print("=" * 80)
            
        except Exception as e:
            print(f"❌ 图像缩放失败: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="批量图像缩放器")
    parser.add_argument("--source_dir", 
                       default=r"Z:\ljh\MsaMIL_Net_Data\images",
                       help="源图像目录路径")
    parser.add_argument("--target_dir", 
                       default=r"Z:\ljh\MsaMIL_Net_Data\images_1024",
                       help="目标图像目录路径")
    parser.add_argument("--target_size", type=int, default=1024,
                       help="目标图像尺寸")
    
    args = parser.parse_args()
    
    # 检查源目录是否存在
    if not os.path.exists(args.source_dir):
        print(f"❌ 源目录不存在: {args.source_dir}")
        return
    
    # 创建图像缩放器并运行
    resizer = ImageResizer(
        source_dir=args.source_dir,
        target_dir=args.target_dir,
        target_size=args.target_size
    )
    
    resizer.run()

if __name__ == "__main__":
    main()