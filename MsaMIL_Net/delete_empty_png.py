#!/usr/bin/env python3
import os
from pathlib import Path

def delete_empty_png_files(directory: str):
    """删除指定目录及其子目录下大小为0KB的PNG文件"""
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"目录不存在: {directory}")
        return
        
    # 递归查找所有PNG文件
    png_files = dir_path.rglob("*.png")
    deleted_count = 0
    
    for png_file in png_files:
        try:
            # 获取文件大小（字节）
            file_size = png_file.stat().st_size
            # 如果文件大小为0字节，则删除
            if file_size == 0:
                png_file.unlink()
                print(f"已删除: {png_file}")
                deleted_count += 1
        except Exception as e:
            print(f"处理文件 {png_file} 时出错: {e}")
    
    print(f"\n在目录 {directory} 中删除了 {deleted_count} 个空PNG文件")

def main():
    # 定义要处理的目录列表
    directories = [
        "/mnt/nas/ljh/MsaMIL_Net_Data/results/patches_5x",
        "/mnt/nas/ljh/MsaMIL_Net_Data/results/patches_10x",
        "/mnt/nas/ljh/MsaMIL_Net_Data/results/patches_20x"
    ]
    
    # 处理每个目录
    for directory in directories:
        print(f"\n正在处理目录: {directory}")
        print("-" * 50)
        delete_empty_png_files(directory)
    
    print("\n所有目录处理完成！")

if __name__ == "__main__":
    main()
