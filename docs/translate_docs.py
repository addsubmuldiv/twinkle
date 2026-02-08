#!/usr/bin/env python3
"""
Comprehensive documentation translation script for Twinkle docs
Translates all files from source/ to source_en/
"""

import os
import shutil
from pathlib import Path

# File mappings: Chinese folder name -> English folder name
FOLDER_MAPPINGS = {
    "使用指引": "Usage Guide",
    "服务端和客户端": "Server and Client",
    "组件": "Components",
    "LRScheduler": "LRScheduler",
    "任务处理器": "Task Processor",
    "指标": "Metrics",
    "损失": "Loss",
    "数据加载": "Data Loading",
    "数据格式": "Data Format",
    "数据集": "Dataset",
    "模型": "Model",
    "模板": "Template",
    "组件化": "Plugin",
    "补丁": "Patch",
    "训练中间件": "Training Middleware",
    "预处理器和过滤器": "Preprocessor and Filter",
}

# File mappings: Chinese filename -> English filename
FILE_MAPPINGS = {
    "快速开始.md": "Quick-Start.md",
    "安装.md": "Installation.md",
    "NPU的支持.md": "NPU-Support.md",
    "魔搭免费资源.md": "ModelScope-Free-Resources.md",
    "概述.md": "Overview.md",
    "服务端.md": "Server.md",
    "Twinkle客户端.md": "Twinkle-Client.md",
    "Tinker兼容客户端.md": "Tinker-Compatible-Client.md",
    "构建指标.md": "Building-Metrics.md",
    "构建损失.md": "Building-Loss.md",
}

def get_all_files(source_dir):
    """Get all files in source directory"""
    all_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if not file.startswith('.'):
                rel_path = os.path.relpath(os.path.join(root, file), source_dir)
                all_files.append(rel_path)
    return sorted(all_files)

def map_path(chinese_path):
    """Map Chinese path to English path"""
    parts = Path(chinese_path).parts
    mapped_parts = []
    
    for part in parts:
        # Check if it's a folder name that needs mapping
        if part in FOLDER_MAPPINGS:
            mapped_parts.append(FOLDER_MAPPINGS[part])
        # Check if it's a file name that needs mapping
        elif part in FILE_MAPPINGS:
            mapped_parts.append(FILE_MAPPINGS[part])
        else:
            # Keep as is (like .md files, .rst files, etc.)
            mapped_parts.append(part)
    
    return str(Path(*mapped_parts))

def main():
    source_dir = Path("/Users/tastelikefeet/code/public/twinkle/docs/source")
    target_dir = Path("/Users/tastelikefeet/code/public/twinkle/docs/source_en")
    
    # Get all files
    all_files = get_all_files(source_dir)
    
    print(f"Found {len(all_files)} files to process")
    print("\nFile mapping plan:")
    print("=" * 80)
    
    for file_path in all_files:
        chinese_path = file_path
        english_path = map_path(file_path)
        print(f"{chinese_path:50} -> {english_path}")
    
    print("\n" + "=" * 80)
    print(f"Total files: {len(all_files)}")

if __name__ == "__main__":
    main()
