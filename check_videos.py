#!/usr/bin/env python
"""检查数据集中视频文件的完整性"""

import subprocess
import sys
from pathlib import Path


def check_video_integrity(video_path: Path) -> bool:
    """使用 FFmpeg 检查视频文件是否完整"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(video_path), "-f", "null", "-"],
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            timeout=30,
        )
        if result.returncode != 0:
            print(f"❌ 损坏: {video_path}")
            print(f"   错误信息: {result.stderr.decode('utf-8', errors='ignore')[:200]}")
            return False
        else:
            print(f"✅ 正常: {video_path}")
            return True
    except subprocess.TimeoutExpired:
        print(f"⚠️  超时: {video_path}")
        return False
    except Exception as e:
        print(f"⚠️  检查失败: {video_path} - {e}")
        return False


def fix_video(video_path: Path, output_path: Path | None = None) -> bool:
    """重新编码视频文件以修复损坏"""
    if output_path is None:
        output_path = video_path.parent / f"{video_path.stem}_fixed{video_path.suffix}"
    
    try:
        cmd = [
            "ffmpeg",
            "-i", str(video_path),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-y",  # 覆盖输出文件
            str(output_path),
        ]
        result = subprocess.run(
            cmd,
            stderr=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            timeout=120,
        )
        if result.returncode == 0:
            print(f"✅ 修复成功: {output_path}")
            return True
        else:
            print(f"❌ 修复失败: {video_path}")
            print(f"   错误: {result.stderr.decode('utf-8', errors='ignore')[:200]}")
            return False
    except Exception as e:
        print(f"⚠️  修复异常: {video_path} - {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("用法: python check_videos.py <数据集根目录> [--fix]")
        print("示例: python check_videos.py /root/public/ghr/datasets/record1")
        print("      python check_videos.py /root/public/ghr/datasets/record1 --fix")
        sys.exit(1)
    
    dataset_root = Path(sys.argv[1])
    fix_mode = "--fix" in sys.argv
    
    if not dataset_root.exists():
        print(f"错误: 目录不存在: {dataset_root}")
        sys.exit(1)
    
    # 查找所有 MP4 文件
    video_files = list(dataset_root.rglob("*.mp4"))
    if not video_files:
        print(f"未在 {dataset_root} 中找到 MP4 文件")
        sys.exit(1)
    
    print(f"找到 {len(video_files)} 个视频文件,开始检查...\n")
    
    damaged_videos = []
    for i, video_path in enumerate(video_files, 1):
        print(f"[{i}/{len(video_files)}] ", end="")
        if not check_video_integrity(video_path):
            damaged_videos.append(video_path)
    
    print(f"\n{'='*60}")
    print(f"检查结果:")
    print(f"  总文件数: {len(video_files)}")
    print(f"  损坏文件: {len(damaged_videos)}")
    print(f"  正常文件: {len(video_files) - len(damaged_videos)}")
    
    if damaged_videos:
        print(f"\n损坏的文件列表:")
        for v in damaged_videos:
            print(f"  - {v}")
        
        if fix_mode:
            print(f"\n开始修复损坏的视频...")
            fixed_count = 0
            for video_path in damaged_videos:
                output_path = video_path.parent / f"{video_path.stem}_fixed{video_path.suffix}"
                if fix_video(video_path, output_path):
                    fixed_count += 1
                    # 备份原文件并替换
                    backup_path = video_path.with_suffix(".mp4.bak")
                    video_path.rename(backup_path)
                    output_path.rename(video_path)
                    print(f"   已替换原文件,备份至: {backup_path}")
            print(f"\n修复完成: {fixed_count}/{len(damaged_videos)} 个文件")
        else:
            print(f"\n提示: 添加 --fix 参数可自动尝试修复损坏的视频")
    else:
        print("\n🎉 所有视频文件均完好!")


if __name__ == "__main__":
    main()
