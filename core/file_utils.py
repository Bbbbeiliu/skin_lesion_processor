"""
文件处理工具
"""
import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict
import re


class FileUtils:
    """文件处理工具类"""

    @staticmethod
    def find_matching_mask(overlay_path: str, mask_dir: str) -> Optional[str]:
        """
        根据原图路径找到对应的分割结果文件

        Args:
            overlay_path: 原图文件路径 (xxx_overlay.png)
            mask_dir: 分割结果文件夹路径

        Returns:
            对应的分割结果文件路径，如果未找到返回None
        """
        try:
            # 获取原图文件名
            overlay_name = Path(overlay_path).name

            # 从原图文件名中提取前缀（移除_overlay后缀）
            if '_overlay' in overlay_name:
                prefix = overlay_name.split('_overlay')[0]
                # 构建对应的分割结果文件名
                mask_name = f"{prefix}_mask.png"
                mask_path = Path(mask_dir) / mask_name

                if mask_path.exists():
                    return str(mask_path)
                else:
                    # 尝试其他可能的扩展名
                    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                        mask_path = Path(mask_dir) / f"{prefix}_mask{ext}"
                        if mask_path.exists():
                            return str(mask_path)

            return None

        except Exception as e:
            print(f"查找分割结果文件时出错: {e}")
            return None

    @staticmethod
    def find_matching_overlay(mask_path: str, overlay_dir: str) -> Optional[str]:
        """
        根据分割结果路径找到对应的原图文件

        Args:
            mask_path: 分割结果文件路径 (xxx_mask.png)
            overlay_dir: 原图文件夹路径

        Returns:
            对应的原图文件路径，如果未找到返回None
        """
        try:
            # 获取分割结果文件名
            mask_name = Path(mask_path).name

            # 从分割结果文件名中提取前缀（移除_mask后缀）
            if '_mask' in mask_name:
                prefix = mask_name.split('_mask')[0]
                # 构建对应的原图文件名
                overlay_name = f"{prefix}_overlay.png"
                overlay_path = Path(overlay_dir) / overlay_name

                if overlay_path.exists():
                    return str(overlay_path)
                else:
                    # 尝试其他可能的扩展名
                    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
                        overlay_path = Path(overlay_dir) / f"{prefix}_overlay{ext}"
                        if overlay_path.exists():
                            return str(overlay_path)

            return None

        except Exception as e:
            print(f"查找原图文件时出错: {e}")
            return None

    @staticmethod
    def validate_image_pair(overlay_path: str, mask_path: str) -> bool:
        """
        验证原图和分割结果文件是否匹配

        Args:
            overlay_path: 原图文件路径
            mask_path: 分割结果文件路径

        Returns:
            是否匹配
        """
        try:
            overlay_name = Path(overlay_path).stem
            mask_name = Path(mask_path).stem

            # 提取前缀进行比较
            overlay_prefix = overlay_name.replace('_overlay', '')
            mask_prefix = mask_name.replace('_mask', '')

            return overlay_prefix == mask_prefix

        except Exception:
            return False

    @staticmethod
    def scan_image_pairs(overlay_dir: str, mask_dir: str) -> List[Tuple[str, str]]:
        """
        扫描文件夹，找到所有匹配的原图和分割结果文件对

        Args:
            overlay_dir: 原图文件夹路径
            mask_dir: 分割结果文件夹路径

        Returns:
            文件对列表 [(overlay_path, mask_path), ...]
        """
        image_pairs = []

        try:
            # 扫描原图文件夹
            overlay_dir_path = Path(overlay_dir)
            mask_dir_path = Path(mask_dir)

            if not overlay_dir_path.exists():
                print(f"原图文件夹不存在: {overlay_dir}")
                return image_pairs

            if not mask_dir_path.exists():
                print(f"分割结果文件夹不存在: {mask_dir}")
                return image_pairs

            # 获取所有原图文件
            overlay_patterns = [
                "*_overlay.png", "*_overlay.jpg", "*_overlay.jpeg",
                "*_overlay.bmp", "*_overlay.tif", "*_overlay.tiff"
            ]

            overlay_files = []
            for pattern in overlay_patterns:
                overlay_files.extend(list(overlay_dir_path.glob(pattern)))

            # 为每个原图文件查找对应的分割结果文件
            for overlay_file in overlay_files:
                mask_path = FileUtils.find_matching_mask(str(overlay_file), mask_dir)
                if mask_path and Path(mask_path).exists():
                    image_pairs.append((str(overlay_file), mask_path))
                else:
                    print(f"警告: 未找到与 {overlay_file.name} 对应的分割结果文件")

            print(f"找到 {len(image_pairs)} 个匹配的图像对")
            return image_pairs

        except Exception as e:
            print(f"扫描图像对时出错: {e}")
            return image_pairs

    @staticmethod
    def get_image_pair_info(overlay_path: str, mask_path: str) -> Dict:
        """
        获取图像对的信息

        Args:
            overlay_path: 原图文件路径
            mask_path: 分割结果文件路径

        Returns:
            图像对信息字典
        """
        try:
            overlay_name = Path(overlay_path).name
            mask_name = Path(mask_path).name
            prefix = overlay_name.split('_overlay')[0] if '_overlay' in overlay_name else Path(overlay_path).stem

            return {
                'prefix': prefix,
                'overlay_name': overlay_name,
                'overlay_path': overlay_path,
                'mask_name': mask_name,
                'mask_path': mask_path,
                'is_valid': FileUtils.validate_image_pair(overlay_path, mask_path)
            }
        except Exception as e:
            print(f"获取图像对信息时出错: {e}")
            return {}