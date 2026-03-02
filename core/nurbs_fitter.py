"""
NURBS曲线拟合模块（现在方法已移到AdvancedImageProcessor中）
"""
try:
    from geomdl import NURBS
    from geomdl import knotvector
    GEOMDL_AVAILABLE = True
except ImportError:
    GEOMDL_AVAILABLE = False
    print("警告: geomdl库未安装，无法使用NURBS曲线拟合。请运行: pip install geomdl")

# 现在所有方法都在AdvancedImageProcessor中
# 这个文件只保留GEOMDL_AVAILABLE变量供其他模块使用