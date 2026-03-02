"""
轮廓类定义
"""
import numpy as np
from PyQt5.QtCore import QPointF, QRectF
from PyQt5.QtGui import QColor, QBrush, QPen
import random
import math


class Contour:
    """轮廓类"""

    def __init__(self, points: np.ndarray, contour_id: int, source_image: str = "", label: int = 0):
        self.id = contour_id
        self.original_points = points  # 原始轮廓点
        self.nurbs_points = []  # NURBS曲线点
        self.position = QPointF(0, 0)  # 轮廓位置
        self.scale = 1.0  # 缩放比例
        self.is_selected = False
        self.source_image = source_image  # 来源图像
        self.bounding_box = QRectF()  # 包围盒
        self.color = self.generate_color(contour_id)
        self.label = label  # 添加：轮廓标号
        self.label_text = str(label) if label > 0 else ""  # 添加：标号文本
        self.label_font_size = 12  # 添加：标号字体大小
        # 添加实际尺寸存储
        self.actual_width_cm = 0.0  # 实际宽度（厘米）
        self.actual_height_cm = 0.0  # 实际高度（厘米）
        self.pixel_scale_mm_per_px = None  # 像素比例尺
        # NURBS参数
        self.nurbs_curve = None  # 存储NURBS曲线对象
        self.precision = 0.5  # 拟合精度

        # 计算初始包围盒
        self.calculate_bounding_box()

    def calculate_bounding_box(self):
        """计算轮廓的包围盒"""
        if len(self.original_points) == 0:
            return

        points = self.original_points.squeeze()
        if points.ndim != 2:
            return

        min_x = np.min(points[:, 0])
        max_x = np.max(points[:, 0])
        min_y = np.min(points[:, 1])
        max_y = np.max(points[:, 1])

        self.bounding_box = QRectF(min_x, min_y, max_x - min_x, max_y - min_y)

    def generate_color(self, idx):
        """生成颜色"""
        colors = [
            QColor(255, 0, 0),  # 红
            QColor(0, 180, 0),  # 深绿
            QColor(0, 100, 255),  # 深蓝
            QColor(255, 180, 0),  # 橙黄
            QColor(180, 0, 180),  # 紫
            QColor(0, 180, 180),  # 青
            QColor(255, 100, 0),  # 橙色
            QColor(150, 0, 255),  # 深紫
        ]
        return colors[idx % len(colors)]

    def get_display_rect(self):
        """获取显示时的矩形（考虑缩放和位置）"""
        if self.bounding_box.isNull():
            return QRectF()

        width = self.bounding_box.width() * self.scale
        height = self.bounding_box.height() * self.scale

        # 计算中心点位置
        center_x = self.position.x() + width / 2
        center_y = self.position.y() + height / 2

        # 返回以中心点为基准的矩形
        return QRectF(center_x - width / 2, center_y - height / 2, width, height)


    # def set_size(self, width_cm: float, height_cm: float, pixels_per_cm: float):
    #     """设置包围盒大小（厘米单位）"""
    #     if self.bounding_box.width() > 0 and self.bounding_box.height() > 0:
    #         width_px = width_cm * pixels_per_cm
    #         height_px = height_cm * pixels_per_cm
    #
    #         scale_x = width_px / self.bounding_box.width()
    #         scale_y = height_px / self.bounding_box.height()
    #         self.scale = min(scale_x, scale_y)  # 保持纵横比
    #
    #         # 存储实际尺寸
    #         self.actual_width_cm = width_cm
    #         self.actual_height_cm = height_cm
    def set_size(self, width_cm: float, height_cm: float, pixels_per_cm: float, pixel_scale_mm_per_px: float = None):
        """设置轮廓尺寸和比例尺"""
        if self.bounding_box.width() > 0 and self.bounding_box.height() > 0:
            width_px = width_cm * pixels_per_cm
            height_px = height_cm * pixels_per_cm

            scale_x = width_px / self.bounding_box.width()
            scale_y = height_px / self.bounding_box.height()
            self.scale = min(scale_x, scale_y)

            # 存储实际尺寸和比例尺
            self.actual_width_cm = width_cm
            self.actual_height_cm = height_cm
            self.pixel_scale_mm_per_px = pixel_scale_mm_per_px

    def update_label_size(self, pixels_per_cm: float):
        """更新标号大小"""
        display_rect = self.get_display_rect()
        if display_rect.isNull():
            self.label_font_size = 12
            return

        # 计算标号直径（包围盒宽度的1/5）
        diameter_px = display_rect.width() / 5

        # 限制在2mm到10mm之间
        min_diameter_px = 2 * pixels_per_cm / 10  # 2mm转换为像素
        max_diameter_px = 10 * pixels_per_cm / 10  # 10mm转换为像素

        diameter_px = max(min_diameter_px, min(diameter_px, max_diameter_px))

        # 根据直径设置字体大小（经验公式）
        self.label_font_size = int(diameter_px * 0.6)

        # 确保最小字体大小
        min_font_size = int(min_diameter_px * 0.4)
        max_font_size = int(max_diameter_px * 0.6)
        self.label_font_size = max(min_font_size, min(self.label_font_size, max_font_size))