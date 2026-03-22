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
        self.scale = 1.0  # 保留，兼容旧代码
        self.scale_x = 1.0  # x 方向缩放
        self.scale_y = 1.0  # y 方向缩放
        self.is_selected = False
        self.source_image = source_image  # 来源图像
        self.bounding_box = QRectF()  # 包围盒
        # self.color = self.generate_color(contour_id) # 基于轮廓id分配颜色
        self.color = self.generate_color(label if label > 0 else contour_id)  # 优先用标号
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
        self.control_points = 50  # 默认控制点数
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
        if self.bounding_box.isNull():
            return QRectF()
        width = self.bounding_box.width() * self.scale_x
        height = self.bounding_box.height() * self.scale_y
        center_x = self.position.x() + width / 2
        center_y = self.position.y() + height / 2
        return QRectF(center_x - width / 2, center_y - height / 2, width, height)

    def get_geometric_center(self) -> QPointF:
        if not self.nurbs_points:
            return QPointF(0, 0)
        sum_x = sum(p.x() for p in self.nurbs_points)
        sum_y = sum(p.y() for p in self.nurbs_points)
        n = len(self.nurbs_points)
        local_center = QPointF(sum_x / n, sum_y / n)
        display_rect = self.get_display_rect()
        if display_rect.isNull():
            return QPointF(0, 0)
        bbox_tl = self.bounding_box.topLeft()
        scale = self.scale
        display_x = display_rect.left() + (local_center.x() - bbox_tl.x()) * scale
        display_y = display_rect.top() + (local_center.y() - bbox_tl.y()) * scale
        return QPointF(display_x, display_y)

    from PyQt5.QtCore import QPointF

    def get_label_position(self, pixels_per_cm: float, font_size_mm: float, min_size_mm: float,
                           step_ratio: float = 0.5):
        """
        在轮廓内部寻找一个能容纳标号的矩形区域，返回其中心点的显示坐标和竖直距离（局部像素）。
        :param pixels_per_cm: 像素/厘米
        :param font_size_mm: 字体高度（毫米）
        :param min_size_mm: 轮廓最小尺寸阈值（毫米），小于此值不标
        :param step_ratio: 扫描步长与字体宽度的比例
        :return: (QPointF or None, float) 显示坐标和竖直距离（局部像素），若未找到则返回 (None, 0)
        """
        if not self.nurbs_points or len(self.nurbs_points) < 3:
            return None, 0

        # 计算局部像素中的字体宽度和最小尺寸阈值
        min_scale = min(self.scale_x, self.scale_y)
        font_size_local = font_size_mm * (pixels_per_cm / 10) / min_scale
        min_size_local = min_size_mm * (pixels_per_cm / 10) / min_scale

        # 如果轮廓包围盒尺寸过小，直接返回
        bbox_width = self.bounding_box.width()
        bbox_height = self.bounding_box.height()
        if bbox_width < min_size_local or bbox_height < min_size_local:
            return None, 0

        pts = [(p.x(), p.y()) for p in self.nurbs_points]
        x_min = min(p[0] for p in pts)
        x_max = max(p[0] for p in pts)
        step = max(1.0, font_size_local * step_ratio)

        best_point = None
        best_score = -1
        best_vertical_dist = 0

        # 从左到右扫描
        x = x_min
        while x <= x_max:
            # 获取竖直线与轮廓的交点
            intersections = []
            for j in range(len(pts)):
                p1 = pts[j]
                p2 = pts[(j + 1) % len(pts)]
                if (p1[0] <= x <= p2[0]) or (p2[0] <= x <= p1[0]):
                    if p1[0] == p2[0]:
                        continue
                    t = (x - p1[0]) / (p2[0] - p1[0])
                    y = p1[1] + t * (p2[1] - p1[1])
                    intersections.append(y)

            if intersections:
                intersections.sort()
                # 将交点配对为区间（假设轮廓是闭合的，交点数为偶数）
                for k in range(0, len(intersections), 2):
                    if k + 1 >= len(intersections):
                        break
                    y1 = intersections[k]
                    y2 = intersections[k + 1]
                    dist_y = y2 - y1
                    if dist_y >= font_size_local:
                        y_mid = (y1 + y2) / 2

                        # 通过 (x, y_mid) 作水平线，检查水平跨度
                        h_intersections = []
                        for j in range(len(pts)):
                            p1 = pts[j]
                            p2 = pts[(j + 1) % len(pts)]
                            if (p1[1] <= y_mid <= p2[1]) or (p2[1] <= y_mid <= p1[1]):
                                if p1[1] == p2[1]:
                                    continue
                                t = (y_mid - p1[1]) / (p2[1] - p1[1])
                                x_h = p1[0] + t * (p2[0] - p1[0])
                                h_intersections.append(x_h)

                        if h_intersections:
                            h_intersections.sort()
                            for hk in range(0, len(h_intersections), 2):
                                if hk + 1 >= len(h_intersections):
                                    break
                                x1_h = h_intersections[hk]
                                x2_h = h_intersections[hk + 1]
                                if x1_h <= x <= x2_h:
                                    dist_x = x2_h - x1_h
                                    if dist_x >= font_size_local:
                                        # 评分：以竖直距离为基准，越大越好
                                        score = dist_y
                                        if score > best_score:
                                            best_score = score
                                            best_vertical_dist = dist_y
                                            # 将局部点转换为显示坐标，使用 scale_x 和 scale_y 分别转换
                                            display_rect = self.get_display_rect()
                                            bbox_tl = self.bounding_box.topLeft()
                                            display_x = display_rect.left() + (x - bbox_tl.x()) * self.scale_x
                                            display_y = display_rect.top() + (y_mid - bbox_tl.y()) * self.scale_y
                                            best_point = QPointF(display_x, display_y)
                                        break  # 只取包含 x 的那个区间
            x += step

        if best_point is not None:
            return best_point, best_vertical_dist
        else:
            return None, 0

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
        """设置轮廓尺寸和比例尺，支持非均匀缩放"""
        if self.bounding_box.width() > 0 and self.bounding_box.height() > 0:
            width_px = width_cm * pixels_per_cm
            height_px = height_cm * pixels_per_cm
            self.scale_x = width_px / self.bounding_box.width()
            self.scale_y = height_px / self.bounding_box.height()
            # 兼容旧代码：scale 设为平均值
            self.scale = (self.scale_x + self.scale_y) / 2.0

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