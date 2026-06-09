"""
轮廓类定义
"""
import numpy as np
from PyQt5.QtCore import QPointF, QRectF
from PyQt5.QtGui import QColor, QBrush, QPen
from typing import List
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
        # --- 新增：质量评估和输出方式 ---
        self.use_nurbs_for_export = True  # DXF导出时是否使用NURBS点（True=NURBS, False=原始点）
        self.fit_quality = None  # 拟合质量评估结果 dict
        # 新增：标签位置相关属性
        self.patient_name = ""  # 病人名字
        self.contour_index = 0  # 该病人的第几个轮廓（从1开始）
        self.label_offset_mm = 3.0  # 标签距离轮廓底部的偏移量（毫米）
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

    def get_display_rect_with_label(self, pixels_per_cm: float, label_height_mm: float = 8.0) -> QRectF:
        """
        获取包含标签空间的显示矩形。

        该方法返回轮廓和标签的合并边界，用于排样算法的碰撞检测。
        标签可能比轮廓更宽（例如患者名较长时），因此需要合并两个矩形。

        :param pixels_per_cm: 像素/厘米
        :param label_height_mm: 标签区域总高度（毫米，未使用，保留以兼容）
        :return: 包含轮廓和标签的显示矩形
        """
        base_rect = self.get_display_rect()
        if base_rect.isNull():
            return QRectF()

        # 获取标签的实际边界（使用 get_label_bounds）
        # 注意：字体大小需要与 canvas_widget.py 中绘制时的值一致（3.0mm）
        label_bounds = self.get_label_bounds(
            pixels_per_cm=pixels_per_cm,
            font_size_mm=3.0,  # 与 canvas_widget.label_font_size_mm 一致
            bg_padding=4.0
        )

        # 如果标签边界无效，只返回轮廓矩形
        if label_bounds.isNull() or label_bounds.width() <= 0 or label_bounds.height() <= 0:
            return base_rect

        # 合并轮廓矩形和标签矩形
        # 使用 normalized() 确保矩形坐标规范（宽高为正）
        combined_rect = base_rect.united(label_bounds).normalized()

        return combined_rect

    def get_label_bounds(self, pixels_per_cm: float, font_size_mm: float = 3.0,
                         bg_padding: float = 4.0) -> QRectF:
        """
        获取标签文本的实际绘制边界（用于精确的碰撞检测）。
        标签位于轮廓下方中心位置。

        :param pixels_per_cm: 像素/厘米
        :param font_size_mm: 字体大小（毫米）
        :param bg_padding: 背景内边距（像素）
        :return: 标签的边界矩形（显示坐标系）
        """
        if len(self.original_points) < 3:
            return QRectF()

        # 计算标签位置（中心点）
        label_pos = self.get_label_position_below(pixels_per_cm, font_size_mm, self.label_offset_mm)
        if label_pos.x() == 0 and label_pos.y() == 0:
            return QRectF()

        # 估算文本大小（像素）
        font_size_px = font_size_mm * pixels_per_cm / 10
        font_size_px = max(6, min(30, font_size_px))

        # 估算文本宽度：假设每个字符宽度约为字体大小的 0.6 倍
        label_text = self.get_full_label_text()
        text_width_px = len(label_text) * font_size_px * 0.6
        # 文本高度约为字体大小的 1.2 倍（行高）
        text_height_px = font_size_px * 1.2

        # 计算标签边界（包含内边距）
        label_bounds = QRectF(
            label_pos.x() - text_width_px / 2 - bg_padding,
            label_pos.y() - text_height_px / 2 - bg_padding,
            text_width_px + bg_padding * 2,
            text_height_px + bg_padding * 2
        )

        return label_bounds

    def get_geometric_center(self) -> QPointF:
        if len(self.original_points) < 3:
            return QPointF(0, 0)
        pts = self.original_points.squeeze()
        if pts.ndim != 2:
            return QPointF(0, 0)
        sum_x = sum(float(p[0]) for p in pts)
        sum_y = sum(float(p[1]) for p in pts)
        n = len(pts)
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
        if len(self.original_points) < 3:
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

        # 使用原始轮廓点
        orig_pts = self.original_points.squeeze()
        if orig_pts.ndim != 2:
            return None, 0
        pts = [(float(p[0]), float(p[1])) for p in orig_pts]
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

    def get_label_position_below(self, pixels_per_cm: float, font_size_mm: float = 3.0,
                                 offset_mm: float = 3.0) -> QPointF:
        """
        计算轮廓下方标号位置，返回显示坐标系中的 QPointF。
        标签位于轮廓包围盒下方中心位置。

        :param pixels_per_cm: 像素/厘米
        :param font_size_mm: 字体大小（毫米），用于计算标签区域高度
        :param offset_mm: 标签距离轮廓底部的偏移量（毫米）
        :return: QPointF 显示坐标，如果轮廓无效返回 QPointF(0, 0)
        """
        if len(self.original_points) < 3:
            return QPointF(0, 0)

        display_rect = self.get_display_rect()
        if display_rect.isNull():
            return QPointF(0, 0)

        # 计算标签位置：轮廓包围盒底部中心向下偏移
        # X坐标：包围盒中心
        label_x = display_rect.center().x()

        # Y坐标：包围盒底部 + 偏移量（转换为像素）
        offset_px = offset_mm * pixels_per_cm / 10  # 毫米转像素
        label_y = display_rect.bottom() + offset_px

        return QPointF(label_x, label_y)

    def get_full_label_text(self) -> str:
        """
        获取完整的标签文本，格式：病人名字-轮廓序号
        例如："张三-1" 表示张三的第1个轮廓

        为避免标签过长遮挡轮廓，患者名限制为4个字符

        :return: 完整标签文本
        """
        if self.patient_name and self.contour_index > 0:
            # 限制患者名长度为4个字符，避免标签过大
            short_name = self.patient_name[:4]
            return f"{short_name}-{self.contour_index}"
        # 回退到旧格式（仅标号）
        return str(self.label) if self.label > 0 else ""

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

    def get_export_points(self) -> List[QPointF]:
        """
        获取用于DXF导出的轮廓点（始终使用原始轮廓点）

        Returns:
            List[QPointF]: 用于导出的点列表（原始轮廓点）
        """
        # 始终返回原始点
        if len(self.original_points) > 0:
            pts = self.original_points.squeeze()
            if pts.ndim == 2 and pts.shape[1] == 2:
                return [QPointF(float(p[0]), float(p[1])) for p in pts]
        return []