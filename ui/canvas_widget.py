"""
画布控件
"""
import numpy as np
import traceback
import random
import math
from typing import List, Optional, Dict, Any
from PyQt5.QtWidgets import QWidget, QApplication  # 添加 QApplication 导入
from PyQt5.QtCore import Qt, pyqtSignal, QPointF, QRectF, QSizeF
from PyQt5.QtGui import (QPainter, QPen, QColor, QBrush, QPainterPath,
                         QFont, QFontMetrics, QMouseEvent, QCursor, QPixmap)

from core.contour import Contour
from core.image_processor import AdvancedImageProcessor, GEOMDL_AVAILABLE


class CanvasWidget(QWidget):
    """画布控件"""
    contour_selected = pyqtSignal(object)  # 轮廓选中信号
    contour_changed = pyqtSignal()  # 轮廓变化信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.contours: List[Contour] = []
        self.selected_contour: Optional[Contour] = None
        self.dragging = False
        self.dragging_handle = False
        self.dragging_canvas = False  # 添加这行
        self.drag_offset = QPointF(0, 0)
        self.drag_start_pos = QPointF(0, 0)
        self.resize_handle_idx = -1
        self.hovered_contour: Optional[Contour] = None
        self.hovered_handle_idx = -1
        self.label_to_image_mapping: Dict[int, str] = {}

        # 标号设置变量
        self.label_font_size_mm = 3.0  # 标号字体大小（毫米）
        self.label_min_size_mm = 3.0  # 最小轮廓尺寸阈值（毫米）

        # 添加工具变量
        self.current_tool = "select"  # select, pan
        self.pan_start_pos = QPointF(0, 0)
        self.pan_offset = QPointF(0, 0)
        self.zoom_factor = 1.0
        self.last_mouse_pos = QPointF(0, 0)

        # 显示选项
        self.show_nurbs_curve = True
        self.show_original_contour = False
        self.show_labels = True
        self.show_bounding_box = True

        # 画布参数
        self.pixels_per_cm = 75.59
        self.canvas_width_cm = 15
        self.canvas_height_cm = 15
        self.canvas_width_px = int(self.canvas_width_cm * self.pixels_per_cm)
        self.canvas_height_px = int(self.canvas_height_cm * self.pixels_per_cm)

        self.setMouseTracking(True)
        self.setFixedSize(self.canvas_width_px, self.canvas_height_px)

        self.setStyleSheet("""
            QWidget {
                background-color: white;
                border: 2px solid #cccccc;
            }
        """)



    def clear(self):
        """清空画布"""
        self.contours.clear()
        self.selected_contour = None
        self.hovered_contour = None
        self.label_to_image_mapping.clear()
        self.update()

    def add_contour(self, points: np.ndarray, source_image: str = "", label: int = 0,
                    precision: float = 0.5, control_points: int = 120) -> Contour:
        contour = Contour(points, len(self.contours), source_image, label)
        contour.precision = precision
        contour.control_points = control_points  # 保存控制点数量

        # 使用 NURBS 拟合（传入控制点数量）
        nurbs_points, nurbs_curve = AdvancedImageProcessor.smooth_contour_with_nurbs(
            points, num_control_points=control_points
        )
        contour.nurbs_points = nurbs_points
        contour.nurbs_curve = nurbs_curve

        self.contours.append(contour)
        self.update()
        return contour

    def get_handle_rect(self, contour: Contour, handle_idx: int) -> QRectF:
        """获取控制点的矩形区域（不考虑视图变换，因为会在外部应用）"""
        if not contour:
            return QRectF()

        rect = contour.get_display_rect()
        if rect.isNull():
            return QRectF()

        handle_size = 10

        if handle_idx == 0:  # 左上
            return QRectF(rect.topLeft() - QPointF(handle_size / 2, handle_size / 2),
                          QSizeF(handle_size, handle_size))
        elif handle_idx == 1:  # 上中
            return QRectF(QPointF(rect.center().x() - handle_size / 2,
                                  rect.top() - handle_size / 2),
                          QSizeF(handle_size, handle_size))
        elif handle_idx == 2:  # 右上
            return QRectF(rect.topRight() - QPointF(handle_size / 2, handle_size / 2),
                          QSizeF(handle_size, handle_size))
        elif handle_idx == 3:  # 右中
            return QRectF(QPointF(rect.right() - handle_size / 2,
                                  rect.center().y() - handle_size / 2),
                          QSizeF(handle_size, handle_size))
        elif handle_idx == 4:  # 右下
            return QRectF(rect.bottomRight() - QPointF(handle_size / 2, handle_size / 2),
                          QSizeF(handle_size, handle_size))
        elif handle_idx == 5:  # 下中
            return QRectF(QPointF(rect.center().x() - handle_size / 2,
                                  rect.bottom() - handle_size / 2),
                          QSizeF(handle_size, handle_size))
        elif handle_idx == 6:  # 左下
            return QRectF(rect.bottomLeft() - QPointF(handle_size / 2, handle_size / 2),
                          QSizeF(handle_size, handle_size))
        elif handle_idx == 7:  # 左中
            return QRectF(QPointF(rect.left() - handle_size / 2,
                                  rect.center().y() - handle_size / 2),
                          QSizeF(handle_size, handle_size))

        return QRectF()

    def draw_cm_grid(self, painter: QPainter):
        """绘制厘米网格（加深颜色）"""
        painter.save()

        # 绘制厘米网格 - 使用深色
        painter.setPen(QPen(QColor(100, 100, 100), 1))

        # 垂直线 (每厘米)
        for x in range(0, self.canvas_width_px, int(self.pixels_per_cm)):
            painter.drawLine(x, 0, x, self.canvas_height_px)

        # 水平线 (每厘米)
        for y in range(0, self.canvas_height_px, int(self.pixels_per_cm)):
            painter.drawLine(0, y, self.canvas_width_px, y)

        # 绘制5厘米网格（更粗、更深色）
        painter.setPen(QPen(QColor(70, 70, 70), 2))
        for x in range(0, self.canvas_width_px, int(5 * self.pixels_per_cm)):
            painter.drawLine(x, 0, x, self.canvas_height_px)
        for y in range(0, self.canvas_height_px, int(5 * self.pixels_per_cm)):
            painter.drawLine(0, y, self.canvas_width_px, y)

        # 绘制边框
        painter.setPen(QPen(QColor(150, 150, 150), 3))
        painter.drawRect(0, 0, self.canvas_width_px - 1, self.canvas_height_px - 1)

        # 绘制刻度标签
        painter.setPen(QColor(80, 80, 80))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)

        for x in range(0, self.canvas_width_px, int(self.pixels_per_cm)):
            cm = x / self.pixels_per_cm
            if cm % 5 == 0 and cm > 0:
                painter.drawText(x + 5, 18, f"{int(cm)}")

        for y in range(0, self.canvas_height_px, int(self.pixels_per_cm)):
            cm = y / self.pixels_per_cm
            if cm % 5 == 0 and cm > 0:
                painter.drawText(5, y - 8, f"{int(cm)}")

        # 画布尺寸标签
        font.setPointSize(12)
        painter.setFont(font)
        painter.drawText(10, 25, f"{self.canvas_width_cm}cm × {self.canvas_height_cm}cm")

        painter.restore()

    def draw_contour(self, painter: QPainter, contour: Contour):
        """绘制单个轮廓"""
        try:
            painter.save()

            display_rect = contour.get_display_rect()
            if display_rect.isNull():
                painter.restore()
                return

            scale_x = contour.scale_x
            scale_y = contour.scale_y
            translate_x = display_rect.left() - contour.bounding_box.left() * scale_x
            translate_y = display_rect.top() - contour.bounding_box.top() * scale_y

            painter.translate(translate_x, translate_y)
            painter.scale(scale_x, scale_y)

            # 绘制原始轮廓 - 使用深灰色
            if self.show_original_contour and len(contour.original_points) > 1:
                painter.setPen(QPen(QColor(150, 150, 150), 3, Qt.DashLine))

                path = QPainterPath()
                points = contour.original_points.squeeze()

                if points.ndim == 2 and len(points) > 0:
                    path.moveTo(float(points[0][0]), float(points[0][1]))
                    for i in range(1, len(points)):
                        path.lineTo(float(points[i][0]), float(points[i][1]))

                    if not path.isEmpty():
                        path.closeSubpath()
                        painter.drawPath(path)

            # 绘制NURBS曲线
            if self.show_nurbs_curve and contour.nurbs_points:
                # 根据是否选中设置颜色和线宽
                if contour.is_selected:
                    pen_color = QColor(0, 100, 255)  # 深蓝色
                    pen_width = 4 / min(scale_x, scale_y)
                else:
                    # 使用轮廓的原始颜色
                    pen_color = contour.color
                    pen_width = 3 / min(scale_x, scale_y)

                painter.setPen(QPen(pen_color, pen_width))

                path = QPainterPath()
                if contour.nurbs_points:
                    path.moveTo(contour.nurbs_points[0])
                    for point in contour.nurbs_points[1:]:
                        path.lineTo(point)

                    # 确保闭合
                    if not path.isEmpty() and contour.nurbs_points[0] != contour.nurbs_points[-1]:
                        path.lineTo(contour.nurbs_points[0])

                    painter.drawPath(path)

            painter.restore()

            # 绘制标号（在屏幕坐标下）
            if self.show_labels and contour.label > 0:
                self.draw_contour_label(painter, contour)

        except Exception as e:
            print(f"绘制轮廓错误: {str(e)}")

    def draw_contour_label(self, painter: QPainter, contour: Contour):
        try:
            painter.save()

            # 获取标号位置和竖直距离
            label_pos, dist_px = contour.get_label_position(
                pixels_per_cm=self.pixels_per_cm,
                font_size_mm=self.label_font_size_mm,
                min_size_mm=self.label_min_size_mm
            )
            if label_pos is None or dist_px <= 0:
                painter.restore()
                return

            # 计算字体像素大小（固定毫米转换为像素）
            font_size_px = self.label_font_size_mm * self.pixels_per_cm / 10
            font_size_px = max(6, min(30, font_size_px))

            font = painter.font()
            font.setPixelSize(int(font_size_px))
            font.setBold(True)
            painter.setFont(font)

            label_text = str(contour.label) if contour.label > 0 else ""
            if not label_text:
                painter.restore()
                return

            fm = QFontMetrics(font)
            text_rect = fm.boundingRect(label_text)
            draw_rect = QRectF(
                label_pos.x() - text_rect.width() / 2,
                label_pos.y() - text_rect.height() / 2,
                text_rect.width(),
                text_rect.height()
            )
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawText(draw_rect, Qt.AlignCenter, label_text)

            painter.restore()
        except Exception as e:
            print(f"绘制标号错误: {str(e)}")

    def draw_selection_indicators(self, painter: QPainter, contour: Contour):
        """绘制选中轮廓的指示器"""
        painter.save()

        # 在显示坐标中绘制
        display_rect = contour.get_display_rect()
        if display_rect.isNull():
            painter.restore()
            return

        # 绘制包围盒
        painter.setPen(QPen(QColor(0, 100, 255), 2, Qt.DashLine))
        painter.drawRect(display_rect)

        # 绘制控制点
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        painter.setBrush(QBrush(QColor(0, 100, 255)))

        handle_size = 8

        # 左上
        painter.drawRect(QRectF(
            display_rect.left() - handle_size / 2,
            display_rect.top() - handle_size / 2,
            handle_size, handle_size
        ))

        # 上中
        painter.drawRect(QRectF(
            display_rect.center().x() - handle_size / 2,
            display_rect.top() - handle_size / 2,
            handle_size, handle_size
        ))

        # 右上
        painter.drawRect(QRectF(
            display_rect.right() - handle_size / 2,
            display_rect.top() - handle_size / 2,
            handle_size, handle_size
        ))

        # 右中
        painter.drawRect(QRectF(
            display_rect.right() - handle_size / 2,
            display_rect.center().y() - handle_size / 2,
            handle_size, handle_size
        ))

        # 右下
        painter.drawRect(QRectF(
            display_rect.right() - handle_size / 2,
            display_rect.bottom() - handle_size / 2,
            handle_size, handle_size
        ))

        # 下中
        painter.drawRect(QRectF(
            display_rect.center().x() - handle_size / 2,
            display_rect.bottom() - handle_size / 2,
            handle_size, handle_size
        ))

        # 左下
        painter.drawRect(QRectF(
            display_rect.left() - handle_size / 2,
            display_rect.bottom() - handle_size / 2,
            handle_size, handle_size
        ))

        # 左中
        painter.drawRect(QRectF(
            display_rect.left() - handle_size / 2,
            display_rect.center().y() - handle_size / 2,
            handle_size, handle_size
        ))

        # 显示尺寸
        painter.setPen(QColor(0, 100, 255))
        font = painter.font()
        font.setBold(True)
        font.setPointSize(10)
        painter.setFont(font)

        width_cm = display_rect.width() / self.pixels_per_cm
        height_cm = display_rect.height() / self.pixels_per_cm
        size_text = f"{width_cm:.1f}cm × {height_cm:.1f}cm"

        # 使用QFontMetrics计算文本宽度
        from PyQt5.QtGui import QFontMetrics
        fm = QFontMetrics(font)
        text_width = fm.width(size_text) + 20  # 加上一些边距
        text_height = fm.height() + 10

        # 在轮廓上方显示尺寸
        text_rect = QRectF(
            display_rect.center().x() - text_width / 2,
            display_rect.top() - text_height - 5,
            text_width,
            text_height
        )
        painter.fillRect(text_rect, QColor(255, 255, 255, 220))
        painter.setPen(QColor(0, 100, 255))
        painter.drawText(text_rect, Qt.AlignCenter, size_text)

        painter.restore()

    def mousePressEvent(self, event):
        try:
            pos = event.pos()
            self.drag_start_pos = pos

            # 处理平移工具
            if (event.button() == Qt.MiddleButton or
                    (event.button() == Qt.LeftButton and self.current_tool == "pan")):
                self.dragging_canvas = True
                self.pan_start_pos = pos
                self.setCursor(Qt.ClosedHandCursor)
                return

            # 检查是否点击了控制点
            if self.selected_contour:
                # 将屏幕坐标转换到显示坐标系（应用视图变换的逆变换）
                display_pos = QPointF(
                    (pos.x() - self.pan_offset.x()) / self.zoom_factor,
                    (pos.y() - self.pan_offset.y()) / self.zoom_factor
                )

                # 将显示坐标转换到轮廓局部坐标系（考虑非均匀缩放）
                rect = self.selected_contour.get_display_rect()
                if not rect.isNull():
                    scale_x = self.selected_contour.scale_x
                    scale_y = self.selected_contour.scale_y
                    bbox = self.selected_contour.bounding_box

                    # 局部坐标 = (显示坐标 - 显示矩形左上角) / 缩放因子 + 包围盒左上角
                    local_x = (display_pos.x() - rect.left()) / scale_x + bbox.left()
                    local_y = (display_pos.y() - rect.top()) / scale_y + bbox.top()
                    local_pos = QPointF(local_x, local_y)

                    # 检查是否点击了控制点（控制点仍基于原始包围盒的局部坐标）
                    handle_size = 10 / min(scale_x, scale_y)  # 保持视觉大小一致
                    bbox = self.selected_contour.bounding_box

                    handles = [
                        QRectF(bbox.left() - handle_size / 2, bbox.top() - handle_size / 2, handle_size, handle_size),
                        QRectF(bbox.center().x() - handle_size / 2, bbox.top() - handle_size / 2, handle_size,
                               handle_size),
                        QRectF(bbox.right() - handle_size / 2, bbox.top() - handle_size / 2, handle_size, handle_size),
                        QRectF(bbox.right() - handle_size / 2, bbox.center().y() - handle_size / 2, handle_size,
                               handle_size),
                        QRectF(bbox.right() - handle_size / 2, bbox.bottom() - handle_size / 2, handle_size,
                               handle_size),
                        QRectF(bbox.center().x() - handle_size / 2, bbox.bottom() - handle_size / 2, handle_size,
                               handle_size),
                        QRectF(bbox.left() - handle_size / 2, bbox.bottom() - handle_size / 2, handle_size,
                               handle_size),
                        QRectF(bbox.left() - handle_size / 2, bbox.center().y() - handle_size / 2, handle_size,
                               handle_size)
                    ]

                    for i, handle_rect in enumerate(handles):
                        if handle_rect.contains(local_pos):
                            self.dragging_handle = True
                            self.resize_handle_idx = i
                            self.update()
                            return

            # 检查是否点击了轮廓或包围盒边框
            clicked_contour = None
            for contour in reversed(self.contours):
                if self.is_point_near_contour_or_border(pos, contour):
                    clicked_contour = contour
                    break

            if clicked_contour:
                self.select_contour(clicked_contour)
                self.dragging = True
                rect = clicked_contour.get_display_rect()
                display_pos = QPointF(
                    (pos.x() - self.pan_offset.x()) / self.zoom_factor,
                    (pos.y() - self.pan_offset.y()) / self.zoom_factor
                )
                self.drag_offset = display_pos - rect.topLeft()
                self.update()
                return

            # 点击空白处取消选中
            if self.selected_contour:
                self.selected_contour.is_selected = False
                self.selected_contour = None
                self.contour_selected.emit(None)
                self.update()

        except Exception as e:
            print(f"鼠标按下事件错误: {str(e)}")

    def mouseMoveEvent(self, event):
        try:
            pos = event.pos()

            # 处理画布平移
            if self.dragging_canvas:
                delta = pos - self.pan_start_pos
                self.pan_offset += delta
                self.pan_start_pos = pos
                self.update()
                return

            # 更新悬停状态
            old_hovered = self.hovered_contour
            old_handle_idx = self.hovered_handle_idx

            self.hovered_contour = None
            self.hovered_handle_idx = -1

            # 检查是否悬停在控制点上
            if self.selected_contour:
                display_pos = QPointF(
                    (pos.x() - self.pan_offset.x()) / self.zoom_factor,
                    (pos.y() - self.pan_offset.y()) / self.zoom_factor
                )

                rect = self.selected_contour.get_display_rect()
                if not rect.isNull():
                    scale_x = self.selected_contour.scale_x
                    scale_y = self.selected_contour.scale_y
                    bbox = self.selected_contour.bounding_box

                    local_x = (display_pos.x() - rect.left()) / scale_x + bbox.left()
                    local_y = (display_pos.y() - rect.top()) / scale_y + bbox.top()
                    local_pos = QPointF(local_x, local_y)

                    handle_size = 10 / min(scale_x, scale_y)
                    bbox = self.selected_contour.bounding_box

                    handles = [
                        QRectF(bbox.left() - handle_size / 2, bbox.top() - handle_size / 2, handle_size, handle_size),
                        QRectF(bbox.center().x() - handle_size / 2, bbox.top() - handle_size / 2, handle_size,
                               handle_size),
                        QRectF(bbox.right() - handle_size / 2, bbox.top() - handle_size / 2, handle_size, handle_size),
                        QRectF(bbox.right() - handle_size / 2, bbox.center().y() - handle_size / 2, handle_size,
                               handle_size),
                        QRectF(bbox.right() - handle_size / 2, bbox.bottom() - handle_size / 2, handle_size,
                               handle_size),
                        QRectF(bbox.center().x() - handle_size / 2, bbox.bottom() - handle_size / 2, handle_size,
                               handle_size),
                        QRectF(bbox.left() - handle_size / 2, bbox.bottom() - handle_size / 2, handle_size,
                               handle_size),
                        QRectF(bbox.left() - handle_size / 2, bbox.center().y() - handle_size / 2, handle_size,
                               handle_size)
                    ]

                    for i, handle_rect in enumerate(handles):
                        if handle_rect.contains(local_pos):
                            self.hovered_handle_idx = i
                            break

            # 检查是否悬停在轮廓上
            if self.hovered_handle_idx == -1:
                for contour in reversed(self.contours):
                    if self.is_point_near_contour_or_border(pos, contour):
                        self.hovered_contour = contour
                        break

            # 更新光标
            if self.hovered_handle_idx != -1:
                if self.hovered_handle_idx in [0, 4]:
                    self.setCursor(Qt.SizeFDiagCursor)
                elif self.hovered_handle_idx in [2, 6]:
                    self.setCursor(Qt.SizeBDiagCursor)
                elif self.hovered_handle_idx in [1, 5]:
                    self.setCursor(Qt.SizeVerCursor)
                elif self.hovered_handle_idx in [3, 7]:
                    self.setCursor(Qt.SizeHorCursor)
            elif self.hovered_contour:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

            if (old_hovered != self.hovered_contour or
                    old_handle_idx != self.hovered_handle_idx):
                self.update()

            # 处理拖动
            if self.dragging and self.selected_contour:
                display_pos = QPointF(
                    (pos.x() - self.pan_offset.x()) / self.zoom_factor,
                    (pos.y() - self.pan_offset.y()) / self.zoom_factor
                )
                new_pos = display_pos - self.drag_offset
                display_rect = self.selected_contour.get_display_rect()
                max_x = (self.width() - display_rect.width() * self.zoom_factor) / self.zoom_factor
                max_y = (self.height() - display_rect.height() * self.zoom_factor) / self.zoom_factor
                new_pos.setX(max(0, min(new_pos.x(), max_x)))
                new_pos.setY(max(0, min(new_pos.y(), max_y)))
                self.selected_contour.position = new_pos
                self.update()

            elif self.dragging_handle and self.selected_contour:
                display_pos = QPointF(
                    (pos.x() - self.pan_offset.x()) / self.zoom_factor,
                    (pos.y() - self.pan_offset.y()) / self.zoom_factor
                )
                self.resize_with_handle(display_pos)

        except Exception as e:
            print(f"鼠标移动事件错误: {str(e)}")

    def mouseReleaseEvent(self, event):
        """鼠标释放事件"""
        self.dragging = False
        self.dragging_handle = False
        self.dragging_canvas = False
        self.resize_handle_idx = -1

        if self.current_tool == "pan":
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def wheelEvent(self, event):
        """鼠标滚轮缩放（以鼠标位置为中心）"""
        # 获取鼠标在画布上的位置
        mouse_pos = event.pos()

        # 保存当前鼠标位置
        self.last_mouse_pos = mouse_pos

        # 计算缩放前的画布坐标
        canvas_pos_before = (mouse_pos - self.pan_offset) / self.zoom_factor

        # 计算缩放因子
        if event.angleDelta().y() > 0:
            zoom_change = 1.1  # 放大
        else:
            zoom_change = 0.9  # 缩小

        # 更新缩放因子
        new_zoom_factor = self.zoom_factor * zoom_change

        # 限制缩放范围
        new_zoom_factor = max(0.1, min(5.0, new_zoom_factor))

        if new_zoom_factor != self.zoom_factor:
            # 计算缩放后的画布坐标
            canvas_pos_after = canvas_pos_before * (self.zoom_factor / new_zoom_factor)

            # 调整平移偏移以保持鼠标位置不变
            new_pan_offset = mouse_pos - canvas_pos_after * new_zoom_factor

            # 更新视图参数
            self.zoom_factor = new_zoom_factor
            self.pan_offset = new_pan_offset

            self.update()

    def resize_with_handle(self, display_pos: QPointF):
        """通过控制点调整大小（支持非均匀缩放）"""
        if not self.selected_contour:
            return

        rect = self.selected_contour.get_display_rect()
        if rect.isNull():
            return

        # 获取当前缩放因子
        scale_x = self.selected_contour.scale_x
        scale_y = self.selected_contour.scale_y
        bbox = self.selected_contour.bounding_box

        # 获取键盘修饰符
        from PyQt5.QtWidgets import QApplication
        modifiers = QApplication.keyboardModifiers()
        keep_aspect = bool(modifiers & Qt.ShiftModifier)

        # 当前宽高（显示像素）
        current_width = rect.width()
        current_height = rect.height()
        if keep_aspect:
            aspect = current_width / current_height if current_height != 0 else 1.0

        # 根据控制点索引调整
        if self.resize_handle_idx == 0:  # 左上
            new_left = display_pos.x()
            new_top = display_pos.y()
            new_width = rect.right() - new_left
            new_height = rect.bottom() - new_top
            if new_width > 0 and new_height > 0:
                if keep_aspect:
                    desired_height = new_width / aspect
                    if desired_height <= new_height:
                        new_height = desired_height
                    else:
                        new_width = new_height * aspect
                self.selected_contour.position = QPointF(new_left, new_top)
                self.selected_contour.scale_x = new_width / bbox.width()
                self.selected_contour.scale_y = new_height / bbox.height()
                self.selected_contour.scale = (self.selected_contour.scale_x + self.selected_contour.scale_y) / 2.0

        elif self.resize_handle_idx == 1:  # 上中
            new_top = display_pos.y()
            new_height = rect.bottom() - new_top
            if new_height > 0:
                if keep_aspect:
                    new_width = new_height * aspect
                    self.selected_contour.position = QPointF(rect.left() - (new_width - current_width) / 2, new_top)
                    self.selected_contour.scale_x = new_width / bbox.width()
                else:
                    self.selected_contour.position = QPointF(rect.left(), new_top)
                self.selected_contour.scale_y = new_height / bbox.height()
                if not keep_aspect:
                    self.selected_contour.scale_x = current_width / bbox.width()
                self.selected_contour.scale = (self.selected_contour.scale_x + self.selected_contour.scale_y) / 2.0

        elif self.resize_handle_idx == 2:  # 右上
            new_right = display_pos.x()
            new_top = display_pos.y()
            new_width = new_right - rect.left()
            new_height = rect.bottom() - new_top
            if new_width > 0 and new_height > 0:
                if keep_aspect:
                    desired_height = new_width / aspect
                    if desired_height <= new_height:
                        new_height = desired_height
                    else:
                        new_width = new_height * aspect
                self.selected_contour.position = QPointF(rect.left(), new_top)
                self.selected_contour.scale_x = new_width / bbox.width()
                self.selected_contour.scale_y = new_height / bbox.height()
                self.selected_contour.scale = (self.selected_contour.scale_x + self.selected_contour.scale_y) / 2.0

        elif self.resize_handle_idx == 3:  # 右中
            new_right = display_pos.x()
            new_width = new_right - rect.left()
            if new_width > 0:
                if keep_aspect:
                    new_height = new_width / aspect
                    self.selected_contour.position = QPointF(rect.left(),
                                                             rect.top() - (new_height - current_height) / 2)
                    self.selected_contour.scale_y = new_height / bbox.height()
                self.selected_contour.scale_x = new_width / bbox.width()
                self.selected_contour.scale = (self.selected_contour.scale_x + self.selected_contour.scale_y) / 2.0

        elif self.resize_handle_idx == 4:  # 右下
            new_right = display_pos.x()
            new_bottom = display_pos.y()
            new_width = new_right - rect.left()
            new_height = new_bottom - rect.top()
            if new_width > 0 and new_height > 0:
                if keep_aspect:
                    desired_height = new_width / aspect
                    if desired_height <= new_height:
                        new_height = desired_height
                    else:
                        new_width = new_height * aspect
                self.selected_contour.scale_x = new_width / bbox.width()
                self.selected_contour.scale_y = new_height / bbox.height()
                self.selected_contour.scale = (self.selected_contour.scale_x + self.selected_contour.scale_y) / 2.0

        elif self.resize_handle_idx == 5:  # 下中
            new_bottom = display_pos.y()
            new_height = new_bottom - rect.top()
            if new_height > 0:
                if keep_aspect:
                    new_width = new_height * aspect
                    self.selected_contour.position = QPointF(rect.left() - (new_width - current_width) / 2, rect.top())
                    self.selected_contour.scale_x = new_width / bbox.width()
                else:
                    self.selected_contour.position = QPointF(rect.left(), rect.top())
                self.selected_contour.scale_y = new_height / bbox.height()
                if not keep_aspect:
                    self.selected_contour.scale_x = current_width / bbox.width()
                self.selected_contour.scale = (self.selected_contour.scale_x + self.selected_contour.scale_y) / 2.0

        elif self.resize_handle_idx == 6:  # 左下
            new_left = display_pos.x()
            new_bottom = display_pos.y()
            new_width = rect.right() - new_left
            new_height = new_bottom - rect.top()
            if new_width > 0 and new_height > 0:
                if keep_aspect:
                    desired_height = new_width / aspect
                    if desired_height <= new_height:
                        new_height = desired_height
                    else:
                        new_width = new_height * aspect
                self.selected_contour.position = QPointF(new_left, rect.top())
                self.selected_contour.scale_x = new_width / bbox.width()
                self.selected_contour.scale_y = new_height / bbox.height()
                self.selected_contour.scale = (self.selected_contour.scale_x + self.selected_contour.scale_y) / 2.0

        elif self.resize_handle_idx == 7:  # 左中
            new_left = display_pos.x()
            new_width = rect.right() - new_left
            if new_width > 0:
                if keep_aspect:
                    new_height = new_width / aspect
                    self.selected_contour.position = QPointF(new_left, rect.top() - (new_height - current_height) / 2)
                    self.selected_contour.scale_y = new_height / bbox.height()
                else:
                    self.selected_contour.position = QPointF(new_left, rect.top())
                self.selected_contour.scale_x = new_width / bbox.width()
                self.selected_contour.scale = (self.selected_contour.scale_x + self.selected_contour.scale_y) / 2.0

        # 发出信号更新UI
        self.contour_selected.emit(self.selected_contour)
        self.update()

    def select_contour(self, contour: Contour):
        """选中轮廓"""
        if self.selected_contour:
            self.selected_contour.is_selected = False

        contour.is_selected = True
        self.selected_contour = contour

        # 发出信号
        self.contour_selected.emit(contour)
        self.update()

    def is_point_near_contour_or_border(self, point: QPointF, contour: Contour, threshold: float = 8) -> bool:
        """判断点是否在轮廓或包围盒附近（考虑视图变换）"""
        try:
            # 获取轮廓的显示矩形
            display_rect = contour.get_display_rect()
            if display_rect.isNull():
                return False

            # 将显示矩形转换为屏幕坐标
            screen_rect = QRectF(
                display_rect.left() * self.zoom_factor + self.pan_offset.x(),
                display_rect.top() * self.zoom_factor + self.pan_offset.y(),
                display_rect.width() * self.zoom_factor,
                display_rect.height() * self.zoom_factor
            )

            # 检查是否在矩形内部
            if screen_rect.contains(point):
                return True

            # 检查是否靠近边框
            border_rect = screen_rect.adjusted(-threshold, -threshold, threshold, threshold)
            if border_rect.contains(point) and not screen_rect.adjusted(threshold, threshold, -threshold,
                                                                        -threshold).contains(point):
                return True

            # 检查是否在轮廓内部
            if self.is_point_in_contour(point, contour):
                return True

            return False

        except Exception as e:
            print(f"判断点是否在轮廓附近错误: {str(e)}")
            return False

    def is_point_in_contour(self, point: QPointF, contour: Contour) -> bool:
        if not contour.nurbs_points:
            return False
        display_rect = contour.get_display_rect()
        if display_rect.isNull():
            return False
        local_x = (point.x() - display_rect.left()) / contour.scale_x + contour.bounding_box.left()
        local_y = (point.y() - display_rect.top()) / contour.scale_y + contour.bounding_box.top()
        local_point = QPointF(local_x, local_y)

        n = len(contour.nurbs_points)
        if n < 3:
            return False
        inside = False
        p1 = contour.nurbs_points[0]
        for i in range(1, n + 1):
            p2 = contour.nurbs_points[i % n]
            if (p1.y() > local_point.y()) != (p2.y() > local_point.y()):
                if local_point.x() < (p2.x() - p1.x()) * (local_point.y() - p1.y()) / (p2.y() - p1.y()) + p1.x():
                    inside = not inside
            p1 = p2
        return inside

    # def refit_all_contours(self, precision: float):
    #     """使用新的精度重新拟合所有轮廓"""
    #     try:
    #         for contour in self.contours:
    #             if len(contour.original_points) >= 3:
    #                 # 重新使用NURBS曲线拟合
    #                 contour.precision = precision
    #                 nurbs_points, nurbs_curve = AdvancedImageProcessor.smooth_contour_with_nurbs(
    #                     contour.original_points, precision
    #                 )
    #                 contour.nurbs_points = nurbs_points
    #                 contour.nurbs_curve = nurbs_curve
    #
    #         self.update()
    #         self.contour_changed.emit()
    #
    #     except Exception as e:
    #         print(f"重新拟合轮廓失败: {str(e)}")
    #         traceback.print_exc()

    def refit_single_contour(self, contour: Contour, num_control_points: int):
        """使用指定的控制点数重新拟合单个轮廓"""
        if len(contour.original_points) < 3:
            return

        # 调用拟合函数
        nurbs_points, nurbs_curve = AdvancedImageProcessor.smooth_contour_with_nurbs(
            contour.original_points,
            precision=0.5,  # 此处 precision 不再起决定作用，但函数需要
            num_control_points=num_control_points
        )
        contour.nurbs_points = nurbs_points
        contour.nurbs_curve = nurbs_curve
        contour.control_points = num_control_points  # 保存当前点数

        self.contour_changed.emit()

    def render_contours_only(self, background_color=Qt.white, draw_labels=True):
        """
        生成仅包含轮廓（和标号）的图像，背景为指定颜色，不包含网格、边框等。
        :param background_color: 背景颜色（默认为白色）
        :param draw_labels: 是否绘制标号（默认 True）
        :return: QPixmap 对象
        """
        pixmap = QPixmap(self.canvas_width_px, self.canvas_height_px)
        pixmap.fill(background_color)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制所有轮廓（不应用视图变换，直接使用轮廓自身的位置和缩放）
        for contour in self.contours:
            self.draw_contour(painter, contour)
            if draw_labels and contour.label > 0 and self.show_labels:
                self.draw_contour_label(painter, contour)

        painter.end()
        return pixmap

    def paintEvent(self, event):
        """绘制事件"""
        try:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.Antialiasing)

            # 绘制白色背景
            painter.fillRect(self.rect(), QColor(255, 255, 255))

            # 保存当前状态
            painter.save()

            # 应用视图变换（先平移后缩放）
            painter.translate(self.pan_offset)
            painter.scale(self.zoom_factor, self.zoom_factor)

            # 绘制厘米网格（在变换后的坐标系中）
            self.draw_cm_grid(painter)

            # 绘制所有轮廓
            for contour in self.contours:
                self.draw_contour(painter, contour)

            # 绘制选中轮廓的包围盒和控制点
            if self.selected_contour and self.show_bounding_box:
                self.draw_selection_indicators(painter, self.selected_contour)

            painter.restore()

        except Exception as e:
            print(f"绘制错误: {str(e)}")
            traceback.print_exc()

    def keyPressEvent(self, event):
        """键盘事件"""
        if event.key() == Qt.Key_Space:
            self.current_tool = "pan" if self.current_tool != "pan" else "select"
            self.updateCursor()
        elif event.key() == Qt.Key_Plus or event.key() == Qt.Key_Equal:
            self.zoom_factor *= 1.1
            self.update()
        elif event.key() == Qt.Key_Minus:
            self.zoom_factor /= 1.1
            self.update()
        elif event.key() == Qt.Key_0 or event.key() == Qt.Key_R:  # 0键或R键还原视图
            self.pan_offset = QPointF(0, 0)
            self.zoom_factor = 1.0
            self.update()
        else:
            super().keyPressEvent(event)

    def updateCursor(self):
        """更新光标"""
        if self.current_tool == "pan":
            self.setCursor(Qt.OpenHandCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def wheelEvent_helper(self, mouse_pos, delta):
        """辅助方法，用于通过代码触发缩放"""
        # 创建一个虚拟的滚轮事件
        from PyQt5.QtCore import QPoint
        from PyQt5.QtGui import QWheelEvent

        # 使用当前保存的鼠标位置，或者使用传入的位置
        if mouse_pos is None:
            mouse_pos = self.last_mouse_pos if self.last_mouse_pos else QPointF(self.width() / 2, self.height() / 2)

        # 创建虚拟事件
        event = QWheelEvent(
            mouse_pos, mouse_pos, QPoint(0, delta), QPoint(0, delta),
            delta, Qt.Vertical, Qt.NoButton, Qt.NoModifier
        )

        self.wheelEvent(event)