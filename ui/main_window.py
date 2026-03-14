"""
主窗口
"""
import sys
import traceback
import json
import math
import random
from pathlib import Path
from typing import Dict
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
                             QCheckBox, QListWidget, QListWidgetItem, QProgressDialog,
                             QFileDialog, QMessageBox, QFormLayout, QMenuBar, QAction,
                             QProgressBar, QApplication, QProgressBar, QRadioButton, QDockWidget)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QDateTime, QPointF, QRectF, pyqtSlot, QMetaObject, Q_ARG
from PyQt5.QtGui import QPalette, QColor

# 修改导入部分
from core.image_processor import AdvancedImageProcessor, GEOMDL_AVAILABLE
from core.dxf_exporter import DXFExporter
from ui.canvas_widget import CanvasWidget
from ui.label_mapping_dialog import LabelMappingDialog
# 在文件顶部导入部分添加
from pathlib import Path
from core.marker_detector import WhiteBallMarkerDetector
from core.laser_controller import LaserController
import os
from ui.simulation_widget import SimulationWidget

class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("皮肤病灶轮廓处理系统")
        self.setGeometry(100, 100, 1400, 900)

        self.current_directory = ""
        self.image_files = []
        self.label_to_image_map: Dict[int, str] = {}  # 添加：标号到图像名称的映射
        self.next_label = 1  # 添加：下一个标号
        self.current_precision = 0.5  # 当前拟合精度
        self.refit_timer = QTimer()  # 用于延迟重新拟合的定时器
        self.refit_timer.setSingleShot(True)
        self.refit_timer.timeout.connect(self.on_refit_timeout)
        # 添加像素比例尺相关变量
        self.pixel_scale_mm_per_px = None
        self.current_overlay_dir = ""
        # 创建激光控制器实例
        self.laser = LaserController()
        # 初始化UI
        self.init_ui()
        # 更新状态显示
        self.update_status_display()
        # 轮廓画布分页
        self.pages_contours = []  # 分页后的轮廓列表
        self.current_page = 0  # 当前显示的页码
        # 创建模拟可视化窗口（可选显示）
        self.simulation_widget = SimulationWidget()
        self.simulation_widget.simulation_completed.connect(self._on_simulation_completed)

        # 可以放在停靠窗口或独立窗口
        sim_dock = QDockWidget("切割模拟", self)
        sim_dock.setWidget(self.simulation_widget)
        sim_dock.setVisible(False)  # 默认隐藏，需要时显示
        self.addDockWidget(Qt.RightDockWidgetArea, sim_dock)
        self.sim_dock = sim_dock

        # 设置异常处理
        sys.excepthook = self.exception_hook

        # 检查geomdl库
        if not GEOMDL_AVAILABLE:
            QMessageBox.warning(self, "库未安装",
                                "geomdl库未安装，将使用贝塞尔曲线进行拟合。\n\n"
                                "要使用更精确的NURBS拟合，请运行:\n"
                                "pip install geomdl")

    def exception_hook(self, exc_type, exc_value, exc_traceback):
        """异常处理钩子"""
        error_msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        print(f"未捕获的异常:\n{error_msg}")

        # 显示错误对话框
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Critical)
        error_dialog.setWindowTitle("程序错误")
        error_dialog.setText(f"发生错误: {str(exc_value)}")
        error_dialog.setDetailedText(error_msg)
        error_dialog.exec_()

    def init_ui(self):
        """初始化UI"""
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # 创建控制面板（左侧）
        control_panel = self.create_control_panel()

        # 创建画布容器（右侧，垂直布局）
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0, 0, 0, 0)

        # 画布
        self.canvas = CanvasWidget()
        self.canvas.contour_selected.connect(self.on_contour_selected)
        self.canvas.contour_changed.connect(self.on_contour_changed)
        canvas_layout.addWidget(self.canvas)

        # 翻页栏（放在画布下方）
        page_widget = QWidget()
        page_layout = QHBoxLayout(page_widget)
        page_layout.setContentsMargins(0, 5, 0, 0)

        self.btn_prev = QPushButton("◀ 上一页")
        self.btn_prev.clicked.connect(self.prev_page)
        self.btn_next = QPushButton("下一页 ▶")
        self.btn_next.clicked.connect(self.next_page)
        self.lbl_page = QLabel("第 1 页 / 共 1 页")
        self.lbl_page.setAlignment(Qt.AlignCenter)

        page_layout.addWidget(self.btn_prev)
        page_layout.addWidget(self.lbl_page)
        page_layout.addWidget(self.btn_next)

        canvas_layout.addWidget(page_widget)

        # 将控制面板和画布容器添加到主布局
        main_layout.addWidget(control_panel, 1)
        main_layout.addWidget(canvas_container, 3)

        # 创建菜单栏
        self.create_menu_bar()

        # 状态栏
        self.statusBar().showMessage("就绪")
        self.status_progress = QProgressBar()
        self.status_progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.status_progress)

        # 添加快捷键
        self.canvas.setFocusPolicy(Qt.StrongFocus)

        # 添加激光控制功能区（如原代码）
        self.init_laser_control_panel()

    def create_control_panel(self) -> QWidget:
        """创建控制面板"""
        from .control_panel import ControlPanel
        return ControlPanel(self)

    def create_menu_bar(self):
        """创建菜单栏"""
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu("文件")

        load_action = QAction("加载多个图像", self)
        load_action.triggered.connect(self.load_images)
        file_menu.addAction(load_action)

        file_menu.addSeparator()

        save_action = QAction("保存轮廓", self)
        save_action.triggered.connect(self.save_contours)
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 视图菜单
        view_menu = menubar.addMenu("视图")

        show_original = QAction("显示原始轮廓", self, checkable=True)
        show_original.setChecked(False)
        show_original.triggered.connect(self.toggle_original_contour)
        view_menu.addAction(show_original)

        show_nurbs = QAction("显示NURBS曲线", self, checkable=True)
        show_nurbs.setChecked(True)
        show_nurbs.triggered.connect(self.toggle_nurbs_curve)
        view_menu.addAction(show_nurbs)

        show_bounding_box = QAction("显示选中轮廓的包围盒", self, checkable=True)
        show_bounding_box.setChecked(True)
        show_bounding_box.triggered.connect(self.toggle_bounding_box)
        view_menu.addAction(show_bounding_box)

        # 添加：显示标号菜单项
        show_labels = QAction("显示轮廓标号", self, checkable=True)
        show_labels.setChecked(True)
        show_labels.triggered.connect(self.toggle_labels)
        view_menu.addAction(show_labels)

        # 工具菜单
        tools_menu = menubar.addMenu("工具")

        arrange_action = QAction("自动排列轮廓", self)
        arrange_action.triggered.connect(self.arrange_contours)
        tools_menu.addAction(arrange_action)

        mapping_action = QAction("查看标号映射", self)
        mapping_action.triggered.connect(self.show_label_mapping_dialog)
        tools_menu.addAction(mapping_action)

    def load_images(self):
        """加载多个图像文件"""
        try:
            files, _ = QFileDialog.getOpenFileNames(
                self, "选择图像文件", "",
                "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*.*)"
            )

            if files:
                self.image_files = files
                # 更新UI显示
                self.findChild(QLabel, "lbl_image_info").setText(f"已选择 {len(files)} 个图像文件")
                self.statusBar().showMessage(f"已加载 {len(files)} 个图像文件")

                # 重置标号
                self.next_label = 1
                self.label_to_image_map.clear()
                self.canvas.label_to_image_mapping.clear()

        except Exception as e:
            QMessageBox.critical(self, "错误", f"加载图像失败: {str(e)}")

    def process_all_images(self):
        """处理所有图像"""
        try:
            if not self.image_files:
                QMessageBox.warning(self, "警告", "请先加载图像文件！")
                return

            # 清空现有轮廓
            self.canvas.clear()
            # 找到列表控件
            # contour_list_widget = self.findChild(QListWidget, "contour_list_widget")
            # if contour_list_widget:
            #     contour_list_widget.clear()

            # 找到选择组
            selection_group = self.findChild(QGroupBox, "selection_group")
            if selection_group:
                selection_group.setEnabled(False)

            # 重置标号和映射
            self.next_label = 1
            self.label_to_image_map.clear()
            self.canvas.label_to_image_mapping.clear()

            contour_count = 0

            # 创建进度对话框
            progress = QProgressDialog("处理图像中...", "取消", 0, len(self.image_files), self)
            progress.setWindowTitle("处理进度")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # 获取处理参数
            kernel_size = self.findChild(QSpinBox, "spin_kernel_size").value()

            for idx, image_path in enumerate(self.image_files):
                progress.setValue(idx)
                progress.setLabelText(f"处理图像 {idx + 1}/{len(self.image_files)}: {Path(image_path).name}")

                if progress.wasCanceled():
                    break

                # 处理图像
                contours_data = AdvancedImageProcessor.load_and_process_image(image_path, kernel_size)

                for contour_points, image_name in contours_data:
                    if len(contour_points) >= 3:
                        # 简化轮廓 - 使用 NURBSFitter 中的方法
                        simplified_points = AdvancedImageProcessor.simplify_contour(contour_points, tolerance=2.0)

                        # 使用NURBS曲线拟合 - 使用 NURBSFitter 中的方法
                        nurbs_points, nurbs_curve = AdvancedImageProcessor.smooth_contour_with_nurbs(
                            simplified_points,
                            self.current_precision
                        )

                        # 检查是否为该图像分配了标号
                        if self.next_label not in self.label_to_image_map:
                            self.label_to_image_map[self.next_label] = image_name
                            self.canvas.label_to_image_mapping[self.next_label] = image_name

                        # 创建轮廓对象
                        contour = self.canvas.add_contour(simplified_points, image_name, self.next_label)

                        contour_count += 1

                # 处理完一个图像后，增加标号（同一个图像的所有轮廓共享一个标号）
                if contours_data:  # 只有该图像有轮廓时才增加标号
                    self.next_label += 1

                QApplication.processEvents()  # 更新UI

            progress.close()

            # 自动排列轮廓
            # 自动标定所有轮廓尺寸
            self.auto_calibrate_contours()
            self.arrange_contours()

            self.statusBar().showMessage(f"已处理 {len(self.image_files)} 个图像，提取到 {contour_count} 个轮廓")

            if contour_count == 0:
                QMessageBox.information(self, "提示", "图像中没有检测到轮廓")
            else:
                method = 'NURBS' if GEOMDL_AVAILABLE else '贝塞尔'
                QMessageBox.information(self, "完成",
                                        f"成功处理 {contour_count} 个轮廓！使用{method}曲线拟合")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理图像失败: {str(e)}")
            traceback.print_exc()

    def arrange_contours(self, margin_mm=1.0):
        try:
            from shapely.geometry import Polygon, Point
            from shapely import affinity, prepared
            from shapely.ops import unary_union
            import math
        except ImportError:
            QMessageBox.critical(self, "缺少依赖", "请安装shapely库：pip install shapely")
            return

        if not self.canvas.contours:
            return

        pixels_per_cm = self.canvas.pixels_per_cm
        container_radius_px = 5 * pixels_per_cm  # 5cm半径
        center_x = self.canvas.canvas_width_px / 2
        center_y = self.canvas.canvas_height_px / 2
        margin_px = margin_mm * (pixels_per_cm / 10)  # 间距转换为像素

        # 构建轮廓多边形数据
        contours_data = []
        for contour in self.canvas.contours:
            if not contour.nurbs_points:
                continue
            display_rect = contour.get_display_rect()
            if display_rect.isNull():
                continue
            pts = []
            scale = contour.scale
            bbox = contour.bounding_box
            for p in contour.nurbs_points:
                x_px = display_rect.left() + (p.x() - bbox.left()) * scale
                y_px = display_rect.top() + (p.y() - bbox.top()) * scale
                pts.append((x_px, y_px))
            original_poly = Polygon(pts)
            if not original_poly.is_valid:
                original_poly = original_poly.buffer(0)
            poly_with_margin = original_poly.buffer(margin_px, join_style=2)
            if not poly_with_margin.is_valid:
                poly_with_margin = poly_with_margin.buffer(0)

            contours_data.append({
                'contour': contour,
                'original_poly': original_poly,
                'poly_with_margin': poly_with_margin,
            })

        # 按面积降序排列
        contours_data.sort(key=lambda x: x['poly_with_margin'].area, reverse=True)

        pages = []  # 每页为已放置轮廓列表

        def poly_in_circle(poly, cx, cy, radius):
            """检查多边形是否完全在圆内"""
            if poly.is_empty:
                return False
            # 快速检查包围盒
            minx, miny, maxx, maxy = poly.bounds
            if (maxx - cx) ** 2 + (maxy - cy) ** 2 > radius ** 2:
                return False
            for x, y in poly.exterior.coords:
                if (x - cx) ** 2 + (y - cy) ** 2 > radius ** 2:
                    return False
            return True

        def try_place(poly, placed_items):
            """尝试将多边形 poly 放入当前页，返回平移向量 (dx, dy) 或 None"""
            poly_bounds = poly.bounds
            poly_width = poly_bounds[2] - poly_bounds[0]
            poly_height = poly_bounds[3] - poly_bounds[1]
            max_dim = max(poly_width, poly_height)

            # 合并已放置区域的禁止多边形
            if placed_items:
                placed_union = unary_union([item['poly_with_margin'] for item in placed_items])
                placed_prep = prepared.prep(placed_union)
            else:
                placed_union = None
                placed_prep = None

            # 采样半径：步长取 max_dim/2，从0到圆半径
            step_r = max_dim / 2
            radii = [i * step_r for i in range(int(container_radius_px / step_r) + 1)]

            for r in radii:
                if r == 0:
                    angles = [0.0]  # 仅尝试圆心
                else:
                    # 根据半径确定角度采样数，使圆周上采样间距约为 max_dim/2
                    circumference = 2 * math.pi * r
                    n_angles = max(1, int(circumference / (max_dim / 2)))
                    n_angles = min(n_angles, 36)  # 上限36
                    angles = [2 * math.pi * i / n_angles for i in range(n_angles)]

                for angle in angles:
                    x = center_x + r * math.cos(angle)
                    y = center_y + r * math.sin(angle)

                    dx = x - (poly_bounds[0] + poly_bounds[2]) / 2
                    dy = y - (poly_bounds[1] + poly_bounds[3]) / 2
                    candidate = affinity.translate(poly, dx, dy)

                    if not poly_in_circle(candidate, center_x, center_y, container_radius_px):
                        continue

                    if placed_prep is not None and placed_prep.intersects(candidate):
                        continue

                    return (dx, dy)

            return None

        # 分配每个轮廓
        for data in contours_data:
            placed = False
            for page in pages:
                vec = try_place(data['poly_with_margin'], page)
                if vec is not None:
                    dx, dy = vec
                    data['original_poly'] = affinity.translate(data['original_poly'], dx, dy)
                    data['poly_with_margin'] = affinity.translate(data['poly_with_margin'], dx, dy)
                    data['contour'].position += QPointF(dx, dy)
                    page.append(data)
                    placed = True
                    break

            if not placed:
                # 尝试新建页面
                vec = try_place(data['poly_with_margin'], [])
                if vec is not None:
                    dx, dy = vec
                    data['original_poly'] = affinity.translate(data['original_poly'], dx, dy)
                    data['poly_with_margin'] = affinity.translate(data['poly_with_margin'], dx, dy)
                    data['contour'].position += QPointF(dx, dy)
                    pages.append([data])
                else:
                    print(f"警告：轮廓 {data['contour'].label} 无法放置")

        # 保存分页结果并显示第一页
        self.pages_contours = pages
        self.current_page = 0
        if pages:
            self.canvas.contours = [item['contour'] for item in pages[0]]
            self.lbl_page.setText(f"第 1 页 / 共 {len(pages)} 页")
        else:
            self.canvas.contours = []
            self.lbl_page.setText("第 0 页 / 共 0 页")
        self.canvas.update()
        self.statusBar().showMessage(f"已自动排列到 {len(pages)} 页")

    def clear_contours(self):
        """清空所有轮廓"""
        self.canvas.clear()
        # 找到列表控件
        # contour_list_widget = self.findChild(QListWidget, "contour_list_widget")
        # if contour_list_widget:
        #     contour_list_widget.clear()

        # 找到选择组
        selection_group = self.findChild(QGroupBox, "selection_group")
        if selection_group:
            selection_group.setEnabled(False)

        # 找到选中信息标签
        lbl_selected_info = self.findChild(QLabel, "lbl_selected_info")
        if lbl_selected_info:
            lbl_selected_info.setText("未选中轮廓")

        self.label_to_image_map.clear()
        self.canvas.label_to_image_mapping.clear()
        self.next_label = 1
        self.statusBar().showMessage("已清空所有轮廓")

    def on_contour_selected(self, contour):
        """轮廓选中事件处理"""
        if contour:
            # 找到选择组
            selection_group = self.findChild(QGroupBox, "selection_group")
            if selection_group:
                selection_group.setEnabled(True)

            # 找到选中信息标签
            lbl_selected_info = self.findChild(QLabel, "lbl_selected_info")
            if lbl_selected_info:
                label_info = f"标号 {contour.label}" if contour.label > 0 else "无标号"
                lbl_selected_info.setText(f"{label_info} - 轮廓 {contour.id} ({contour.source_image})")

            # 更新尺寸显示
            rect = contour.get_display_rect()
            width_cm = rect.width() / self.canvas.pixels_per_cm
            height_cm = rect.height() / self.canvas.pixels_per_cm

            # 找到尺寸输入框
            spin_width = self.findChild(QDoubleSpinBox, "spin_width")
            spin_height = self.findChild(QDoubleSpinBox, "spin_height")

            if spin_width:
                spin_width.setValue(width_cm)
            if spin_height:
                spin_height.setValue(height_cm)
        else:
            selection_group = self.findChild(QGroupBox, "selection_group")
            if selection_group:
                selection_group.setEnabled(False)

            lbl_selected_info = self.findChild(QLabel, "lbl_selected_info")
            if lbl_selected_info:
                lbl_selected_info.setText("未选中轮廓")

    def apply_contour_size(self):
        """应用轮廓尺寸"""
        if not self.canvas.selected_contour:
            QMessageBox.warning(self, "警告", "请先选中一个轮廓！")
            return

        # 找到尺寸输入框
        spin_width = self.findChild(QDoubleSpinBox, "spin_width")
        spin_height = self.findChild(QDoubleSpinBox, "spin_height")

        if not spin_width or not spin_height:
            return

        width_cm = spin_width.value()
        height_cm = spin_height.value()

        self.canvas.selected_contour.set_size(width_cm, height_cm, self.canvas.pixels_per_cm)

        # 更新标号大小
        self.canvas.selected_contour.update_label_size(self.canvas.pixels_per_cm)

        self.canvas.update()
        self.statusBar().showMessage(f"已设置轮廓尺寸: {width_cm}cm × {height_cm}cm")

    # def on_contour_list_item_clicked(self, item):
    #     """轮廓列表项点击"""
    #     try:
    #         # 从项目数据中获取轮廓ID
    #         contour_id = item.data(Qt.UserRole)
    #
    #         # 查找并选中对应的轮廓
    #         for contour in self.canvas.contours:
    #             if contour.id == contour_id:
    #                 self.canvas.select_contour(contour)
    #                 break
    #
    #     except Exception as e:
    #         print(f"选择轮廓错误: {str(e)}")

    def toggle_original_contour(self, checked):
        """切换原始轮廓显示"""
        self.canvas.show_original_contour = checked
        self.canvas.update()

    def toggle_nurbs_curve(self, checked):
        """切换NURBS曲线显示"""
        self.canvas.show_nurbs_curve = checked
        self.canvas.update()

    def toggle_bounding_box(self, checked):
        """切换包围盒显示"""
        self.canvas.show_bounding_box = checked
        self.canvas.update()

    def toggle_labels(self, checked):
        """切换标号显示"""
        self.canvas.show_labels = checked
        self.canvas.update()

    def show_label_mapping_dialog(self):
        """显示标号映射对话框"""
        if not self.label_to_image_map:
            QMessageBox.information(self, "提示", "当前没有标号映射信息，请先处理图像。")
            return

        dialog = LabelMappingDialog(self.label_to_image_map, self)
        dialog.exec_()

    def on_precision_changed(self, value):
        """精度滑块值改变事件"""
        # 更新精度值显示
        precision_percent = value
        lbl_precision_value = self.findChild(QLabel, "lbl_precision_value")
        if lbl_precision_value:
            # 将50-150映射到50%-150%显示
            display_percent = precision_percent
            lbl_precision_value.setText(f"{display_percent}%")

        # 计算精度值 (0.5-1.5)
        # 将滑块值50-150映射到0.5-1.5
        new_precision = 0.5 + (precision_percent - 50) * 0.01
        # 限制在0.5-1.5范围内（理论上不会超出，但为了安全）
        new_precision = max(0.5, min(2, new_precision))

        # 只有当精度变化足够大时才重新拟合
        if abs(new_precision - self.current_precision) > 0.005:
            self.current_precision = new_precision

            # 启动定时器进行延迟重新拟合
            self.refit_timer.start(200)  # 200毫秒后重新拟合
            self.statusBar().showMessage(f"正在调整拟合精度到 {precision_percent}%...")

    def on_refit_timeout(self):
        """定时器超时，重新拟合所有轮廓"""
        if self.canvas.contours:
            self.canvas.refit_all_contours(self.current_precision)
            self.statusBar().showMessage(f"已重新拟合所有轮廓 (精度: {int(self.current_precision * 100)}%)")

    def on_contour_changed(self):
        """轮廓变化事件"""
        self.statusBar().showMessage("轮廓已更新")

    def save_contours(self):
        """保存轮廓数据，支持多种格式"""
        try:
            # 扩展文件选择对话框的过滤器
            file_path, selected_filter = QFileDialog.getSaveFileName(
                self, "保存轮廓数据", "",
                "DXF文件 (*.dxf);;JSON文件 (*.json);;EZCAD文件 (*.ezd);;PNG图像 (*.png);;BMP图像 (*.bmp);;JPEG图像 (*.jpg *.jpeg);;所有文件 (*.*)"
            )

            if not file_path:
                return

            # 获取文件扩展名（小写）
            ext = Path(file_path).suffix.lower()

            # ========== 图像格式导出（不需要轮廓） ==========
            if ext in ['.png', '.bmp', '.jpg', '.jpeg'] or any(x in selected_filter for x in ['PNG', 'BMP', 'JPEG']):
                # 补充默认扩展名
                if not ext:
                    if 'PNG' in selected_filter:
                        file_path += '.png'
                    elif 'BMP' in selected_filter:
                        file_path += '.bmp'
                    elif 'JPEG' in selected_filter:
                        file_path += '.jpg'
                    else:
                        file_path += '.png'

                # 抓取画布内容并保存
                # 生成仅包含轮廓和标号的图像
                pixmap = self.canvas.render_contours_only(background_color=Qt.white, draw_labels=True)
                if file_path.lower().endswith(('.jpg', '.jpeg')):
                    pixmap.save(file_path, "JPEG", quality=95)
                else:
                    pixmap.save(file_path)

                self.statusBar().showMessage(f"画布图像已保存到: {file_path}")
                QMessageBox.information(self, "成功", "画布图像已保存！")
                return

            # ========== 以下格式需要轮廓存在 ==========
            if not self.canvas.contours:
                QMessageBox.warning(self, "警告", "没有轮廓数据可以保存！")
                return

            # DXF 格式
            if ext == '.dxf' or (selected_filter == "DXF文件 (*.dxf)" and not ext):
                if not ext:
                    file_path += '.dxf'
                success = DXFExporter.export_to_dxf(
                    self.canvas.contours,
                    self.canvas.pixels_per_cm,
                    file_path,
                    label_font_size_mm=self.canvas.label_font_size_mm*0.5,
                    label_min_size_mm=self.canvas.label_min_size_mm
                )
                if success:
                    self.statusBar().showMessage(f"轮廓数据已保存为DXF: {file_path}")
                    QMessageBox.information(self, "成功", f"成功保存 {len(self.canvas.contours)} 个轮廓到DXF文件！")
                else:
                    QMessageBox.critical(self, "错误", "保存DXF文件失败！")
                return

            # EZCAD 格式（复用 DXF 逻辑）
            if ext == '.ezd' or (selected_filter == "EZCAD文件 (*.ezd)" and not ext):
                if not ext:
                    file_path += '.ezd'
                success = DXFExporter.export_to_dxf(
                    self.canvas.contours,
                    self.canvas.pixels_per_cm,
                    file_path,
                    label_font_size_mm=self.canvas.label_font_size_mm,
                    label_min_size_mm=self.canvas.label_min_size_mm
                )
                if success:
                    self.statusBar().showMessage(f"轮廓数据已保存为EZCAD格式: {file_path}")
                    QMessageBox.information(self, "成功", f"成功保存 {len(self.canvas.contours)} 个轮廓到EZCAD文件！")
                else:
                    QMessageBox.critical(self, "错误", "保存EZCAD文件失败！")
                return

            # ---------- JSON 格式（默认） ----------
            if not file_path.lower().endswith('.json'):
                file_path += '.json'

            data = []
            for contour in self.canvas.contours:
                points_list = contour.original_points.tolist() if hasattr(contour.original_points, 'tolist') else []
                contour_data = {
                    "id": contour.id,
                    "label": contour.label,
                    "source_image": contour.source_image,
                    "position": {
                        "x": float(contour.position.x()),
                        "y": float(contour.position.y())
                    },
                    "scale": float(contour.scale),
                    "precision": contour.precision,
                    "original_points": points_list,
                    "nurbs_points": [
                        {"x": float(p.x()), "y": float(p.y())}
                        for p in contour.nurbs_points
                    ],
                    "color": {
                        "r": contour.color.red(),
                        "g": contour.color.green(),
                        "b": contour.color.blue()
                    }
                }
                data.append(contour_data)

            # 添加标号映射信息
            mapping_data = {
                "label_mapping": [
                    {"label": label, "image": image}
                    for label, image in self.label_to_image_map.items()
                ]
            }
            data.append({"metadata": mapping_data})

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.statusBar().showMessage(f"轮廓数据已保存到: {file_path}")
            QMessageBox.information(self, "成功", f"成功保存 {len(data) - 1} 个轮廓数据！")

        except ImportError as e:
            QMessageBox.critical(self, "DXF导出错误", f"{str(e)}\n\n请安装ezdxf库: pip install ezdxf")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存文件失败: {str(e)}")

    def auto_calibrate_contours(self):
        """自动标定轮廓尺寸"""
        try:
            if not self.canvas.contours:
                QMessageBox.warning(self, "警告", "没有轮廓可以标定，请先处理图像！")
                return

            # 查找原始图片文件夹（假设在mask文件夹的同级overlays文件夹）
            if not self.current_overlay_dir and self.image_files:
                mask_dir = Path(self.image_files[0]).parent
                overlay_dir = mask_dir.parent / "overlays"
                if overlay_dir.exists():
                    self.current_overlay_dir = str(overlay_dir)
                else:
                    QMessageBox.warning(self, "警告", "找不到原始图片文件夹（overlays）！")
                    return

            # 创建进度对话框
            progress = QProgressDialog("自动标定中...", "取消", 0, len(self.canvas.contours), self)
            progress.setWindowTitle("自动标定")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            # 创建检测器
            detector = WhiteBallMarkerDetector(ball_diameter_mm=10)

            # 遍历所有轮廓，按来源图像分组处理
            images_processed = {}
            success_count = 0

            for i, contour in enumerate(self.canvas.contours):
                progress.setValue(i)
                QApplication.processEvents()

                if progress.wasCanceled():
                    break

                source_image = contour.source_image
                if not source_image or source_image in images_processed:
                    continue

                # 从掩膜文件名构建原始图片文件名
                if '_mask' in source_image:
                    overlay_filename = source_image.replace('_mask', '_overlay')
                else:
                    # 尝试直接添加 _overlay 后缀
                    base_name = Path(source_image).stem
                    overlay_filename = f"{base_name}_overlay.png"

                # 查找原始图片文件
                overlay_path = Path(self.current_overlay_dir) / overlay_filename

                if not overlay_path.exists():
                    # 尝试其他扩展名
                    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                        alt_path = Path(self.current_overlay_dir) / f"{Path(overlay_filename).stem}{ext}"
                        if alt_path.exists():
                            overlay_path = alt_path
                            break

                if not overlay_path.exists():
                    continue

                # 处理原始图片获取比例尺
                result = detector.process_single_image(str(overlay_path), None)

                if result and result['detected'] and result['pixel_scale']:
                    pixel_scale = result['pixel_scale']  # mm/px
                    images_processed[source_image] = pixel_scale
                    success_count += 1

            progress.close()

            # 应用比例尺到轮廓
            if images_processed:
                applied_count = 0

                for contour in self.canvas.contours:
                    source_image = contour.source_image
                    if source_image in images_processed:
                        pixel_scale = images_processed[source_image]

                        # 获取轮廓的原始像素尺寸
                        original_width_px = contour.bounding_box.width()
                        original_height_px = contour.bounding_box.height()

                        if original_width_px > 0 and original_height_px > 0:
                            # 计算实际尺寸（毫米）
                            actual_width_mm = original_width_px * pixel_scale
                            actual_height_mm = original_height_px * pixel_scale

                            # 转换为厘米
                            actual_width_cm = actual_width_mm / 10
                            actual_height_cm = actual_height_mm / 10

                            # 应用尺寸到轮廓
                            # contour.set_size(actual_width_cm, actual_height_cm, self.canvas.pixels_per_cm)
                            contour.set_size(actual_width_cm, actual_height_cm, self.canvas.pixels_per_cm, pixel_scale)
                            applied_count += 1

                self.canvas.update()
                self.statusBar().showMessage(f"已自动标定 {applied_count} 个轮廓")

                # 更新选中的轮廓显示
                if self.canvas.selected_contour:
                    self.on_contour_selected(self.canvas.selected_contour)

                QMessageBox.information(self, "成功",
                                        f"自动标定完成！\n"
                                        f"成功处理 {success_count} 个原始图像\n"
                                        f"应用比例尺到 {applied_count} 个轮廓")
            else:
                QMessageBox.warning(self, "警告",
                                    "无法完成自动标定！\n"
                                    "请检查：\n"
                                    "1. overlays文件夹是否存在\n"
                                    "2. 原始图片命名是否正确（xxx_overlay.png）\n"
                                    "3. 图片中是否包含白色小球标志物")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"自动标定失败: {str(e)}")
            traceback.print_exc()

    def calibrate_selected_contour(self):
        """标定选中轮廓的尺寸"""
        if not self.canvas.selected_contour:
            QMessageBox.warning(self, "警告", "请先选中一个轮廓！")
            return

        try:
            contour = self.canvas.selected_contour
            source_image = contour.source_image

            if not source_image:
                QMessageBox.warning(self, "警告", "选中轮廓没有来源图像信息！")
                return

            # 查找原始图片文件夹
            if not self.current_overlay_dir and self.image_files:
                mask_dir = Path(self.image_files[0]).parent
                overlay_dir = mask_dir.parent / "overlays"
                if overlay_dir.exists():
                    self.current_overlay_dir = str(overlay_dir)
                else:
                    QMessageBox.warning(self, "警告", "找不到原始图片文件夹（overlays）！")
                    return

            # 从掩膜文件名构建原始图片文件名
            if '_mask' in source_image:
                overlay_filename = source_image.replace('_mask', '_overlay')
            else:
                base_name = Path(source_image).stem
                overlay_filename = f"{base_name}_overlay.png"

            # 查找原始图片文件
            overlay_path = Path(self.current_overlay_dir) / overlay_filename
            if not overlay_path.exists():
                for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                    alt_path = Path(self.current_overlay_dir) / f"{Path(overlay_filename).stem}{ext}"
                    if alt_path.exists():
                        overlay_path = alt_path
                        break

            if not overlay_path.exists():
                QMessageBox.warning(self, "警告", f"找不到对应的原始图片：{overlay_filename}")
                return

            # 处理原始图片获取比例尺
            detector = WhiteBallMarkerDetector(ball_diameter_mm=10)
            result = detector.process_single_image(str(overlay_path), None)

            if result and result['detected'] and result['pixel_scale']:
                pixel_scale = result['pixel_scale']  # mm/px

                # 获取轮廓的原始像素尺寸
                original_width_px = contour.bounding_box.width()
                original_height_px = contour.bounding_box.height()

                if original_width_px > 0 and original_height_px > 0:
                    # 计算实际尺寸（毫米）
                    actual_width_mm = original_width_px * pixel_scale
                    actual_height_mm = original_height_px * pixel_scale

                    # 转换为厘米
                    actual_width_cm = actual_width_mm / 10
                    actual_height_cm = actual_height_mm / 10

                    # 应用尺寸到轮廓
                    contour.set_size(actual_width_cm, actual_height_cm, self.canvas.pixels_per_cm)

                    # 更新标号大小
                    contour.update_label_size(self.canvas.pixels_per_cm)

                    self.canvas.update()
                    self.on_contour_selected(contour)  # 更新UI显示

                    self.statusBar().showMessage(
                        f"已标定轮廓 {contour.label} 的尺寸：{actual_width_cm:.1f}cm × {actual_height_cm:.1f}cm")
                    QMessageBox.information(self, "成功",
                                            f"标定成功！\n比例尺：{pixel_scale:.6f} mm/px\n"
                                            f"轮廓尺寸：{actual_width_cm:.1f}cm × {actual_height_cm:.1f}cm")
            else:
                QMessageBox.warning(self, "警告",
                                    "无法检测白色小球标志物！\n请确保原始图片包含完整的白色小球。")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"标定失败: {str(e)}")
            traceback.print_exc()

    def set_tool(self, tool_name):
        """设置当前工具"""
        self.canvas.current_tool = tool_name

        # 更新按钮状态
        for btn_name, btn_tool in [("btn_select", "select"), ("btn_pan", "pan")]:
            btn = self.findChild(QPushButton, btn_name)
            if btn:
                btn.setChecked(tool_name == btn_tool)

        # 更新光标
        if tool_name == "pan":
            self.canvas.setCursor(Qt.OpenHandCursor)
        else:
            self.canvas.setCursor(Qt.ArrowCursor)

        self.statusBar().showMessage(f"当前工具: {'选择' if tool_name == 'select' else '移动'}")

    def zoom_in(self):
        """放大"""
        self.canvas.zoom_factor *= 1.2
        self.canvas.zoom_factor = min(5.0, self.canvas.zoom_factor)
        self.canvas.update()
        self.statusBar().showMessage(f"缩放: {self.canvas.zoom_factor:.1f}x")

    def zoom_out(self):
        """缩小"""
        self.canvas.zoom_factor /= 1.2
        self.canvas.zoom_factor = max(0.1, self.canvas.zoom_factor)
        self.canvas.update()
        self.statusBar().showMessage(f"缩放: {self.canvas.zoom_factor:.1f}x")

    def reset_view(self):
        """重置画布视图（还原缩放和平移）"""
        self.canvas.pan_offset = QPointF(0, 0)
        self.canvas.zoom_factor = 1.0
        self.canvas.update()
        self.statusBar().showMessage("视图已重置")

    def init_laser_control_panel(self):
        """初始化激光控制面板"""
        # 创建激光控制面板（可放在工具栏或侧边栏）
        laser_panel = QWidget()
        layout = QVBoxLayout()

        # 模式选择
        mode_group = QGroupBox("工作模式")
        mode_layout = QHBoxLayout()

        self.sim_radio = QRadioButton("模拟模式")
        self.hardware_radio = QRadioButton("硬件模式")

        # 根据当前模式设置单选按钮
        if self.laser.simulation_mode:
            self.sim_radio.setChecked(True)
        else:
            self.hardware_radio.setChecked(True)

        self.sim_radio.toggled.connect(self.on_mode_changed)
        self.hardware_radio.toggled.connect(self.on_mode_changed)

        mode_layout.addWidget(self.sim_radio)
        mode_layout.addWidget(self.hardware_radio)
        mode_group.setLayout(mode_layout)

        # 状态显示
        status_group = QGroupBox("系统状态")
        status_layout = QVBoxLayout()

        self.mode_label = QLabel("模式: 模拟")
        self.hardware_label = QLabel("硬件: 未连接")
        self.status_label = QLabel("状态: 就绪")

        status_layout.addWidget(self.mode_label)
        status_layout.addWidget(self.hardware_label)
        status_layout.addWidget(self.status_label)
        status_group.setLayout(status_layout)

        # 控制按钮
        btn_layout = QHBoxLayout()

        self.test_btn = QPushButton("测试连接")
        self.test_btn.clicked.connect(self.test_hardware_connection)

        self.cut_btn = QPushButton("执行激光切割")
        self.cut_btn.clicked.connect(self.execute_laser_cutting)
        self.cut_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")

        self.stop_btn = QPushButton("停止")
        self.stop_btn.clicked.connect(self.stop_cutting)
        self.stop_btn.setEnabled(False)

        btn_layout.addWidget(self.test_btn)
        btn_layout.addWidget(self.cut_btn)
        btn_layout.addWidget(self.stop_btn)

        # 添加到主布局
        layout.addWidget(mode_group)
        layout.addWidget(status_group)
        layout.addLayout(btn_layout)
        layout.addStretch()

        laser_panel.setLayout(layout)

        # 添加到主窗口（根据您的布局调整位置）
        # 例如，如果使用QDockWidget:
        dock = QDockWidget("激光控制", self)
        dock.setWidget(laser_panel)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

        # 或者添加到现有的控制面板中
        # self.control_panel.addWidget(laser_panel)

    def on_mode_changed(self):
        """模式切换"""
        if self.sim_radio.isChecked():
            success = self.laser.switch_mode("simulation")
            if success:
                self.status_label.setText("状态: 已切换到模拟模式")
            else:
                self.sim_radio.setChecked(True)
        else:
            success = self.laser.switch_mode("hardware")
            if success:
                self.status_label.setText("状态: 已切换到硬件模式")
            else:
                QMessageBox.warning(self, "警告", "无法切换到硬件模式，硬件可能未连接")
                self.sim_radio.setChecked(True)

        self.update_status_display()

    def test_hardware_connection(self):
        """测试硬件连接"""
        if self.laser.simulation_mode:
            QMessageBox.information(self, "测试结果",
                                    "当前为模拟模式，无需硬件连接。\n"
                                    "所有功能都可以正常使用。")
        else:
            QApplication.setOverrideCursor(Qt.WaitCursor)
            success, msg = self.laser.initialize()
            QApplication.restoreOverrideCursor()

            if success:
                QMessageBox.information(self, "测试结果",
                                        f"硬件连接测试成功！\n{msg}")
            else:
                QMessageBox.warning(self, "测试结果",
                                    f"硬件连接测试失败：\n{msg}\n\n"
                                    "建议切换到模拟模式继续使用其他功能。")

    def update_status_display(self):
        """更新状态显示"""
        status = self.laser.get_status_info()

        mode_text = "模拟" if status["simulation_mode"] else "硬件"
        self.mode_label.setText(f"模式: {mode_text}")

        if status["simulation_mode"]:
            self.hardware_label.setText("硬件: 模拟模式（无需连接）")
            self.hardware_label.setStyleSheet("color: #666666;")
        else:
            if status["initialized"]:
                self.hardware_label.setText("硬件: 已连接")
                self.hardware_label.setStyleSheet("color: #4CAF50;")
            else:
                self.hardware_label.setText("硬件: 未连接")
                self.hardware_label.setStyleSheet("color: #f44336;")

        if status["is_marking"]:
            self.status_label.setText("状态: 切割中...")
            self.status_label.setStyleSheet("color: #FF9800; font-weight: bold;")
        else:
            self.status_label.setText("状态: 就绪")
            self.status_label.setStyleSheet("color: #4CAF50;")

    def execute_laser_cutting(self):
        """执行激光切割 - 修复状态问题"""
        # 1. 检查是否正在切割
        if self.laser.is_marking:
            QMessageBox.warning(self, "警告", "正在切割中，请等待完成")
            return

        # 2. 检查文件
        dxf_path = "output/current_design.dxf"
        if not os.path.exists(dxf_path):
            # 创建测试文件
            self._create_test_dxf(dxf_path)

        # 3. 设置UI状态
        self.cut_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

        # 4. 开始切割
        def callback(msg):
            # 修复：只在主线程中更新UI
            QMetaObject.invokeMethod(self, "_handle_callback",
                                     Qt.QueuedConnection,
                                     Q_ARG(str, msg))

        success, message = self.laser.load_and_execute_dxf(dxf_path, callback)


        if not success:
            QMessageBox.warning(self, "错误", message)
            self._reset_buttons()

        # 如果是模拟模式，显示可视化
        if self.laser.simulation_mode:
            # 显示模拟窗口
            self.sim_dock.setVisible(True)

            # 获取预计时间
            cut_time = self.laser.config.get("simulation_settings", {}).get("cutting_time", 5)

            # 启动可视化模拟
            self.simulation_widget.start_simulation(cut_time)

            # 更新状态
            self.status_label.setText("状态: 模拟切割中...")

    def _on_simulation_completed(self):
        """模拟可视化完成"""
        self.status_label.setText("状态: 模拟切割完成")

    @pyqtSlot(str)
    def _handle_callback(self, msg):
        """处理回调消息"""
        self.status_label.setText(f"状态: {msg}")

        # 修复：当收到"已完成"消息时重置按钮
        if "已完成" in msg or "完成" in msg or "取消" in msg or "出错" in msg:
            # 延迟一下确保状态更新完成
            QTimer.singleShot(200, self._reset_buttons)

    def _reset_buttons(self):
        """重置按钮状态"""
        self.cut_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def stop_cutting(self):
        """停止切割"""
        if not self.laser.is_marking:
            return

        self.laser.stop_cutting()
        self.status_label.setText("状态: 正在停止...")

        # 等待一会儿再检查是否真的停止了
        QTimer.singleShot(1000, self._check_if_stopped)

    def _check_if_stopped(self):
        """检查是否已停止"""
        if not self.laser.is_marking:
            self.status_label.setText("状态: 已停止")
            self._reset_buttons()
        else:
            # 如果还在切割，再等一会儿
            QTimer.singleShot(500, self._check_if_stopped)

    def _create_test_dxf(self, filepath):
        """创建测试DXF文件"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write("0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF")
        print(f"已创建测试文件: {filepath}")

    def prev_page(self):
        """显示上一页"""
        if self.pages_contours and self.current_page > 0:
            self.current_page -= 1
            self.canvas.contours = [item['contour'] for item in self.pages_contours[self.current_page]]
            self.canvas.update()
            self.lbl_page.setText(f"第 {self.current_page + 1} 页 / 共 {len(self.pages_contours)} 页")

    def next_page(self):
        """显示下一页"""
        if self.pages_contours and self.current_page < len(self.pages_contours) - 1:
            self.current_page += 1
            self.canvas.contours = [item['contour'] for item in self.pages_contours[self.current_page]]
            self.canvas.update()
            self.lbl_page.setText(f"第 {self.current_page + 1} 页 / 共 {len(self.pages_contours)} 页")