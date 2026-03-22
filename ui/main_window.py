"""
主窗口
"""
import sys
import traceback
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
                             QCheckBox, QListWidget, QListWidgetItem, QProgressDialog,
                             QFileDialog, QMessageBox, QFormLayout, QMenuBar, QAction,
                             QProgressBar, QApplication, QProgressBar, QRadioButton, QDockWidget)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer, QDateTime, QPointF, QRectF, pyqtSlot, QMetaObject, Q_ARG, QObject, QThread
from PyQt5.QtGui import QPalette, QColor

from core.contour import Contour
# 修改导入部分
from core.image_processor import AdvancedImageProcessor, GEOMDL_AVAILABLE
from core.dxf_exporter import DXFExporter
from core.cloud_manager import CloudDataManager

from ui.canvas_widget import CanvasWidget
from ui.label_mapping_dialog import LabelMappingDialog
# 在文件顶部导入部分添加
from pathlib import Path
from core.marker_detector import WhiteBallMarkerDetector
from core.laser_controller import LaserController
import os
from ui.simulation_widget import SimulationWidget


class DownloadWorker(QObject):
    progress = pyqtSignal(str)
    finished = pyqtSignal(tuple)   # (mask_files, overlay_map)
    error = pyqtSignal(str)

    def __init__(self, cloud_manager, patient_names):
        super().__init__()
        self.cloud_manager = cloud_manager
        self.patient_names = patient_names
        self._canceled = False

    def cancel(self):
        self._canceled = True

    def run(self):
        try:
            mask_files, overlay_map = self.cloud_manager.download_patients(
                self.patient_names,
                progress_callback=self.progress.emit
            )
            if self._canceled:
                self.finished.emit(([], {}))
            else:
                self.finished.emit((mask_files, overlay_map))
        except Exception as e:
            self.error.emit(str(e))

class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("皮肤病灶轮廓处理系统")
        self.setGeometry(100, 100, 1400, 900)

        self.current_directory = ""
        self.image_files = []
        self.label_to_image_map: Dict[int, str] = {}  # 添加：标号到图像名称的映射
        self.label_metadata: Dict[int, dict] = {}  # 标号元数据
        self.next_label = 1  # 添加：下一个标号
        self.next_contour_id = 0  # 全局唯一轮廓ID计数器
        # self.current_precision = 0.5  # 当前拟合精度
        # self.refit_timer = QTimer()  # 用于延迟重新拟合的定时器
        # self.refit_timer.setSingleShot(True)
        # self.refit_timer.timeout.connect(self.on_refit_timeout)
        self.control_points_timer = QTimer()
        self.control_points_timer.setSingleShot(True)
        self.control_points_timer.timeout.connect(self.on_control_points_timeout)
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
        # 添加手动输入尺寸相关变量
        self._updating_spins = False
        self.current_aspect_ratio = 1.0
        # 可以放在停靠窗口或独立窗口
        sim_dock = QDockWidget("切割模拟", self)
        sim_dock.setWidget(self.simulation_widget)
        sim_dock.setVisible(False)  # 默认隐藏，需要时显示
        self.addDockWidget(Qt.RightDockWidgetArea, sim_dock)
        self.sim_dock = sim_dock
        self.lock_aspect = True  # 默认锁定高宽比
        self.label_metadata: Dict[int, dict] = {}
        # 设置异常处理
        sys.excepthook = self.exception_hook

        # 检查geomdl库
        if not GEOMDL_AVAILABLE:
            QMessageBox.warning(self, "库未安装",
                                "geomdl库未安装，将使用贝塞尔曲线进行拟合。\n\n"
                                "要使用更精确的NURBS拟合，请运行:\n"
                                "pip install geomdl")

        # 云端数据管理
        self.cloud_manager = CloudDataManager(
            app_id="wx727c965326d8f905",
            app_secret="f8ee8710362411c0e8686c5aae39e5ef",
            env_id="cloud1-4gut65zm8fa5f13d"
        )
        self.overlay_map = {}  # 存储 mask 文件名到 overlay 路径的映射

        # 创建云端数据停靠窗口
        self.create_cloud_dock()

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
        self.control_panel = self.create_control_panel()

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
        main_layout.addWidget(self.control_panel, 1)
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

        self.control_panel.lock_state_changed.connect(self.on_aspect_lock_changed)

    def _add_contours_common(self, image_files: List[str], source: str = "local") -> bool:
        """公共添加轮廓逻辑，source: 'local' 或 'cloud'"""
        if not image_files:
            return False

        new_contours, has_processed = self._extract_contours_from_images(image_files, source=source)
        if not new_contours:
            if has_processed:
                QMessageBox.information(self, "提示", "所选图像中没有检测到轮廓")
            return False

        # 收集现有全部轮廓（从所有页面中提取）
        all_contours = []
        if hasattr(self, 'pages_contours') and self.pages_contours:
            for page in self.pages_contours:
                for item in page:
                    all_contours.append(item['contour'])
        all_contours.extend(new_contours)

        # 将全部轮廓赋给画布
        self.canvas.contours = all_contours

        # 自动标定所有轮廓（如果存在 overlay 映射）
        self.auto_calibrate_contours()

        # 全局重新排样（会生成新的分页）
        self.global_arrange_contours()

        # 清除选中轮廓
        if self.canvas.selected_contour:
            self.canvas.selected_contour.is_selected = False
            self.canvas.selected_contour = None
            self.canvas.contour_selected.emit(None)

        self.update()
        self.statusBar().showMessage(f"已添加 {len(new_contours)} 个新轮廓，并重新排样")
        return True

    def _create_contour_from_points(self, points: np.ndarray, source_image: str, label: int) -> Contour:
        """
        根据原始轮廓点创建Contour对象，自动分配唯一ID，并进行NURBS拟合。
        """
        self.next_contour_id += 1
        contour = Contour(points, self.next_contour_id, source_image, label)

        # 简化轮廓（可选，与原流程一致）
        simplified_points = AdvancedImageProcessor.simplify_contour(contour.original_points, tolerance=2.0)

        # NURBS拟合，控制点数从界面滑块获取（默认120）
        control_points = self.control_panel.slider_control_points.value() if hasattr(self, 'control_panel') else 120
        nurbs_points, nurbs_curve = AdvancedImageProcessor.smooth_contour_with_nurbs(
            simplified_points,
            num_control_points=control_points
        )
        contour.nurbs_points = nurbs_points
        contour.nurbs_curve = nurbs_curve
        contour.control_points = control_points

        return contour

    def _extract_contours_from_images(self, image_files: List[str], source: str = "local") -> Tuple[
        List[Contour], bool]:
        """
        从图像文件中提取轮廓（不添加至画布），返回新轮廓列表，并更新标号映射。
        如果图像已存在，会弹窗询问用户是否仍然添加。
        """
        new_contours = []
        has_processed = False
        for image_path in image_files:
            print(f"[DEBUG] 正在处理图像: {image_path}")  # 为了调试添加这行
            image_name = Path(image_path).name
            # 检查是否已存在该图像
            existing_label = None
            for label, img in self.label_to_image_map.items():
                if img == image_name:
                    existing_label = label
                    break

            if existing_label is not None:
                reply = QMessageBox.question(
                    self,
                    "重复图像",
                    f"图像 '{image_name}' 已经存在（标号 {existing_label}）。\n"
                    f"是否仍然添加该图像？（将分配新标号）",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply == QMessageBox.No:
                    continue
                # 用户选择添加，分配新标号
                label = self.next_label
                self.label_to_image_map[label] = image_name
                self.canvas.label_to_image_mapping[label] = image_name
                self.next_label += 1
                self.label_metadata[label] = {
                    "source": source,
                    "created": QDateTime.currentDateTime(),
                    "deleted": False,
                    "image_path": image_path  # 完整路径
                }
                has_processed = True
            else:
                label = self.next_label
                self.label_to_image_map[label] = image_name
                self.canvas.label_to_image_mapping[label] = image_name
                self.next_label += 1
                self.label_metadata[label] = {
                    "source": source,
                    "created": QDateTime.currentDateTime(),
                    "deleted": False,
                    "image_path": image_path  # 完整路径
                }
                has_processed = True

            # 提取轮廓
            contours_data = AdvancedImageProcessor.load_and_process_image(image_path, kernel_size=5)
            if not contours_data:
                continue

            for contour_points, _ in contours_data:
                if len(contour_points) < 3:
                    continue
                new_contour = self._create_contour_from_points(contour_points, image_name, label)
                new_contours.append(new_contour)

        return new_contours, has_processed

    def _update_height_from_width(self, new_width_cm=None):
        """锁定状态下，根据当前宽度按原始比例更新高度输入框（不触发应用尺寸）"""
        if not self.canvas.selected_contour:
            return
        contour = self.canvas.selected_contour
        spin_width = self.control_panel.spin_width
        spin_height = self.control_panel.spin_height
        if not spin_width or not spin_height:
            return

        if new_width_cm is None:
            new_width_cm = spin_width.value()

        # 使用原始包围盒比例，而不是当前显示比例
        bbox = contour.bounding_box
        if bbox.width() <= 0 or bbox.height() <= 0:
            return
        aspect = bbox.height() / bbox.width()  # 原始高度/宽度比
        new_height_cm = new_width_cm * aspect

        self._updating_spins = True
        spin_height.blockSignals(True)
        spin_height.setValue(new_height_cm)
        spin_height.blockSignals(False)
        self._updating_spins = False

    def _update_width_from_height(self):
        """锁定状态下，根据当前高度按比例更新宽度输入框（一般不会用到，但保留）"""
        if not self.canvas.selected_contour:
            return
        spin_width = self.control_panel.spin_width
        spin_height = self.control_panel.spin_height
        if not spin_width or not spin_height:
            return
        rect = self.canvas.selected_contour.get_display_rect()
        if rect.isNull():
            return
        current_width_cm = rect.width() / self.canvas.pixels_per_cm
        current_height_cm = rect.height() / self.canvas.pixels_per_cm
        if current_height_cm == 0:
            return
        aspect = current_width_cm / current_height_cm
        new_width_cm = spin_height.value() * aspect
        self._updating_spins = True
        spin_width.blockSignals(True)
        spin_width.setValue(new_width_cm)
        spin_width.blockSignals(False)
        self._updating_spins = False

    def add_contours(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择要添加的图像文件", "",
            "图像文件 (*.png *.jpg *.jpeg *.bmp);;所有文件 (*.*)"
        )
        self._add_contours_common(files, source="local")

    def add_images_and_process(self, image_files: List[str], source: str = "local"):
        """添加并处理指定图像文件（用于云端增量添加）"""
        self._add_contours_common(image_files, source)

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
        arrange_action.triggered.connect(self.rearrange_current_page)
        tools_menu.addAction(arrange_action)

        mapping_action = QAction("查看标号映射", self)
        mapping_action.triggered.connect(self.show_label_mapping_dialog)
        tools_menu.addAction(mapping_action)

    def delete_selected_contour(self):
        """删除当前选中的轮廓"""
        if not self.canvas.selected_contour:
            QMessageBox.warning(self, "警告", "请先选中一个轮廓")
            return

        # 收集现有全部轮廓
        if not hasattr(self, 'pages_contours') or not self.pages_contours:
            return

        all_contours = [item['contour'] for page in self.pages_contours for item in page]
        deleted_label = self.canvas.selected_contour.label
        all_contours = [c for c in all_contours if c is not self.canvas.selected_contour]

        if not all_contours:
            # 没有轮廓了，清空画布和分页
            self.canvas.clear()
            self.pages_contours = []
            self.canvas.contours = []
            self.canvas.selected_contour = None
            self.canvas.contour_selected.emit(None)
            # 标记所有标号已删除
            for label in self.label_metadata:
                self.label_metadata[label]["deleted"] = True
            self.statusBar().showMessage("已删除所有轮廓")
            self.update()
            return

        # 检查被删轮廓的标号是否还有剩余轮廓
        remaining_labels = {c.label for c in all_contours}
        if deleted_label not in remaining_labels:
            self.label_metadata[deleted_label]["deleted"] = True

        # 重新排样
        self.canvas.contours = all_contours
        self.global_arrange_contours()
        self.statusBar().showMessage("已删除选中轮廓，并重新排样")

        # 清除选中状态
        if self.canvas.selected_contour:
            self.canvas.selected_contour.is_selected = False
            self.canvas.selected_contour = None
            self.canvas.contour_selected.emit(None)
        self.update()

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

    def process_all_images(self, source: str = "local"):
        """处理所有图像，提取轮廓并排样"""
        try:
            if not self.image_files:
                QMessageBox.warning(self, "警告", "请先加载图像文件！")
                return

            # 清空现有轮廓和映射
            self.canvas.clear()
            self.label_to_image_map.clear()
            self.canvas.label_to_image_mapping.clear()
            self.label_metadata.clear()  # 清空元数据
            self.next_label = 1
            self.next_contour_id = 0

            # 禁用选择组
            self.control_panel.selection_group.setEnabled(False)

            # 获取处理参数
            kernel_size = self.findChild(QSpinBox, "spin_kernel_size").value()

            # 创建进度对话框
            progress = QProgressDialog("处理图像中...", "取消", 0, len(self.image_files), self)
            progress.setWindowTitle("处理进度")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

            all_contours = []  # 临时存储所有轮廓

            for idx, image_path in enumerate(self.image_files):
                progress.setValue(idx)
                progress.setLabelText(f"处理图像 {idx + 1}/{len(self.image_files)}: {Path(image_path).name}")

                if progress.wasCanceled():
                    break

                # 处理图像，获取轮廓点集
                contours_data = AdvancedImageProcessor.load_and_process_image(image_path, kernel_size)
                if not contours_data:
                    continue

                # 分配标号
                label = self.next_label
                image_name = Path(image_path).name
                self.label_to_image_map[label] = image_name
                self.canvas.label_to_image_mapping[label] = image_name
                self.next_label += 1

                # 记录元数据
                self.label_metadata[label] = {
                    "source": source,
                    "created": QDateTime.currentDateTime(),
                    "deleted": False,
                    "image_path": image_path   # 完整路径
                }

                for contour_points, _ in contours_data:
                    if len(contour_points) < 3:
                        continue
                    simplified_points = AdvancedImageProcessor.simplify_contour(contour_points, tolerance=2.0)
                    contour = self._create_contour_from_points(simplified_points, image_name, label)
                    all_contours.append(contour)

                QApplication.processEvents()  # 更新UI

            progress.close()

            if not all_contours:
                QMessageBox.information(self, "提示", "图像中没有检测到轮廓")
                return

            # 将全部轮廓赋给画布
            self.canvas.contours = all_contours

            # 自动标定所有轮廓尺寸
            self.auto_calibrate_contours()

            # 全局排样
            self.global_arrange_contours()

            # 更新状态栏
            self.statusBar().showMessage(f"已处理 {len(self.image_files)} 个图像，提取到 {len(all_contours)} 个轮廓")

            # 提示完成
            method = 'NURBS' if GEOMDL_AVAILABLE else '贝塞尔'
            QMessageBox.information(self, "完成",
                                    f"成功处理 {len(all_contours)} 个轮廓！使用{method}曲线拟合")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"处理图像失败: {str(e)}")
            traceback.print_exc()

    def global_arrange_contours(self, margin_mm=1.0):
        """自动排列所有轮廓到画布页面（极坐标排样）"""
        try:
            from shapely.geometry import Polygon, Point
            from shapely import affinity, prepared
            from shapely.ops import unary_union
            import math
        except ImportError:
            QMessageBox.critical(self, "缺少依赖", "请安装shapely库：pip install shapely")
            return

        # 创建进度提示对话框
        progress = QProgressDialog("正在自动排样，请稍候...", None, 0, 0, self)
        progress.setWindowTitle("自动排样")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)  # 立即显示
        progress.show()
        QApplication.processEvents()  # 确保对话框显示

        try:
            if not self.canvas.contours:
                progress.close()
                return

            pixels_per_cm = self.canvas.pixels_per_cm
            container_radius_px = 5 * pixels_per_cm  # 5cm半径
            center_x = self.canvas.canvas_width_px / 2
            center_y = self.canvas.canvas_height_px / 2
            margin_px = margin_mm * (pixels_per_cm / 10)  # 间距转换为像素

            # 构建轮廓多边形数据
            contours_data = []
            total = len(self.canvas.contours)
            for idx, contour in enumerate(self.canvas.contours):
                # 定期处理事件，保持界面响应
                if idx % 10 == 0:
                    QApplication.processEvents()

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
                    # 如果变为 MultiPolygon，取面积最大的
                    if original_poly.geom_type == 'MultiPolygon':
                        original_poly = max(original_poly.geoms, key=lambda p: p.area)
                poly_with_margin = original_poly.buffer(margin_px, join_style=2)
                if not poly_with_margin.is_valid:
                    poly_with_margin = poly_with_margin.buffer(0)
                    if poly_with_margin.geom_type == 'MultiPolygon':
                        poly_with_margin = max(poly_with_margin.geoms, key=lambda p: p.area)

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
                # 逐个检查多边形外环上的每个顶点
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

                # 合并已放置区域的所有带间距多边形，形成联合多边形，表示禁止区域
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
            total_contours = len(contours_data)
            for i, data in enumerate(contours_data):
                # 定期处理事件
                if i % 10 == 0:
                    QApplication.processEvents()

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

        finally:
            # 无论成功或异常，关闭进度对话框
            progress.close()

    def rearrange_current_page(self, margin_mm=1.0):
        """仅对当前页的轮廓进行极坐标重新排列，放不下的轮廓与最后一页合并重排"""
        try:
            from shapely.geometry import Polygon
            from shapely import affinity, prepared
            from shapely.ops import unary_union
            import math
        except ImportError:
            QMessageBox.critical(self, "缺少依赖", "请安装shapely库：pip install shapely")
            return
        # 防御：如果 margin_mm 不是数值（如 bool），则使用默认值
        # if not isinstance(margin_mm, (int, float)):
        print(f"margin_mm: = {margin_mm}")
        # if not isinstance(margin_mm, (int, float)):
        #     print(f"margin_mm不是int或float")
        # margin_mm = 1.0

        # 如果尚未分页，回退到全局排样
        if not hasattr(self, 'pages_contours') or not self.pages_contours:
            self.global_arrange_contours(margin_mm)
            return

        current_idx = self.current_page
        total_pages = len(self.pages_contours)
        if current_idx >= total_pages:
            return

        # 画布参数
        pixels_per_cm = self.canvas.pixels_per_cm
        container_radius_px = 5 * pixels_per_cm
        center_x = self.canvas.canvas_width_px / 2
        center_y = self.canvas.canvas_height_px / 2
        margin_px = margin_mm * (pixels_per_cm / 10)
        # print(f"rearrange: pixels_per_cm = {self.canvas.pixels_per_cm}, margin_mm = {margin_mm}")
        # print(f"margin_px = {margin_px}")

        # 辅助函数：尝试在容器内放置多边形
        def try_place(poly, placed_items):
            """尝试将 poly 放入容器，返回 (成功标志, dx, dy, 放置后的多边形)"""
            poly_bounds = poly.bounds
            poly_width = poly_bounds[2] - poly_bounds[0]
            poly_height = poly_bounds[3] - poly_bounds[1]
            max_dim = max(poly_width, poly_height)

            # 构建已放置区域的禁止多边形
            if placed_items:
                placed_polys = [item['poly_with_margin'] for item in placed_items]
                placed_union = unary_union(placed_polys)
                placed_prep = prepared.prep(placed_union)
            else:
                placed_prep = None

            # 从圆心向外螺旋搜索
            step_r = max_dim / 2
            max_r = container_radius_px
            radii = [i * step_r for i in range(int(max_r / step_r) + 1)]

            for r in radii:
                if r == 0:
                    angles = [0.0]
                else:
                    circumference = 2 * math.pi * r
                    n_angles = max(1, int(circumference / (max_dim / 2)))
                    n_angles = min(n_angles, 36)
                    angles = [2 * math.pi * i / n_angles for i in range(n_angles)]

                for angle in angles:
                    x = center_x + r * math.cos(angle)
                    y = center_y + r * math.sin(angle)
                    dx = x - (poly_bounds[0] + poly_bounds[2]) / 2
                    dy = y - (poly_bounds[1] + poly_bounds[3]) / 2
                    candidate = affinity.translate(poly, dx, dy)

                    # 检查是否完全在圆内
                    if candidate.is_empty:
                        continue
                    minx, miny, maxx, maxy = candidate.bounds
                    if (maxx - center_x) ** 2 + (maxy - center_y) ** 2 > container_radius_px ** 2:
                        continue
                    out_of_circle = False
                    for xc, yc in candidate.exterior.coords:
                        if (xc - center_x) ** 2 + (yc - center_y) ** 2 > container_radius_px ** 2:
                            out_of_circle = True
                            break
                    if out_of_circle:
                        continue

                    # 检查与已放置区域是否重叠
                    if placed_prep is not None and placed_prep.intersects(candidate):
                        continue

                    return True, dx, dy, candidate

            return False, 0, 0, None

        # 辅助函数：批量放置轮廓（修改轮廓的 position 并更新多边形）
        def place_items(items, container_placed_items):
            """将 items 中的轮廓逐个尝试放置到容器中，返回 (成功放置的列表, 失败的列表)"""
            placed = []
            failed = []
            for item in items:
                poly = item['poly_with_margin']
                success, dx, dy, new_poly = try_place(poly, container_placed_items + placed)
                if success:
                    # 更新轮廓位置
                    item['contour'].position += QPointF(dx, dy)
                    # 更新多边形
                    item['original_poly'] = affinity.translate(item['original_poly'], dx, dy)
                    item['poly_with_margin'] = new_poly
                    placed.append(item)
                else:
                    failed.append(item)
            return placed, failed

        # ---------- 1. 重新计算当前页所有轮廓的实时多边形 ----------
        current_page_data = self.pages_contours[current_idx]
        for item in current_page_data:
            contour = item['contour']
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
            item['original_poly'] = original_poly
            item['poly_with_margin'] = poly_with_margin

        # 按面积降序排列
        current_page_data.sort(key=lambda x: x['poly_with_margin'].area, reverse=True)

        # ---------- 2. 在当前页尝试放置 ----------
        placed_current, overflow = place_items(current_page_data, [])

        # 更新当前页数据为成功放置的轮廓
        self.pages_contours[current_idx] = placed_current

        # 如果没有溢出，直接结束
        if not overflow:
            self.canvas.contours = [item['contour'] for item in placed_current]
            self.canvas.update()
            self.lbl_page.setText(f"第 {current_idx + 1} 页 / 共 {total_pages} 页")
            self.statusBar().showMessage("当前页重排完成")
            return

        # ---------- 3. 有溢出：获取最后一页 ----------
        last_idx = total_pages - 1
        if last_idx == current_idx:
            # 当前页就是最后一页，且放不下，则需要新建一页
            new_page_data = []
            placed_new, still_overflow = place_items(overflow, [])
            if placed_new:
                self.pages_contours.append(placed_new)
            if still_overflow:
                # 极端情况：即使新页也放不下（可能轮廓太大），强制放在圆心并警告
                for item in still_overflow:
                    print(f"警告：轮廓 {item['contour'].label} 尺寸过大，无法放入新页，已强制放置在圆心")
                    poly = item['poly_with_margin']
                    dx = center_x - (poly.bounds[0] + poly.bounds[2]) / 2
                    dy = center_y - (poly.bounds[1] + poly.bounds[3]) / 2
                    item['contour'].position += QPointF(dx, dy)
                    item['original_poly'] = affinity.translate(item['original_poly'], dx, dy)
                    item['poly_with_margin'] = affinity.translate(poly, dx, dy)
                    placed_new.append(item)
                self.pages_contours.append(placed_new)
            self.canvas.contours = [item['contour'] for item in placed_current]
            self.canvas.update()
            self.lbl_page.setText(f"第 {current_idx + 1} 页 / 共 {len(self.pages_contours)} 页")
            self.statusBar().showMessage(f"当前页重排完成，溢出轮廓已放入新页")
            return

        # 否则，获取最后一页的数据
        last_page_data = self.pages_contours[last_idx]

        # ---------- 4. 重新计算最后一页轮廓的实时多边形 ----------
        for item in last_page_data:
            contour = item['contour']
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
            item['original_poly'] = original_poly
            item['poly_with_margin'] = poly_with_margin

        # ---------- 5. 将溢出轮廓与最后一页轮廓合并，重新排最后一页 ----------
        combined = overflow + last_page_data
        combined.sort(key=lambda x: x['poly_with_margin'].area, reverse=True)

        placed_combined, still_overflow = place_items(combined, [])
        self.pages_contours[last_idx] = placed_combined

        # 如果 still_overflow 非空，说明最后一页也放不下，需要新建页面
        while still_overflow:
            new_page_data = []
            placed_new, still_overflow = place_items(still_overflow, [])
            if placed_new:
                self.pages_contours.append(placed_new)
            if still_overflow:
                # 极端情况，强制放置
                for item in still_overflow:
                    print(f"警告：轮廓 {item['contour'].label} 尺寸过大，无法放入新页，已强制放置在圆心")
                    poly = item['poly_with_margin']
                    dx = center_x - (poly.bounds[0] + poly.bounds[2]) / 2
                    dy = center_y - (poly.bounds[1] + poly.bounds[3]) / 2
                    item['contour'].position += QPointF(dx, dy)
                    item['original_poly'] = affinity.translate(item['original_poly'], dx, dy)
                    item['poly_with_margin'] = affinity.translate(poly, dx, dy)
                    placed_new.append(item)
                self.pages_contours.append(placed_new)
                still_overflow = []  # 强制结束

        # ---------- 6. 更新画布显示当前页 ----------
        self.canvas.contours = [item['contour'] for item in placed_current]
        self.canvas.update()
        self.lbl_page.setText(f"第 {current_idx + 1} 页 / 共 {len(self.pages_contours)} 页")
        self.statusBar().showMessage(f"当前页重排完成，溢出轮廓已移至最后页处理")

    def rearrange_process(self):
        """重排当前页轮廓的包装函数，用于按钮信号连接，确保间距参数正确传递"""
        self.rearrange_current_page(1.0)

    def clear_contours(self):
        """清空所有轮廓"""
        self.canvas.clear()
        # 使用 control_panel 中的 selection_group 和 lbl_selected_info
        self.control_panel.selection_group.setEnabled(False)
        self.control_panel.lbl_selected_info.setText("未选中轮廓")

        self.label_to_image_map.clear()
        self.canvas.label_to_image_mapping.clear()
        self.next_label = 1
        self.statusBar().showMessage("已清空所有轮廓")

    def on_contour_selected(self, contour):
        if contour:
            self.control_panel.selection_group.setEnabled(True)
            label_info = f"标号 {contour.label}" if contour.label > 0 else "无标号"
            self.control_panel.lbl_selected_info.setText(f"{label_info} - 轮廓 {contour.id} ({contour.source_image})")
            rect = contour.get_display_rect()
            width_cm = rect.width() / self.canvas.pixels_per_cm
            height_cm = rect.height() / self.canvas.pixels_per_cm
            self._updating_spins = True
            self.control_panel.spin_width.setValue(width_cm)
            self.control_panel.spin_height.setValue(height_cm)
            locked = self.control_panel.lock_btn.isChecked()
            self.control_panel.spin_height.setEnabled(not locked)
            if locked:
                self._update_height_from_width()
            self._updating_spins = False

            # 同步控制点滑块和数值框
            control_points = getattr(contour, 'control_points', 120)  # 默认120
            # 阻塞信号，避免触发重新拟合
            self.control_panel.slider_control_points.blockSignals(True)
            self.control_panel.spin_control_points.blockSignals(True)
            self.control_panel.slider_control_points.setValue(control_points)
            self.control_panel.spin_control_points.setValue(control_points)
            self.control_panel.slider_control_points.blockSignals(False)
            self.control_panel.spin_control_points.blockSignals(False)
        else:
            self.control_panel.selection_group.setEnabled(False)
            self.control_panel.lbl_selected_info.setText("未选中轮廓")

    def on_spin_width_changed(self, value):
        if self._updating_spins or not self.canvas.selected_contour:
            return

        # 锁定状态下同步高度，并传入新宽度值
        if self.control_panel.lock_btn.isChecked():
            self._update_height_from_width(value)

        # 最终应用尺寸
        self.apply_contour_size()

    def on_spin_height_changed(self, value):
        if self._updating_spins or not self.canvas.selected_contour:
            return
        self.apply_contour_size()

    def on_aspect_lock_changed(self, locked):
        if not self.canvas.selected_contour:
            return
        if locked:
            self._update_height_from_width()
            self.apply_contour_size()
        else:
            self.control_panel.spin_height.setEnabled(True)
            self.apply_contour_size()

    def on_download_finished(self, result):
        self.progress_dialog.close()
        mask_files, overlay_map = result
        if not mask_files:
            QMessageBox.information(self, "提示", "没有下载到任何mask文件")
            return

        self.overlay_map = overlay_map
        self.image_files = mask_files
        self.process_all_images(source="cloud")  # 标记云端来源

    def on_download_add(self):
        selected_items = self.patient_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请至少选择一个患者")
            return
        patient_names = [item.text() for item in selected_items]

        reply = QMessageBox.question(self, "确认", f"确定将选中的 {len(patient_names)} 个患者数据添加到当前项目吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        self.download_thread = QThread()
        self.download_worker = DownloadWorker(self.cloud_manager, patient_names)
        self.download_worker.moveToThread(self.download_thread)

        self.download_worker.progress.connect(self.on_download_progress)
        self.download_worker.finished.connect(self.on_download_add_finished)
        self.download_worker.error.connect(self.on_download_error)
        self.download_thread.started.connect(self.download_worker.run)

        # 关键修改：确保线程先退出再销毁
        self.download_worker.finished.connect(self.download_thread.quit)
        self.download_thread.finished.connect(self.download_thread.deleteLater)
        self.download_thread.finished.connect(self.download_worker.deleteLater)
        self.download_worker.error.connect(self.download_thread.quit)

        self.progress_dialog = QProgressDialog("正在下载患者数据...", "取消", 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.canceled.connect(self.download_worker.cancel)

        self.download_thread.start()
        self.progress_dialog.show()

    def on_download_add_finished(self, result):
        self.progress_dialog.close()
        mask_files, overlay_map = result
        if not mask_files:
            QMessageBox.information(self, "提示", "没有下载到任何mask文件")
            return

        self.overlay_map.update(overlay_map)
        self.add_images_and_process(mask_files, source="cloud")  # 显示调用标记来源
        # 强制释放内存和OpenCV资源
        import gc
        gc.collect()
        cv2.destroyAllWindows()

    def apply_contour_size(self):
        if not self.canvas.selected_contour:
            QMessageBox.warning(self, "警告", "请先选中一个轮廓！")
            return

        contour = self.canvas.selected_contour
        spin_width = self.control_panel.spin_width
        spin_height = self.control_panel.spin_height

        if not spin_width or not spin_height:
            return

        width_cm = spin_width.value()
        height_cm = spin_height.value()

        # 获取轮廓原始包围盒像素尺寸
        bbox = contour.bounding_box
        if bbox.width() <= 0 or bbox.height() <= 0:
            return

        # 根据锁定状态决定缩放方式
        if self.control_panel.lock_btn.isChecked():
            # 锁定模式：按宽度等比例缩放
            target_width_px = width_cm * self.canvas.pixels_per_cm
            scale = target_width_px / bbox.width()
            # 计算新的高度（用于更新输入框）
            new_height_px = bbox.height() * scale
            new_height_cm = new_height_px / self.canvas.pixels_per_cm
            # 更新高度输入框（不触发信号）
            self._updating_spins = True
            spin_height.blockSignals(True)
            spin_height.setValue(new_height_cm)
            spin_height.blockSignals(False)
            self._updating_spins = False
            # 关键修改：同时设置 scale_x 和 scale_y
            contour.scale_x = scale
            contour.scale_y = scale
            contour.scale = scale  # 保持兼容
            # 更新实际尺寸记录
            contour.actual_width_cm = width_cm
            contour.actual_height_cm = new_height_cm
        else:
            # 非锁定模式：使用 set_size 方法（允许拉伸）
            contour.set_size(width_cm, height_cm, self.canvas.pixels_per_cm)

        # 更新标号大小
        contour.update_label_size(self.canvas.pixels_per_cm)

        self.canvas.update()
        self.statusBar().showMessage(f"已设置轮廓尺寸: {width_cm:.2f}cm × {height_cm:.2f}cm")

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

        dialog = LabelMappingDialog(self.label_to_image_map, self.label_metadata, self)
        dialog.exec_()

    # def on_precision_changed(self, value):
    #     """精度滑块值改变事件"""
    #     # 更新精度值显示
    #     precision_percent = value
    #     lbl_precision_value = self.findChild(QLabel, "lbl_precision_value")
    #     if lbl_precision_value:
    #         # 将50-150映射到50%-150%显示
    #         display_percent = precision_percent
    #         lbl_precision_value.setText(f"{display_percent}%")
    #
    #     # 计算精度值 (0.5-1.5)
    #     # 将滑块值50-150映射到0.5-1.5
    #     new_precision = 0.5 + (precision_percent - 50) * 0.01
    #     # 限制在0.5-1.5范围内（理论上不会超出，但为了安全）
    #     new_precision = max(0.5, min(2, new_precision))
    #
    #     # 只有当精度变化足够大时才重新拟合
    #     if abs(new_precision - self.current_precision) > 0.005:
    #         self.current_precision = new_precision
    #
    #         # 启动定时器进行延迟重新拟合
    #         self.refit_timer.start(200)  # 200毫秒后重新拟合
    #         self.statusBar().showMessage(f"正在调整拟合精度到 {precision_percent}%...")
    #
    # def on_refit_timeout(self):
    #     """定时器超时，重新拟合所有轮廓"""
    #     if self.canvas.contours:
    #         self.canvas.refit_all_contours(self.current_precision)
    #         self.statusBar().showMessage(f"已重新拟合所有轮廓 (精度: {int(self.current_precision * 100)}%)")

    def on_control_points_changed(self, value):
        # 更新数值框显示
        spin = self.findChild(QSpinBox, "spin_control_points")
        if spin:
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)

        # 更新标签显示
        lbl = self.findChild(QLabel, "lbl_control_points")
        if lbl:
            lbl.setText(f"{value} 个点")

        # 启动定时器延迟重绘
        self.control_points_timer.start(300)

    def on_control_points_spin_changed(self, value):
        # 更新滑块
        slider = self.findChild(QSlider, "slider_control_points")
        if slider:
            slider.blockSignals(True)
            slider.setValue(value)
            slider.blockSignals(False)

        # 更新标签
        lbl = self.findChild(QLabel, "lbl_control_points")
        if lbl:
            lbl.setText(f"{value} 个点")

        self.control_points_timer.start(300)

    def on_control_points_timeout(self):
        """定时器超时，重新拟合选中的轮廓"""
        if not self.canvas.selected_contour:
            return

        slider = self.findChild(QSlider, "slider_control_points")
        if not slider:
            return

        num_points = slider.value()
        contour = self.canvas.selected_contour

        # 重新拟合该轮廓
        self.canvas.refit_single_contour(contour, num_points)

        # 更新画布
        self.canvas.update()
        self.statusBar().showMessage(f"已重新拟合轮廓 {contour.label}，控制点数: {num_points}")

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
                    "control_points": contour.control_points,
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
        """自动标定轮廓尺寸（优先使用 overlay_map）"""
        try:
            if not self.canvas.contours:
                QMessageBox.warning(self, "警告", "没有轮廓可以标定，请先处理图像！")
                return

            # 如果存在 overlay_map，使用映射标定
            if hasattr(self, 'overlay_map') and self.overlay_map:
                self._calibrate_with_map()
                return

            # 原有逻辑（基于本地文件结构）
            if not self.current_overlay_dir and self.image_files:
                mask_dir = Path(self.image_files[0]).parent
                overlay_dir = mask_dir.parent / "overlays"
                if overlay_dir.exists():
                    self.current_overlay_dir = str(overlay_dir)
                else:
                    QMessageBox.warning(self, "警告", "找不到原始图片文件夹（overlays）！")
                    return

            # 创建检测器
            detector = WhiteBallMarkerDetector(ball_diameter_mm=10)
            progress = QProgressDialog("自动标定中...", "取消", 0, len(self.canvas.contours), self)
            progress.setWindowTitle("自动标定")
            progress.setWindowModality(Qt.WindowModal)
            progress.show()

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

                # 构建原始图片文件名
                if '_mask' in source_image:
                    overlay_filename = source_image.replace('_mask', '_overlay')
                else:
                    base_name = Path(source_image).stem
                    overlay_filename = f"{base_name}_overlay.png"

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

                result = detector.process_single_image(str(overlay_path), None)
                if result and result['detected'] and result['pixel_scale']:
                    pixel_scale = result['pixel_scale']
                    images_processed[source_image] = pixel_scale
                    success_count += 1

            progress.close()

            # 应用比例尺
            applied_count = 0
            for contour in self.canvas.contours:
                if contour.source_image in images_processed:
                    pixel_scale = images_processed[contour.source_image]
                    orig_w = contour.bounding_box.width()
                    orig_h = contour.bounding_box.height()
                    if orig_w > 0 and orig_h > 0:
                        w_mm = orig_w * pixel_scale
                        h_mm = orig_h * pixel_scale
                        contour.set_size(w_mm / 10, h_mm / 10, self.canvas.pixels_per_cm, pixel_scale)
                        applied_count += 1

            self.canvas.update()
            self.statusBar().showMessage(f"已自动标定 {applied_count} 个轮廓")
            if self.canvas.selected_contour:
                self.on_contour_selected(self.canvas.selected_contour)

            if applied_count == 0:
                QMessageBox.warning(self, "警告", "自动标定未成功，请检查overlay文件或手动标定。")
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

    def _calibrate_with_map(self):
        """使用 overlay_map 进行标定，若不存在则回退到本地查找"""
        detector = WhiteBallMarkerDetector(ball_diameter_mm=10)
        progress = QProgressDialog("自动标定中...", "取消", 0, len(self.canvas.contours), self)
        progress.setWindowTitle("自动标定")
        progress.setWindowModality(Qt.WindowModal)
        progress.show()

        applied_count = 0
        for i, contour in enumerate(self.canvas.contours):
            progress.setValue(i)
            QApplication.processEvents()
            if progress.wasCanceled():
                break

            source_image = contour.source_image
            # 优先使用 overlay_map
            if source_image in self.overlay_map:
                overlay_path = self.overlay_map[source_image]
            else:
                # 回退到本地查找（原有的自动标定逻辑）
                if not self.current_overlay_dir and self.image_files:
                    mask_dir = Path(self.image_files[0]).parent
                    overlay_dir = mask_dir.parent / "overlays"
                    if overlay_dir.exists():
                        self.current_overlay_dir = str(overlay_dir)
                if not self.current_overlay_dir:
                    continue

                # 构建 overlay 文件名
                if '_mask' in source_image:
                    overlay_filename = source_image.replace('_mask', '_overlay')
                else:
                    base_name = Path(source_image).stem
                    overlay_filename = f"{base_name}_overlay.png"

                overlay_path = Path(self.current_overlay_dir) / overlay_filename
                if not overlay_path.exists():
                    # 尝试其他扩展名
                    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                        alt_path = Path(self.current_overlay_dir) / f"{Path(overlay_filename).stem}{ext}"
                        if alt_path.exists():
                            overlay_path = alt_path
                            break
                    else:
                        continue  # 未找到对应文件

            # 使用检测器处理
            result = detector.process_single_image(str(overlay_path), None)
            if result and result['detected'] and result['pixel_scale']:
                pixel_scale = result['pixel_scale']
                orig_w = contour.bounding_box.width()
                orig_h = contour.bounding_box.height()
                if orig_w > 0 and orig_h > 0:
                    w_mm = orig_w * pixel_scale
                    h_mm = orig_h * pixel_scale
                    contour.set_size(w_mm / 10, h_mm / 10, self.canvas.pixels_per_cm, pixel_scale)
                    applied_count += 1

        progress.close()
        self.canvas.update()
        self.statusBar().showMessage(f"已自动标定 {applied_count} 个轮廓")
        if self.canvas.selected_contour:
            self.on_contour_selected(self.canvas.selected_contour)

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

            # 清除选中轮廓
            if self.canvas.selected_contour:
                self.canvas.selected_contour.is_selected = False
                self.canvas.selected_contour = None
                self.canvas.contour_selected.emit(None)  # 通知主窗口更新 UI

            self.canvas.update()
            self.lbl_page.setText(f"第 {self.current_page + 1} 页 / 共 {len(self.pages_contours)} 页")

    def next_page(self):
        """显示下一页"""
        if self.pages_contours and self.current_page < len(self.pages_contours) - 1:
            self.current_page += 1
            self.canvas.contours = [item['contour'] for item in self.pages_contours[self.current_page]]

            # 清除选中轮廓
            if self.canvas.selected_contour:
                self.canvas.selected_contour.is_selected = False
                self.canvas.selected_contour = None
                self.canvas.contour_selected.emit(None)

            self.canvas.update()
            self.lbl_page.setText(f"第 {self.current_page + 1} 页 / 共 {len(self.pages_contours)} 页")

    def create_cloud_dock(self):
        """创建云端数据停靠面板"""
        from PyQt5.QtWidgets import QDockWidget, QAbstractItemView

        dock = QDockWidget("云端数据", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)

        widget = QWidget()
        layout = QVBoxLayout(widget)

        # 按钮行
        btn_layout = QHBoxLayout()
        self.btn_refresh = QPushButton("刷新患者列表")
        self.btn_refresh.clicked.connect(self.on_refresh_patient_list)
        self.btn_download = QPushButton("下载并处理选中患者")
        self.btn_download.clicked.connect(self.on_download_selected)
        btn_layout.addWidget(self.btn_refresh)
        btn_layout.addWidget(self.btn_download)
        layout.addLayout(btn_layout)

        self.btn_download_add = QPushButton("添加选中患者数据")
        self.btn_download_add.clicked.connect(self.on_download_add)
        btn_layout.addWidget(self.btn_download_add)

        # 患者列表（多选）
        self.patient_list = QListWidget()
        self.patient_list.setSelectionMode(QAbstractItemView.MultiSelection)
        layout.addWidget(self.patient_list)

        # 状态标签
        self.lbl_cloud_status = QLabel("就绪")
        layout.addWidget(self.lbl_cloud_status)

        dock.setWidget(widget)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def on_refresh_patient_list(self):
        """刷新患者列表"""
        self.lbl_cloud_status.setText("正在获取患者列表...")
        QApplication.processEvents()
        try:
            patients = self.cloud_manager.fetch_patient_names()
            self.patient_list.clear()
            self.patient_list.addItems(patients)
            self.lbl_cloud_status.setText(f"获取到 {len(patients)} 个患者")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"获取患者列表失败: {str(e)}")
            self.lbl_cloud_status.setText("获取失败")

    def on_download_selected(self):
        """下载选中患者并处理"""
        selected_items = self.patient_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "警告", "请至少选择一个患者")
            return
        patient_names = [item.text() for item in selected_items]

        reply = QMessageBox.question(self, "确认", f"确定下载并处理选中的 {len(patient_names)} 个患者吗？",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply != QMessageBox.Yes:
            return

        # 启动下载线程
        self.download_thread = QThread()
        self.download_worker = DownloadWorker(self.cloud_manager, patient_names)
        self.download_worker.moveToThread(self.download_thread)

        self.download_worker.progress.connect(self.on_download_progress)
        self.download_worker.finished.connect(self.on_download_finished)
        self.download_worker.error.connect(self.on_download_error)

        # 线程生命周期管理（确保线程正确退出后再销毁）
        self.download_worker.finished.connect(self.download_thread.quit)
        self.download_worker.error.connect(self.download_thread.quit)
        self.download_thread.finished.connect(self.download_worker.deleteLater)
        self.download_thread.finished.connect(self.download_thread.deleteLater)

        self.download_thread.started.connect(self.download_worker.run)

        self.progress_dialog = QProgressDialog("正在下载患者数据...", "取消", 0, 0, self)
        self.progress_dialog.setWindowModality(Qt.WindowModal)
        self.progress_dialog.canceled.connect(self.download_worker.cancel)

        self.download_thread.start()
        self.progress_dialog.show()

    def on_download_progress(self, msg):
        self.progress_dialog.setLabelText(msg)

    def on_download_finished(self, result):
        self.progress_dialog.close()
        mask_files, overlay_map = result
        if not mask_files:
            QMessageBox.information(self, "提示", "没有下载到任何mask文件")
            return

        self.overlay_map = overlay_map
        self.image_files = mask_files
        self.process_all_images(source="cloud")  # 明确指定云端来源

        # 强制释放内存和OpenCV资源
        import gc
        gc.collect()
        cv2.destroyAllWindows()

    def on_download_error(self, err_msg):
        self.progress_dialog.close()
        QMessageBox.critical(self, "下载错误", err_msg)