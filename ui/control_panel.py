"""
控制面板
"""
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
                             QPushButton, QLabel, QSlider, QSpinBox, QDoubleSpinBox,
                             QCheckBox, QListWidget, QListWidgetItem, QFormLayout,
                             QFrame, QListWidget)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QFont


class ControlPanel(QWidget):
    """控制面板"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.setMaximumWidth(350)
        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)

        # 文件操作组
        file_group = QGroupBox("文件操作")
        file_layout = QVBoxLayout()

        btn_load_images = QPushButton("📁 加载多个图像文件")
        btn_load_images.setObjectName("btn_load_images")
        btn_load_images.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-weight: bold;
                background-color: #4CAF50;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        btn_load_images.clicked.connect(self.main_window.load_images)
        file_layout.addWidget(btn_load_images)

        btn_process_all = QPushButton("🔄 处理所有图像")
        btn_process_all.setObjectName("btn_process_all")
        btn_process_all.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-weight: bold;
                background-color: #2196F3;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
        """)
        btn_process_all.clicked.connect(self.main_window.process_all_images)
        file_layout.addWidget(btn_process_all)

        lbl_image_info = QLabel("未加载图像")
        lbl_image_info.setObjectName("lbl_image_info")
        lbl_image_info.setAlignment(Qt.AlignCenter)
        lbl_image_info.setStyleSheet("padding: 10px; background-color: #f8f9fa; border-radius: 5px;")
        lbl_image_info.setWordWrap(True)
        file_layout.addWidget(lbl_image_info)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # 拟合精度控制组
        precision_group = QGroupBox("拟合精度控制")
        precision_layout = QVBoxLayout()

        # 精度标签
        precision_label_layout = QHBoxLayout()
        lbl_precision_title = QLabel("拟合精度:")
        lbl_precision_value = QLabel("50%")
        lbl_precision_value.setObjectName("lbl_precision_value")
        lbl_precision_value.setStyleSheet("font-weight: bold; font-size: 14px; color: #2196F3;")
        precision_label_layout.addWidget(lbl_precision_title)
        precision_label_layout.addStretch()
        precision_label_layout.addWidget(lbl_precision_value)
        precision_layout.addLayout(precision_label_layout)

        # 精度滑块
        slider_precision = QSlider(Qt.Horizontal)
        slider_precision.setObjectName("slider_precision")
        slider_precision.setRange(50, 200)
        slider_precision.setValue(100)
        slider_precision.setTickPosition(QSlider.TicksBelow)
        slider_precision.setTickInterval(25)
        slider_precision.setStyleSheet("""
            QSlider::groove:horizontal {
                height: 8px;
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:0.5 #FFC107, stop:1 #F44336);
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #2196F3;
                width: 20px;
                height: 20px;
                margin: -6px 0;
                border-radius: 10px;
            }
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #4CAF50, stop:0.5 #FFC107, stop:1 #F44336);
                border-radius: 4px;
            }
        """)
        slider_precision.valueChanged.connect(self.main_window.on_precision_changed)
        precision_layout.addWidget(slider_precision)

        # 精度描述
        precision_desc_layout = QHBoxLayout()
        lbl_low = QLabel("高精度")
        lbl_low.setStyleSheet("color: #4CAF50; font-weight: bold;")
        lbl_medium = QLabel("较高")
        lbl_medium.setStyleSheet("color: #FFC107; font-weight: bold;")
        lbl_high = QLabel("极高精度")
        lbl_high.setStyleSheet("color: #F44336; font-weight: bold;")
        precision_desc_layout.addWidget(lbl_low)
        precision_desc_layout.addStretch()
        precision_desc_layout.addWidget(lbl_medium)
        precision_desc_layout.addStretch()
        precision_desc_layout.addWidget(lbl_high)
        precision_layout.addLayout(precision_desc_layout)

        # 说明文字
        lbl_precision_info = QLabel("调整滑块实时更新所有轮廓的拟合精度\n50%-100%：高精度范围\n100%-150%：极高精度范围")
        lbl_precision_info.setStyleSheet(
            "font-size: 11px; color: #666; padding: 5px; background-color: #f9f9f9; border-radius: 5px;")
        lbl_precision_info.setWordWrap(True)
        precision_layout.addWidget(lbl_precision_info)

        precision_group.setLayout(precision_layout)
        layout.addWidget(precision_group)

        # 处理参数组
        params_group = QGroupBox("处理参数")
        params_layout = QFormLayout()

        spin_kernel_size = QSpinBox()
        spin_kernel_size.setObjectName("spin_kernel_size")
        spin_kernel_size.setRange(1, 11)
        spin_kernel_size.setValue(5)
        spin_kernel_size.setSingleStep(2)
        spin_kernel_size.setStyleSheet("padding: 5px;")
        params_layout.addRow("闭运算核大小:", spin_kernel_size)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # 选中轮廓操作组
        selection_group = QGroupBox("选中轮廓操作")
        selection_group.setObjectName("selection_group")
        selection_group.setEnabled(False)  # 初始禁用
        selection_layout = QFormLayout()

        lbl_selected_info = QLabel("未选中轮廓")
        lbl_selected_info.setObjectName("lbl_selected_info")
        lbl_selected_info.setStyleSheet("font-weight: bold; color: #2196F3; padding: 5px;")
        selection_layout.addRow("状态:", lbl_selected_info)

        spin_width = QDoubleSpinBox()
        spin_width.setObjectName("spin_width")
        spin_width.setRange(0.1, 30.0)
        spin_width.setValue(5.0)
        spin_width.setSuffix(" cm")
        spin_width.setDecimals(1)
        spin_width.setSingleStep(0.1)
        spin_width.setStyleSheet("padding: 5px;")
        selection_layout.addRow("宽度:", spin_width)

        spin_height = QDoubleSpinBox()
        spin_height.setObjectName("spin_height")
        spin_height.setRange(0.1, 30.0)
        spin_height.setValue(5.0)
        spin_height.setSuffix(" cm")
        spin_height.setDecimals(1)
        spin_height.setSingleStep(0.1)
        spin_height.setStyleSheet("padding: 5px;")
        selection_layout.addRow("高度:", spin_height)

        btn_apply_size = QPushButton("应用尺寸")
        btn_apply_size.setObjectName("btn_apply_size")
        btn_apply_size.clicked.connect(self.main_window.apply_contour_size)
        selection_layout.addRow(btn_apply_size)

        selection_group.setLayout(selection_layout)
        layout.addWidget(selection_group)

        # 在selection_group中，在btn_apply_size后面添加：
        btn_calibrate_selected = QPushButton("自动标定选中轮廓")
        btn_calibrate_selected.setObjectName("btn_calibrate_selected")
        btn_calibrate_selected.clicked.connect(self.main_window.calibrate_selected_contour)
        selection_layout.addRow(btn_calibrate_selected)

        # 显示选项组
        display_group = QGroupBox("显示选项")
        display_layout = QVBoxLayout()

        cb_show_original = QCheckBox("显示原始轮廓")
        cb_show_original.setObjectName("cb_show_original")
        cb_show_original.setChecked(False)
        cb_show_original.toggled.connect(self.main_window.toggle_original_contour)
        display_layout.addWidget(cb_show_original)

        cb_show_nurbs = QCheckBox("显示NURBS曲线")
        cb_show_nurbs.setObjectName("cb_show_nurbs")
        cb_show_nurbs.setChecked(True)
        cb_show_nurbs.toggled.connect(self.main_window.toggle_nurbs_curve)
        display_layout.addWidget(cb_show_nurbs)

        cb_show_bounding_box = QCheckBox("显示选中轮廓的包围盒")
        cb_show_bounding_box.setObjectName("cb_show_bounding_box")
        cb_show_bounding_box.setChecked(True)
        cb_show_bounding_box.toggled.connect(self.main_window.toggle_bounding_box)
        display_layout.addWidget(cb_show_bounding_box)

        # 添加：显示标号复选框
        cb_show_labels = QCheckBox("显示轮廓标号")
        cb_show_labels.setObjectName("cb_show_labels")
        cb_show_labels.setChecked(True)
        cb_show_labels.toggled.connect(self.main_window.toggle_labels)
        display_layout.addWidget(cb_show_labels)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        # 在显示选项组后面添加工具组
        tools_group = QGroupBox("画布工具")
        tools_layout = QHBoxLayout()

        btn_select = QPushButton("🖱️")
        btn_select.setObjectName("btn_select")
        btn_select.setCheckable(True)
        btn_select.setChecked(True)
        btn_select.clicked.connect(lambda: self.main_window.set_tool("select"))
        tools_layout.addWidget(btn_select)

        btn_pan = QPushButton("✋")
        btn_pan.setObjectName("btn_pan")
        btn_pan.setCheckable(True)
        btn_pan.clicked.connect(lambda: self.main_window.set_tool("pan"))
        tools_layout.addWidget(btn_pan)

        btn_zoom_in = QPushButton("➕")
        btn_zoom_in.setObjectName("btn_zoom_in")
        btn_zoom_in.clicked.connect(self.main_window.zoom_in)
        tools_layout.addWidget(btn_zoom_in)

        btn_zoom_out = QPushButton("➖")
        btn_zoom_out.setObjectName("btn_zoom_out")
        btn_zoom_out.clicked.connect(self.main_window.zoom_out)
        tools_layout.addWidget(btn_zoom_out)

        btn_reset_view = QPushButton("🔄")
        btn_reset_view.setObjectName("btn_reset_view")
        btn_reset_view.clicked.connect(self.main_window.reset_view)
        tools_layout.addWidget(btn_reset_view)

        # 添加快捷键提示
        btn_reset_view.setToolTip("重置缩放和平移 (快捷键: R)")
        tools_group.setLayout(tools_layout)
        layout.addWidget(tools_group)


        # 轮廓操作组
        contour_group = QGroupBox("轮廓操作")
        contour_layout = QVBoxLayout()

        btn_clear = QPushButton("🗑️ 清空所有轮廓")
        btn_clear.setObjectName("btn_clear")
        btn_clear.clicked.connect(self.main_window.clear_contours)
        contour_layout.addWidget(btn_clear)

        btn_arrange = QPushButton("📐 自动排列轮廓")
        btn_arrange.setObjectName("btn_arrange")
        btn_arrange.clicked.connect(self.main_window.arrange_contours)
        contour_layout.addWidget(btn_arrange)

        # 只保留保存按钮，移除加载按钮
        btn_save = QPushButton("💾 保存轮廓")
        btn_save.setObjectName("btn_save")
        btn_save.clicked.connect(self.main_window.save_contours)
        contour_layout.addWidget(btn_save)

        contour_group.setLayout(contour_layout)
        layout.addWidget(contour_group)

        # # 轮廓列表组
        # layout.addWidget(QLabel("📋 轮廓列表:"))
        # contour_list_widget = QListWidget()
        # contour_list_widget.setObjectName("contour_list_widget")
        # contour_list_widget.itemClicked.connect(self.main_window.on_contour_list_item_clicked)
        # layout.addWidget(contour_list_widget)

        # 添加：标号映射列表按钮
        btn_show_label_mapping = QPushButton("📌 查看标号映射列表")
        btn_show_label_mapping.setObjectName("btn_show_label_mapping")
        btn_show_label_mapping.clicked.connect(self.main_window.show_label_mapping_dialog)
        btn_show_label_mapping.setStyleSheet("""
            QPushButton {
                padding: 10px;
                font-weight: bold;
                background-color: #FF9800;
                color: white;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        layout.addWidget(btn_show_label_mapping)

        layout.addStretch()


