"""
标号映射对话框
"""
from PyQt5.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel,
                             QTableWidget, QTableWidgetItem, QPushButton,
                             QMessageBox, QFileDialog, QFrame)
from PyQt5.QtCore import Qt, QDateTime
from PyQt5.QtGui import QBrush
from typing import Dict


class LabelMappingDialog(QDialog):
    """标号映射对话框"""

    def __init__(self, label_to_image_map: Dict[int, str], label_metadata: Dict[int, dict], parent=None):
        super().__init__(parent)
        self.main_window = parent  # 保存主窗口引用，用于获取轮廓颜色
        self.label_to_image_map = label_to_image_map
        self.label_metadata = label_metadata
        self.setWindowTitle("标号映射列表")
        self.setGeometry(200, 200, 900, 500)  # 适当加宽以容纳新列

        self.init_ui()

    def init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)

        # 标题
        title_label = QLabel("标号 ↔ 图像文件映射")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; padding: 10px;")
        layout.addWidget(title_label)

        # 添加分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        layout.addWidget(line)

        # 创建表格（7列）
        self.table_widget = QTableWidget()
        self.table_widget.setColumnCount(7)
        self.table_widget.setHorizontalHeaderLabels(
            ["颜色", "标号", "图像文件名", "完整路径", "云端/本地", "创建时间", "已删除"]
        )

        # 设置颜色列宽度固定
        self.table_widget.setColumnWidth(0, 30)
        # 其他列自动拉伸
        self.table_widget.horizontalHeader().setStretchLastSection(True)
        self.table_widget.setAlternatingRowColors(True)
        self.table_widget.setSelectionBehavior(QTableWidget.SelectRows)
        self.table_widget.setEditTriggers(QTableWidget.NoEditTriggers)

        # 填充表格数据
        self.populate_table()

        layout.addWidget(self.table_widget)

        # 按钮区域
        button_layout = QHBoxLayout()

        btn_copy = QPushButton("复制列表")
        btn_copy.clicked.connect(self.copy_to_clipboard)
        button_layout.addWidget(btn_copy)

        btn_save = QPushButton("保存为文本文件")
        btn_save.clicked.connect(self.save_to_file)
        button_layout.addWidget(btn_save)

        btn_close = QPushButton("关闭")
        btn_close.clicked.connect(self.close)
        button_layout.addWidget(btn_close)

        layout.addLayout(button_layout)

    def _get_contour_color_for_label(self, label: int):
        """从主窗口的画布中查找该标号对应的第一个轮廓的颜色"""
        if not hasattr(self, 'main_window'):
            return None
        for contour in self.main_window.canvas.contours:
            if contour.label == label:
                return contour.color
        return None

    def populate_table(self):
        """填充表格数据"""
        if not self.label_to_image_map:
            self.table_widget.setRowCount(1)
            for col in range(self.table_widget.columnCount()):
                self.table_widget.setItem(0, col, QTableWidgetItem("无数据"))
            return

        # 按标号排序
        sorted_labels = sorted(self.label_to_image_map.keys())
        self.table_widget.setRowCount(len(sorted_labels))

        for row, label in enumerate(sorted_labels):
            image_name = self.label_to_image_map[label]
            metadata = self.label_metadata.get(label, {})
            source = metadata.get("source", "local")
            created = metadata.get("created")
            deleted = metadata.get("deleted", False)

            source_text = "云端" if source == "cloud" else "本地"
            created_text = created.toString("yyyy-MM-dd hh:mm:ss") if created else "-"
            deleted_text = "是" if deleted else "否"

            # 颜色列
            color = self._get_contour_color_for_label(label)
            color_item = QTableWidgetItem()
            if color is not None:
                color_item.setBackground(QBrush(color))
            color_item.setFlags(color_item.flags() & ~Qt.ItemIsEditable)
            self.table_widget.setItem(row, 0, color_item)

            # 标号
            label_item = QTableWidgetItem(str(label))
            label_item.setTextAlignment(Qt.AlignCenter)
            self.table_widget.setItem(row, 1, label_item)

            # 图像文件名
            name_item = QTableWidgetItem(image_name)
            self.table_widget.setItem(row, 2, name_item)

            # 完整路径（这里只显示文件名，因为原路径可能不完整）
            path_item = QTableWidgetItem(f"来自: {image_name}")
            # 完整路径（从元数据获取实际路径）
            full_path = metadata.get("image_path", "")
            path_item = QTableWidgetItem(f'来自: {full_path if full_path else "未知路径"}')
            self.table_widget.setItem(row, 3, path_item)

            # 云端/本地
            self.table_widget.setItem(row, 4, QTableWidgetItem(source_text))
            # 创建时间
            self.table_widget.setItem(row, 5, QTableWidgetItem(created_text))
            # 已删除
            self.table_widget.setItem(row, 6, QTableWidgetItem(deleted_text))

        # 调整列宽
        self.table_widget.resizeColumnsToContents()

    def copy_to_clipboard(self):
        """复制列表到剪贴板"""
        if not self.label_to_image_map:
            QMessageBox.information(self, "提示", "没有数据可复制")
            return

        text = "标号 ↔ 图像文件映射列表\n"
        text += "=" * 90 + "\n"
        text += f"{'标号':<6} {'图像文件名':<30} {'来源':<6} {'创建时间':<20} {'已删除':<6}\n"
        text += "-" * 90 + "\n"

        sorted_labels = sorted(self.label_to_image_map.keys())
        for label in sorted_labels:
            image_name = self.label_to_image_map[label]
            metadata = self.label_metadata.get(label, {})
            source = "云端" if metadata.get("source") == "cloud" else "本地"
            created = metadata.get("created", "")
            created_str = created.toString("yyyy-MM-dd hh:mm:ss") if created else "-"
            deleted = "是" if metadata.get("deleted") else "否"
            text += f"{label:<6} {image_name:<30} {source:<6} {created_str:<20} {deleted:<6}\n"

        text += "=" * 90 + "\n"
        text += f"总计: {len(sorted_labels)} 个标号\n"

        from PyQt5.QtWidgets import QApplication
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

        QMessageBox.information(self, "成功", "标号映射列表已复制到剪贴板")

    def save_to_file(self):
        """保存到文本文件"""
        if not self.label_to_image_map:
            QMessageBox.information(self, "提示", "没有数据可保存")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存标号映射列表", "",
            "文本文件 (*.txt);;所有文件 (*.*)"
        )

        if not file_path:
            return

        if not file_path.lower().endswith('.txt'):
            file_path += '.txt'

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("标号 ↔ 图像文件映射列表\n")
                f.write("=" * 90 + "\n")
                f.write(f"{'标号':<6} {'图像文件名':<30} {'来源':<6} {'创建时间':<20} {'已删除':<6}\n")
                f.write("-" * 90 + "\n")

                sorted_labels = sorted(self.label_to_image_map.keys())
                for label in sorted_labels:
                    image_name = self.label_to_image_map[label]
                    metadata = self.label_metadata.get(label, {})
                    source = "云端" if metadata.get("source") == "cloud" else "本地"
                    created = metadata.get("created", "")
                    created_str = created.toString("yyyy-MM-dd hh:mm:ss") if created else "-"
                    deleted = "是" if metadata.get("deleted") else "否"
                    f.write(f"{label:<6} {image_name:<30} {source:<6} {created_str:<20} {deleted:<6}\n")

                f.write("=" * 90 + "\n")
                f.write(f"总计: {len(sorted_labels)} 个标号\n")
                f.write(f"生成时间: {QDateTime.currentDateTime().toString('yyyy-MM-dd HH:mm:ss')}\n")

            QMessageBox.information(self, "成功", f"标号映射列表已保存到:\n{file_path}")

        except Exception as e:
            QMessageBox.critical(self, "错误", f"保存文件失败: {str(e)}")