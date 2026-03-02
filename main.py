#!/usr/bin/env python3
"""
皮肤病灶轮廓处理系统 - 主入口
"""
import sys
import traceback
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
from ui.main_window import MainWindow


def main():
    """主函数"""
    try:
        app = QApplication(sys.argv)
        app.setStyle('Fusion')

        # 设置应用程序样式
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(240, 240, 240))
        palette.setColor(QPalette.WindowText, Qt.black)
        app.setPalette(palette)

        # 设置应用程序信息
        app.setApplicationName("皮肤病灶轮廓处理系统")
        app.setOrganizationName("SkinLesionProcessor")
        app.setOrganizationDomain("sklprocessor.com")

        window = MainWindow()
        window.show()

        sys.exit(app.exec_())

    except Exception as e:
        print(f"程序启动失败: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()