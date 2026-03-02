from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
import random


class SimulationWidget(QWidget):
    """简单的激光切割模拟可视化"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()
        self.reset()

    def init_ui(self):
        self.setMinimumSize(400, 300)
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                border: 1px solid #555;
                border-radius: 5px;
            }
        """)

    def reset(self):
        self.progress = 0
        self.laser_pos = (50, 150)
        self.trail_points = []
        self.is_active = False
        self.update()

    def start_simulation(self, duration=5.0):
        """开始模拟"""
        self.reset()
        self.is_active = True

        # 创建随机切割路径（模拟）
        self.trail_points = []
        for i in range(20):
            x = random.randint(50, 350)
            y = random.randint(50, 250)
            self.trail_points.append((x, y))

        # 动画定时器
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_animation)
        interval = int((duration * 1000) / len(self.trail_points))
        self.timer.start(interval)

    def update_animation(self):
        """更新动画"""
        self.progress += 1

        if self.progress < len(self.trail_points):
            self.laser_pos = self.trail_points[self.progress]
            self.update()
        else:
            self.timer.stop()
            self.is_active = False
            # 完成信号
            self.simulation_completed.emit()

    def paintEvent(self, event):
        """绘制模拟界面"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制背景
        painter.fillRect(self.rect(), QColor(43, 43, 43))

        # 绘制网格
        painter.setPen(QPen(QColor(60, 60, 60), 1))
        for x in range(0, self.width(), 20):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 20):
            painter.drawLine(0, y, self.width(), y)

        # 绘制切割轨迹
        if self.trail_points:
            painter.setPen(QPen(QColor(0, 255, 0, 100), 2))
            for i in range(1, min(self.progress, len(self.trail_points))):
                x1, y1 = self.trail_points[i - 1]
                x2, y2 = self.trail_points[i]
                painter.drawLine(x1, y1, x2, y2)

        # 绘制激光头
        if self.is_active:
            # 激光束
            painter.setPen(QPen(QColor(255, 50, 50, 200), 3))
            painter.drawLine(self.laser_pos[0], self.laser_pos[1],
                             self.laser_pos[0], self.laser_pos[1] + 30)

            # 激光头
            painter.setBrush(QBrush(QColor(255, 100, 100)))
            painter.setPen(QPen(QColor(255, 200, 200), 2))
            painter.drawEllipse(self.laser_pos[0] - 10, self.laser_pos[1] - 10, 20, 20)

            # 光晕效果
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor(255, 100, 100, 100), 1))
            painter.drawEllipse(self.laser_pos[0] - 15, self.laser_pos[1] - 15, 30, 30)

        # 绘制进度条
        if self.is_active:
            progress_width = int(self.width() * self.progress / len(self.trail_points))
            painter.fillRect(10, self.height() - 30, progress_width, 20, QColor(0, 150, 255, 150))
            painter.setPen(QColor(255, 255, 255))
            painter.drawRect(10, self.height() - 30, self.width() - 20, 20)

            # 进度文本
            percent = int(100 * self.progress / len(self.trail_points))
            painter.drawText(self.width() // 2 - 20, self.height() - 15, f"{percent}%")

        # 标题
        painter.setPen(QColor(200, 200, 200))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(10, 20, "激光切割模拟")

    simulation_completed = pyqtSignal()