from pathlib import Path

import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import json


class HSVThresholdAdjuster:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("HSV阈值调整工具 - 白色小球检测")
        self.root.geometry("1400x900")

        # 图像变量
        self.original_image = None
        self.hsv_image = None
        self.current_mask = None
        self.display_image = None

        # HSV阈值变量
        self.h_min = tk.IntVar(value=0)
        self.h_max = tk.IntVar(value=179)
        self.s_min = tk.IntVar(value=0)
        self.s_max = tk.IntVar(value=50)
        self.v_min = tk.IntVar(value=180)
        self.v_max = tk.IntVar(value=255)

        # 形态学参数
        self.open_size = tk.IntVar(value=3)
        self.close_size = tk.IntVar(value=5)
        self.dilate_size = tk.IntVar(value=3)

        # 其他参数
        self.min_area = tk.IntVar(value=100)
        self.min_circularity = tk.DoubleVar(value=0.6)

        # 当前图像路径
        self.current_image_path = None

        # 设置UI
        self.setup_ui()

        # 加载预设（如果存在）
        self.load_preset()

    def setup_ui(self):
        """设置用户界面"""
        # 主框架
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 左侧控制面板
        control_frame = tk.LabelFrame(main_frame, text="控制面板", padx=10, pady=10)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # 右侧图像显示区域
        image_frame = tk.LabelFrame(main_frame, text="图像显示")
        image_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # 图像显示画布
        self.image_canvas = tk.Canvas(image_frame, bg="gray", width=800, height=600)
        self.image_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 图像标签
        self.image_label = tk.Label(image_frame, text="请加载图像", font=("Arial", 14))
        self.image_label.pack(side=tk.BOTTOM, pady=5)

        # 按钮区域
        button_frame = tk.Frame(control_frame)
        button_frame.pack(fill=tk.X, pady=(0, 20))

        tk.Button(button_frame, text="加载图像", command=self.load_image,
                  width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="保存阈值", command=self.save_preset,
                  width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="加载阈值", command=self.load_preset_dialog,
                  width=15).pack(side=tk.LEFT, padx=5)
        tk.Button(button_frame, text="重置阈值", command=self.reset_thresholds,
                  width=15).pack(side=tk.LEFT, padx=5)

        # HSV阈值调节区域
        self.create_hsv_controls(control_frame)

        # 形态学参数调节区域
        self.create_morphology_controls(control_frame)

        # 其他参数调节区域
        self.create_other_controls(control_frame)

        # 显示模式选择
        self.create_display_mode_controls(control_frame)

        # 底部信息区域
        info_frame = tk.Frame(control_frame)
        info_frame.pack(fill=tk.X, pady=(20, 0))

        self.info_label = tk.Label(info_frame, text="就绪", justify=tk.LEFT)
        self.info_label.pack(fill=tk.X)

    def create_hsv_controls(self, parent):
        """创建HSV控制滑块"""
        hsv_frame = tk.LabelFrame(parent, text="HSV阈值调整", padx=10, pady=10)
        hsv_frame.pack(fill=tk.X, pady=(0, 10))

        # H最小值
        tk.Label(hsv_frame, text="H 最小值 (0-179):").grid(row=0, column=0, sticky=tk.W, pady=5)
        tk.Scale(hsv_frame, from_=0, to=179, orient=tk.HORIZONTAL,
                 variable=self.h_min, length=300, command=self.update_display).grid(row=0, column=1, pady=5)
        tk.Label(hsv_frame, textvariable=self.h_min).grid(row=0, column=2, padx=10)

        # H最大值
        tk.Label(hsv_frame, text="H 最大值 (0-179):").grid(row=1, column=0, sticky=tk.W, pady=5)
        tk.Scale(hsv_frame, from_=0, to=179, orient=tk.HORIZONTAL,
                 variable=self.h_max, length=300, command=self.update_display).grid(row=1, column=1, pady=5)
        tk.Label(hsv_frame, textvariable=self.h_max).grid(row=1, column=2, padx=10)

        # S最小值
        tk.Label(hsv_frame, text="S 最小值 (0-255):").grid(row=2, column=0, sticky=tk.W, pady=5)
        tk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                 variable=self.s_min, length=300, command=self.update_display).grid(row=2, column=1, pady=5)
        tk.Label(hsv_frame, textvariable=self.s_min).grid(row=2, column=2, padx=10)

        # S最大值
        tk.Label(hsv_frame, text="S 最大值 (0-255):").grid(row=3, column=0, sticky=tk.W, pady=5)
        tk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                 variable=self.s_max, length=300, command=self.update_display).grid(row=3, column=1, pady=5)
        tk.Label(hsv_frame, textvariable=self.s_max).grid(row=3, column=2, padx=10)

        # V最小值
        tk.Label(hsv_frame, text="V 最小值 (0-255):").grid(row=4, column=0, sticky=tk.W, pady=5)
        tk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                 variable=self.v_min, length=300, command=self.update_display).grid(row=4, column=1, pady=5)
        tk.Label(hsv_frame, textvariable=self.v_min).grid(row=4, column=2, padx=10)

        # V最大值
        tk.Label(hsv_frame, text="V 最大值 (0-255):").grid(row=5, column=0, sticky=tk.W, pady=5)
        tk.Scale(hsv_frame, from_=0, to=255, orient=tk.HORIZONTAL,
                 variable=self.v_max, length=300, command=self.update_display).grid(row=5, column=1, pady=5)
        tk.Label(hsv_frame, textvariable=self.v_max).grid(row=5, column=2, padx=10)

    def create_morphology_controls(self, parent):
        """创建形态学参数控制滑块"""
        morph_frame = tk.LabelFrame(parent, text="形态学参数", padx=10, pady=10)
        morph_frame.pack(fill=tk.X, pady=(0, 10))

        # 开运算大小
        tk.Label(morph_frame, text="开运算核大小 (奇数):").grid(row=0, column=0, sticky=tk.W, pady=5)
        tk.Scale(morph_frame, from_=1, to=15, orient=tk.HORIZONTAL,
                 variable=self.open_size, length=300, command=self.update_display).grid(row=0, column=1, pady=5)
        tk.Label(morph_frame, textvariable=self.open_size).grid(row=0, column=2, padx=10)

        # 闭运算大小
        tk.Label(morph_frame, text="闭运算核大小 (奇数):").grid(row=1, column=0, sticky=tk.W, pady=5)
        tk.Scale(morph_frame, from_=1, to=15, orient=tk.HORIZONTAL,
                 variable=self.close_size, length=300, command=self.update_display).grid(row=1, column=1, pady=5)
        tk.Label(morph_frame, textvariable=self.close_size).grid(row=1, column=2, padx=10)

        # 膨胀大小
        tk.Label(morph_frame, text="膨胀核大小 (奇数):").grid(row=2, column=0, sticky=tk.W, pady=5)
        tk.Scale(morph_frame, from_=1, to=15, orient=tk.HORIZONTAL,
                 variable=self.dilate_size, length=300, command=self.update_display).grid(row=2, column=1, pady=5)
        tk.Label(morph_frame, textvariable=self.dilate_size).grid(row=2, column=2, padx=10)

    def create_other_controls(self, parent):
        """创建其他参数控制"""
        other_frame = tk.LabelFrame(parent, text="其他参数", padx=10, pady=10)
        other_frame.pack(fill=tk.X, pady=(0, 10))

        # 最小面积
        tk.Label(other_frame, text="最小面积:").grid(row=0, column=0, sticky=tk.W, pady=5)
        tk.Scale(other_frame, from_=10, to=5000, orient=tk.HORIZONTAL,
                 variable=self.min_area, length=300, command=self.update_display).grid(row=0, column=1, pady=5)
        tk.Label(other_frame, textvariable=self.min_area).grid(row=0, column=2, padx=10)

        # 最小圆形度
        tk.Label(other_frame, text="最小圆形度 (0-1):").grid(row=1, column=0, sticky=tk.W, pady=5)
        tk.Scale(other_frame, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
                 variable=self.min_circularity, length=300, command=self.update_display).grid(row=1, column=1, pady=5)
        tk.Label(other_frame, textvariable=self.min_circularity).grid(row=1, column=2, padx=10)

    def create_display_mode_controls(self, parent):
        """创建显示模式控制"""
        mode_frame = tk.LabelFrame(parent, text="显示模式", padx=10, pady=10)
        mode_frame.pack(fill=tk.X)

        self.display_mode = tk.StringVar(value="mask")

        tk.Radiobutton(mode_frame, text="原始图像", variable=self.display_mode,
                       value="original", command=self.update_display).pack(anchor=tk.W)
        tk.Radiobutton(mode_frame, text="HSV图像", variable=self.display_mode,
                       value="hsv", command=self.update_display).pack(anchor=tk.W)
        tk.Radiobutton(mode_frame, text="掩码图像", variable=self.display_mode,
                       value="mask", command=self.update_display).pack(anchor=tk.W)
        tk.Radiobutton(mode_frame, text="叠加效果", variable=self.display_mode,
                       value="overlay", command=self.update_display).pack(anchor=tk.W)
        tk.Radiobutton(mode_frame, text="轮廓检测", variable=self.display_mode,
                       value="contour", command=self.update_display).pack(anchor=tk.W)

    def load_image(self):
        """加载图像文件"""
        file_path = filedialog.askopenfilename(
            title="选择图像文件",
            filetypes=[("图像文件", "*.jpg *.jpeg *.png *.bmp *.tiff"), ("所有文件", "*.*")]
        )

        if file_path:
            try:
                # 读取图像
                self.current_image_path = file_path
                self.original_image = cv2.imread(file_path)

                if self.original_image is None:
                    raise ValueError("无法读取图像文件")

                # 转换为HSV
                self.hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)

                # 更新显示
                self.update_display()

                # 更新标签
                self.image_label.config(text=os.path.basename(file_path))
                self.info_label.config(text=f"图像已加载: {os.path.basename(file_path)}")

            except Exception as e:
                tk.messagebox.showerror("错误", f"加载图像失败: {str(e)}")

    def update_display(self, *args):
        """更新显示"""
        if self.original_image is None:
            return

        try:
            # 获取当前阈值
            h_min = self.h_min.get()
            h_max = self.h_max.get()
            s_min = self.s_min.get()
            s_max = self.s_max.get()
            v_min = self.v_min.get()
            v_max = self.v_max.get()

            # 获取形态学参数
            open_size = self.open_size.get()
            close_size = self.close_size.get()
            dilate_size = self.dilate_size.get()

            # 确保核大小为奇数
            open_size = max(1, open_size // 2 * 2 + 1)
            close_size = max(1, close_size // 2 * 2 + 1)
            dilate_size = max(1, dilate_size // 2 * 2 + 1)

            # 创建初始掩码
            lower_bound = np.array([h_min, s_min, v_min])
            upper_bound = np.array([h_max, s_max, v_max])
            initial_mask = cv2.inRange(self.hsv_image, lower_bound, upper_bound)

            # 形态学处理
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
            mask = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel_open)

            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
            mask = cv2.dilate(mask, kernel_dilate, iterations=1)

            # 应用最小面积和最小圆形度过滤到掩码本身
            min_area = self.min_area.get()
            min_circularity = self.min_circularity.get()

            if min_area > 0 or min_circularity > 0.1:
                # 查找轮廓
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 创建空白掩码
                filtered_mask = np.zeros_like(mask)

                for contour in contours:
                    area = cv2.contourArea(contour)

                    # 面积过滤
                    if area < min_area:
                        continue

                    # 圆形度过滤（如果需要）
                    if min_circularity > 0.1:  # 只有当圆形度阈值有意义时才计算
                        perimeter = cv2.arcLength(contour, True)
                        if perimeter > 0:
                            circularity = 4 * np.pi * area / (perimeter * perimeter)
                        else:
                            circularity = 0

                        if circularity < min_circularity:
                            continue

                    # 符合条件，绘制到过滤后的掩码中
                    cv2.drawContours(filtered_mask, [contour], -1, 255, -1)  # -1表示填充

                # 使用过滤后的掩码
                mask = filtered_mask

            self.current_mask = mask

            # 根据显示模式选择显示的图像
            display_mode = self.display_mode.get()

            if display_mode == "original":
                display_img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
            elif display_mode == "hsv":
                display_img = cv2.cvtColor(self.hsv_image, cv2.COLOR_HSV2RGB)
            elif display_mode == "mask":
                display_img = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            elif display_mode == "overlay":
                # 创建叠加图像
                overlay = self.original_image.copy()
                overlay[mask > 0] = [0, 255, 0]  # 绿色覆盖
                display_img = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            elif display_mode == "contour":
                # 查找轮廓并绘制
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour_img = self.original_image.copy()

                # 统计信息
                valid_contours = []
                min_area_contour = self.min_area.get()
                min_circularity_contour = self.min_circularity.get()

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area < min_area_contour:
                        continue

                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                    else:
                        circularity = 0

                    if circularity >= min_circularity_contour:
                        valid_contours.append(contour)

                # 绘制所有轮廓
                cv2.drawContours(contour_img, contours, -1, (0, 0, 255), 2)  # 红色：所有轮廓

                # 绘制符合条件的轮廓
                cv2.drawContours(contour_img, valid_contours, -1, (0, 255, 0), 3)  # 绿色：有效轮廓

                # 添加文本信息
                cv2.putText(contour_img, f"Total: {len(contours)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(contour_img, f"Valid: {len(valid_contours)}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # 为每个有效轮廓添加信息
                for i, contour in enumerate(valid_contours[:3]):  # 只显示前3个
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        area = cv2.contourArea(contour)
                        cv2.putText(contour_img, f"A={area:.0f}", (cX - 30, cY - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                display_img = cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB)

            # 调整图像大小以适应画布
            canvas_width = self.image_canvas.winfo_width()
            canvas_height = self.image_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                # 计算缩放比例
                img_height, img_width = display_img.shape[:2]
                scale = min(canvas_width / img_width, canvas_height / img_height) * 0.9

                if scale < 1:
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    display_img = cv2.resize(display_img, (new_width, new_height))

            # 转换为PIL图像并显示
            pil_image = Image.fromarray(display_img)
            tk_image = ImageTk.PhotoImage(pil_image)

            # 更新画布
            self.image_canvas.delete("all")
            self.image_canvas.create_image(canvas_width // 2, canvas_height // 2,
                                           image=tk_image, anchor=tk.CENTER)
            self.image_canvas.image = tk_image  # 保持引用

            # 更新信息
            mask_area = np.sum(mask > 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            mask_percentage = mask_area / total_pixels * 100

            # 统计轮廓信息
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    continue

                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0

                if circularity >= min_circularity:
                    valid_count += 1

            info_text = (f"掩码面积: {mask_area} 像素 ({mask_percentage:.2f}%)\n"
                         f"HSV范围: H[{h_min}-{h_max}] S[{s_min}-{s_max}] V[{v_min}-{v_max}]\n"
                         f"形态学: 开{open_size} 闭{close_size} 膨胀{dilate_size}\n"
                         f"过滤条件: 面积≥{min_area}, 圆形度≥{min_circularity:.2f}\n"
                         f"轮廓数量: {len(contours)}个, 有效: {valid_count}个")

            self.info_label.config(text=info_text)

        except Exception as e:
            self.info_label.config(text=f"错误: {str(e)}")

    def save_preset(self):
        """保存当前阈值预设"""
        if self.original_image is None:
            tk.messagebox.showwarning("警告", "请先加载图像")
            return

        preset = {
            "h_min": self.h_min.get(),
            "h_max": self.h_max.get(),
            "s_min": self.s_min.get(),
            "s_max": self.s_max.get(),
            "v_min": self.v_min.get(),
            "v_max": self.v_max.get(),
            "open_size": self.open_size.get(),
            "close_size": self.close_size.get(),
            "dilate_size": self.dilate_size.get(),
            "min_area": self.min_area.get(),
            "min_circularity": float(self.min_circularity.get())
        }

        file_path = filedialog.asksaveasfilename(
            title="保存阈值预设",
            defaultextension=".json",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w') as f:
                    json.dump(preset, f, indent=4)
                self.info_label.config(text=f"预设已保存: {file_path}")
            except Exception as e:
                tk.messagebox.showerror("错误", f"保存失败: {str(e)}")

    def load_preset(self):
        """加载预设（从默认位置）"""
        default_path = "../resources/HSV_ValueSet/Value1.json"

        if os.path.exists(default_path):
            print("Loading preset ValueSet")
            try:
                with open(default_path, 'r') as f:
                    preset = json.load(f)
                self.apply_preset(preset)
                self.info_label.config(text=f"已加载预设: {default_path}")
            except:
                pass

    def load_preset_dialog(self):
        """通过对话框加载预设"""
        file_path = filedialog.askopenfilename(
            title="加载阈值预设",
            filetypes=[("JSON文件", "*.json"), ("所有文件", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'r') as f:
                    preset = json.load(f)
                self.apply_preset(preset)
                self.info_label.config(text=f"预设已加载: {file_path}")
            except Exception as e:
                tk.messagebox.showerror("错误", f"加载失败: {str(e)}")

    def apply_preset(self, preset):
        """应用预设值"""
        self.h_min.set(preset.get("h_min", 0))
        self.h_max.set(preset.get("h_max", 179))
        self.s_min.set(preset.get("s_min", 0))
        self.s_max.set(preset.get("s_max", 50))
        self.v_min.set(preset.get("v_min", 180))
        self.v_max.set(preset.get("v_max", 255))
        self.open_size.set(preset.get("open_size", 3))
        self.close_size.set(preset.get("close_size", 5))
        self.dilate_size.set(preset.get("dilate_size", 3))
        self.min_area.set(preset.get("min_area", 100))
        self.min_circularity.set(preset.get("min_circularity", 0.6))

        # 更新显示
        if self.original_image is not None:
            self.update_display()

    def reset_thresholds(self):
        """重置阈值到默认值"""
        self.h_min.set(0)
        self.h_max.set(179)
        self.s_min.set(0)
        self.s_max.set(50)
        self.v_min.set(180)
        self.v_max.set(255)
        self.open_size.set(3)
        self.close_size.set(5)
        self.dilate_size.set(3)
        self.min_area.set(100)
        self.min_circularity.set(0.6)

        # 更新显示
        if self.original_image is not None:
            self.update_display()

    def get_current_thresholds(self):
        """获取当前阈值设置"""
        return {
            "h_min": self.h_min.get(),
            "h_max": self.h_max.get(),
            "s_min": self.s_min.get(),
            "s_max": self.s_max.get(),
            "v_min": self.v_min.get(),
            "v_max": self.v_max.get(),
            "open_size": self.open_size.get(),
            "close_size": self.close_size.get(),
            "dilate_size": self.dilate_size.get(),
            "min_area": self.min_area.get(),
            "min_circularity": float(self.min_circularity.get())
        }

    def run(self):
        """运行应用程序"""
        self.root.mainloop()


# 检测器类（使用从GUI获得的阈值）
class OptimizedWhiteBallDetector:
    def __init__(self, threshold_preset=None):
        # 默认参数
        self.thresholds = {
            "h_min": 0,
            "h_max": 179,
            "s_min": 0,
            "s_max": 50,
            "v_min": 180,
            "v_max": 255,
            "open_size": 3,
            "close_size": 5,
            "dilate_size": 3,
            "min_area": 100,
            "min_circularity": 0.6
        }

        # 如果提供了预设，则使用
        if threshold_preset:
            self.thresholds.update(threshold_preset)

        # 小球直径
        self.ball_diameter_mm = 10

        # 预处理核
        self.open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.thresholds["open_size"], self.thresholds["open_size"])
        )
        self.close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.thresholds["close_size"], self.thresholds["close_size"])
        )
        self.dilate_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.thresholds["dilate_size"], self.thresholds["dilate_size"])
        )

    def create_mask(self, image):
        """创建掩码"""
        # 转换为HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # HSV阈值分割
        lower_bound = np.array([
            self.thresholds["h_min"],
            self.thresholds["s_min"],
            self.thresholds["v_min"]
        ])
        upper_bound = np.array([
            self.thresholds["h_max"],
            self.thresholds["s_max"],
            self.thresholds["v_max"]
        ])

        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        # 形态学处理
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)
        mask = cv2.dilate(mask, self.dilate_kernel, iterations=1)

        return mask

    def find_best_ball(self, mask, image):
        """寻找最佳的小球轮廓"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_contour = None
        best_score = 0

        for contour in contours:
            area = cv2.contourArea(contour)

            # 面积过滤
            if area < self.thresholds["min_area"]:
                continue

            # 计算周长和圆形度
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0

            # 圆形度过滤
            if circularity < self.thresholds["min_circularity"]:
                continue

            # 椭圆拟合
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (center, axes, angle) = ellipse
                    major_axis = max(axes)
                    minor_axis = min(axes)
                    ellipse_ratio = minor_axis / major_axis if major_axis > 0 else 0
                except:
                    ellipse_ratio = 0
            else:
                ellipse_ratio = 0

            # 计算综合评分
            score = circularity * 0.6 + ellipse_ratio * 0.4

            if score > best_score:
                best_score = score
                best_contour = contour

        return best_contour, best_score

    def calculate_pixel_scale(self, contour):
        """计算像素比例尺"""
        if contour is None or len(contour) < 5:
            return None

        try:
            # 使用椭圆拟合
            ellipse = cv2.fitEllipse(contour)
            (center, axes, angle) = ellipse
            major_axis = max(axes)

            # 使用长轴作为直径
            pixel_diameter = major_axis
            scale = self.ball_diameter_mm / pixel_diameter

            return scale, pixel_diameter
        except:
            # 如果椭圆拟合失败，使用最小外接圆
            (circle_center, circle_radius) = cv2.minEnclosingCircle(contour)
            pixel_diameter = circle_radius * 2
            scale = self.ball_diameter_mm / pixel_diameter

            return scale, pixel_diameter

    def process_image(self, image_path, output_dir=None):
        """处理单张图像"""
        image = cv2.imread(image_path)
        if image is None:
            print(f"无法读取图像: {image_path}")
            return None

        original = image.copy()

        # 创建掩码
        mask = self.create_mask(image)

        # 寻找最佳小球
        best_contour, score = self.find_best_ball(mask, image)

        # 计算比例尺
        scale_info = None
        if best_contour is not None:
            scale_info = self.calculate_pixel_scale(best_contour)

        # 绘制结果
        result = image.copy()
        if best_contour is not None:
            # 绘制轮廓
            cv2.drawContours(result, [best_contour], -1, (0, 255, 0), 3)

            # 绘制椭圆
            if len(best_contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(best_contour)
                    cv2.ellipse(result, ellipse, (0, 165, 255), 2)
                except:
                    pass

            # 绘制最小外接圆
            (circle_center, circle_radius) = cv2.minEnclosingCircle(best_contour)
            cv2.circle(result, (int(circle_center[0]), int(circle_center[1])),
                       int(circle_radius), (255, 0, 255), 2)

            # 添加文本
            cv2.putText(result, f"Score: {score:.3f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            if scale_info:
                scale, pixel_diameter = scale_info
                cv2.putText(result, f"Scale: {scale:.6f} mm/px", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(result, f"Diameter: {pixel_diameter:.1f} px", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = Path(image_path).stem

            cv2.imwrite(os.path.join(output_dir, f"{filename}_mask.png"), mask)
            cv2.imwrite(os.path.join(output_dir, f"{filename}_result.png"), result)

            # 保存结果文本
            with open(os.path.join(output_dir, f"{filename}_results.txt"), 'w') as f:
                f.write(f"图像: {filename}\n")
                f.write(f"检测状态: {'成功' if best_contour is not None else '失败'}\n")

                if best_contour is not None:
                    f.write(f"形状评分: {score:.3f}\n")

                    if scale_info:
                        scale, pixel_diameter = scale_info
                        f.write(f"像素直径: {pixel_diameter:.2f} px\n")
                        f.write(f"实际直径: {self.ball_diameter_mm} mm\n")
                        f.write(f"像素比例尺: {scale:.6f} mm/px\n")
                        f.write(f"或: {1 / scale:.2f} px/mm\n")

        return {
            'original': original,
            'mask': mask,
            'result': result,
            'contour': best_contour,
            'score': score,
            'scale': scale_info[0] if scale_info else None
        }


# 主程序
if __name__ == "__main__":
    # 运行阈值调整工具
    print("正在启动HSV阈值调整工具...")
    print("使用说明:")
    print("1. 点击'加载图像'选择要处理的图片")
    print("2. 调整HSV滑块直到白色小球被正确分割")
    print("3. 调整形态学参数优化分割效果")
    print("4. 使用'轮廓检测'模式查看检测结果")
    print("5. 点击'保存阈值'保存最佳参数")
    print("6. 关闭窗口后可以使用OptimizedWhiteBallDetector进行批量处理")
    print()

    app = HSVThresholdAdjuster()
    app.run()

    # 获取调整后的阈值
    print("\n阈值调整完成!")
    print("当前阈值设置:")
    thresholds = app.get_current_thresholds()
    for key, value in thresholds.items():
        print(f"  {key}: {value}")

    # 询问是否使用这些阈值进行批量处理
    print("\n是否使用这些阈值进行批量处理？")
    print("1. 将阈值保存到文件供后续使用")
    print("2. 使用以下代码进行批量处理:")
    print()
    print("```python")
    print("detector = OptimizedWhiteBallDetector(threshold_preset={")
    for key, value in thresholds.items():
        print(f"    '{key}': {value},")
    print("})")
    print("detector.process_folder('your_folder_path', 'output_folder')")
    print("```")