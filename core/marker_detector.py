import cv2
import numpy as np
import os
import glob
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import ndimage


class WhiteBallMarkerDetector:
    def __init__(self, ball_diameter_mm=10):
        # 参数配置
        self.ball_diameter_mm = ball_diameter_mm  # 小球实际直径10mm

        # 三套参数配置（优先级从高到低）
        self.param_sets = [
            # Value1 - 最高优先级
            {
                "name": "Value1",
                "h_min": 47,
                "h_max": 122,
                "s_min": 7,
                "s_max": 88,
                "v_min": 148,
                "v_max": 255,
                "open_size": 3,
                "close_size": 5,
                "dilate_size": 3,
                "min_area": 5000,
                "min_circularity": 0.6
            },
            # Value2 - 次优先级
            {
                "name": "Value2",
                "h_min": 0,
                "h_max": 179,
                "s_min": 0,
                "s_max": 50,
                "v_min": 180,
                "v_max": 255,
                "open_size": 3,
                "close_size": 5,
                "dilate_size": 3,
                "min_area": 5000,
                "min_circularity": 0.6
            },
            # 可以在这里插入 Value4
            # {
            #     "name": "Value4",
            #     "h_min": 50,
            #     "h_max": 120,
            #     "s_min": 10,
            #     "s_max": 70,
            #     "v_min": 160,
            #     "v_max": 255,
            #     "open_size": 5,
            #     "close_size": 7,
            #     "dilate_size": 4,
            #     "min_area": 5000,
            #     "min_circularity": 0.6
            # },
            # Value3 - 最低优先级
            {
                "name": "Value3",
                "h_min": 40,
                "h_max": 122,
                "s_min": 5,
                "s_max": 88,
                "v_min": 148,
                "v_max": 255,
                "open_size": 3,
                "close_size": 8,
                "dilate_size": 3,
                "min_area": 5000,
                "min_circularity": 0.6
            }
        ]

        # 统一的形状检测参数
        self.min_circularity = 0.6  # 最小圆形度
        self.min_ellipse_ratio = 0.4  # 最小椭圆长短轴比
        self.min_area = 5000  # 最小轮廓面积
        self.max_area_ratio = 0.03  # 最大面积占图像比例

        # 当前使用的参数集
        self.current_params = None
        self.param_set_used = None  # 记录使用的是哪套参数

    def set_current_params(self, params):
        """设置当前使用的参数"""
        self.current_params = params
        self.param_set_used = params["name"]

        # 使用参数集中的形状参数，但优先级低于统一参数
        # 这里我们使用统一参数，所以忽略参数集中的min_circularity
        pass

    def simple_color_segmentation(self, image):
        """简化的颜色分割 - 使用当前参数集的HSV阈值"""
        if self.current_params is None:
            raise ValueError("没有设置当前参数集")

        # 转换为HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 使用当前参数集的阈值
        lower_bound = np.array([
            self.current_params["h_min"],
            self.current_params["s_min"],
            self.current_params["v_min"]
        ])
        upper_bound = np.array([
            self.current_params["h_max"],
            self.current_params["s_max"],
            self.current_params["v_max"]
        ])

        # 创建掩码
        mask = cv2.inRange(hsv, lower_bound, upper_bound)

        return mask, hsv

    def apply_morphological_operations(self, mask):
        """应用形态学操作 - 使用当前参数集的形态学参数"""
        if self.current_params is None:
            raise ValueError("没有设置当前参数集")

        # 确保核大小为奇数
        open_size = self.current_params["open_size"]
        close_size = self.current_params["close_size"]
        dilate_size = self.current_params["dilate_size"]

        open_size = open_size if open_size % 2 == 1 else open_size + 1
        close_size = close_size if close_size % 2 == 1 else close_size + 1
        dilate_size = dilate_size if dilate_size % 2 == 1 else dilate_size + 1

        # 开运算去除小噪点
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)

        # 闭运算填充小孔洞
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)

        # 填充剩余孔洞
        filled = ndimage.binary_fill_holes(closed).astype(np.uint8) * 255

        # 膨胀以连接邻近区域
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        dilated = cv2.dilate(filled, kernel_dilate, iterations=1)

        return dilated

    def detect_and_select_best_candidate(self, mask, image):
        """检测并选择最佳候选轮廓 - 使用统一的形状检测参数"""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        img_height, img_width = image.shape[:2]
        image_area = img_height * img_width
        candidates = []

        # 过滤轮廓
        for contour in contours:
            area = cv2.contourArea(contour)

            # 面积过滤 - 使用统一参数
            if (area < self.min_area or area > image_area * self.max_area_ratio):
                continue

            # 周长计算
            perimeter = cv2.arcLength(contour, True)

            # 圆形度计算
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
            else:
                circularity = 0

            # 凸包检测
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0

            # 椭圆拟合（如果点数足够）
            ellipse = None
            ellipse_ratio = 0
            if len(contour) >= 5:
                try:
                    ellipse = cv2.fitEllipse(contour)
                    (center, axes, angle) = ellipse
                    major_axis = max(axes)
                    minor_axis = min(axes)
                    ellipse_ratio = minor_axis / major_axis if major_axis > 0 else 0
                except:
                    pass

            # 最小外接圆
            (circle_center, circle_radius) = cv2.minEnclosingCircle(contour)

            candidate = {
                'contour': contour,
                'area': area,
                'circularity': circularity,
                'convexity': convexity,
                'ellipse': ellipse,
                'ellipse_ratio': ellipse_ratio,
                'circle_center': (int(circle_center[0]), int(circle_center[1])),
                'circle_radius': circle_radius,
                'hull': hull
            }

            # 形状评分 - 使用统一参数中的min_circularity和min_ellipse_ratio
            shape_score = 0
            if candidate['circularity'] >= self.min_circularity:
                circularity_score = candidate['circularity']
            else:
                circularity_score = 0

            ellipse_score = 0
            if candidate['ellipse'] is not None:
                if candidate['ellipse_ratio'] >= self.min_ellipse_ratio:
                    ellipse_score = candidate['ellipse_ratio']

            convexity_score = candidate['convexity']

            # 综合评分
            candidate['shape_score'] = (
                    circularity_score * 0.4 +
                    ellipse_score * 0.3 +
                    convexity_score * 0.3
            )

            # 使用参数集中的min_circularity进行过滤
            if self.current_params and candidate['circularity'] < self.current_params["min_circularity"]:
                continue

            candidates.append(candidate)

        if not candidates:
            return None

        # 按评分排序并选择最佳
        candidates.sort(key=lambda x: x['shape_score'], reverse=True)

        # 如果最高分太低，可能没有合适的候选
        if not candidates or candidates[0]['shape_score'] < 0.4:
            return None

        best_candidate = candidates[0]

        # 计算质心
        M = cv2.moments(best_candidate['contour'])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = best_candidate['circle_center']

        best_candidate['centroid'] = (cX, cY)

        return best_candidate

    def calculate_pixel_scale(self, candidate):
        """计算像素比例尺"""
        if candidate is None:
            return None

        # 使用椭圆长轴或外接圆直径作为像素直径
        if candidate['ellipse'] is not None:
            (center, axes, angle) = candidate['ellipse']
            major_axis = max(axes)
            pixel_diameter = major_axis
        else:
            pixel_diameter = candidate['circle_radius'] * 2

        # 计算比例尺：mm/pixel
        scale = self.ball_diameter_mm / pixel_diameter if pixel_diameter > 0 else None

        return scale, pixel_diameter

    def create_comparison_image(self, original, mask, candidate, pixel_scale=None):
        """创建包含三个子图的对比图像"""
        # 创建大图，包含三个子图
        height, width = original.shape[:2]
        fig_width = width * 3
        fig_height = height

        # 创建空白大图
        comparison = np.ones((fig_height, fig_width, 3), dtype=np.uint8) * 50  # 灰色背景

        # 1. 原始图像
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        comparison[0:height, 0:width] = original_rgb

        # 添加子图标题 - 加大字体
        cv2.putText(comparison, "Original Image", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # 2. 阈值分割掩码图像
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        comparison[0:height, width:width * 2] = mask_rgb

        # 添加子图标题 - 加大字体
        cv2.putText(comparison, "Segmentation Mask", (width + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # 3. 叠加圆形轮廓检测结果
        result = original.copy()

        if candidate is not None:
            # 绘制轮廓
            cv2.drawContours(result, [candidate['contour']], -1, (0, 255, 0), 2)

            # 绘制椭圆（如果检测到）
            if candidate['ellipse'] is not None:
                cv2.ellipse(result, candidate['ellipse'], (0, 165, 255), 2)

            # 绘制最小外接圆
            cv2.circle(result, candidate['circle_center'],
                       int(candidate['circle_radius']), (255, 0, 255), 2)

            # 绘制质心
            cv2.circle(result, candidate['centroid'], 5, (0, 0, 255), -1)

            # 在图像上显示关键信息
            info_lines = [
                f"Params: {self.param_set_used}",
                f"Score: {candidate['shape_score']:.3f}",
                f"Circularity: {candidate['circularity']:.2f}",
                f"Area: {candidate['area']:.0f} px"
            ]

            if pixel_scale:
                info_lines.append(f"Scale: {pixel_scale:.4f} mm/px")

            # 选择合适的位置显示信息 - 左上角
            x_pos = 20
            y_pos = 50
            line_height = 35  # 行间距

            # 创建半透明背景框
            bg_height = len(info_lines) * line_height + 20
            bg_width = 300  # 稍微加宽以容纳参数集名称
            overlay = result.copy()

            # 绘制半透明矩形背景
            cv2.rectangle(overlay, (x_pos - 10, y_pos - 30),
                          (x_pos + bg_width, y_pos + bg_height),
                          (0, 0, 0), -1)

            # 添加透明度
            alpha = 0.7
            result = cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0)

            # 绘制白色边框
            cv2.rectangle(result, (x_pos - 10, y_pos - 30),
                          (x_pos + bg_width, y_pos + bg_height),
                          (255, 255, 255), 1)

            # 添加信息文本 - 加大字体和粗细
            for i, line in enumerate(info_lines):
                # 参数集名称用黄色突出显示
                color = (255, 255, 0) if i == 0 else (255, 255, 255)
                cv2.putText(result, line, (x_pos, y_pos + i * line_height),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        comparison[0:height, width * 2:width * 3] = result_rgb

        # 添加子图标题 - 加大字体
        cv2.putText(comparison, "Detection Result", (width * 2 + 10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # 添加分割线
        cv2.line(comparison, (width, 0), (width, height), (200, 200, 200), 2)
        cv2.line(comparison, (width * 2, 0), (width * 2, height), (200, 200, 200), 2)

        return comparison

    def try_detection_with_params(self, image, params):
        """使用指定的参数集尝试检测"""
        # 设置当前参数
        self.set_current_params(params)

        # 1. 简化的颜色分割
        color_mask, hsv = self.simple_color_segmentation(image)

        # 2. 形态学处理
        processed_mask = self.apply_morphological_operations(color_mask)

        # 3. 检测并选择最佳候选
        best_candidate = self.detect_and_select_best_candidate(processed_mask, image)

        return best_candidate, processed_mask

    def process_single_image(self, image_path, output_dir=None):
        """处理单张图像 - 依次尝试多套参数"""
        # 读取图像
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"错误: 无法读取图像 {image_path}")
            return None

        original = image.copy()
        height, width = image.shape[:2]

        # 初始化变量
        best_candidate = None
        processed_mask = None
        pixel_scale = None
        param_set_used = None

        # 依次尝试三套参数
        for params in self.param_sets:
            try:
                candidate, mask = self.try_detection_with_params(image, params)

                if candidate is not None:
                    best_candidate = candidate
                    processed_mask = mask
                    param_set_used = params["name"]

                    # 计算像素比例尺
                    pixel_scale, pixel_diameter = self.calculate_pixel_scale(best_candidate)

                    print(f"  使用参数集 {params['name']}: 检测成功")
                    print(f"    评分: {best_candidate['shape_score']:.3f}, "
                          f"圆形度: {best_candidate['circularity']:.2f}, "
                          f"面积: {best_candidate['area']:.0f} px")

                    if pixel_scale:
                        print(f"    比例尺: {pixel_scale:.6f} mm/px")

                    break  # 找到有效的参数集，跳出循环
                else:
                    print(f"  使用参数集 {params['name']}: 未检测到")

            except Exception as e:
                print(f"  使用参数集 {params['name']} 时出错: {str(e)}")
                continue

        # 创建对比图像
        comparison_image = self.create_comparison_image(original, processed_mask, best_candidate, pixel_scale)

        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = Path(image_path).stem

            # 保存对比图像
            cv2.imwrite(os.path.join(output_dir, f"{filename}_comparison.png"),
                        cv2.cvtColor(comparison_image, cv2.COLOR_RGB2BGR))

        # 返回检测结果
        detected = best_candidate is not None
        return {
            'detected': detected,
            'param_set': param_set_used,
            'score': best_candidate['shape_score'] if detected else 0,
            'circularity': best_candidate['circularity'] if detected else 0,
            'area': best_candidate['area'] if detected else 0,
            'pixel_scale': pixel_scale if detected else None,
            'comparison_image': comparison_image
        }

    def process_folder(self, folder_path, output_folder=None):
        """处理文件夹中的所有图像"""
        if output_folder is None:
            output_folder = os.path.join(folder_path, "detection_results")

        os.makedirs(output_folder, exist_ok=True)

        # 查找所有图像文件
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        image_files = []

        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))

        if not image_files:
            print(f"在文件夹中未找到图像文件: {folder_path}")
            return 0, 0

        print(f"找到 {len(image_files)} 个图像文件")
        print("=" * 80)

        # 处理每个图像
        detection_count = 0

        for i, img_file in enumerate(image_files):
            print(f"\n处理 [{i + 1}/{len(image_files)}]: {os.path.basename(img_file)}")

            try:
                result = self.process_single_image(img_file, output_folder)

                if result and result['detected']:
                    detection_count += 1
                    print(f"  ✓ 检测成功 (使用参数集: {result['param_set']})")
                else:
                    print(f"  ✗ 所有参数集均未检测到")

            except Exception as e:
                print(f"  处理失败: {str(e)}")

        # 输出统计结果
        print("\n" + "=" * 80)
        print(f"处理完成!")
        print(f"成功检测: {detection_count}/{len(image_files)}")
        print(f"检测成功率: {detection_count / len(image_files) * 100:.1f}%")
        print(f"结果保存到: {output_folder}")

        return detection_count, len(image_files)


def main():
    """主函数"""
    # 创建检测器，设置小球直径为10mm
    detector = WhiteBallMarkerDetector(ball_diameter_mm=10)

    # 处理文件夹
    detection_count, total_images = detector.process_folder(
        folder_path="D:/Appdevelop/.venv/skin_lesion_processor/resources/overlays",
        output_folder="D:/Appdevelop/.venv/skin_lesion_processor/resources/detection_results"
    )

    print(f"\n最终统计: 成功检测 {detection_count} 个，总共 {total_images} 个图像")


if __name__ == "__main__":
    main()