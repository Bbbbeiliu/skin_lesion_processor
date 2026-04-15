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

        # 多套参数配置
        self.param_sets = [
            {
                "name": "Value1",
                "h_min": 47, "h_max": 122, "s_min": 7, "s_max": 88, "v_min": 148, "v_max": 255,
                "open_size": 3, "close_size": 5, "dilate_size": 3, "min_area": 5000, "min_circularity": 0.6
            },
            {
                "name": "Value2",
                "h_min": 0, "h_max": 179, "s_min": 0, "s_max": 50, "v_min": 180, "v_max": 255,
                "open_size": 3, "close_size": 5, "dilate_size": 3, "min_area": 5000, "min_circularity": 0.6
            },
            {
                "name": "Value3",
                "h_min": 40, "h_max": 122, "s_min": 5, "s_max": 88, "v_min": 148, "v_max": 255,
                "open_size": 3, "close_size": 8, "dilate_size": 3, "min_area": 5000, "min_circularity": 0.6
            },
            {
                "name": "Value4",
                "h_min": 38, "h_max": 110, "s_min": 0, "s_max": 115, "v_min": 169, "v_max": 255,
                "open_size": 12, "close_size": 15, "dilate_size": 1, "min_area": 5000, "min_circularity": 0.75
            }
        ]

        # 统一的形状检测参数
        self.min_circularity = 0.6
        self.min_ellipse_ratio = 0.4
        self.min_area = 5000
        self.max_area_ratio = 0.03

        self.current_params = None
        self.param_set_used = None

    def set_current_params(self, params):
        self.current_params = params
        self.param_set_used = params["name"]

    def simple_color_segmentation(self, image):
        if self.current_params is None:
            raise ValueError("没有设置当前参数集")
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower = np.array([self.current_params["h_min"], self.current_params["s_min"], self.current_params["v_min"]])
        upper = np.array([self.current_params["h_max"], self.current_params["s_max"], self.current_params["v_max"]])
        mask = cv2.inRange(hsv, lower, upper)
        return mask, hsv

    def apply_morphological_operations(self, mask):
        if self.current_params is None:
            raise ValueError("没有设置当前参数集")
        open_size = self.current_params["open_size"]
        close_size = self.current_params["close_size"]
        dilate_size = self.current_params["dilate_size"]
        open_size = open_size if open_size % 2 == 1 else open_size + 1
        close_size = close_size if close_size % 2 == 1 else close_size + 1
        dilate_size = dilate_size if dilate_size % 2 == 1 else dilate_size + 1

        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        filled = ndimage.binary_fill_holes(closed).astype(np.uint8) * 255
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        dilated = cv2.dilate(filled, kernel_dilate, iterations=1)
        return dilated

    def detect_and_select_best_candidate(self, mask, image):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        img_height, img_width = image.shape[:2]
        image_area = img_height * img_width
        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > image_area * self.max_area_ratio:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0
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
            shape_score = 0
            circularity_score = candidate['circularity'] if candidate['circularity'] >= self.min_circularity else 0
            ellipse_score = candidate['ellipse_ratio'] if (candidate['ellipse'] is not None and candidate['ellipse_ratio'] >= self.min_ellipse_ratio) else 0
            convexity_score = candidate['convexity']
            candidate['shape_score'] = circularity_score * 0.4 + ellipse_score * 0.3 + convexity_score * 0.3
            if self.current_params and candidate['circularity'] < self.current_params["min_circularity"]:
                continue
            candidates.append(candidate)

        if not candidates:
            return None
        candidates.sort(key=lambda x: x['shape_score'], reverse=True)
        if candidates[0]['shape_score'] < 0.4:
            return None
        best = candidates[0]
        M = cv2.moments(best['contour'])
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = best['circle_center']
        best['centroid'] = (cX, cY)
        return best

    def calculate_pixel_scale(self, candidate):
        if candidate is None:
            return None, None
        if candidate['ellipse'] is not None:
            (center, axes, angle) = candidate['ellipse']
            major_axis = max(axes)
            pixel_diameter = major_axis
        else:
            pixel_diameter = candidate['circle_radius'] * 2
        scale = self.ball_diameter_mm / pixel_diameter if pixel_diameter > 0 else None
        return scale, pixel_diameter

    def create_comparison_image(self, original, mask, candidate, pixel_scale=None, error_msg=None):
        """创建对比图，处理 mask/candidate 为 None 的情况"""
        height, width = original.shape[:2]
        # 创建三块画布
        comparison = np.ones((height, width * 3, 3), dtype=np.uint8) * 50

        # 1. 原始图像
        original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        comparison[0:height, 0:width] = original_rgb
        cv2.putText(comparison, "Original Image", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # 2. 阈值分割掩码（可能为 None）
        if mask is not None:
            mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        else:
            mask_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(mask_rgb, "No mask available", (50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        comparison[0:height, width:width*2] = mask_rgb
        cv2.putText(comparison, "Segmentation Mask", (width + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # 3. 检测结果图
        result = original.copy()
        detected = candidate is not None
        if detected:
            cv2.drawContours(result, [candidate['contour']], -1, (0, 255, 0), 2)
            if candidate['ellipse'] is not None:
                cv2.ellipse(result, candidate['ellipse'], (0, 165, 255), 2)
            cv2.circle(result, candidate['circle_center'], int(candidate['circle_radius']), (255, 0, 255), 2)
            cv2.circle(result, candidate['centroid'], 5, (0, 0, 255), -1)

        # 构建信息文本（使用更大字体显示 Score 和参数集）
        info_lines = []
        if error_msg:
            info_lines.append(f"Error: {error_msg}")
        elif detected:
            info_lines.append(f"Params: {self.param_set_used}")
            info_lines.append(f"Score: {candidate['shape_score']:.3f}")
            info_lines.append(f"Circularity: {candidate['circularity']:.2f}")
            info_lines.append(f"Area: {candidate['area']:.0f} px")
            if pixel_scale:
                info_lines.append(f"Scale: {pixel_scale:.4f} mm/px")
        else:
            info_lines.append("No marker detected")
            info_lines.append(f"Last params tried: {self.param_set_used if self.param_set_used else 'None'}")

        # 绘制半透明背景框
        x_pos, y_pos = 20, 50
        line_height_small = 30
        line_height_big = 40   # 用于 Score 和 Params 的大行距
        font_small = 0.8
        font_big = 1.5         # 相当于14号左右
        thickness_small = 2
        thickness_big = 3

        # 计算总高度
        total_lines = len(info_lines)
        # 前两行（Params 和 Score）用大字体，其余用小字体
        big_lines = 2 if detected and not error_msg else 0
        if error_msg:
            big_lines = 1  # 错误信息也用大字体
        total_height = big_lines * line_height_big + (total_lines - big_lines) * line_height_small + 20
        bg_width = 320

        overlay = result.copy()
        cv2.rectangle(overlay, (x_pos - 10, y_pos - 30), (x_pos + bg_width, y_pos + total_height), (0, 0, 0), -1)
        result = cv2.addWeighted(overlay, 0.7, result, 0.3, 0)
        cv2.rectangle(result, (x_pos - 10, y_pos - 30), (x_pos + bg_width, y_pos + total_height), (255, 255, 255), 1)

        # 绘制文本
        current_y = y_pos
        for i, line in enumerate(info_lines):
            if i < big_lines:
                font = font_big
                thickness = thickness_big
                line_height = line_height_big
                color = (0, 255, 255) if "Params" in line or "Score" in line else (255, 255, 255)
            else:
                font = font_small
                thickness = thickness_small
                line_height = line_height_small
                color = (255, 255, 255)
            cv2.putText(result, line, (x_pos, current_y), cv2.FONT_HERSHEY_SIMPLEX, font, color, thickness)
            current_y += line_height

        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        comparison[0:height, width*2:width*3] = result_rgb
        cv2.putText(comparison, "Detection Result", (width*2 + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)

        # 分割线
        cv2.line(comparison, (width, 0), (width, height), (200, 200, 200), 2)
        cv2.line(comparison, (width*2, 0), (width*2, height), (200, 200, 200), 2)
        return comparison

    def try_detection_with_params(self, image, params):
        self.set_current_params(params)
        color_mask, hsv = self.simple_color_segmentation(image)
        processed_mask = self.apply_morphological_operations(color_mask)
        best_candidate = self.detect_and_select_best_candidate(processed_mask, image)
        return best_candidate, processed_mask

    def process_single_image(self, image_path, output_dir=None):
        """处理单张图像，总是生成对比图并保存"""
        # 读取图像，处理失败情况
        image = cv2.imread(str(image_path))
        error_msg = None
        if image is None:
            error_msg = f"Cannot read image: {image_path}"
            # 创建一个空白图像作为占位
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(image, "IMAGE READ ERROR", (100, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # 直接生成错误对比图并返回
            comparison = self.create_comparison_image(image, None, None, None, error_msg)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                filename = Path(image_path).stem
                cv2.imwrite(os.path.join(output_dir, f"{filename}_comparison.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
            return {
                'detected': False,
                'param_set': None,
                'score': 0,
                'circularity': 0,
                'area': 0,
                'pixel_scale': None,
                'comparison_image': comparison,
                'error': error_msg
            }

        original = image.copy()
        best_candidate = None
        processed_mask = None
        pixel_scale = None
        param_set_used = None

        # 依次尝试多套参数
        for params in self.param_sets:
            try:
                candidate, mask = self.try_detection_with_params(image, params)
                if candidate is not None:
                    best_candidate = candidate
                    processed_mask = mask
                    param_set_used = params["name"]
                    pixel_scale, _ = self.calculate_pixel_scale(best_candidate)
                    print(f"  使用参数集 {params['name']}: 检测成功")
                    print(f"    评分: {best_candidate['shape_score']:.3f}, 圆形度: {best_candidate['circularity']:.2f}, 面积: {best_candidate['area']:.0f} px")
                    if pixel_scale:
                        print(f"    比例尺: {pixel_scale:.6f} mm/px")
                    break
                else:
                    print(f"  使用参数集 {params['name']}: 未检测到")
            except Exception as e:
                print(f"  使用参数集 {params['name']} 时出错: {str(e)}")
                continue

        # 如果所有参数都失败，保留最后尝试的 mask（任意一个，这里取最后一个 params 的 mask）
        if best_candidate is None and processed_mask is None:
            # 尝试获取最后一个有效的 mask（简单重试一次最后一个参数集获取 mask）
            try:
                _, last_mask = self.try_detection_with_params(image, self.param_sets[-1])
                processed_mask = last_mask
            except:
                processed_mask = None

        # 创建对比图（总是执行）
        comparison = self.create_comparison_image(original, processed_mask, best_candidate, pixel_scale, error_msg)

        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            filename = Path(image_path).stem
            cv2.imwrite(os.path.join(output_dir, f"{filename}_comparison.png"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))

        detected = best_candidate is not None
        return {
            'detected': detected,
            'param_set': param_set_used,
            'score': best_candidate['shape_score'] if detected else 0,
            'circularity': best_candidate['circularity'] if detected else 0,
            'area': best_candidate['area'] if detected else 0,
            'pixel_scale': pixel_scale if detected else None,
            'comparison_image': comparison
        }

    def process_folder(self, folder_path, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(folder_path, "detection_results")
        os.makedirs(output_folder, exist_ok=True)

        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(folder_path, ext)))

        if not image_files:
            print(f"在文件夹中未找到图像文件: {folder_path}")
            return 0, 0

        print(f"找到 {len(image_files)} 个图像文件")
        print("=" * 80)

        detection_count = 0
        for i, img_file in enumerate(image_files):
            print(f"\n处理 [{i+1}/{len(image_files)}]: {os.path.basename(img_file)}")
            try:
                result = self.process_single_image(img_file, output_folder)
                if result and result['detected']:
                    detection_count += 1
                    print(f"  ✓ 检测成功 (使用参数集: {result['param_set']})")
                else:
                    print(f"  ✗ 未检测到")
            except Exception as e:
                print(f"  处理失败: {str(e)}")
                # 即使异常，也尝试生成一个错误对比图（简单处理：创建空白图）
                try:
                    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(dummy_img, f"PROCESSING ERROR: {str(e)}", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    comparison = self.create_comparison_image(dummy_img, None, None, None, str(e))
                    out_path = os.path.join(output_folder, f"{Path(img_file).stem}_error.png")
                    cv2.imwrite(out_path, cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
                except:
                    pass

        print("\n" + "=" * 80)
        print(f"处理完成! 成功检测: {detection_count}/{len(image_files)}")
        print(f"检测成功率: {detection_count/len(image_files)*100:.1f}%")
        print(f"结果保存到: {output_folder}")
        return detection_count, len(image_files)


def main():
    detector = WhiteBallMarkerDetector(ball_diameter_mm=10)
    detector.process_folder(
        folder_path="../analyze/asserts",
        output_folder="D:/Appdevelop/.venv/skin_lesion_processor/core/temp_detection_results"
    )


if __name__ == "__main__":
    main()