"""
图像处理器
"""
import cv2
import numpy as np
import traceback
import math
from pathlib import Path
from typing import List, Tuple, Any, Optional
from PyQt5.QtCore import QPointF
from .file_utils import FileUtils

try:
    from geomdl import NURBS
    from geomdl import knotvector
    GEOMDL_AVAILABLE = True
except ImportError:
    GEOMDL_AVAILABLE = False
    print("警告: geomdl库未安装，无法使用NURBS曲线拟合。请运行: pip install geomdl")


class AdvancedImageProcessor:
    """高级图像处理器，提供更好的轮廓拟合"""

    @staticmethod
    def load_and_process_image(image_path: str, kernel_size: int = 3) -> List[Tuple[np.ndarray, str]]:
        """
        加载并处理图像，提取轮廓
        Returns:
            (轮廓点, 图像名称) 元组列表
        """
        try:
            image_name = Path(image_path).name
            contours = []

            # 读取图像
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"无法读取图像: {image_path}")
                return contours

            # 二值化
            if len(np.unique(image)) > 2:
                _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            else:
                binary = image

            # 形态学操作：闭运算连接相邻区域
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

            # 查找轮廓
            contours_data, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours_data:
                # 过滤掉太小的轮廓
                area = cv2.contourArea(contour)
                if area < 100:  # 提高最小面积阈值
                    continue

                # 使用更精确的轮廓近似
                epsilon = 0.001 * cv2.arcLength(contour, True)  # 更小的epsilon以保留更多细节
                approx = cv2.approxPolyDP(contour, epsilon, True)

                # 转换为numpy数组并确保维度正确
                if len(approx) >= 3:
                    points = approx.squeeze()
                    if points.ndim == 1:
                        points = points.reshape(-1, 2)

                    # 确保轮廓闭合
                    if not np.array_equal(points[0], points[-1]):
                        points = np.vstack([points, points[0]])

                    contours.append((points, image_name))

            return contours

        except Exception as e:
            print(f"处理图像时发生错误 {image_path}: {str(e)}")
            traceback.print_exc()
            return []

    @staticmethod
    def simplify_contour(points: np.ndarray, tolerance: float = 1.0) -> np.ndarray:
        """使用Ramer-Douglas-Peucker算法简化轮廓"""
        if len(points) < 3:
            return points

        def rdp_recursive(point_list, start_idx, end_idx):
            if end_idx <= start_idx + 1:
                return []

            # 找到离线段最远的点
            max_dist = 0
            max_idx = start_idx

            start_pt = point_list[start_idx]
            end_pt = point_list[end_idx]

            for i in range(start_idx + 1, end_idx):
                pt = point_list[i]
                # 计算点到线段的距离
                if np.array_equal(start_pt, end_pt):
                    dist = np.linalg.norm(pt - start_pt)
                else:
                    line_len = np.linalg.norm(end_pt - start_pt)
                    if line_len == 0:
                        dist = np.linalg.norm(pt - start_pt)
                    else:
                        t = max(0, min(1, np.dot(pt - start_pt, end_pt - start_pt) / (line_len * line_len)))
                        projection = start_pt + t * (end_pt - start_pt)
                        dist = np.linalg.norm(pt - projection)

                if dist > max_dist:
                    max_dist = dist
                    max_idx = i

            result = []
            if max_dist > tolerance:
                # 递归处理
                rec1 = rdp_recursive(point_list, start_idx, max_idx)
                rec2 = rdp_recursive(point_list, max_idx, end_idx)

                result = rec1 + [point_list[max_idx]] + rec2

            return result

        # 执行RDP算法
        simplified = [points[0]] + rdp_recursive(points, 0, len(points) - 1) + [points[-1]]

        return np.array(simplified)

    @staticmethod
    def smooth_contour_with_nurbs(points: np.ndarray, precision: float = 0.5,
                                  num_control_points: Optional[int] = None) -> Tuple[List[QPointF], Any]:
        """
        使用闭合NURBS曲线平滑轮廓，控制点按曲率自适应采样
        Args:
            points: 原始轮廓点 (numpy数组，形状为 (N, 2) 或 (N, 1, 2))
            precision: 拟合精度 (0.0-1.0) - 影响简化容差和采样点数量
            num_control_points: 指定的控制点数量（优先级高于 precision）
        Returns:
            (NURBS曲线点列表, NURBS曲线对象) 元组
        """
        if not GEOMDL_AVAILABLE:
            print("警告: geomdl库未安装，使用贝塞尔曲线替代")
            _, nurbs_points = AdvancedImageProcessor.smooth_contour_with_cubic_bezier(points, int(precision * 40 + 10))
            return nurbs_points, None

        if len(points) < 3:
            return [], None

        points = points.squeeze()
        if points.ndim != 2 or points.shape[1] != 2:
            return [], None

        # 确保轮廓闭合
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack([points, points[0]])

        try:
            # 计算简化容差（基于精度）
            simplify_tolerance = max(0.5, min(5.0, (1.0 - precision) * 10))
            degree = 3 if precision > 0.3 else 2

            # 第一步：简化轮廓（RDP）
            simplified_points = AdvancedImageProcessor.simplify_contour(points, simplify_tolerance)
            if len(simplified_points) < 3:
                simplified_points = points

            # 确定目标控制点数量
            if num_control_points is not None:
                target_control_points = max(degree + 1, min(500, num_control_points))
            else:
                target_control_points = max(10, min(50, int(pow(precision, 1.5) * 40 + 10)))

            # --- 均匀采样（回退至原始方案）---
            src_points = simplified_points  # 列表
            src_len = len(src_points)

            if src_len <= target_control_points:
                selected_indices = list(range(src_len))
            else:
                step = src_len / target_control_points
                selected_indices = [int(i * step) for i in range(target_control_points)]

            control_points_list = []
            for idx in selected_indices:
                pt = src_points[idx]
                if len(pt) == 2:
                    control_points_list.append([float(pt[0]), float(pt[1]), 1.0])
                else:
                    control_points_list.append([float(pt[0]), float(pt[1]), float(pt[2]) if len(pt) > 2 else 1.0])
            # --- 均匀采样结束 ---

            # 确保控制点数量满足最低要求（degree+1）
            while len(control_points_list) < degree + 1:
                pt = simplified_points[-1]
                control_points_list.append([float(pt[0]), float(pt[1]), 1.0])

            # 创建NURBS曲线
            curve = NURBS.Curve()
            curve.degree = degree
            curve.ctrlpts = control_points_list

            # 自动生成节点向量
            curve.knotvector = knotvector.generate(curve.degree, len(curve.ctrlpts))

            # 设置采样点数（用于曲线求值）
            sample_points = max(100, min(300, int(precision * 200 + 100)))
            curve.sample_size = sample_points

            # 计算曲线上的点
            curve_points = curve.evalpts

            # 转换为QPointF列表
            nurbs_points = [QPointF(float(p[0]), float(p[1])) for p in curve_points]

            # 确保曲线闭合
            if len(nurbs_points) > 1 and nurbs_points[0] != nurbs_points[-1]:
                nurbs_points.append(nurbs_points[0])

            return nurbs_points, curve

        except Exception as e:
            print(f"NURBS曲线拟合错误: {str(e)}")
            traceback.print_exc()
            # 失败时回退到贝塞尔曲线
            _, nurbs_points = AdvancedImageProcessor.smooth_contour_with_cubic_bezier(points, int(precision * 40 + 10))
            return nurbs_points, None

    @staticmethod
    def smooth_contour_with_cubic_bezier(points: np.ndarray, num_control_points: int = 20) -> Tuple[
        List[QPointF], List[QPointF]]:
        """
        使用闭合三次贝塞尔曲线平滑轮廓（备选方案）
        """
        if len(points) < 3:
            return [], []

        points = points.squeeze()
        if points.ndim != 2 or points.shape[1] != 2:
            return [], []

        # 确保是闭合轮廓
        if not np.array_equal(points[0], points[-1]):
            points = np.vstack([points, points[0]])

        n_points = len(points)
        if n_points <= 1:
            return [], []

        # 1. 计算曲线总长度
        total_length = 0
        segment_lengths = []
        for i in range(n_points - 1):
            dx = points[i + 1][0] - points[i][0]
            dy = points[i + 1][1] - points[i][1]
            length = math.sqrt(dx * dx + dy * dy)
            segment_lengths.append(length)
            total_length += length

        if total_length == 0:
            return [], []

        # 2. 等弧长选取控制点
        control_points = []
        num_control_points = min(num_control_points, n_points)
        target_spacing = total_length / num_control_points

        current_length = 0
        next_target = target_spacing

        # 第一个点
        control_points.append(QPointF(float(points[0][0]), float(points[0][1])))

        for i in range(n_points - 1):
            seg_len = segment_lengths[i]

            while current_length + seg_len >= next_target and len(control_points) < num_control_points:
                # 在段内线性插值
                t = (next_target - current_length) / seg_len
                x = points[i][0] + t * (points[i + 1][0] - points[i][0])
                y = points[i][1] + t * (points[i + 1][1] - points[i][1])
                control_points.append(QPointF(float(x), float(y)))
                next_target += target_spacing

            current_length += seg_len

        # 确保首尾一致形成闭合曲线
        if len(control_points) > 1 and control_points[0] != control_points[-1]:
            control_points.append(control_points[0])

        # 3. 使用三次贝塞尔曲线平滑（Catmull-Rom样条）
        if len(control_points) >= 4:
            bezier_points = []
            n = len(control_points) - 1  # 因为首尾相同

            for i in range(n):
                # Catmull-Rom样条控制点
                p0 = control_points[i - 1] if i > 0 else control_points[n - 1]
                p1 = control_points[i]
                p2 = control_points[i + 1]
                p3 = control_points[i + 2] if i < n - 1 else control_points[1]

                # 转换为三次贝塞尔控制点
                b0 = p1
                b1 = p1 + (p2 - p0) / 6.0
                b2 = p2 - (p3 - p1) / 6.0
                b3 = p2

                # 生成贝塞尔曲线段
                for t in np.linspace(0, 1, 30):
                    u = 1 - t
                    x = (u ** 3 * b0.x() + 3 * u ** 2 * t * b1.x() +
                         3 * u * t ** 2 * b2.x() + t ** 3 * b3.x())
                    y = (u ** 3 * b0.y() + 3 * u ** 2 * t * b1.y() +
                         3 * u * t ** 2 * b2.y() + t ** 3 * b3.y())
                    bezier_points.append(QPointF(x, y))

            return control_points, bezier_points

        return [], []