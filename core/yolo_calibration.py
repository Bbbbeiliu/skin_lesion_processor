"""
YOLO 标志物检测标定模块
功能：使用 YOLOv8 ONNX 模型检测白色小球，计算像素比例尺 (mm/px)
封装为独立类，供 marker_detector.py 集成调用
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List


class YOLOMarkerDetector:
    """YOLO 标志物检测器，用于自动标定像素比例尺"""

    def __init__(self, model_path: str, conf_threshold: float = 0.5, nms_threshold: float = 0.5,
                 input_size: int = 640, ball_diameter_mm: float = 10.0):
        """
        初始化检测器
        :param model_path: ONNX 模型文件路径
        :param conf_threshold: 置信度阈值
        :param nms_threshold: NMS 阈值
        :param input_size: 模型输入尺寸（正方形）
        :param ball_diameter_mm: 标志物实际直径（毫米）
        """
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.ball_diameter_mm = ball_diameter_mm
        self.net = None
        self._load_model()

    def _load_model(self):
        """加载 ONNX 模型"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
        print(f"[YOLO] 加载模型: {self.model_path}")
        self.net = cv2.dnn.readNetFromONNX(self.model_path)
        print("[YOLO] 模型加载成功")

    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Tuple[float, int, int, int, int], Tuple[int, int]]:
        """预处理图像，返回 blob、预处理参数和原始尺寸"""
        original_h, original_w = image.shape[:2]
        scale = min(self.input_size / original_w, self.input_size / original_h)
        new_w = int(original_w * scale)
        new_h = int(original_h * scale)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_top = (self.input_size - new_h) // 2
        pad_left = (self.input_size - new_w) // 2
        padded[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        norm = rgb.astype(np.float32) / 255.0
        chw = norm.transpose(2, 0, 1)
        blob = np.expand_dims(chw, axis=0)

        return blob, (scale, pad_left, pad_top, new_w, new_h), (original_h, original_w)

    def _postprocess(self, outputs: np.ndarray, preprocess_info: Tuple, original_shape: Tuple) -> List:
        """后处理模型输出，返回检测框列表 [[x1,y1,x2,y2,conf], ...]"""
        scale, pad_left, pad_top, new_w, new_h = preprocess_info
        original_h, original_w = original_shape

        predictions = outputs[0].transpose(1, 0)  # (8400,5)
        bbox_data = predictions[:, :4]   # cx, cy, w, h 归一化（相对于640）
        scores = predictions[:, 4]

        keep = scores > self.conf_threshold
        if not np.any(keep):
            return []
        bbox_data = bbox_data[keep]
        scores = scores[keep]

        cx, cy, w, h = bbox_data[:, 0], bbox_data[:, 1], bbox_data[:, 2], bbox_data[:, 3]
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        boxes_640 = np.column_stack([x1, y1, x2, y2])

        boxes_no_pad = boxes_640.copy()
        boxes_no_pad[:, [0, 2]] -= pad_left
        boxes_no_pad[:, [1, 3]] -= pad_top
        boxes_original = boxes_no_pad / scale

        boxes_original[:, [0, 2]] = np.clip(boxes_original[:, [0, 2]], 0, original_w)
        boxes_original[:, [1, 3]] = np.clip(boxes_original[:, [1, 3]], 0, original_h)

        indices = cv2.dnn.NMSBoxes(
            boxes_original.tolist(),
            scores.tolist(),
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold
        )

        detections = []
        if len(indices) > 0:
            indices = indices.flatten()
            for idx in indices:
                x1, y1, x2, y2 = boxes_original[idx]
                conf = scores[idx]
                detections.append([int(x1), int(y1), int(x2), int(y2), float(conf)])

        # 按面积降序排序，取最大者
        if detections:
            detections.sort(key=lambda d: (d[2]-d[0])*(d[3]-d[1]), reverse=True)
        return detections

    def get_detection_info(self, image_path: str) -> Tuple[Optional[List], Optional[float]]:
        """
        从图像中检测标志物，返回 (检测框列表, 比例尺)
        检测框列表格式: [[x1,y1,x2,y2,conf], ...]（已按面积排序）
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"[YOLO] 无法读取图像: {image_path}")
            return None, None

        blob, pre_info, orig_shape = self._preprocess(img)
        self.net.setInput(blob)
        outputs = self.net.forward()
        detections = self._postprocess(outputs, pre_info, orig_shape)

        if not detections:
            print(f"[YOLO] 未检测到标志物: {image_path}")
            return None, None

        # 取面积最大的检测框作为标志物
        best = detections[0]
        x1, y1, x2, y2, conf = best
        width = x2 - x1
        height = y2 - y1
        pixel_diameter = (width + height) / 2.0
        if pixel_diameter <= 0:
            return None, None

        scale_mm_per_px = self.ball_diameter_mm / pixel_diameter
        print(f"[YOLO] 检测成功: 置信度={conf:.3f}, 像素直径={pixel_diameter:.1f}px, 比例尺={scale_mm_per_px:.6f} mm/px")
        return detections, scale_mm_per_px

    def get_scale_from_image(self, image_path: str) -> Optional[float]:
        """兼容旧接口，仅返回比例尺"""
        _, scale = self.get_detection_info(image_path)
        return scale