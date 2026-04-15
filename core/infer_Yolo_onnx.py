import cv2
import numpy as np
import os
from pathlib import Path
import time

# ========== 配置参数 ==========
ONNX_MODEL_PATH = r"best.onnx"
INPUT_FOLDER = r"E:/SkinLesion/Test"
OUTPUT_FOLDER = r"E:/SkinLesion/Test/opencv_dnn_inference/results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

MODEL_INPUT_SIZE = 640
CLASS_NAMES = ['scar']  # 单类别

CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.5
MAX_DETECTIONS = 1  # 与原pt推理的 max_det=1 保持一致

# ========== 加载模型 ==========
print(f"正在加载模型: {ONNX_MODEL_PATH}")
net = cv2.dnn.readNetFromONNX(ONNX_MODEL_PATH)
print("模型加载成功。")

# 验证模型输入输出形状
print("=" * 50)
print("模型信息确认:")
print(f"  输出形状确认: (1, 5, 8400)")
print(f"  格式说明: [批次大小, 特征维度, 锚点数量]")
print(f"  特征维度5对应: [中心点x, 中心点y, 宽度, 高度, 置信度]")
print("=" * 50)


# ========== 图像预处理函数 ==========
def preprocess_image(image_path):
    """
    将单张图片预处理为模型输入的blob。
    步骤：读取 -> 缩放+填充至640x640 -> 归一化 -> 转换维度 (HWC to BCHW)
    """
    # 读取原始图像
    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"错误：无法读取图像 {image_path}")
        return None, None, None, None

    h, w = img_original.shape[:2]

    # 计算缩放比例，并保持长宽比进行缩放
    scale = min(MODEL_INPUT_SIZE / w, MODEL_INPUT_SIZE / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    # 缩放图像
    img_resized = cv2.resize(img_original, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 创建640x640的灰色画布 (RGB: 114, 114, 114)
    img_padded = np.full((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3), 114, dtype=np.uint8)
    top = (MODEL_INPUT_SIZE - new_h) // 2
    left = (MODEL_INPUT_SIZE - new_w) // 2
    img_padded[top:top + new_h, left:left + new_w, :] = img_resized

    # 转换为模型输入需要的格式: BGR -> RGB, HWC -> CHW, 归一化
    img_rgb = cv2.cvtColor(img_padded, cv2.COLOR_BGR2RGB)
    img_normalized = img_rgb.astype(np.float32) / 255.0
    img_chw = img_normalized.transpose(2, 0, 1)  # 从 (H,W,C) 变为 (C,H,W)

    # 添加批处理维度: (C,H,W) -> (1,C,H,W)
    blob = np.expand_dims(img_chw, axis=0)

    return blob, img_original, (scale, top, left, new_w, new_h), (h, w)


# ========== 后处理函数（针对单类别模型优化） ==========
def postprocess_output(output, original_shape, preprocess_info, conf_thresh=0.5, nms_thresh=0.5, max_det=1):
    """
    处理单类别YOLOv8模型输出，形状为 (1, 5, 8400)
    输出格式: [中心点x, 中心点y, 宽度, 高度, 置信度]
    """
    scale, pad_top, pad_left, new_w, new_h = preprocess_info
    original_h, original_w = original_shape

    # 1. 转换输出格式: (1, 5, 8400) -> (8400, 5)
    predictions = output[0].transpose(1, 0)  # 转置后形状: (8400, 5)

    # 2. 分离各个分量
    # predictions[:, 0:4]: [cx, cy, w, h] (归一化坐标，相对于640x640)
    # predictions[:, 4]: confidence (置信度分数)
    bbox_data = predictions[:, :4]  # 边界框参数 [8400, 4]
    scores = predictions[:, 4]  # 置信度分数 [8400]

    # 3. 根据置信度阈值进行初步筛选
    keep_indices = scores > conf_thresh

    if not np.any(keep_indices):
        # 没有检测到任何目标
        return np.array([])

    # 应用筛选
    bbox_data = bbox_data[keep_indices]
    scores = scores[keep_indices]

    # 4. 将边界框格式从 (cx, cy, w, h) 转换为 (x1, y1, x2, y2)
    cx, cy, w, h = bbox_data[:, 0], bbox_data[:, 1], bbox_data[:, 2], bbox_data[:, 3]
    x1 = cx - w / 2
    y1 = cy - h / 2
    x2 = cx + w / 2
    y2 = cy + h / 2

    # 将所有框组合为 [x1, y1, x2, y2] 格式
    boxes_640 = np.column_stack([x1, y1, x2, y2])

    # 5. 将坐标从640x640画布映射回原始图像尺寸
    # 5.1 先减去填充偏移量
    boxes_no_pad = boxes_640.copy()
    boxes_no_pad[:, [0, 2]] -= pad_left  # 减去左边的填充
    boxes_no_pad[:, [1, 3]] -= pad_top  # 减去顶部的填充

    # 5.2 再除以缩放比例，映射回原始图像尺寸
    boxes_original = boxes_no_pad / scale

    # 6. 确保坐标不超出原始图像边界
    boxes_original[:, [0, 2]] = np.clip(boxes_original[:, [0, 2]], 0, original_w)
    boxes_original[:, [1, 3]] = np.clip(boxes_original[:, [1, 3]], 0, original_h)

    # 7. 应用非极大值抑制 (NMS) 以去除重叠框
    indices = cv2.dnn.NMSBoxes(
        boxes_original.tolist(),
        scores.tolist(),
        score_threshold=conf_thresh,
        nms_threshold=nms_thresh
    )

    # 8. 收集并限制检测结果数量
    detections = []
    if len(indices) > 0:
        indices = indices.flatten()[:max_det]  # 限制最大检测数量
        for idx in indices:
            x1, y1, x2, y2 = boxes_original[idx]
            confidence = scores[idx]

            detections.append([
                int(x1), int(y1), int(x2), int(y2),  # 边界框坐标
                float(confidence),  # 置信度
                0  # 类别ID (只有一类，所以固定为0)
            ])

    return np.array(detections) if detections else np.array([])


# ========== 主推理流程 ==========
def main():
    # 获取输入文件夹中的所有图片
    image_extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(Path(INPUT_FOLDER).glob(ext))

    if not image_paths:
        print(f"在文件夹 {INPUT_FOLDER} 中未找到图片。")
        return

    print(f"找到 {len(image_paths)} 张图片，开始推理...")
    print(f"置信度阈值: {CONFIDENCE_THRESHOLD}, 最大检测数: {MAX_DETECTIONS}")
    print("-" * 50)

    # 统计信息
    total_images = len(image_paths)
    processed_count = 0
    detection_count = 0
    no_detection_images = []  # 新增：记录未检测到目标的图片
    start_time = time.time()

    for img_path in image_paths:
        img_path_str = str(img_path)
        img_name = img_path.name
        processed_count += 1
        print(f"[{processed_count}/{total_images}] 处理: {img_name}")

        # 1. 预处理
        blob, img_original, preprocess_info, original_shape = preprocess_image(img_path_str)
        if blob is None:
            print(f"  跳过: 无法读取或预处理图像")
            continue

        # 2. 推理
        net.setInput(blob)
        outputs = net.forward()  # 输出形状: (1, 5, 8400)

        # 3. 后处理
        detections = postprocess_output(
            outputs, original_shape, preprocess_info,
            conf_thresh=CONFIDENCE_THRESHOLD,
            nms_thresh=NMS_THRESHOLD,
            max_det=MAX_DETECTIONS
        )

        # 4. 可视化并保存结果
        img_result = img_original.copy()
        if len(detections) > 0:
            detection_count += 1
            for det in detections:
                x1, y1, x2, y2, conf, cls_id = det

                # 修复：将 cls_id 转换为整数
                cls_id = int(cls_id)

                # 绘制边界框
                cv2.rectangle(img_result, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # 准备标签文本
                label = f"{CLASS_NAMES[cls_id]}: {conf:.2f}"

                # 计算文本尺寸
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )

                # 绘制文本背景
                text_y1 = max(y1, text_height + 5)
                cv2.rectangle(img_result,
                              (int(x1), int(text_y1 - text_height - 5)),
                              (int(x1 + text_width), int(text_y1)),
                              (0, 255, 0), -1)

                # 绘制文本
                cv2.putText(img_result, label,
                            (int(x1), int(text_y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                print(
                    f"  检测到: {CLASS_NAMES[cls_id]}, 置信度: {conf:.4f}, 位置: [{x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}]")
        else:
            # 记录未检测到目标的图片
            no_detection_images.append(img_name)
            print(f"  未检测到目标 (置信度 < {CONFIDENCE_THRESHOLD})")

        # 保存结果图片
        output_path = os.path.join(OUTPUT_FOLDER, img_name)
        cv2.imwrite(output_path, img_result)
        print(f"  结果已保存: {output_path}")
        print()

    # 打印统计信息
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / total_images if total_images > 0 else 0

    print("=" * 50)
    print("推理完成！")
    print(f"处理图片总数: {total_images} 张")
    print(f"检测到目标的图片: {detection_count} 张")
    print(f"未检测到目标的图片: {len(no_detection_images)} 张")
    print(f"总耗时: {total_time:.2f} 秒")
    print(f"平均每张图片: {avg_time:.3f} 秒")
    print(f"所有结果保存在: {OUTPUT_FOLDER}")
    print("-" * 50)

    # 输出未检测到目标的图片列表
    if no_detection_images:
        print("未检测到目标的图片列表:")
        for i, img_name in enumerate(no_detection_images, 1):
            print(f"  {i:3d}. {img_name}")

        # 可选：将列表保存到文本文件
        no_detection_file = os.path.join(OUTPUT_FOLDER, "no_detection_list.txt")
        with open(no_detection_file, 'w', encoding='utf-8') as f:
            f.write("未检测到目标的图片列表:\n")
            for img_name in no_detection_images:
                f.write(f"{img_name}\n")
        print(f"未检测到目标的图片列表已保存到: {no_detection_file}")
    else:
        print("所有图片都检测到了目标！")

    print("=" * 50)


if __name__ == "__main__":
    main()