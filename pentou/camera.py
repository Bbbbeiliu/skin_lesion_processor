import cv2
import numpy as np
import json

def nothing(x):
    pass

def save_hsv_params(filename, h_min, h_max, s_min, s_max, v_min, v_max):
    params = {
        'H_min': h_min, 'H_max': h_max,
        'S_min': s_min, 'S_max': s_max,
        'V_min': v_min, 'V_max': v_max
    }
    with open(filename, 'w') as f:
        json.dump(params, f, indent=4)
    print(f"参数已保存至 {filename}")

def load_hsv_params(filename='hsv_params.json'):
    try:
        with open(filename, 'r') as f:
            params = json.load(f)
        return (params['H_min'], params['H_max'],
                params['S_min'], params['S_max'],
                params['V_min'], params['V_max'])
    except FileNotFoundError:
        print("未找到保存的参数文件")
        return None

def main():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)   # 请根据实际情况修改
    if not cap.isOpened():
        print("无法打开摄像头")
        return

    cv2.namedWindow('Mask')
    cv2.namedWindow('Result')

    # 创建HSV滑动条
    cv2.createTrackbar('H Min', 'Mask', 0, 180, nothing)
    cv2.createTrackbar('H Max', 'Mask', 180, 180, nothing)
    cv2.createTrackbar('S Min', 'Mask', 0, 255, nothing)
    cv2.createTrackbar('S Max', 'Mask', 255, 255, nothing)
    cv2.createTrackbar('V Min', 'Mask', 0, 255, nothing)
    cv2.createTrackbar('V Max', 'Mask', 50, 255, nothing)

    # 形态学操作滑动条（可选）
    cv2.createTrackbar('Morph Kernel', 'Mask', 3, 20, nothing)
    cv2.createTrackbar('Morph Iter', 'Mask', 1, 5, nothing)
    cv2.createTrackbar('Min Area', 'Mask', 500, 5000, nothing)

    print("控制说明：按 's' 保存当前HSV参数，按 'l' 加载上次保存的参数，按 'q' 退出")

    # 用于首次显示尺寸的标记
    size_printed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 获取图像尺寸并输出一次
        if not size_printed:
            height, width = frame.shape[:2]
            print(f"摄像头图像尺寸：宽度 = {width} 像素，高度 = {height} 像素")
            size_printed = True

        # 获取滑动条值
        h_min = cv2.getTrackbarPos('H Min', 'Mask')
        h_max = cv2.getTrackbarPos('H Max', 'Mask')
        s_min = cv2.getTrackbarPos('S Min', 'Mask')
        s_max = cv2.getTrackbarPos('S Max', 'Mask')
        v_min = cv2.getTrackbarPos('V Min', 'Mask')
        v_max = cv2.getTrackbarPos('V Max', 'Mask')
        kernel_size = cv2.getTrackbarPos('Morph Kernel', 'Mask')
        morph_iter = cv2.getTrackbarPos('Morph Iter', 'Mask')
        min_area = cv2.getTrackbarPos('Min Area', 'Mask')

        # 确保最小值不大于最大值
        if h_min > h_max: h_min, h_max = h_max, h_min
        if s_min > s_max: s_min, s_max = s_max, s_min
        if v_min > v_max: v_min, v_max = v_max, v_min

        # 生成掩膜
        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower, upper)

        # 形态学开运算
        if kernel_size > 0:
            kernel = np.ones((kernel_size, kernel_size), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=morph_iter)

        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 绘制结果
        result = frame.copy()
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)  # 绿色轮廓

        # 筛选并标记矩形，同时输出宽度和内部像素个数
        contour_idx = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                contour_idx += 1
                x, y, w, h = cv2.boundingRect(cnt)   # 获取包围矩形

                # 输出轮廓信息（宽度和内部像素个数）
                print(f"轮廓 {contour_idx}: 宽度 = {w} 像素, 内部像素个数 = {int(area)}")

                # 可选：绘制矩形并标注面积（仅对四边形绘制蓝色矩形）
                epsilon = 0.02 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) == 4:
                    cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(result, f'Area:{int(area)}', (x, y-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 在图像上显示尺寸信息
        cv2.putText(result, f"Size: {width}x{height}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow('Mask', mask)
        cv2.imshow('Result', result)

        # 按键处理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # 保存当前HSV参数
            h_min = cv2.getTrackbarPos('H Min', 'Mask')
            h_max = cv2.getTrackbarPos('H Max', 'Mask')
            s_min = cv2.getTrackbarPos('S Min', 'Mask')
            s_max = cv2.getTrackbarPos('S Max', 'Mask')
            v_min = cv2.getTrackbarPos('V Min', 'Mask')
            v_max = cv2.getTrackbarPos('V Max', 'Mask')
            save_hsv_params('hsv_params.json', h_min, h_max, s_min, s_max, v_min, v_max)
        elif key == ord('l'):
            params = load_hsv_params('hsv_params.json')
            if params:
                h_min, h_max, s_min, s_max, v_min, v_max = params
                cv2.setTrackbarPos('H Min', 'Mask', h_min)
                cv2.setTrackbarPos('H Max', 'Mask', h_max)
                cv2.setTrackbarPos('S Min', 'Mask', s_min)
                cv2.setTrackbarPos('S Max', 'Mask', s_max)
                cv2.setTrackbarPos('V Min', 'Mask', v_min)
                cv2.setTrackbarPos('V Max', 'Mask', v_max)
                print("已加载保存的参数")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()