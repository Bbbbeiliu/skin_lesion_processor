# mark_automation.py - 支持多轮预览和标刻
# -*- coding: utf-8 -*-
import argparse
import ctypes
import os
import sys
import io
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def main():
    parser = argparse.ArgumentParser(description='EzCAD2 自动化标刻工具')
    parser.add_argument('dxf_path', help='DXF文件路径')
    parser.add_argument('--preview', action='store_true', help='仅红光预览（一次）')
    parser.add_argument('--mark', action='store_true', help='仅实际标刻（一次）')
    parser.add_argument('--preview_count', type=int, help='红光预览次数')
    parser.add_argument('--mark_count', type=int, help='实际标刻次数')
    parser.add_argument('--debug', action='store_true', help='调试模式：弹出设备参数对话框，并打印加工参数')
    args = parser.parse_args()

    print("=" * 60)
    print(f"参数: dxf={args.dxf_path}, preview={args.preview}, mark={args.mark}, "
          f"preview_count={args.preview_count}, mark_count={args.mark_count}, debug={args.debug}")
    print("=" * 60)

    # 确定SDK目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sdk_dir = os.path.join(script_dir, "..", "ezcad_sdk")
    print(f"脚本目录: {script_dir}")
    print(f"SDK目录: {sdk_dir}")

    if not os.path.isdir(sdk_dir):
        print(f"错误：SDK目录不存在: {sdk_dir}")
        sys.exit(1)

    # 列出SDK目录内容（调试用）
    print("SDK目录内容:")
    for f in os.listdir(sdk_dir):
        print(f"  {f}")

    # 检查必要文件
    required_files = [
        "EZCAD.CFG",
        "PARAM/MarkParam.lib",
        "MarkEzd.dll",
        "Lmc1.dll"
    ]
    for rel_path in required_files:
        full_path = os.path.join(sdk_dir, rel_path)
        if not os.path.exists(full_path):
            print(f"错误：缺少必要文件 {full_path}")
            sys.exit(1)
        else:
            print(f"文件存在: {full_path}")

    # 切换工作目录到 SDK 目录
    os.chdir(sdk_dir)
    print(f"工作目录已切换到：{os.getcwd()}")

    # 加载 DLL
    try:
        dll_path = os.path.join(sdk_dir, "MarkEzd.dll")
        ezd = ctypes.CDLL(dll_path)
        print("MarkEzd.dll 加载成功")
    except Exception as e:
        print(f"加载 MarkEzd.dll 失败：{e}")
        sys.exit(1)

    def check_ret(ret, msg):
        if ret != 0:
            print(f"错误：{msg}，返回值={ret}")
            ezd.lmc1_Close()
            sys.exit(ret)
        else:
            print(f"成功：{msg}")

    # ---------- 函数原型定义 ----------
    ezd.lmc1_Initial2.argtypes = [ctypes.c_wchar_p, ctypes.c_bool]
    ezd.lmc1_Initial2.restype = ctypes.c_int

    ezd.lmc1_ClearEntLib.argtypes = []
    ezd.lmc1_ClearEntLib.restype = ctypes.c_int

    ezd.lmc1_LoadEzdFile.argtypes = [ctypes.c_wchar_p]
    ezd.lmc1_LoadEzdFile.restype = ctypes.c_int

    ezd.lmc1_AddFileToLib.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p,
                                      ctypes.c_double, ctypes.c_double, ctypes.c_double,
                                      ctypes.c_int, ctypes.c_double, ctypes.c_int, ctypes.c_bool]
    ezd.lmc1_AddFileToLib.restype = ctypes.c_int

    ezd.lmc1_GetEntSize.argtypes = [ctypes.c_wchar_p,
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                    ctypes.POINTER(ctypes.c_double)]
    ezd.lmc1_GetEntSize.restype = ctypes.c_int

    ezd.lmc1_MoveEnt.argtypes = [ctypes.c_wchar_p, ctypes.c_double, ctypes.c_double]
    ezd.lmc1_MoveEnt.restype = ctypes.c_int

    ezd.lmc1_RedLightMarkContour.argtypes = []
    ezd.lmc1_RedLightMarkContour.restype = ctypes.c_int

    ezd.lmc1_Mark.argtypes = [ctypes.c_bool]
    ezd.lmc1_Mark.restype = ctypes.c_int

    ezd.lmc1_Close.argtypes = []
    ezd.lmc1_Close.restype = ctypes.c_int

    # 调试模式下需要 lmc1_SetDevCfg
    if args.debug:
        ezd.lmc1_SetDevCfg.argtypes = []
        ezd.lmc1_SetDevCfg.restype = ctypes.c_int

    # 定义 lmc1_GetPenParam4
    ezd.lmc1_GetPenParam4.argtypes = [
        ctypes.c_int,                     # nPenNo
        ctypes.c_wchar_p,                 # sPenName
        ctypes.POINTER(ctypes.c_int),     # clr
        ctypes.POINTER(ctypes.c_bool),    # bDisableMark
        ctypes.POINTER(ctypes.c_bool),    # bUseDefParam
        ctypes.POINTER(ctypes.c_int),     # nMarkLoop
        ctypes.POINTER(ctypes.c_double),  # dMarkSpeed
        ctypes.POINTER(ctypes.c_double),  # dPowerRatio
        ctypes.POINTER(ctypes.c_double),  # dCurrent
        ctypes.POINTER(ctypes.c_int),     # nFreq
        ctypes.POINTER(ctypes.c_double),  # dQPulseWidth
        ctypes.POINTER(ctypes.c_int),     # nStartTC
        ctypes.POINTER(ctypes.c_int),     # nLaserOffTC
        ctypes.POINTER(ctypes.c_int),     # nEndTC
        ctypes.POINTER(ctypes.c_int),     # nPolyTC
        ctypes.POINTER(ctypes.c_double),  # dJumpSpeed
        ctypes.POINTER(ctypes.c_int),     # nMinJumpDelayTCUs
        ctypes.POINTER(ctypes.c_int),     # nMaxJumpDelayTCUs
        ctypes.POINTER(ctypes.c_double),  # dJumpLengthLimit
        ctypes.POINTER(ctypes.c_double),  # dPointTime
        ctypes.POINTER(ctypes.c_bool),    # nSpiSpiContinueMode
        ctypes.POINTER(ctypes.c_int),     # nSpiWave
        ctypes.POINTER(ctypes.c_int),     # nYagMarkMode
        ctypes.POINTER(ctypes.c_bool),    # bPulsePointMode
        ctypes.POINTER(ctypes.c_int),     # nPulseNum
        ctypes.POINTER(ctypes.c_bool),    # bEnableACCMode
        ctypes.POINTER(ctypes.c_double),  # dEndComp
        ctypes.POINTER(ctypes.c_double),  # dAccDist
        ctypes.POINTER(ctypes.c_double),  # dBreakAngle
        ctypes.POINTER(ctypes.c_bool),    # bWobbleMode
        ctypes.POINTER(ctypes.c_double),  # bWobbleDiameter
        ctypes.POINTER(ctypes.c_double)   # bWobbleDist
    ]
    ezd.lmc1_GetPenParam4.restype = ctypes.c_int

    # ---------- 辅助函数 ----------
    def import_dxf_and_center(dxf_path, ent_name="AutoDXF"):
        print(f"检查DXF文件: {dxf_path}")
        if not os.path.isfile(dxf_path):
            print(f"错误：DXF文件不存在: {dxf_path}")
            return False, 0, 0

        ret = ezd.lmc1_AddFileToLib(dxf_path, ent_name, 0.0, 0.0, 0.0, 0, 1.0, 0, False)
        if ret != 0:
            print(f"导入DXF失败，返回值={ret}")
            return False, 0, 0
        print(f"导入DXF成功，对象名称: {ent_name}")

        minx = ctypes.c_double()
        miny = ctypes.c_double()
        maxx = ctypes.c_double()
        maxy = ctypes.c_double()
        z = ctypes.c_double()
        ret = ezd.lmc1_GetEntSize(ent_name, ctypes.byref(minx), ctypes.byref(miny),
                                  ctypes.byref(maxx), ctypes.byref(maxy), ctypes.byref(z))
        if ret != 0:
            print(f"获取边界失败，返回值={ret}")
            return False, 0, 0
        center_x = (minx.value + maxx.value) / 2.0
        center_y = (miny.value + maxy.value) / 2.0
        move_x = -center_x
        move_y = -center_y
        ret = ezd.lmc1_MoveEnt(ent_name, move_x, move_y)
        if ret != 0:
            print(f"移动对象失败，返回值={ret}")
            return False, 0, 0
        print(f"对象 {ent_name} 已居中，原中心 ({center_x:.2f}, {center_y:.2f})")
        return True, move_x, move_y

    # ---------- 初始化 ----------
    print("\n正在初始化板卡...")
    ret = ezd.lmc1_Initial2(os.getcwd(), False)
    check_ret(ret, "初始化SDK")

    # ---------- 加载默认参数模板 ----------
    default_template = os.path.join(os.path.dirname(sdk_dir), "template", "template.ezd")
    if os.path.isfile(default_template):
        print(f"\n加载参数模板：{default_template}")
        ret = ezd.lmc1_LoadEzdFile(default_template)
        check_ret(ret, "加载参数模板")
        # 清空模板中的图形对象，只保留设备参数
        ret = ezd.lmc1_ClearEntLib()
        check_ret(ret, "清空图形对象")
    else:
        print("\n警告：未找到默认参数模板，将使用 SDK 默认参数。")

    # ---------- 调试模式：弹出设备参数对话框并显示笔号参数 ----------
    if args.debug:
        print("\n--- 调试模式：弹出设备参数对话框，请查看红光指示输出口、开始标刻端口等配置 ---")
        print("关闭对话框后继续执行...")
        ret = ezd.lmc1_SetDevCfg()
        check_ret(ret, "打开设备参数对话框")
        print("设备参数对话框已关闭，继续执行。")

        print("\n--- 当前加工参数（笔号0）---")
        # 准备变量
        pen_no = 0
        pen_name = ctypes.create_unicode_buffer(256)
        clr = ctypes.c_int()
        b_disable = ctypes.c_bool()
        b_use_def = ctypes.c_bool()
        n_mark_loop = ctypes.c_int()
        d_mark_speed = ctypes.c_double()
        d_power_ratio = ctypes.c_double()
        d_current = ctypes.c_double()
        n_freq = ctypes.c_int()
        d_q_pulse_width = ctypes.c_double()
        n_start_tc = ctypes.c_int()
        n_laser_off_tc = ctypes.c_int()
        n_end_tc = ctypes.c_int()
        n_poly_tc = ctypes.c_int()
        d_jump_speed = ctypes.c_double()
        n_min_jump_delay = ctypes.c_int()
        n_max_jump_delay = ctypes.c_int()
        d_jump_len_limit = ctypes.c_double()
        d_point_time = ctypes.c_double()
        b_spi_cont = ctypes.c_bool()
        n_spi_wave = ctypes.c_int()
        n_yag_mode = ctypes.c_int()
        b_pulse_point = ctypes.c_bool()
        n_pulse_num = ctypes.c_int()
        b_enable_acc = ctypes.c_bool()
        d_end_comp = ctypes.c_double()
        d_acc_dist = ctypes.c_double()
        d_break_angle = ctypes.c_double()
        b_wobble_mode = ctypes.c_bool()
        d_wobble_dia = ctypes.c_double()
        d_wobble_dist = ctypes.c_double()

        ret = ezd.lmc1_GetPenParam4(
            pen_no, pen_name, ctypes.byref(clr), ctypes.byref(b_disable),
            ctypes.byref(b_use_def), ctypes.byref(n_mark_loop), ctypes.byref(d_mark_speed),
            ctypes.byref(d_power_ratio), ctypes.byref(d_current), ctypes.byref(n_freq),
            ctypes.byref(d_q_pulse_width), ctypes.byref(n_start_tc), ctypes.byref(n_laser_off_tc),
            ctypes.byref(n_end_tc), ctypes.byref(n_poly_tc), ctypes.byref(d_jump_speed),
            ctypes.byref(n_min_jump_delay), ctypes.byref(n_max_jump_delay), ctypes.byref(d_jump_len_limit),
            ctypes.byref(d_point_time), ctypes.byref(b_spi_cont), ctypes.byref(n_spi_wave),
            ctypes.byref(n_yag_mode), ctypes.byref(b_pulse_point), ctypes.byref(n_pulse_num),
            ctypes.byref(b_enable_acc), ctypes.byref(d_end_comp), ctypes.byref(d_acc_dist),
            ctypes.byref(d_break_angle), ctypes.byref(b_wobble_mode), ctypes.byref(d_wobble_dia),
            ctypes.byref(d_wobble_dist)
        )
        check_ret(ret, "读取笔号0参数")

        # 打印关键参数
        print(f"  笔名称: {pen_name.value}")
        print(f"  加工次数: {n_mark_loop.value}")
        print(f"  标刻速度: {d_mark_speed.value:.2f} mm/s")
        print(f"  功率百分比: {d_power_ratio.value:.1f} %")
        print(f"  电流: {d_current.value:.2f} A")
        print(f"  频率: {n_freq.value} Hz")
        print(f"  Q脉冲宽度: {d_q_pulse_width.value:.1f} us")
        print(f"  开光延时: {n_start_tc.value} us")
        print(f"  关光延时: {n_laser_off_tc.value} us")
        print(f"  结束延时: {n_end_tc.value} us")
        print(f"  拐角延时: {n_poly_tc.value} us")
        print(f"  跳转速度: {d_jump_speed.value:.2f} mm/s")
        print(f"  打点时间: {d_point_time.value:.2f} ms")
        print("--- 参数显示完毕 ---\n")

    # ---------- 导入 DXF 并居中 ----------
    print("\n导入 DXF 文件...")
    ok, _, _ = import_dxf_and_center(args.dxf_path, "Content")
    if not ok:
        ezd.lmc1_Close()
        sys.exit(1)

    # ---------- 执行操作 ----------
    # 如果指定了预览次数和标刻次数，执行多轮
    if args.preview_count is not None or args.mark_count is not None:
        preview_count = args.preview_count if args.preview_count else 0
        mark_count = args.mark_count if args.mark_count else 0

        for i in range(preview_count):
            print(f"\n红光预览 ({i+1}/{preview_count})...")
            ret = ezd.lmc1_RedLightMarkContour()
            check_ret(ret, f"红光预览 {i+1}")
            time.sleep(0.5)  # 短暂间隔，让红光显示明显

        for i in range(mark_count):
            print(f"\n实际标刻 ({i+1}/{mark_count})...")
            ret = ezd.lmc1_Mark(False)
            check_ret(ret, f"标刻 {i+1}")
            time.sleep(1)   # 标刻间隔，避免过热（可根据需要调整）

    # 否则根据原有参数执行单次
    elif args.preview:
        print("\n红光预览...")
        ret = ezd.lmc1_RedLightMarkContour()
        check_ret(ret, "红光预览")
        time.sleep(2)  # 保持2秒便于观察
    elif args.mark:
        print("\n开始标刻...")
        ret = ezd.lmc1_Mark(False)
        check_ret(ret, "标刻完成")
    else:
        # 默认：预览+标刻各一次（保持兼容）
        print("\n红光预览...")
        ret = ezd.lmc1_RedLightMarkContour()
        check_ret(ret, "红光预览")
        time.sleep(2)
        print("\n开始标刻...")
        ret = ezd.lmc1_Mark(False)
        check_ret(ret, "标刻完成")

    ezd.lmc1_Close()
    print("\n所有操作完成，板卡已关闭。")

if __name__ == "__main__":
    main()