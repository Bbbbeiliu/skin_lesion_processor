# mark_automation.py - 32位Python脚本，用于自动化标刻
# -*- coding: utf-8 -*-
"""
================================================================================
脚本名称：mark_automation.py
功能描述：自动化调用金橙子EZCAD2 SDK，实现DXF文件导入、居中，并支持单独红光预览或实际标刻。
          用于替代手动点击EZCAD2界面操作，可与主程序通过子进程集成。

使用方法：
    仅红光预览：  python mark_automation.py --preview <dxf_file_path>
    仅实际加工：  python mark_automation.py --mark <dxf_file_path>
    完整流程：    python mark_automation.py <dxf_file_path>   (预览+加工)

参数说明：
    --preview   只执行红光预览，不进行标刻
    --mark      只执行实际标刻（前提是已经预览过，板卡未重新初始化）
    （无参数）   先预览后标刻，完整流程

依赖环境及目录结构要求见脚本头部注释。
================================================================================
"""
# mark_automation.py - 支持直接DXF或模板模式
# -*- coding: utf-8 -*-
import argparse
import ctypes
import os
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

def main():
    parser = argparse.ArgumentParser(description='EzCAD2 自动化标刻工具')
    parser.add_argument('dxf_path', help='DXF文件路径')
    parser.add_argument('--preview', action='store_true', help='仅红光预览')
    parser.add_argument('--mark', action='store_true', help='仅实际加工')
    parser.add_argument('--template', help='模板文件路径 (.ezd)')
    parser.add_argument('--replace', default='PLACEHOLDER', help='模板中要替换的对象名称，默认为 PLACEHOLDER')
    args = parser.parse_args()

    # 确定SDK目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sdk_dir = os.path.join(script_dir, "..", "ezcad_sdk")
    os.chdir(sdk_dir)
    print(f"工作目录切换到：{os.getcwd()}")

    # 加载 DLL
    try:
        dll_path = os.path.join(sdk_dir, "MarkEzd.dll")
        ezd = ctypes.CDLL(dll_path)
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

    ezd.lmc1_DeleteEnt.argtypes = [ctypes.c_wchar_p]
    ezd.lmc1_DeleteEnt.restype = ctypes.c_int

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

    ezd.lmc1_RedLightMark.argtypes = []
    ezd.lmc1_RedLightMark.restype = ctypes.c_int

    ezd.lmc1_Mark.argtypes = [ctypes.c_bool]
    ezd.lmc1_Mark.restype = ctypes.c_int

    ezd.lmc1_Close.argtypes = []
    ezd.lmc1_Close.restype = ctypes.c_int

    # ---------- 辅助函数 ----------
    def import_dxf_and_center(dxf_path, ent_name="AutoDXF"):
        """导入DXF并居中，返回 (是否成功, 移动X, 移动Y)"""
        ret = ezd.lmc1_AddFileToLib(dxf_path, ent_name, 0.0, 0.0, 0.0, 0, 1.0, 0, False)
        if ret != 0:
            print(f"导入DXF失败，返回值={ret}")
            return False, 0, 0
        # 获取边界
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
    ret = ezd.lmc1_Initial2(os.getcwd(), False)
    check_ret(ret, "初始化SDK")

    # 清空数据库
    ret = ezd.lmc1_ClearEntLib()
    check_ret(ret, "清空数据库")

    # ---------- 模式分支 ----------
    if args.template:
        print(f"使用模板：{args.template}")
        # 加载模板
        ret = ezd.lmc1_LoadEzdFile(args.template)
        check_ret(ret, "加载模板")
        # 删除占位对象
        ret = ezd.lmc1_DeleteEnt(args.replace)
        check_ret(ret, f"删除占位对象 {args.replace}")
        # 导入新DXF并居中
        ok, _, _ = import_dxf_and_center(args.dxf_path)
        if not ok:
            ezd.lmc1_Close()
            sys.exit(1)
    else:
        # 直接导入DXF并居中
        print("直接导入DXF模式")
        ok, _, _ = import_dxf_and_center(args.dxf_path)
        if not ok:
            ezd.lmc1_Close()
            sys.exit(1)

    # ---------- 执行操作 ----------
    if args.preview:
        print("红光预览...")
        ret = ezd.lmc1_RedLightMark()
        check_ret(ret, "红光预览")
    elif args.mark:
        print("开始标刻...")
        ret = ezd.lmc1_Mark(False)
        check_ret(ret, "标刻完成")
    else:  # 默认：预览+标刻
        print("红光预览...")
        ret = ezd.lmc1_RedLightMark()
        check_ret(ret, "红光预览")
        print("开始标刻...")
        ret = ezd.lmc1_Mark(False)
        check_ret(ret, "标刻完成")

    ezd.lmc1_Close()
    print("所有操作完成，板卡已关闭。")

if __name__ == "__main__":
    main()