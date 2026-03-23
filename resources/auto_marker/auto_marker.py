# auto_marker.py
import threading
import sys
import os

# 确保可以导入同级模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import *
import utils
import receiver
import worker
import gui


def main():
    # 设置日志
    log = utils.setup_logger('auto_marker')
    log.info("启动 AutoMarker 程序")

    # 检查必要的文件是否存在
    if not os.path.exists(PYTHON32_EXE):
        log.error(f"32位 Python 未找到: {PYTHON32_EXE}")
        sys.exit(1)
    if not os.path.exists(MARK_SCRIPT):
        log.error(f"标刻脚本未找到: {MARK_SCRIPT}")
        sys.exit(1)

    # 任务队列（线程安全）
    task_queue = []  # 列表，每个元素是任务字典

    # 启动 HTTP 接收服务
    def start_receiver():
        receiver.start_server(task_queue, log)

    # 创建 GUI（不再传入 process_func，只传 receiver 启动函数）
    gui_instance = gui.AppGUI(task_queue, start_receiver)

    # 启动后台任务处理（只启动一次）
    threading.Thread(target=worker.process_queue, args=(task_queue, gui_instance.update_job_status),
                     daemon=True).start()

    # 运行 GUI
    gui_instance.run()


if __name__ == '__main__':
    main()