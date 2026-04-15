# config.py
import os

# 基础路径（相对于此文件所在目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESOURCES_DIR = os.path.dirname(BASE_DIR)          # resources 目录
SDK_DIR = os.path.join(RESOURCES_DIR, 'ezcad_sdk')
SCRIPTS_DIR = os.path.join(RESOURCES_DIR, 'scripts')
RECEIVED_DIR = os.path.join(RESOURCES_DIR, 'received_files')
PYTHON32_EXE = os.path.join(RESOURCES_DIR, 'python32', 'python.exe')
DEFAULT_TEMPLATE_PATH = os.path.join(RESOURCES_DIR, 'template', 'template.ezd')

# 确保目录存在
os.makedirs(RECEIVED_DIR, exist_ok=True)

# HTTP 服务端口
SERVER_PORT = 5000

# 标刻脚本路径
MARK_SCRIPT = os.path.join(SCRIPTS_DIR, 'mark_automation.py')

# 任务执行参数（可在此调整预览和标刻次数）
PREVIEW_COUNT = 1   # 红光预览次数
MARK_COUNT = 1      # 实际标刻次数