@echo off
chcp 65001 >nul
title 激光切割控制系统 - 优化打包

echo ==============================================
echo     激光切割控制系统 - 优化打包脚本
echo ==============================================

echo [1/4] 分析项目依赖...
python -c "
import sys
import subprocess

# 生成实际导入的模块列表
modules_to_include = [
    'PyQt5', 'numpy', 'cv2', 'geomdl', 'ezdxf', 
    'scipy', 'PIL', 'json', 'pathlib', 'typing',
    'traceback', 'math', 'random', 'threading',
    'os', 'sys', 'time'
]

print('需要包含的核心模块:')
for mod in modules_to_include:
    print(f'  - {mod}')

# 检查这些模块是否可导入
print('\n检查模块导入状态:')
for mod in modules_to_include:
    try:
        __import__(mod)
        print(f'  ✓ {mod}')
    except ImportError as e:
        print(f'  ✗ {mod}: {e}')
"

echo.
echo [2/4] 清理之前的构建...
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist
if exist "__pycache__" rmdir /s /q __pycache__
del /q *.spec 2>nul

echo.
echo [3/4] 生成优化的spec文件...
python -c "
import sys

spec_content = '''# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.building.api import PYZ, EXE, COLLECT
from PyInstaller.building.build_main import Analysis
import os

# 排除不必要的包
excludes = [
    # 开发工具
    'pytest', 'pytest-*', 'unittest', 'doctest', 'test', 
    'setuptools', 'pip', 'wheel', 'distutils',
    
    # IDE和文档工具
    'ipython', 'jupyter', 'spyder', 'sphinx', 'docutils',
    
    # 数据科学库（未使用）
    'pandas', 'tensorflow', 'torch', 'keras', 'sklearn', 
    'statsmodels', 'networkx', 'seaborn', 'plotly',
    
    # Web和网络相关
    'flask', 'django', 'requests', 'urllib3', 'asyncio',
    'aiohttp', 'tornado', 'sqlalchemy', 'sqlite3',
    
    # 其他
    'matplotlib',  # 虽然marker_detector.py导入了，但实际代码中未使用
    'notebook', 'bokeh', 'dash',
    
    # 排除一些大型的C扩展（如果不需要）
    'scipy.sparse', 'scipy.optimize', 'scipy.signal',
]

# 只包含必要的隐藏导入
hiddenimports = [
    'PyQt5.sip',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
    'scipy.special._ufuncs_cxx',
    'scipy.special.cython_special',
    'scipy.integrate',
    'scipy._lib',
    'scipy._lib.messagestream',
    'scipy.sparse.csgraph._validation',
    'numpy.core._multiarray_umath',
    'numpy.lib.format',
    'cv2',
    'geomdl.NURBS',
    'geomdl.knotvector',
    'geomdl.fitting',
    'ezdxf.entities',
    'ezdxf.lldxf.const',
    'ezdxf.lldxf.tags',
    'ezdxf.math',
    'ezdxf.tools',
]

# 分析主文件
a = Analysis(
    ['main.py'],
    pathex=[os.getcwd()],
    binaries=[],
    datas=[
        ('resources/*', 'resources'),
        ('*.json', '.'),
        ('core/*.py', 'core'),
        ('ui/*.py', 'ui'),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=1,  # 启用优化
)

# 创建PYZ
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# 单文件EXE
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='激光切割控制系统',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # 使用UPX压缩
    console=False,
    icon='resources/laser.ico' if os.path.exists('resources/laser.ico') else None,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

print('Spec文件生成完成！')
print('排除的包:', excludes)
'''

with open('optimized.spec', 'w', encoding='utf-8') as f:
    f.write(spec_content)
print('已生成 optimized.spec 文件')
"

echo.
echo [4/4] 开始打包（使用UPX压缩）...
echo 注意：如果UPX未安装，打包体积会稍大一些
echo.

:: 检查UPX是否可用
where upx >nul 2>&1
if errorlevel 1 (
    echo [警告] UPX未找到，打包体积会较大
    echo 建议安装UPX: https://upx.github.io/
    timeout /t 3 >nul
)

:: 执行打包
pyinstaller optimized.spec --clean

echo.
echo ==============================================
echo 打包完成！
echo.
echo 生成的文件位置：dist\激光切割控制系统.exe
echo.
echo 文件大小信息：
for %%F in (dist\激光切割控制系统.exe) do (
    set size=%%~zF
    set /a size_mb=!size! / 1048576
    echo 最终文件大小: !size_mb! MB
)

echo.
echo 测试运行（10秒后自动退出）...
timeout /t 10 >nul
start "" "dist\激光切割控制系统.exe"

echo.
echo 按任意键退出...
pause >nul