@echo off
set PYTHONIOENCODING=utf-8
echo 正在调试底层标刻脚本...
echo 使用 DXF 文件: ..\received_files\test.dxf
echo 将加载模板并弹出设备参数对话框，请查看参数配置。
echo 关闭对话框后，将显示当前加工参数，然后红光预览（重复10次）。
echo 预览结束后，按 Enter 键将执行 3 次实际标刻。
echo.
rem 当前目录已经是 resources\scripts，直接运行同目录下的 mark_automation.py
..\python32\python.exe mark_automation.py ..\received_files\test.dxf --preview --debug
echo.
echo 脚本执行完毕，按任意键关闭窗口...
pause > nul