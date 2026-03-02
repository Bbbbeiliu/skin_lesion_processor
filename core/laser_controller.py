import ctypes
import os
import json
import random
import time
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import threading


class LaserController:
    """激光切割控制器 - 修复状态管理"""

    def __init__(self):
        self.dll_loaded = False
        self.initialized = False
        self.is_marking = False  # 直接使用普通变量
        self.stop_requested = False
        self.hardware_available = False
        self.simulation_mode = True

        # 加载配置
        self.config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """加载激光参数配置"""
        config_path = Path("laser_config.json")
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            return {
                "operating_mode": "simulation",
                "simulation_settings": {"cutting_time": 5.0}
            }

    def initialize(self) -> Tuple[bool, str]:
        """初始化激光控制卡"""
        if self.simulation_mode:
            self.initialized = True
            return True, "模拟模式已就绪"
        return False, "无需初始化模拟模式"

    def load_and_execute_dxf(self, dxf_path: str, callback=None) -> Tuple[bool, str]:
        """
        加载DXF文件并执行切割
        修复：确保状态正确管理
        """
        if self.is_marking:
            return False, "正在切割中，请等待完成"

        if not os.path.exists(dxf_path):
            return False, f"DXF文件不存在: {dxf_path}"

        # 清除之前的停止请求
        self.stop_requested = False

        # 在新线程中执行
        thread = threading.Thread(
            target=self._execute_cutting,
            args=(dxf_path, callback),
            daemon=True
        )
        thread.start()

        mode = "模拟" if self.simulation_mode else "硬件"
        return True, f"{mode}切割已开始"

    def _execute_cutting(self, dxf_path: str, callback=None):
        """执行切割（核心修复）"""
        try:
            # 1. 设置切割状态
            self.is_marking = True

            if callback:
                callback("开始切割...")

            # 2. 执行切割（模拟或真实）
            if self.simulation_mode:
                self._simulate_cut(dxf_path, callback)
            else:
                self._real_cut(dxf_path, callback)

        except Exception as e:
            if callback:
                callback(f"切割出错: {str(e)}")
        finally:
            # 3. 关键修复：无论如何都重置状态
            time.sleep(0.1)  # 小延迟确保回调完成
            self.is_marking = False
            self.stop_requested = False

            # 4. 发送最终状态消息
            if callback and not self.stop_requested:
                # 再发一次完成消息，确保UI收到
                time.sleep(0.05)
                callback("切割已完成")

    def _simulate_cut(self, dxf_path: str, callback):
        """增强版模拟切割 - 基于图形的真实模拟"""
        try:
            # 1. 分析DXF文件，获取基本信息
            file_info = self._analyze_dxf_file(dxf_path)

            if callback:
                callback(f"分析文件: {file_info['entity_count']}个图形元素")
                time.sleep(0.5)

            # 2. 计算切割长度和时间
            if file_info['total_length'] > 0:
                # 根据长度和速度计算时间
                speed = self.config.get("pen_params", {}).get("speed", 100)  # mm/s
                estimated_time = file_info['total_length'] / speed
                estimated_time = max(3, min(60, estimated_time))  # 限制在3-60秒
            else:
                estimated_time = 5.0

            if callback:
                callback(f"预计切割时间: {estimated_time:.1f}秒")
                time.sleep(0.5)

            # 3. 模拟切割过程（更真实的进度）
            steps = int(estimated_time * 4)  # 每0.25秒一个进度点

            for i in range(steps + 1):
                if self.stop_requested:
                    if callback:
                        callback("切割已取消")
                    return

                time.sleep(0.25)

                # 模拟不同阶段的反馈
                if callback and steps > 0:
                    progress = int(i * 100 / steps)

                    # 不同的进度阶段显示不同的信息
                    if progress < 30:
                        status = "初始化振镜..."
                    elif progress < 60:
                        status = "激光预热..."
                    elif progress < 80:
                        status = "轮廓切割..."
                    else:
                        status = "收尾处理..."

                    # 模拟随机事件
                    if random.random() < 0.1 and progress > 20 and progress < 90:
                        callback(f"{status} {progress}% (焦点调整)")
                        time.sleep(0.2)
                    else:
                        callback(f"{status} {progress}%")

            # 4. 模拟完成后的处理
            if callback:
                time.sleep(0.3)
                callback("切割完成，检查质量...")
                time.sleep(0.3)

                # 模拟质量检查结果
                if random.random() < 0.9:  # 90%成功率
                    callback("✓ 切割质量: 优秀")
                else:
                    callback("⚠ 切割质量: 一般（边缘轻微毛刺）")

                time.sleep(0.3)
                callback("模拟切割完成！")

                # 生成模拟报告
                self._generate_simulation_report(dxf_path, file_info, estimated_time)

        except Exception as e:
            if callback:
                callback(f"模拟出错: {str(e)}")

    def _real_cut(self, dxf_path: str, callback):
        """真实硬件切割"""
        # 简化的硬件调用
        # 这里应该是调用EZCAD DLL的真实代码
        if callback:
            callback("硬件切割模式（暂未实现）")
        time.sleep(3)  # 模拟硬件处理时间

    def stop_cutting(self):
        """停止切割"""
        if self.is_marking:
            self.stop_requested = True
            # 如果是硬件模式，调用停止函数
            if not self.simulation_mode and hasattr(self, 'ezd'):
                try:
                    self.ezd.lmc1_StopMark()
                except:
                    pass

    def shutdown(self):
        """关闭控制器"""
        if self.is_marking:
            self.stop_cutting()
            time.sleep(0.5)

        if self.dll_loaded and hasattr(self, 'ezd'):
            try:
                self.ezd.lmc1_Close()
            except:
                pass

    def __del__(self):
        self.shutdown()

    def get_status_info(self):
        """获取状态信息"""
        return {
            "is_marking": self.is_marking,
            "simulation_mode": self.simulation_mode,
            "stop_requested": self.stop_requested
        }

    def _analyze_dxf_file(self, dxf_path: str) -> dict:
        """分析DXF文件，获取基本信息"""
        try:
            import os

            # 尝试用简单方法分析DXF（如果不复杂的话）
            file_size = os.path.getsize(dxf_path)

            # 读取文件前几行，尝试识别实体
            entity_count = 0
            total_length = 0

            try:
                with open(dxf_path, 'r', encoding='utf-8', errors='ignore') as f:
                    lines = f.readlines()

                    # 简单的DXF解析（仅用于演示）
                    for i, line in enumerate(lines):
                        if "LINE" in line.upper() and i < len(lines) - 4:
                            entity_count += 1
                            total_length += 10  # 假设每条线10mm
                        elif "CIRCLE" in line.upper():
                            entity_count += 1
                            total_length += 31.4  # 假设圆周长31.4mm
                        elif "ARC" in line.upper():
                            entity_count += 1
                            total_length += 15.7  # 假设弧长15.7mm
                        elif "POLYLINE" in line.upper():
                            entity_count += 1
                            total_length += 50  # 假设多段线50mm

            except:
                # 如果解析失败，使用估算
                entity_count = max(1, file_size // 500)
                total_length = entity_count * 20

            return {
                "file_size_kb": file_size / 1024,
                "entity_count": entity_count,
                "total_length": total_length,
                "estimated_time": total_length / 100  # 假设100mm/s速度
            }

        except:
            # 默认值
            return {
                "file_size_kb": 100,
                "entity_count": 10,
                "total_length": 500,
                "estimated_time": 5
            }

    def _generate_simulation_report(self, dxf_path: str, file_info: dict, actual_time: float):
        """生成模拟切割报告"""
        import json
        from datetime import datetime

        report = {
            "timestamp": datetime.now().isoformat(),
            "dxf_file": dxf_path,
            "simulation_mode": True,
            "file_info": file_info,
            "cutting_parameters": {
                "speed": self.config.get("pen_params", {}).get("speed", 100),
                "power": self.config.get("pen_params", {}).get("power", 80),
                "frequency": self.config.get("pen_params", {}).get("frequency", 20000)
            },
            "simulation_results": {
                "estimated_time": file_info.get("estimated_time", 5),
                "actual_time": actual_time,
                "success": True,
                "quality": "优秀" if random.random() < 0.9 else "良好"
            }
        }

        # 保存报告
        report_dir = Path("reports")
        report_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_dir / f"simulation_{timestamp}.json"

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print(f"模拟报告已保存: {report_file}")

        # 生成简短的文本报告
        text_report = f"""模拟切割报告
    ================
    时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    文件: {Path(dxf_path).name}
    图形元素: {file_info['entity_count']}个
    切割长度: {file_info['total_length']:.1f}mm
    切割时间: {actual_time:.1f}秒
    切割速度: {self.config.get('pen_params', {}).get('speed', 100)}mm/s
    切割功率: {self.config.get('pen_params', {}).get('power', 80)}%
    质量评级: {report['simulation_results']['quality']}
    """

        text_file = report_dir / f"simulation_{timestamp}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            f.write(text_report)

        return str(report_file)
