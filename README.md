# 皮肤病灶轮廓处理系统

> 一站式皮肤病灶轮廓处理桌面应用——支持图像轮廓提取、NURBS曲线拟合、自动排样、激光切割模拟与云端数据管理

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyQt5](https://img.shields.io/badge/PyQt5-GUI-green)](https://www.riverbankcomputing.com/software/pyqt/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 目录

- [功能特性](#-功能特性)
- [技术栈](#-技术栈)
- [安装说明](#-安装说明)
- [快速开始](#-快速开始)
- [项目结构](#-项目结构)
- [使用指南](#-使用指南)
- [配置说明](#-配置说明)
- [常见问题](#-常见问题)
- [更新日志](#-更新日志)
- [许可证](#-许可证)
- [贡献指南](#-贡献指南)

---

## 功能特性

### 核心功能

| 功能 | 说明 |
|------|------|
| **图像轮廓提取** | 基于 OpenCV 自动检测病灶轮廓，支持闭运算预处理，自动过滤小面积噪点 |
| **NURBS/贝塞尔拟合** | 使用 NURBS 或三次贝塞尔曲线平滑拟合轮廓，动态调整控制点数量，实时预览 |
| **尺寸自动标定** | 通过检测 overlay 图像中直径10mm的白色小球自动计算像素比例尺 |
| **自动排样** | 基于 Shapely 的极坐标排样算法，将轮廓紧凑排列在直径10cm圆形画布中 |
| **激光切割模拟** | 内置激光控制器，支持模拟模式与硬件模式，实时显示切割轨迹与进度 |
| **云端数据集成** | 通过微信云开发 API 获取患者数据，批量下载 mask 与 overlay 文件 |

### 导出支持

- **DXF** - 用于激光切割机
- **JSON** - 数据交换与存档
- **图像** - PNG/JPEG/BMP 画布截图

### 交互功能

- 轮廓拖拽、缩放、非均匀拉伸
- 包围盒与控制点显示
- 原始轮廓/NURBS曲线/标号显示切换
- 多图像管理与标号映射

---

## 技术栈

```
Python 3.8+
├── PyQt5          # GUI 框架
├── OpenCV         # 图像处理
├── NumPy/SciPy    # 数值计算
├── Shapely        # 几何排样
├── geomdl         # NURBS 曲线拟合
├── ezdxf          # DXF 文件导出
└── requests       # 云端 API 调用
```

---

## 安装说明

### 环境要求

- Python 3.8 或更高版本
- Windows 10/11（推荐）

### 安装步骤

1. **克隆仓库**
   ```bash
   git clone https://github.com/Bbbbeiliu/skin_lesion_processor.git
   cd skin_lesion_processor
   ```

2. **创建虚拟环境（推荐）**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

### 依赖列表

```
PyQt5>=5.15.0
opencv-python>=4.5.0
numpy>=1.19.0
scipy>=1.5.0
shapely>=1.8.0
geomdl>=1.2.0
ezdxf>=0.17.0
requests>=2.25.0
```

---

## 快速开始

### 启动程序

```bash
python main.py
```

或使用批处理文件（Windows）：
```bash
run.bat
```

### 基本操作流程

```
1. 加载图像 → 点击"加载多个图像文件"选择 mask 图像（黑白二值图）
2. 处理图像 → 点击"处理所有图像"提取轮廓并自动标定尺寸
3. 调整轮廓 → 选中轮廓后修改尺寸、控制点数，支持拖拽与缩放
4. 自动排样 → 点击"自动排列轮廓"进行极坐标排样，自动分页
5. 导出数据 → 选择 DXF、JSON 或图像格式导出
6. 激光切割 → 切换到"激光控制"面板，选择模拟模式执行切割
```

---

## 项目结构

```
skin_lesion_processor/
├── main.py                    # 程序入口
├── setup.py                   # 安装配置
├── requirements.txt           # 依赖列表
├── laser_config.json          # 激光配置文件
├── run.bat                    # Windows 启动脚本
│
├── core/                      # 核心功能模块
│   ├── image_processor.py     # 图像处理
│   ├── contour.py             # 轮廓检测
│   ├── dxf_exporter.py        # DXF 导出
│   ├── laser_controller.py    # 激光控制
│   └── marker_detector.py     # 标记检测
│
├── ui/                        # 用户界面
│   ├── main_window.py         # 主窗口
│   ├── canvas_widget.py       # 画布组件
│   ├── control_panel.py       # 控制面板
│   └── label_mapping_dialog.py # 标号映射对话框
│
├── utils/                     # 工具函数
│
├── config/                    # 配置文件
│
├── resources/                 # 资源文件
│   ├── auto_marker/           # 自动标记模块
│   ├── ezcad_sdk/             # EZCAD SDK
│   ├── scripts/               # 脚本文件
│   └── template/              # 模板文件
│
├── output/                    # 输出目录（运行时生成）
├── workspace/                 # 工作空间（运行时生成）
└── reports/                   # 报告目录（运行时生成）
```

---

## 使用指南

### 云端数据使用

1. 点击右侧"云端数据"停靠窗口的"刷新患者列表"
2. 选择一名或多名患者
3. 选择操作方式：
   - **下载并处理选中患者** - 完全替换当前项目
   - **添加选中患者数据** - 增量添加到当前项目
4. 系统自动标定轮廓尺寸（通过 overlay 图像中的白色小球）

### 图像命名规范

- Mask 文件：`*_mask.png`（黑白二值图）
- Overlay 文件：`*_overlay.png`（包含标定小球）
- 两者需配套使用，文件名前缀需匹配

---

## 配置说明

### 激光配置 (laser_config.json)

```json
{
  "simulator_mode": true,
  "cutting_speed": 100,
  "power": 80
}
```

| 参数 | 说明 |
|------|------|
| simulator_mode | 模拟模式开关 |
| cutting_speed | 切割速度 (mm/s) |
| power | 激光功率 (%) |

---

## 常见问题

### Q: 为什么自动标定没有成功？

**A:** 请确保：
- overlay 图像中包含完整的白色小球（直径 10mm）
- 文件命名符合 `*_overlay.png` 格式
- 与 mask 文件名前缀匹配

### Q: 云端下载后程序闪退？

**A:** 该问题已通过优化线程生命周期管理和内存回收解决。如仍出现，请检查：
- 网络连接是否稳定
- 文件完整性是否正常
- Python 版本是否为 3.8+

### Q: 控制点滑块不更新？

**A:** 选中不同轮廓时，控制点滑块会自动同步当前轮廓的控制点数。如有异常请重启程序。

---

## 更新日志

### v1.2.0 (2026-06-08)

- [新增] 添加 Git 版本控制与 GitHub 备份
- [新增] 完善 README 文档与许可证框架
- [修复] 云端闪退问题
- [修复] 来源标注问题
- [修复] 清空轮廓残留问题
- [优化] 多线程下载性能

### v1.1.0

- [新增] 自动标记功能
- [新增] 标号映射表导出
- [优化] 轮廓拟合算法

### v1.0.0

- [新增] 基础轮廓处理功能
- [新增] DXF 导出功能
- [新增] 激光切割模拟

---

## 许可证

本项目采用 **MIT 许可证** - 详见 [LICENSE](LICENSE) 文件

```
MIT License

Copyright (c) 2026 Bbbbeiliu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本仓库
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 提交 Pull Request

---

## 联系方式

- 仓库地址：[https://github.com/Bbbbeiliu/skin_lesion_processor](https://github.com/Bbbbeiliu/skin_lesion_processor)
- Issue 跟踪：[GitHub Issues](https://github.com/Bbbbeiliu/skin_lesion_processor/issues)

---

<p align="center">
  <b>皮肤病灶轮廓处理系统</b> © 2026 Bbbbeiliu
</p>
