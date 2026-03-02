"""
DXF导出器
"""
import traceback
from typing import List
from core.contour import Contour


class DXFExporter:
    """DXF导出器"""

    @staticmethod
    def export_to_dxf(contours: List[Contour], pixels_per_cm: float, filename: str):
        """将轮廓导出为DXF文件"""
        try:
            try:
                import ezdxf
            except ImportError:
                raise ImportError("需要安装ezdxf库才能导出DXF文件。请运行: pip install ezdxf")

            # 创建DXF文档
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()

            # 添加图层
            doc.layers.add("CONTOURS", color=1)
            doc.layers.add("DIMENSIONS", color=5)

            # 设置单位：毫米
            doc.header['$INSUNITS'] = 4

            # 添加调试信息
            print(f"\n=== DXF导出调试信息 ===")
            print(f"像素比例: {pixels_per_cm} 像素/厘米")
            print(f"轮廓数量: {len(contours)}")

            # 添加每个轮廓
            for i, contour in enumerate(contours):
                if not contour.nurbs_points:
                    print(f"轮廓 {i} 无NURBS点，跳过")
                    continue

                # 获取轮廓在画布上的显示矩形（这是自动排列后的位置）
                display_rect = contour.get_display_rect()

                print(f"轮廓 {i} (标号 {contour.label}):")
                print(f"  画布位置: ({display_rect.x():.1f}, {display_rect.y():.1f}) 像素")
                print(f"  画布尺寸: {display_rect.width():.1f}x{display_rect.height():.1f} 像素")
                print(f"  轮廓缩放: {contour.scale:.3f}")
                print(f"  NURBS点数: {len(contour.nurbs_points)}")

                # 将NURBS点转换为DXF点（毫米单位）
                points = []

                for j, nurbs_point in enumerate(contour.nurbs_points):
                    # 关键步骤：直接将NURBS点转换为画布上的显示坐标
                    # 1. 获取轮廓的变换参数
                    scale = contour.scale
                    bbox = contour.bounding_box

                    # 2. 将NURBS点从局部坐标转换为画布上的全局坐标
                    # 公式：画布坐标 = position + (nurbs_point - bbox_左上角) * scale
                    x_px = display_rect.left() + (nurbs_point.x() - bbox.left()) * scale
                    y_px = display_rect.top() + (nurbs_point.y() - bbox.top()) * scale

                    # 3. 转换为毫米（画布1cm = 10mm）
                    # 注意：这里使用统一的像素比例尺转换
                    x_mm = x_px * 10 / pixels_per_cm
                    y_mm = y_px * 10 / pixels_per_cm

                    # 4. DXF中Y轴向上为正，屏幕Y轴向下为正，需要翻转
                    # 画布高度是15cm = 150mm
                    y_mm = 150 - y_mm

                    points.append((x_mm, y_mm))

                    # 打印前几个点用于调试
                    if j < 3:
                        print(
                            f"    点 {j}: 局部({nurbs_point.x():.1f}, {nurbs_point.y():.1f}) -> 画布({x_px:.1f}, {y_px:.1f}) -> DXF({x_mm:.1f}, {y_mm:.1f})")

                # 创建闭合的LWPolyline
                if len(points) >= 2:
                    # 确保闭合
                    if points[0] != points[-1]:
                        points.append(points[0])

                    msp.add_lwpolyline(points, dxfattribs={
                        'layer': 'CONTOURS',
                        'closed': True,
                        'lineweight': 5  # 线宽设置为0.05毫米
                    })

                    print(f"  导出成功: {len(points)} 个点")

                # 添加轮廓中心点标记（可选，用于调试）
                center_x = (display_rect.left() + display_rect.width() / 2) * 10 / pixels_per_cm
                center_y = 150 - ((display_rect.top() + display_rect.height() / 2) * 10 / pixels_per_cm)

                msp.add_point((center_x, center_y), dxfattribs={'layer': 'DIMENSIONS'})

            # 保存文件
            doc.saveas(filename)
            print(f"\n成功导出 {len([c for c in contours if c.nurbs_points])} 个轮廓到 {filename}")
            print(f"DXF文件尺寸: 150mm x 150mm (15cm x 15cm)")

            return True

        except Exception as e:
            print(f"\n导出DXF失败: {str(e)}")
            traceback.print_exc()
            return False