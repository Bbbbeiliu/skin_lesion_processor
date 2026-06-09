"""
DXF导出器
"""
import traceback
from typing import List
from core.contour import Contour


class DXFExporter:
    """DXF导出器"""

    @staticmethod
    def export_to_dxf(contours: List[Contour], pixels_per_cm: float, filename: str,
                      label_font_size_mm: float = 5.0, label_min_size_mm: float = 5.0):
        """将轮廓导出为DXF文件（整体缩小至原尺寸的1/2.3）"""
        try:
            import ezdxf
            from ezdxf.enums import TextEntityAlignment
        except ImportError:
            raise ImportError("需要安装ezdxf库才能导出DXF文件。请运行: pip install ezdxf")

        # 缩放因子：1/2.3 ≈ 0.4347826087
        SCALE_FACTOR = 1.0 / 2.3

        # 创建DXF文档
        doc = ezdxf.new('R2010')
        msp = doc.modelspace()

        # 添加图层
        doc.layers.add("CONTOURS", color=1)
        doc.layers.add("DIMENSIONS", color=5)
        doc.layers.add("LABELS", color=7)  # 黑色，与画布一致

        # 设置单位：毫米
        doc.header['$INSUNITS'] = 4

        # 添加调试信息
        print(f"\n=== DXF导出调试信息 ===")
        print(f"像素比例: {pixels_per_cm} 像素/厘米")
        print(f"轮廓数量: {len(contours)}")
        print(f"标号字体: {label_font_size_mm}mm, 最小尺寸阈值: {label_min_size_mm}mm")
        print(f"全局缩放因子: {SCALE_FACTOR:.3f} (1/2.3)")

        # 统计计数器
        exported_count = 0
        skipped_count = 0

        # 添加每个轮廓
        for i, contour in enumerate(contours):
            # 直接使用原始轮廓点进行导出
            export_points = contour.get_export_points()
            if not export_points:
                print(f"轮廓 {i} 无导出点，跳过")
                skipped_count += 1
                continue

            # 统计
            exported_count += 1

            # 获取轮廓在画布上的显示矩形
            display_rect = contour.get_display_rect()

            print(f"轮廓 {i} (标号 {contour.label}):")
            print(f"  画布位置: ({display_rect.x():.1f}, {display_rect.y():.1f}) 像素")
            print(f"  画布尺寸: {display_rect.width():.1f}x{display_rect.height():.1f} 像素")
            print(f"  轮廓缩放: {contour.scale:.3f}")
            print(f"  原始点数: {len(export_points)}")

            # 将原始轮廓点转换为DXF点（毫米单位），并整体缩放
            points = []
            scale = contour.scale
            bbox = contour.bounding_box

            for j, pt in enumerate(export_points):
                # 计算画布上的显示坐标
                x_px = display_rect.left() + (pt.x() - bbox.left()) * scale
                y_px = display_rect.top() + (pt.y() - bbox.top()) * scale

                # 转换为毫米并缩放
                x_mm = x_px * 10 / pixels_per_cm * SCALE_FACTOR
                y_mm = y_px * 10 / pixels_per_cm * SCALE_FACTOR

                # 翻转Y轴
                y_mm = (150 * SCALE_FACTOR) - y_mm

                points.append((x_mm, y_mm))

                # 打印前几个点用于调试
                if j < 3:
                    print(
                        f"    点 {j}: 局部({pt.x():.1f}, {pt.y():.1f}) -> 画布({x_px:.1f}, {y_px:.1f}) -> 缩放后DXF({x_mm:.1f}, {y_mm:.1f})")

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

            # ---------- 导出标号（与画布逻辑一致）----------
            if contour.label > 0:
                # 使用新的定位方法：标签在轮廓下方
                label_pos = contour.get_label_position_below(
                    pixels_per_cm=pixels_per_cm,
                    font_size_mm=label_font_size_mm,
                    offset_mm=contour.label_offset_mm
                )

                if label_pos.x() != 0 or label_pos.y() != 0:
                    label_x_mm = label_pos.x() * 10 / pixels_per_cm * SCALE_FACTOR
                    label_y_mm = (150 - label_pos.y() * 10 / pixels_per_cm) * SCALE_FACTOR

                    # 获取完整标签文本（包含病人名字和序号）
                    label_text = contour.get_full_label_text()

                    text = msp.add_text(
                        label_text,
                        dxfattribs={
                            'layer': 'LABELS',
                            'height': label_font_size_mm * SCALE_FACTOR,  # 标号字体高度也缩放
                            'color': 7,  # 黑色
                        }
                    )
                    text.set_placement((label_x_mm, label_y_mm), align=TextEntityAlignment.MIDDLE_CENTER)
                    print(f"  标号 '{label_text}' 已添加，缩放后位置 ({label_x_mm:.1f}, {label_y_mm:.1f})mm")
                else:
                    print(f"  轮廓 {contour.label} 无合适标号位置，跳过")

            # 添加轮廓中心点标记（可选，用于调试），也进行缩放
            center_x = (display_rect.left() + display_rect.width() / 2) * 10 / pixels_per_cm * SCALE_FACTOR
            center_y = (150 - (display_rect.top() + display_rect.height() / 2) * 10 / pixels_per_cm) * SCALE_FACTOR
            msp.add_point((center_x, center_y), dxfattribs={'layer': 'DIMENSIONS'})

        # 保存文件
        doc.saveas(filename)
        print(f"\n=== DXF导出统计 ===")
        print(f"总轮廓数: {len(contours)}, 成功导出: {exported_count}, 跳过: {skipped_count}")
        print(f"输出方式: 原始轮廓点（保留完整形状细节）")
        print(f"文件已保存: {filename}")
        print(f"缩放后DXF文件有效尺寸: 约 {150 * SCALE_FACTOR:.1f}mm x {150 * SCALE_FACTOR:.1f}mm (原150mm)")

        return True