import ezdxf
from typing import List
from core.contour import Contour
from ezdxf.enums import TextEntityAlignment

class DXFExporter:
    @staticmethod
    def export_to_dxf(contours: List[Contour], pixels_per_cm: float, filename: str):
        try:
            # 创建DXF文档
            doc = ezdxf.new('R2010')
            msp = doc.modelspace()

            # 添加图层
            doc.layers.add("CONTOURS", color=1)      # 红色
            doc.layers.add("DIMENSIONS", color=5)    # 蓝色
            doc.layers.add("LABELS", color=6)        # 青色，用于标号

            # 设置单位：毫米
            doc.header['$INSUNITS'] = 4

            print(f"\n=== DXF导出调试信息 ===")
            print(f"像素比例: {pixels_per_cm} 像素/厘米")
            print(f"轮廓数量: {len(contours)}")

            for i, contour in enumerate(contours):
                if not contour.nurbs_points:
                    print(f"轮廓 {i} 无NURBS点，跳过")
                    continue

                display_rect = contour.get_display_rect()
                scale = contour.scale
                bbox = contour.bounding_box

                # ---------- 导出轮廓线 ----------
                points = []
                for j, nurbs_point in enumerate(contour.nurbs_points):
                    x_px = display_rect.left() + (nurbs_point.x() - bbox.left()) * scale
                    y_px = display_rect.top() + (nurbs_point.y() - bbox.top()) * scale
                    x_mm = x_px * 10 / pixels_per_cm
                    y_mm = 150 - (y_px * 10 / pixels_per_cm)   # 翻转Y
                    points.append((x_mm, y_mm))

                if len(points) >= 2:
                    if points[0] != points[-1]:
                        points.append(points[0])

                    msp.add_lwpolyline(points, dxfattribs={
                        'layer': 'CONTOURS',
                        'closed': True,
                        'lineweight': 5
                    })

                # ---------- 导出标号（仅当轮廓有标号时）----------
                # ---------- 导出标号（仅当轮廓有标号时）----------
                if contour.label > 0:
                    # 计算轮廓显示矩形
                    display_rect = contour.get_display_rect()
                    # 计算轮廓宽度（毫米），判断是否太小
                    width_mm = display_rect.width() * 10 / pixels_per_cm
                    if width_mm < 5:
                        continue  # 跳过标号

                    # 标号中心点（显示矩形中心）
                    center_x_px = display_rect.center().x()
                    center_y_px = display_rect.center().y()

                    # 转换为毫米坐标（Y轴翻转）
                    label_x_mm = center_x_px * 10 / pixels_per_cm
                    label_y_mm = 150 - (center_y_px * 10 / pixels_per_cm)

                    # 计算字体高度（毫米）：轮廓宽度的1/5，限制在1mm～5mm
                    font_height_mm = width_mm / 5
                    min_font_mm = 1.0
                    max_font_mm = 5.0
                    font_height_mm = max(min_font_mm, min(font_height_mm, max_font_mm))

                    # 添加文本
                    text = msp.add_text(
                        str(contour.label),
                        dxfattribs={
                            'layer': 'LABELS',
                            'height': font_height_mm,
                            'color': 6,  # 青色
                        }
                    )
                    # 设置文本对齐方式为居中
                    text.set_placement((label_x_mm, label_y_mm), align=TextEntityAlignment.MIDDLE_CENTER)

                # ---------- 可选：保留原有的中心点标记 ----------
                center_x = (display_rect.left() + display_rect.width() / 2) * 10 / pixels_per_cm
                center_y = 150 - ((display_rect.top() + display_rect.height() / 2) * 10 / pixels_per_cm)
                msp.add_point((center_x, center_y), dxfattribs={'layer': 'DIMENSIONS'})

            doc.saveas(filename)
            print(f"\n成功导出 {len([c for c in contours if c.nurbs_points])} 个轮廓到 {filename}")
            return True

        except Exception as e:
            print(f"\n导出DXF失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False