import html

def create_flowchart_svg(filename="flowchart_restart_pso.svg"):
    # --- 1. 設定 (Colors & Style) ---
    width, height = 600, 950
    
    colors = {
        "bg": "#FFFFFF",
        "node_fill": "#FFFFFF",
        "node_stroke": "#2D3436",
        "start_fill": "#2D3436",     # 開始/終了（黒背景）
        "start_text": "#FFFFFF",
        "parallel_fill": "#F0F4F8",  # 並列処理ブロック背景
        "parallel_stroke": "#0984E3",
        "decision_fill": "#FFF8E1",  # 分岐（薄い黄色）
        "decision_stroke": "#F1C40F",
        "explosion_fill": "#FFEBEE", # リスタート（薄い赤）
        "explosion_stroke": "#D63031",
        "text_main": "#2D3436",
        "text_sub": "#636E72",
        "line": "#2D3436",
        "line_accent": "#D63031"     # 赤線
    }

    font_family = "Yu Gothic, Meiryo, Hiragino Kaku Gothic ProN, sans-serif"

    svg_content = []

    # --- 2. SVG ヘルパー関数 ---
    def add_rect(x, y, w, h, fill, stroke, r=6, stroke_width=1.5, filter_id=""):
        filter_attr = f'filter="url(#{filter_id})"' if filter_id else ""
        svg_content.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" {filter_attr} />'
        )

    # 菱形（Decision用）
    def add_diamond(cx, cy, w, h, fill, stroke, filter_id=""):
        points = f"{cx},{cy-h/2} {cx+w/2},{cy} {cx},{cy+h/2} {cx-w/2},{cy}"
        filter_attr = f'filter="url(#{filter_id})"' if filter_id else ""
        svg_content.append(
            f'<polygon points="{points}" fill="{fill}" stroke="{stroke}" stroke-width="1.5" {filter_attr} />'
        )

    # 角丸の長円（Start/End用）
    def add_pill(x, y, w, h, fill, stroke, filter_id=""):
        r = h / 2
        filter_attr = f'filter="url(#{filter_id})"' if filter_id else ""
        svg_content.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" {filter_attr} />'
        )

    def add_text(x, y, text, size=14, color=colors["text_main"], weight="normal", align="middle"):
        safe_text = html.escape(text)
        svg_content.append(
            f'<text x="{x}" y="{y}" font-family="{font_family}" font-size="{size}" '
            f'fill="{color}" font-weight="{weight}" text-anchor="{align}" '
            f'dominant-baseline="middle">{safe_text}</text>'
        )

    def add_arrow(x1, y1, x2, y2, color=colors["line"], width=1.5, marker="arrowHead"):
        marker_attr = f'marker-end="url(#{marker})"' if marker else ""
        svg_content.append(
            f'<path d="M {x1} {y1} L {x2} {y2}" stroke="{color}" stroke-width="{width}" '
            f'fill="none" {marker_attr} />'
        )

    def add_path(d, color=colors["line"], width=1.5, marker="arrowHead", dash=""):
        marker_attr = f'marker-end="url(#{marker})"' if marker else ""
        dash_attr = f'stroke-dasharray="{dash}"' if dash else ""
        svg_content.append(
            f'<path d="{d}" stroke="{color}" stroke-width="{width}" '
            f'fill="none" {marker_attr} {dash_attr} />'
        )

    # --- 3. 描画開始 ---
    svg_content.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    svg_content.append(f'''
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background-color:{colors['bg']};">
<defs>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur in="SourceAlpha" stdDeviation="2"/>
        <feOffset dx="0" dy="2" result="offsetblur"/>
        <feComponentTransfer><feFuncA type="linear" slope="0.2"/></feComponentTransfer>
        <feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <marker id="arrowHead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" fill="{colors['line']}" />
    </marker>
    <marker id="arrowHeadRed" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" fill="{colors['line_accent']}" />
    </marker>
</defs>
    ''')

    cx = width / 2  # 中央X座標
    
    # --- ノード定義 (位置計算) ---
    # 各ノードの高さと間隔
    y_pos = 40
    step_y = 90
    
    # 1. Start
    add_pill(cx - 60, y_pos, 120, 40, colors["start_fill"], colors["start_fill"], filter_id="softShadow")
    add_text(cx, y_pos + 20, "開始", color=colors["start_text"], weight="bold")
    
    prev_y = y_pos + 40
    y_pos += 70

    # 2. Init
    add_rect(cx - 110, y_pos, 220, 50, colors["node_fill"], colors["node_stroke"], filter_id="softShadow")
    add_text(cx, y_pos + 25, "初期化 (粒子生成, pBest)", size=13)
    add_arrow(cx, prev_y, cx, y_pos)
    
    prev_y = y_pos + 50
    y_pos += 80

    # 3. Loop Start (Decision)
    loop_start_y = y_pos
    add_diamond(cx, y_pos + 30, 180, 60, colors["decision_fill"], colors["decision_stroke"], filter_id="softShadow")
    add_text(cx, y_pos + 30, "t <= Tmax ?", size=13)
    add_arrow(cx, prev_y, cx, y_pos) # Init -> Loop
    
    prev_y = y_pos + 60
    y_pos += 90

    # 4. Penalty Update
    add_rect(cx - 100, y_pos, 200, 40, colors["node_fill"], colors["node_stroke"], filter_id="softShadow")
    add_text(cx, y_pos + 20, "動的ペナルティ更新", size=13)
    add_arrow(cx, prev_y, cx, y_pos)
    add_text(cx + 10, prev_y + 15, "Yes", size=11, align="start") # Loop Yes label

    prev_y = y_pos + 40
    y_pos += 70

    # 5. Parallel Block (Group)
    p_h = 70
    p_w = 260
    add_rect(cx - p_w/2, y_pos, p_w, p_h, colors["parallel_fill"], colors["parallel_stroke"], stroke_width=2, filter_id="softShadow")
    add_text(cx, y_pos + 25, "並列評価プロセス (Workers)", size=14, weight="bold", color=colors["parallel_stroke"])
    add_text(cx, y_pos + 50, "優先度エンコーディング → 適応度計算", size=12, color=colors["text_sub"])
    add_arrow(cx, prev_y, cx, y_pos)

    prev_y = y_pos + p_h
    y_pos += 90

    # 6. Update Best & Stagnation Count
    add_rect(cx - 130, y_pos, 260, 50, colors["node_fill"], colors["node_stroke"], filter_id="softShadow")
    add_text(cx, y_pos + 18, "pBest / gBest 更新", size=13)
    add_text(cx, y_pos + 38, "停滞カウンタ更新 (Stagnation)", size=12, color=colors["text_sub"])
    add_arrow(cx, prev_y, cx, y_pos)

    prev_y = y_pos + 50
    y_pos += 90

    # 7. Stagnation Check (Decision)
    stag_y = y_pos
    add_diamond(cx, y_pos + 40, 240, 80, colors["explosion_fill"], colors["explosion_stroke"], filter_id="softShadow")
    add_text(cx, y_pos + 30, "停滞カウンタ > 閾値?", size=13, weight="bold", color=colors["line_accent"])
    add_text(cx, y_pos + 50, "(Stagnation Check)", size=11, color=colors["line_accent"])
    add_arrow(cx, prev_y, cx, y_pos)

    # --- Branch: Explosion (Yes) ---
    # 右への分岐
    ex_x = cx + 180
    ex_y = y_pos + 20
    add_path(f"M {cx + 120} {y_pos + 40} L {ex_x} {y_pos + 40} L {ex_x} {y_pos + 80}", color=colors["line_accent"], marker="arrowHeadRed")
    add_text(cx + 140, y_pos + 30, "Yes", color=colors["line_accent"], size=12)

    # Explosion Process Node
    add_rect(ex_x - 70, y_pos + 80, 140, 50, "#FFEBEE", "#D63031", filter_id="softShadow")
    add_text(ex_x, y_pos + 98, "Explosion", size=13, weight="bold", color="#D63031")
    add_text(ex_x, y_pos + 118, "(エリート以外再初期化)", size=10, color="#D63031")

    # Merge back line
    merge_y = y_pos + 180
    add_path(f"M {ex_x} {y_pos + 130} L {ex_x} {merge_y} L {cx} {merge_y}", color=colors["line_accent"], marker="arrowHeadRed")

    # --- Branch: Normal (No) ---
    # まっすぐ下へ (No label)
    add_arrow(cx, y_pos + 80, cx, merge_y + 30) # Diamond to lBest (skipping merge point slightly visually)
    # 矢印を描き直してマージ地点を通過させる
    # Diamond Bottom -> lBest Top
    
    y_pos = merge_y + 30

    # 8. Spatial lBest
    add_rect(cx - 100, y_pos, 200, 40, colors["node_fill"], colors["node_stroke"], filter_id="softShadow")
    add_text(cx, y_pos + 20, "空間的lBest計算", size=13)
    # Noラベル
    add_text(cx + 15, stag_y + 90, "No", size=11)

    prev_y = y_pos + 40
    y_pos += 70

    # 9. Update V, X
    add_rect(cx - 110, y_pos, 220, 50, colors["node_fill"], colors["node_stroke"], filter_id="softShadow")
    add_text(cx, y_pos + 25, "速度 V・位置 X 更新", size=13)
    add_arrow(cx, prev_y, cx, y_pos)

    # Loop Back Line
    # Box Bottom -> Left -> Up -> Right -> Loop Diamond
    box_bottom_y = y_pos + 50
    left_x = cx - 200
    add_path(f"M {cx} {box_bottom_y} L {cx} {box_bottom_y + 20} L {left_x} {box_bottom_y + 20} L {left_x} {loop_start_y + 30} L {cx - 90} {loop_start_y + 30}", marker="arrowHead")
    add_text(left_x + 10, loop_start_y + 20, "Next Generation", size=11, align="start", color=colors["text_sub"])

    # --- End Branch (From Loop Diamond) ---
    # Diamond Right -> Down -> End
    end_x = cx + 220
    # Loop Diamond Right to End Node
    add_path(f"M {cx + 90} {loop_start_y + 30} L {end_x} {loop_start_y + 30} L {end_x} {900} L {cx + 60} {900}", marker="arrowHead")
    add_text(cx + 105, loop_start_y + 20, "No (Finish)", size=11, align="start")

    # 10. End Node
    add_pill(cx - 60, 880, 120, 40, colors["start_fill"], colors["start_fill"], filter_id="softShadow")
    add_text(cx, 900, "終了", color=colors["start_text"], weight="bold")


    svg_content.append("</svg>")
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_content))
    print(f"Generated successfully: {filename}")

if __name__ == "__main__":
    create_flowchart_svg()