import os

def create_mcop_concept_fixed_v2():
    # --- 1. デザイン設定 ---
    width, height = 700, 500
    
    # 配色
    colors = {
        "bg": "#FFFFFF",
        "text_main": "#2D3436",
        "text_sub": "#636E72",
        "feasible_area": "#E0F7FA",   
        "infeasible_area": "#FFEBEE", 
        "line_axis": "#2D3436",
        "line_constraint": "#D63031", 
        "point_best": "#0984E3",      
        "point_bad": "#D63031",       
        "point_normal": "#B2BEC3"     
    }
    
    font_family = "'Noto Sans CJK JP', 'Yu Gothic', 'Meiryo', sans-serif"

    # SVGヘッダー
    svg = [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background-color:{colors["bg"]}; font-family:{font_family};">',
        '<defs>',
        # ↓↓↓ ここを修正しました（先頭に f を追加） ↓↓↓
        f'<marker id="arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="{colors["line_axis"]}" /></marker>',
        f'<marker id="axis_arrow" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="{colors["line_axis"]}" /></marker>',
        # ↑↑↑↑↑↑
        '<filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur in="SourceAlpha" stdDeviation="2"/><feOffset dx="1" dy="1" result="offsetblur"/><feComponentTransfer><feFuncA type="linear" slope="0.3"/></feComponentTransfer><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
        '</defs>'
    ]

    # --- 2. レイアウト計算 ---
    margin_left = 80
    margin_right = 50
    margin_top = 50
    margin_bottom = 60

    x_origin = margin_left
    y_origin = height - margin_bottom
    x_max = width - margin_right
    y_max = margin_top

    # 制約ライン (X軸の60%の位置)
    constraint_x = x_origin + (x_max - x_origin) * 0.6

    # --- 3. 背景領域の描画 ---
    # 実行可能領域
    svg.append(f'<rect x="{x_origin}" y="{y_max}" width="{constraint_x - x_origin}" height="{y_origin - y_max}" fill="{colors["feasible_area"]}" opacity="0.6" />')
    # 制約違反領域
    svg.append(f'<rect x="{constraint_x}" y="{y_max}" width="{x_max - constraint_x}" height="{y_origin - y_max}" fill="{colors["infeasible_area"]}" opacity="0.6" />')

    # 領域ラベル (文字サイズ+2, 透明度削除)
    label_y = y_origin - 20
    svg.append(f'<text x="{x_origin + (constraint_x - x_origin)/2}" y="{label_y}" font-size="20" font-weight="bold" fill="{colors["point_best"]}" text-anchor="middle">実行可能領域</text>')
    svg.append(f'<text x="{constraint_x + (x_max - constraint_x)/2}" y="{label_y}" font-size="20" font-weight="bold" fill="{colors["point_bad"]}" text-anchor="middle">制約違反領域</text>')

    # --- 4. 軸と制約ライン ---
    # X軸 (Delay)
    svg.append(f'<line x1="{x_origin}" y1="{y_origin}" x2="{x_max}" y2="{y_origin}" stroke="{colors["line_axis"]}" stroke-width="2" marker-end="url(#arrow)" />')
    svg.append(f'<text x="{x_max}" y="{y_origin + 40}" font-size="16" font-weight="bold" text-anchor="end" fill="{colors["text_main"]}">遅延 (Delay) → 大</text>')
    
    # Y軸
    svg.append(f'<line x1="{x_origin}" y1="{y_origin}" x2="{x_origin}" y2="{y_max}" stroke="{colors["line_axis"]}" stroke-width="2" marker-end="url(#axis_arrow)" />')
    svg.append(f'<text x="{x_origin}" y="{y_max - 30}" font-size="18" font-weight="bold" text-anchor="start" fill="{colors["text_main"]}">帯域幅 (Bandwidth)</text>')
    svg.append(f'<text x="{x_origin}" y="{y_max - 10}" font-size="16" text-anchor="start" fill="{colors["text_main"]}">↑ 大</text>')
    
    # 制約ライン (D_max)
    svg.append(f'<line x1="{constraint_x}" y1="{y_max}" x2="{constraint_x}" y2="{y_origin}" stroke="{colors["line_constraint"]}" stroke-width="2" stroke-dasharray="6,4" />')
    svg.append(f'<text x="{constraint_x}" y="{y_max - 10}" font-size="18" font-weight="bold" fill="{colors["line_constraint"]}" text-anchor="middle">遅延制約 D<tspan baseline-shift="sub" font-size="smaller">max</tspan></text>')

    # --- 5. 解のプロット ---
    points = [
        (0.2, 0.3, "normal", "解C"),
        (0.35, 0.45, "normal", ""),
        (0.45, 0.2, "normal", ""),
        (0.55, 0.75, "best", "解B (最適解)"), 
        (0.75, 0.85, "bad", "解A (帯域最大だが違反)"),
        (0.8, 0.5, "bad", ""),
        (0.9, 0.3, "bad", "")
    ]

    for xr, yr, ptype, label in points:
        px = x_origin + (x_max - x_origin) * xr
        py = y_origin - (y_origin - y_max) * yr
        
        radius = 6
        fill_color = colors["point_normal"]
        stroke_width = 0
        stroke_color = "none"
        
        if ptype == "best":
            radius = 9
            fill_color = colors["point_best"]
            stroke_color = "#FFFFFF"
            stroke_width = 2
            svg.append(f'<circle cx="{px}" cy="{py}" r="{radius+4}" fill="none" stroke="{colors["point_best"]}" stroke-width="1.5" opacity="0.6" />')
        elif ptype == "bad":
            fill_color = colors["point_bad"]
            
        svg.append(f'<circle cx="{px}" cy="{py}" r="{radius}" fill="{fill_color}" stroke="{stroke_color}" stroke-width="{stroke_width}" filter="url(#softShadow)" />')
        
        if label:
            text_y = py - 15
            font_w = "normal"
            if ptype == "best":
                text_y = py - 22
                font_w = "bold"
            svg.append(f'<text x="{px}" y="{text_y}" font-size="14" font-weight="{font_w}" fill="{colors["text_main"]}" text-anchor="middle" style="text-shadow: 1px 1px 0 #FFF;">{label}</text>')

    # --- 6. 補足説明 ---
    best_pt = points[3]
    bx = x_origin + (x_max - x_origin) * best_pt[0]
    by = y_origin - (y_origin - y_max) * best_pt[1]
    
    bubble_x = bx - 130
    bubble_y = by - 30
    
    start_x, start_y = bubble_x + 50, bubble_y + 10
    end_x, end_y = bx - 15, by - 5
    
    ctrl_x = (start_x + end_x) / 2
    ctrl_y = (start_y + end_y) / 2 - 10 
    
    svg.append(f'<path d="M {start_x} {start_y} Q {ctrl_x} {ctrl_y} {end_x} {end_y}" stroke="{colors["text_sub"]}" stroke-width="1.5" fill="none" marker-end="url(#arrow)" />')
    
    svg.append(f'<text x="{bubble_x + 30}" y="{bubble_y}" font-size="15" fill="{colors["text_main"]}" text-anchor="end">制約を満たす中で<tspan x="{bubble_x + 30}" dy="1.4em" font-weight="bold" fill="{colors["point_best"]}">最も帯域が高い点</tspan></text>')

    svg.append('</svg>')
    
    with open("fig2_7_mcop_concept_v2.svg", "w", encoding="utf-8") as f:
        f.write("\n".join(svg))
    print("Generated: fig2_7_mcop_concept_v2.svg")

if __name__ == "__main__":
    create_mcop_concept_fixed_v2()