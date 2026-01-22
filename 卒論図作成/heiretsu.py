import html

def create_final_svg_v3(filename="parallel_model_final_v3.svg"):
    # --- 1. 設定 (Colors & Style) ---
    width, height = 800, 650
    
    colors = {
        "bg": "#FFFFFF",
        "master_fill": "#F0F4F8",
        "master_stroke": "#2D3436",
        "worker_fill": "#FFFFFF",
        "worker_stroke": "#636E72",
        "accent": "#0984E3",         # 青
        "text_main": "#2D3436",
        "text_sub": "#636E72",
        "line_sync": "#D63031",      # 赤
        "data_label": "#0984E3"      # データフロー文字色
    }

    # フォント設定
    font_family = "Yu Gothic, Meiryo, Hiragino Kaku Gothic ProN, sans-serif"

    svg_content = []

    # --- 2. SVG ヘルパー関数 ---
    def add_rect(x, y, w, h, fill, stroke, r=6, filter_id=""):
        filter_attr = f'filter="url(#{filter_id})"' if filter_id else ""
        svg_content.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" ry="{r}" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" {filter_attr} />'
        )

    def add_text(x, y, text, size=14, color=colors["text_main"], weight="normal", align="middle", bg=False):
        # ★重要★ 特殊文字（&など）をエスケープ処理
        safe_text = html.escape(text)
        
        # 背景（座布団）の描画
        if bg:
            tw = len(text) * size * 0.9
            th = size * 1.2
            if align == "middle": rect_x = x - tw/2
            elif align == "start": rect_x = x - 5
            else: rect_x = x
            svg_content.append(
                f'<rect x="{rect_x}" y="{y - size/2 - 2}" width="{tw}" height="{th}" fill="#FFFFFF" opacity="0.85" />'
            )
            
        svg_content.append(
            f'<text x="{x}" y="{y}" font-family="{font_family}" font-size="{size}" '
            f'fill="{color}" font-weight="{weight}" text-anchor="{align}" '
            f'dominant-baseline="middle">{safe_text}</text>'
        )

    def add_path(d, color, width=1.5, dash="", marker_end=""):
        attr_list = [
            f'd="{d}"',
            f'stroke="{color}"',
            f'stroke-width="{width}"',
            'fill="none"'
        ]
        if dash: attr_list.append(f'stroke-dasharray="{dash}"')
        if marker_end: attr_list.append(f'marker-end="url(#{marker_end})"')
        svg_content.append(f'<path {" ".join(attr_list)} />')

    # --- 3. 描画開始 ---
    svg_content.append('<?xml version="1.0" encoding="UTF-8" standalone="no"?>')
    svg_content.append(f'''
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="background-color:{colors['bg']};">
<defs>
    <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
        <feGaussianBlur in="SourceAlpha" stdDeviation="3"/>
        <feOffset dx="0" dy="2" result="offsetblur"/>
        <feComponentTransfer><feFuncA type="linear" slope="0.2"/></feComponentTransfer>
        <feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
    <marker id="arrowHead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
        <polygon points="0 0, 10 3.5, 0 7" fill="{colors['master_stroke']}" />
    </marker>
</defs>
    ''')

    cx = width / 2
    
    # Master (Top)
    m1_y, m1_h, m1_w = 40, 90, 380
    
    # Workers
    w_y, w_h, w_w = 220, 220, 160
    gap = 60
    worker_xs = [cx - (w_w*1.5 + gap), cx - w_w/2, cx + w_w/2 + gap]

    # Master (Bottom)
    m2_y, m2_h, m2_w = 540, 90, 380

    # --- Draw Content ---

    # 1. Master Top
    add_rect(cx - m1_w/2, m1_y, m1_w, m1_h, colors["master_fill"], colors["master_stroke"], filter_id="softShadow")
    add_text(cx, m1_y + 25, "マスタープロセス (制御・管理)", size=16, weight="bold")
    add_text(cx - 170, m1_y + 55, "1. 粒子更新 (Update Position)", size=13, align="start")
    add_text(cx - 170, m1_y + 75, "2. リスタート判定 (Stagnation Check)", size=13, align="start")

    # 2. Scatter Lines
    fork_y = (m1_y + m1_h + w_y) / 2
    add_path(f"M {cx} {m1_y + m1_h} L {cx} {fork_y}", colors["master_stroke"])
    add_path(f"M {worker_xs[0] + w_w/2} {fork_y} L {worker_xs[-1] + w_w/2} {fork_y}", colors["master_stroke"])
    
    add_text(cx + 10, fork_y - 15, "粒子データ配布 (Scatter)", size=12, color=colors["data_label"], align="start", bg=True)
    add_text(cx + 10, fork_y, "Position x", size=11, color=colors["data_label"], align="start", bg=True)

    # 3. Workers
    for i, wx in enumerate(worker_xs):
        center_w = wx + w_w/2
        add_path(f"M {center_w} {fork_y} L {center_w} {w_y}", colors["master_stroke"], marker_end="arrowHead")
        
        add_rect(wx, w_y, w_w, w_h, colors["worker_fill"], colors["worker_stroke"], filter_id="softShadow")
        title = f"Worker {i+1}" if i < 2 else "Worker N"
        add_text(center_w, w_y + 20, title, size=14, weight="bold", color=colors["accent"])

        bx, bw, bh = wx + 10, w_w - 20, 40
        by = w_y + 45
        
        add_rect(bx, by, bw, bh, "#FFF8E1", "#FFC107")
        add_text(center_w, by + 20, "経路生成 (Encode)", size=12, weight="bold")
        
        add_rect(bx, by + 50, bw, bh, "#F5F5F5", "#B0BEC5")
        add_text(center_w, by + 70, "属性・制約計算", size=12, color=colors["text_sub"])
        
        add_rect(bx, by + 100, bw, bh, "#F5F5F5", "#B0BEC5")
        add_text(center_w, by + 120, "適応度評価", size=12, color=colors["text_sub"])

        if i == 1:
             add_text(wx + w_w + gap/2, w_y + w_h/2, "...", size=30, color="#B0BEC5", weight="bold")

    # 4. Gather Lines
    join_y = (w_y + w_h + m2_y) / 2
    sync_y = join_y + 10
    
    for wx in worker_xs:
        center_w = wx + w_w/2
        add_path(f"M {center_w} {w_y + w_h} L {center_w} {join_y}", colors["master_stroke"])
    
    add_path(f"M {worker_xs[0] + w_w/2} {join_y} L {worker_xs[-1] + w_w/2} {join_y}", colors["master_stroke"])
    add_path(f"M {cx} {join_y} L {cx} {m2_y}", colors["master_stroke"], marker_end="arrowHead")

    add_text(cx + 10, join_y - 15, "評価結果集約 (Gather)", size=12, color=colors["data_label"], align="start", bg=True)
    # ↓ここが修正された箇所です
    add_text(cx + 10, join_y, "Fitness & QoS", size=11, color=colors["data_label"], align="start", bg=True)

    # 5. Sync Barrier
    barrier_w = m2_w + 40
    add_path(f"M {cx - barrier_w/2} {sync_y} L {cx + barrier_w/2} {sync_y}", colors["line_sync"], width=2, dash="6, 4")
    add_text(cx + barrier_w/2 + 10, sync_y, "Sync (同期)", size=12, color=colors["line_sync"], align="start")

    # 6. Master Bottom
    add_rect(cx - m2_w/2, m2_y, m2_w, m2_h, colors["master_fill"], colors["master_stroke"], filter_id="softShadow")
    add_text(cx, m2_y + 25, "マスタープロセス (集計)", size=16, weight="bold")
    add_text(cx - 170, m2_y + 55, "3. pBest / gBest 更新", size=13, align="start")
    add_text(cx - 170, m2_y + 75, "4. 次世代へ (Next Gen)", size=13, align="start")

    svg_content.append("</svg>")
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(svg_content))
    print(f"Generated successfully: {filename}")

if __name__ == "__main__":
    create_final_svg_v3()