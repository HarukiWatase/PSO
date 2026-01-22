import os
import math

def create_chap2_diagrams_final_corrected():
    # --- 1. デザイン設定 ---
    width, height = 800, 550  # 高さを少し調整
    
    colors = {
        "bg": "#FFFFFF",
        "node_fill": "#FFFFFF",
        "node_stroke": "#2D3436",
        "link_normal": "#B2BEC3",      # 無向リンク（グレー）
        "link_text": "#636E72",
        "text_main": "#2D3436",
        "text_sub": "#636E72",
        
        # ハイライト用
        "accent_blue": "#0984E3",      # パス (青)
        "accent_orange": "#E17055",    # ウォーク (オレンジ)
        "accent_green": "#00b894",     # トレイル (緑)
    }

    font_family = "'Yu Gothic', 'Meiryo', 'Hiragino Kaku Gothic ProN', sans-serif"

    # --- 2. SVG ヘルパー ---
    def get_svg_header(w, h):
        return [
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="background-color:{colors["bg"]}; font-family:{font_family};">',
            '<defs>',
            # ドロップシャドウ
            '<filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur in="SourceAlpha" stdDeviation="2"/><feOffset dx="1" dy="1" result="offsetblur"/><feComponentTransfer><feFuncA type="linear" slope="0.3"/></feComponentTransfer><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
            # 経路用マーカー（矢印）
            f'<marker id="arrowBlue" markerWidth="10" markerHeight="7" refX="24" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="{colors["accent_blue"]}" /></marker>',
            f'<marker id="arrowOrange" markerWidth="10" markerHeight="7" refX="24" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="{colors["accent_orange"]}" /></marker>',
            f'<marker id="arrowGreen" markerWidth="10" markerHeight="7" refX="24" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="{colors["accent_green"]}" /></marker>',
            '</defs>'
        ]

    def save_svg(filename, content):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(content + ['</svg>']))
        print(f"Generated: {filename}")

    # --- 3. グラフデータ定義 ---
    # ノード座標（全体を少し上にシフトして下部スペースを確保）
    offset_y = -30
    nodes = {
        "d": (400, 300 + offset_y), "b": (400, 160 + offset_y), "f": (400, 440 + offset_y),
        "a": (260, 200 + offset_y), "c": (540, 200 + offset_y),
        "e": (260, 400 + offset_y), "g": (540, 400 + offset_y)
    }

    # リンク定義 (u, v, weight)
    links_data = [
        ("e", "a", 2), ("a", "b", 1), ("b", "d", 3),
        ("d", "c", 4), ("c", "g", 5), ("g", "f", 8),
        ("f", "e", 7), ("d", "e", 6)
    ]

    # --- 描画関数 ---
    def draw_graph_scene(svg, path_sequence=[], highlight_color=colors["accent_blue"], show_labels=True):
        
        # 1. ベースの無向リンクを描画
        for u, v, w in links_data:
            x1, y1 = nodes[u]
            x2, y2 = nodes[v]
            
            # 線
            svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colors["link_normal"]}" stroke-width="1.5" />')
            
            # 重みラベル
            mx, my = (x1+x2)/2, (y1+y2)/2
            dx, dy = x2-x1, y2-y1
            dist = math.sqrt(dx*dx + dy*dy)
            ox, oy = -dy/dist * 15, dx/dist * 15
            
            svg.append(f'<rect x="{mx+ox-10}" y="{my+oy-10}" width="20" height="20" rx="4" fill="{colors["bg"]}" opacity="0.9" />')
            svg.append(f'<text x="{mx+ox}" y="{my+oy}" dy="5" font-size="12" font-weight="bold" fill="{colors["link_text"]}" text-anchor="middle">{w}</text>')

        # 2. 経路の描画（矢印付き）
        if path_sequence:
            marker = "arrowBlue"
            if highlight_color == colors["accent_orange"]: marker = "arrowOrange"
            elif highlight_color == colors["accent_green"]: marker = "arrowGreen"

            for i in range(len(path_sequence) - 1):
                u_node, v_node = path_sequence[i], path_sequence[i+1]
                x1, y1 = nodes[u_node]
                x2, y2 = nodes[v_node]
                
                svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{highlight_color}" stroke-width="3" marker-end="url(#{marker})" />')

        # 3. ノード描画
        for n, (cx, cy) in nodes.items():
            svg.append(f'<circle cx="{cx}" cy="{cy}" r="22" fill="{colors["node_fill"]}" stroke="{colors["node_stroke"]}" stroke-width="2" filter="url(#softShadow)" />')
            svg.append(f'<text x="{cx}" y="{cy}" dy="6" font-size="16" font-weight="bold" fill="{colors["text_main"]}" text-anchor="middle">{n}</text>')

        # 4. 順序バッジ & Start/Goalラベルの動的配置
        if path_sequence:
            # 訪問回数カウント用
            visit_count = {n: 0 for n in nodes}
            
            # Start/Goal の判定
            start_node = path_sequence[0]
            goal_node = path_sequence[-1]

            # 順序バッジ表示
            for i, n in enumerate(path_sequence):
                nx, ny = nodes[n]
                
                # 複数回訪問時の位置ずらし (右上に並べる)
                current_visit = visit_count[n]
                offset_x = 20 + (current_visit * 18)
                offset_y = -20
                
                visit_count[n] += 1
                
                svg.append(f'<circle cx="{nx+offset_x}" cy="{ny+offset_y}" r="9" fill="{highlight_color}" stroke="{colors["bg"]}" stroke-width="1"/>')
                svg.append(f'<text x="{nx+offset_x}" y="{ny+offset_y}" dy="4" font-size="11" font-weight="bold" fill="#FFFFFF" text-anchor="middle">{i+1}</text>')

            # Start / Goal ラベル (経路の始点と終点に配置)
            if show_labels:
                sx, sy = nodes[start_node]
                # Startはノードの下
                svg.append(f'<text x="{sx}" y="{sy+45}" font-size="14" font-weight="bold" fill="{colors["text_main"]}" text-anchor="middle">Start</text>')
                
                gx, gy = nodes[goal_node]
                # Goalはノードの上 (Startと同じノードの場合の考慮が必要だが、簡易的に上)
                # もしStart=Goalの場合（閉路）、上下に分ける
                goal_y_offset = -40
                if start_node == goal_node:
                    svg.append(f'<text x="{gx}" y="{gy+65}" font-size="14" font-weight="bold" fill="{colors["text_main"]}" text-anchor="middle">Goal</text>')
                else:
                    svg.append(f'<text x="{gx}" y="{gy-40}" font-size="14" font-weight="bold" fill="{colors["text_main"]}" text-anchor="middle">Goal</text>')

        # 経路テキストを図の下部に表示
        if path_sequence:
            path_str = " → ".join(path_sequence)
            svg.append(f'<text x="{width/2}" y="{height-30}" font-size="16" font-weight="bold" fill="{colors["text_main"]}" text-anchor="middle">経路: {path_str}</text>')
        elif show_labels: # 経路がない図2.1の場合
            # デフォルトのStart/Goal (g -> c) を表示しておく
            sx, sy = nodes["g"]
            svg.append(f'<text x="{sx}" y="{sy+45}" font-size="14" font-weight="bold" fill="{colors["text_main"]}" text-anchor="middle">Start</text>')
            cx, cy = nodes["c"]
            svg.append(f'<text x="{cx}" y="{cy-40}" font-size="14" font-weight="bold" fill="{colors["text_main"]}" text-anchor="middle">Goal</text>')


    # ==========================================
    # 図生成実行
    # ==========================================

    # 図2.1 グラフ定義
    def create_fig2_1():
        svg = get_svg_header(800, 550)
        draw_graph_scene(svg, path_sequence=[], show_labels=True)
        save_svg("fig2_1_graph_def.svg", svg)

    # 図2.2 ウォーク (Walk)
    # 経路: g -> f -> e -> a -> b -> d -> e
    # 始点: g, 終点: e
    def create_fig2_2():
        svg = get_svg_header(800, 550)
        path = ["g", "f", "e", "a", "b", "d", "e"]
        draw_graph_scene(svg, path_sequence=path, highlight_color=colors["accent_orange"])
        save_svg("fig2_2_walk.svg", svg)

    # 図2.3 トレイル (Trail)
    # 経路: d -> c -> g -> f -> e -> a -> b -> d
    # 始点: d, 終点: d (閉路)
    def create_fig2_3():
        svg = get_svg_header(800, 550)
        path = ["d", "c", "g", "f", "e", "a", "b", "d"]
        draw_graph_scene(svg, path_sequence=path, highlight_color=colors["accent_green"])
        save_svg("fig2_3_trail.svg", svg)

    # 図2.4 パス (Path)
    # 経路: a -> b -> d -> c
    # 始点: a, 終点: c
    def create_fig2_4():
        svg = get_svg_header(800, 550)
        path = ["a", "b", "d", "c"]
        draw_graph_scene(svg, path_sequence=path, highlight_color=colors["accent_blue"])
        save_svg("fig2_4_path.svg", svg)

    create_fig2_1()
    create_fig2_2()
    create_fig2_3()
    create_fig2_4()

if __name__ == "__main__":
    create_chap2_diagrams_final_corrected()