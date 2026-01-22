import os
import math

def create_chap2_remaining_diagrams():
    # --- 1. デザイン設定 (統一) ---
    colors = {
        "bg": "#FFFFFF",
        "node_fill": "#FFFFFF",
        "node_stroke": "#2D3436",
        "link_normal": "#B2BEC3",      
        "link_text": "#636E72",
        "text_main": "#2D3436",
        "text_sub": "#636E72",
        
        "accent_blue": "#0984E3",      # 実行可能解・パス (青)
        "accent_red": "#D63031",       # 制約違反・ボトルネック (赤)
        "feasible_bg": "#E0F7FA",      # 実行可能ラベル背景
        "violation_bg": "#FFEBEE"      # 違反ラベル背景
    }

    font_family = "'Yu Gothic', 'Meiryo', 'Hiragino Kaku Gothic ProN', sans-serif"

    def get_svg_header(w, h):
        return [
            '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="background-color:{colors["bg"]}; font-family:{font_family};">',
            '<defs>',
            '<filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur in="SourceAlpha" stdDeviation="2"/><feOffset dx="1" dy="1" result="offsetblur"/><feComponentTransfer><feFuncA type="linear" slope="0.3"/></feComponentTransfer><feMerge><feMergeNode/><feMergeNode in="SourceGraphic"/></feMerge></filter>',
            f'<marker id="arrowBlue" markerWidth="10" markerHeight="7" refX="24" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="{colors["accent_blue"]}" /></marker>',
            f'<marker id="arrowRed" markerWidth="10" markerHeight="7" refX="24" refY="3.5" orient="auto"><polygon points="0 0, 10 3.5, 0 7" fill="{colors["accent_red"]}" /></marker>',
            '</defs>'
        ]

    def save_svg(filename, content):
        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(content + ['</svg>']))
        print(f"Generated: {filename}")

    # ==========================================
    # 図2.5 MBL概念図 (横一列のパス)
    # ==========================================
    def create_fig2_5_mbl():
        w, h = 800, 350
        svg = get_svg_header(w, h)
        
        # ノード配置 (S -> v1 -> v2 -> D)
        y_pos = 180
        nodes = [
            ("S", 100, y_pos),
            ("v1", 300, y_pos),
            ("v2", 500, y_pos),
            ("D", 700, y_pos)
        ]
        
        # リンクデータ (帯域幅Mbps)
        # S-v1: 200, v1-v2: 100(Bottle), v2-D: 120
        links = [200, 100, 120]
        
        # リンク描画
        for i, bw in enumerate(links):
            u_node = nodes[i]
            v_node = nodes[i+1]
            
            # ボトルネックのみ赤、他は青
            is_bottleneck = (bw == 100)
            color = colors["accent_red"] if is_bottleneck else colors["accent_blue"]
            marker = "arrowRed" if is_bottleneck else "arrowBlue"
            
            # 帯域幅を線の太さで表現 (スケーリング)
            stroke_width = bw / 10 
            
            x1, y1 = u_node[1], u_node[2]
            x2, y2 = v_node[1], v_node[2]
            
            # パスとしての矢印
            svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{stroke_width}" marker-end="url(#{marker})" opacity="0.8" />')
            
            # ラベル
            cx = (x1 + x2) / 2
            svg.append(f'<text x="{cx}" y="{y1 - 30}" font-size="16" font-weight="bold" fill="{color}" text-anchor="middle">{bw} Mbps</text>')
            
            if is_bottleneck:
                svg.append(f'<text x="{cx}" y="{y1 + 40}" font-size="14" font-weight="bold" fill="{colors["accent_red"]}" text-anchor="middle">ボトルネック (最小値)</text>')

        # ノード描画
        for label, nx, ny in nodes:
            svg.append(f'<circle cx="{nx}" cy="{ny}" r="22" fill="{colors["node_fill"]}" stroke="{colors["node_stroke"]}" stroke-width="2" filter="url(#softShadow)" />')
            svg.append(f'<text x="{nx}" y="{ny}" dy="6" font-size="16" font-weight="bold" fill="{colors["text_main"]}" text-anchor="middle">{label}</text>')
        
        # Start/Goalラベル
        svg.append(f'<text x="{nodes[0][1]}" y="{nodes[0][2]+45}" font-size="14" font-weight="bold" fill="{colors["text_main"]}" text-anchor="middle">Start</text>')
        svg.append(f'<text x="{nodes[-1][1]}" y="{nodes[-1][2]+45}" font-size="14" font-weight="bold" fill="{colors["text_main"]}" text-anchor="middle">Goal</text>')

        save_svg("fig2_5_mbl_concept.svg", svg)

    # ==========================================
    # 図2.7 MCOP具体例 (並列経路 A vs B)
    # ==========================================
    def create_fig2_7_mcop():
        w, h = 800, 500
        svg = get_svg_header(w, h)
        
        # 始点・終点
        start_x, end_x = 100, 700
        mid_y = 250
        
        # 経路A (上, 赤, 違反)
        path_a_y = 120
        # 中継ノード座標
        path_a_nodes = [
            ("S", start_x, mid_y),
            ("a1", 300, path_a_y),
            ("a2", 500, path_a_y),
            ("D", end_x, mid_y)
        ]
        
        # 経路Aデータ: (帯域, 遅延)
        # S->a1: 200M, 10ms
        # a1->a2: 100M, 10ms (Bottle)
        # a2->D: 150M, 10ms
        data_a = [
            (200, 10), (100, 10), (150, 10)
        ]
        
        # 経路B (下, 青, 実行可能)
        path_b_y = 380
        path_b_nodes = [
            ("S", start_x, mid_y),
            ("b1", 300, path_b_y),
            ("b2", 500, path_b_y),
            ("D", end_x, mid_y)
        ]
        
        # 経路Bデータ
        # S->b1: 80M, 5ms (Bottle)
        # b1->b2: 120M, 5ms
        # b2->D: 130M, 5ms
        data_b = [
            (80, 5), (120, 5), (130, 5)
        ]

        # --- 描画ループ ---
        
        # 1. 経路A描画
        for i in range(3):
            u = path_a_nodes[i]
            v = path_a_nodes[i+1]
            bw, delay = data_a[i]
            
            # リンク
            svg.append(f'<line x1="{u[1]}" y1="{u[2]}" x2="{v[1]}" y2="{v[2]}" stroke="{colors["accent_red"]}" stroke-width="3" marker-end="url(#arrowRed)" />')
            
            # ラベル (線の中点)
            cx, cy = (u[1]+v[1])/2, (u[2]+v[2])/2
            # 少し上に帯域、下に遅延
            offset_y = -15 if i != 1 else -20 # 中央は見やすく
            
            svg.append(f'<text x="{cx}" y="{cy + offset_y}" font-size="12" font-weight="bold" fill="{colors["accent_red"]}" text-anchor="middle">{bw} Mbps</text>')
            svg.append(f'<text x="{cx}" y="{cy - offset_y + 10}" font-size="12" fill="{colors["text_sub"]}" text-anchor="middle">{delay} ms</text>')

        # 2. 経路B描画
        for i in range(3):
            u = path_b_nodes[i]
            v = path_b_nodes[i+1]
            bw, delay = data_b[i]
            
            svg.append(f'<line x1="{u[1]}" y1="{u[2]}" x2="{v[1]}" y2="{v[2]}" stroke="{colors["accent_blue"]}" stroke-width="3" marker-end="url(#arrowBlue)" />')
            
            cx, cy = (u[1]+v[1])/2, (u[2]+v[2])/2
            offset_y = -15 if i != 1 else -20
            
            svg.append(f'<text x="{cx}" y="{cy + offset_y}" font-size="12" font-weight="bold" fill="{colors["accent_blue"]}" text-anchor="middle">{bw} Mbps</text>')
            svg.append(f'<text x="{cx}" y="{cy - offset_y + 10}" font-size="12" fill="{colors["text_sub"]}" text-anchor="middle">{delay} ms</text>')

        # 3. ノード描画 (S, D, 中継)
        # Setで重複排除して描画
        all_nodes = path_a_nodes + path_b_nodes
        drawn_nodes = set()
        for label, nx, ny in all_nodes:
            if (nx, ny) in drawn_nodes: continue
            drawn_nodes.add((nx, ny))
            
            svg.append(f'<circle cx="{nx}" cy="{ny}" r="20" fill="{colors["node_fill"]}" stroke="{colors["node_stroke"]}" stroke-width="2" filter="url(#softShadow)" />')
            # 中継ノードのラベルは非表示または小さく（図.pdfに合わせるならなしでもよいが、位置確認用に）
            if label in ["S", "D"]:
                svg.append(f'<text x="{nx}" y="{ny}" dy="6" font-size="16" font-weight="bold" fill="{colors["text_main"]}" text-anchor="middle">{label}</text>')
            else:
                 svg.append(f'<circle cx="{nx}" cy="{ny}" r="4" fill="{colors["text_sub"]}" />')

        # Start/Goal ラベル
        svg.append(f'<text x="{start_x}" y="{mid_y+45}" font-size="14" font-weight="bold" fill="{colors["text_main"]}" text-anchor="middle">Start</text>')
        svg.append(f'<text x="{end_x}" y="{mid_y+45}" font-size="14" font-weight="bold" fill="{colors["text_main"]}" text-anchor="middle">Goal</text>')

        # 4. 評価ボックス (図の上部・下部に配置)
        
        # 上部: 経路A評価
        # "経路A (ボトルネック 100Mbps, 遅延 30ms) -> 制約違反"
        svg.append(f'<rect x="200" y="30" width="400" height="30" rx="4" fill="{colors["violation_bg"]}" stroke="{colors["accent_red"]}" />')
        svg.append(f'<text x="400" y="50" font-size="13" font-weight="bold" fill="{colors["accent_red"]}" text-anchor="middle">経路A: Min=100Mbps, Delay=30ms (× 制約違反)</text>')

        # 下部: 経路B評価
        # "経路B (ボトルネック 80Mbps, 遅延 15ms) -> 実行可能"
        svg.append(f'<rect x="200" y="450" width="400" height="30" rx="4" fill="{colors["feasible_bg"]}" stroke="{colors["accent_blue"]}" />')
        svg.append(f'<text x="400" y="470" font-size="13" font-weight="bold" fill="{colors["accent_blue"]}" text-anchor="middle">経路B: Min=80Mbps, Delay=15ms (○ 実行可能解)</text>')

        # 制約条件等のメモ
        svg.append(f'<text x="100" y="30" font-size="12" fill="{colors["text_main"]}" text-anchor="start" font-weight="bold">[目的] ボトルネック最大化</text>')
        svg.append(f'<text x="100" y="50" font-size="12" fill="{colors["text_main"]}" text-anchor="start" font-weight="bold">[制約] 合計遅延 20ms以内</text>')

        save_svg("fig2_7_mcop_example.svg", svg)

    create_fig2_5_mbl()
    create_fig2_7_mcop()

if __name__ == "__main__":
    create_chap2_remaining_diagrams()