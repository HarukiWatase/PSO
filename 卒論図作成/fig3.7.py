import matplotlib.pyplot as plt
import math

# --- デザイン設定 ---
colors = {
    "bg": "#FFFFFF",
    "black": "#000000",       # 主線・文字
    "gray": "#808080",        # 補助線・注釈
    "light_gray": "#D0D0D0",  # 薄い背景
    "white": "#FFFFFF",       # 塗りつぶし
    "arrow_fill": "#000000"
}

# SVG生成用ヘルパー関数
def get_svg_header(w, h):
    return [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="background-color:{colors["bg"]}; font-family:\'Yu Gothic\', \'Meiryo\', sans-serif;">',
        '<defs>',
        f'<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="4" orient="auto"><polygon points="0 0, 12 4, 0 8" fill="{colors["arrow_fill"]}" /></marker>',
        '</defs>'
    ]

def save_svg(filename, content):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(content + ['</svg>']))
    print(f"Generated: {filename}")

# --- 描画パーツ ---
def draw_text(svg, x, y, text, size=18, color=colors["black"], weight="normal", font_style="normal", anchor="middle", baseline="middle"):
    svg.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" font-style="{font_style}" fill="{color}" text-anchor="{anchor}" dominant-baseline="{baseline}">{text}</text>')

def draw_line(svg, x1, y1, x2, y2, color=colors["black"], width=1, arrow=False, dash=None):
    marker = 'marker-end="url(#arrow)"' if arrow else ""
    dash_attr = f'stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" {marker} {dash_attr} />')

def draw_rect(svg, x, y, w, h, fill, stroke, stroke_width=2):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" />')

# =========================================================
# Fig 3.7 修正版 (右側見切れ対応)
# =========================================================
def create_fig3_7_structure_fixed():
    # 幅を900から1100に拡張して、右側の要素を収める
    w, h = 1100, 350
    svg = get_svg_header(w, h)

    # レイアウト設定
    table_x, table_y = 180, 80
    row_h = 50
    cols = [
        {"name": "粒子ID", "width": 80},
        {"name": "位置 (Position)", "width": 140},
        {"name": "速度 (Velocity)", "width": 140},
        {"name": "評価値 (Fitness)", "width": 140},
        {"name": "pBest", "width": 100},
        {"name": "gBest", "width": 100},
    ]
    total_w = sum(c["width"] for c in cols)

    # --- 1. 世代遷移矢印 (左) ---
    arrow_y = table_y + row_h * 2.5
    draw_text(svg, 90, arrow_y - 30, "世代", size=18, weight="bold")
    draw_text(svg, 90, arrow_y, "j - 1", size=22, weight="bold", font_style="italic")
    draw_line(svg, 130, arrow_y, table_x - 10, arrow_y, width=3, arrow=True)

    # --- 2. 世代遷移矢印 (右) ---
    # テーブルの右端座標
    end_x = table_x + total_w
    
    # 矢印とテキストの描画
    draw_line(svg, end_x + 10, arrow_y, end_x + 60, arrow_y, width=3, arrow=True)
    draw_text(svg, end_x + 100, arrow_y - 30, "世代", size=18, weight="bold")
    draw_text(svg, end_x + 100, arrow_y, "j + 1", size=22, weight="bold", font_style="italic")

    # --- 3. テーブルタイトル ---
    draw_text(svg, table_x + total_w/2, table_y - 30, "世代 j における粒子群の情報", size=20, weight="bold")

    # --- 4. ヘッダー描画 ---
    cur_x = table_x
    cur_y = table_y
    
    # ヘッダー背景
    draw_rect(svg, cur_x, cur_y, total_w, row_h, colors["black"], colors["black"])
    
    for col in cols:
        draw_text(svg, cur_x + col["width"]/2, cur_y + row_h/2, col["name"], color=colors["white"], weight="bold", size=16)
        # 列区切り線 (白)
        if col != cols[-1]:
            draw_line(svg, cur_x + col["width"], cur_y, cur_x + col["width"], cur_y + row_h, color=colors["white"], width=1)
        cur_x += col["width"]

    # --- 5. データ行描画 ---
    # 数式表記のための下付き文字用パーツ
    def mk_sub(txt): return f'<tspan baseline-shift="sub" font-size="70%">{txt}</tspan>'
    
    rows_data = [
        ["1", f"x{mk_sub('j,1')}", f"v{mk_sub('j,1')}", f"f(x{mk_sub('j,1')})", "pBest1", "gBest"],
        ["2", f"x{mk_sub('j,2')}", f"v{mk_sub('j,2')}", f"f(x{mk_sub('j,2')})", "pBest2", "gBest"],
        ["...", "...", "...", "...", "...", "..."],
        ["Np", f"x{mk_sub('j,Np')}", f"v{mk_sub('j,Np')}", f"f(x{mk_sub('j,Np')})", f"pBest{mk_sub('Np')}", "gBest"],
    ]

    cur_y += row_h
    for r_idx, row in enumerate(rows_data):
        cur_x = table_x
        # 行全体の枠
        draw_rect(svg, cur_x, cur_y, total_w, row_h, "none", colors["black"])
        
        for c_idx, cell in enumerate(row):
            w = cols[c_idx]["width"]
            # セル枠
            draw_rect(svg, cur_x, cur_y, w, row_h, "none", colors["black"])
            
            f_style = "italic" if c_idx > 0 and r_idx != 2 else "normal" # 3行目(...)は通常
            f_weight = "bold" if c_idx == 0 or r_idx == 2 else "normal"
            
            draw_text(svg, cur_x + w/2, cur_y + row_h/2, cell, size=18, font_style=f_style, weight=f_weight)
            cur_x += w
        cur_y += row_h

    save_svg("fig3_7_pso_structure_fixed.svg", svg)

if __name__ == "__main__":
    create_fig3_7_structure_fixed()