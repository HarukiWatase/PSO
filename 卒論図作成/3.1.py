import matplotlib.pyplot as plt
import math

# --- モダンデザイン設定 ---
colors = {
    "bg": "#FFFFFF",
    "text_main": "#2C3E50",   # 濃いスレートブルー
    "text_sub": "#3B3F3F",    # グレー
    "accent_blue": "#2D84BD", # 明るい青
    "accent_red": "#E74C3C",  # 赤
    "light_bg": "#F4F6F7",    # 極薄いグレー
    "border": "#BDC3C7",      # 薄い境界線
    "white": "#FFFFFF",
    "arrow_fill": "#2C3E50"
}

# フォント設定
fonts = {
    "main": "'Yu Gothic', 'Meiryo', sans-serif",         # 日本語・通常テキスト
    "math": "'Times New Roman', 'Times', serif"          # 数式用 (x, v, i, t, f...)
}

# SVG生成用ヘルパー関数
def get_svg_header(w, h):
    return [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="background-color:{colors["bg"]};">',
        '<defs>',
        f'<marker id="arrow" markerWidth="12" markerHeight="12" refX="10" refY="4" orient="auto"><polygon points="0 0, 12 4, 0 8" fill="{colors["arrow_fill"]}" /></marker>',
        '</defs>'
    ]

def save_svg(filename, content):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(content + ['</svg>']))
    print(f"Generated: {filename}")

# --- 描画パーツ ---
# font_familyを指定できるように拡張
def draw_text(svg, x, y, text, size=27, color=colors["text_main"], weight="normal", font_style="normal", anchor="middle", baseline="middle", font_family=fonts["main"]):
    svg.append(f'<text x="{x}" y="{y}" font-family="{font_family}" font-size="{size}" font-weight="{weight}" font-style="{font_style}" fill="{color}" text-anchor="{anchor}" dominant-baseline="{baseline}">{text}</text>')

def draw_circle(svg, x, y, r, fill, stroke, stroke_width=2, dash=None):
    dash_attr = f'stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" {dash_attr} />')

def draw_line(svg, x1, y1, x2, y2, color=colors["text_main"], width=1, arrow=False, dash=None):
    marker = 'marker-end="url(#arrow)"' if arrow else ""
    dash_attr = f'stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" {marker} {dash_attr} />')

def draw_rect(svg, x, y, w, h, fill, stroke="none", stroke_width=0):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" />')


# =========================================================
# Fig 3.6 モダン・数式フォント版
# =========================================================
def create_fig3_6_final():
    w, h = 600, 500
    svg = get_svg_header(w, h)

    # 1. 座標軸
    origin_x, origin_y = 50, 460
    axis_len_x = 490
    axis_len_y = 450
    
    draw_line(svg, origin_x, origin_y, origin_x, origin_y - axis_len_y, width=2, arrow=True)
    draw_line(svg, origin_x, origin_y, origin_x + axis_len_x, origin_y, width=2, arrow=True)
    draw_text(svg, origin_x + axis_len_x - 10, origin_y + 25, "探索空間", size=25, color=colors["text_sub"], anchor="end")

    # 2. 等高線
    cx, cy = 290, 225 
    radii = [220, 190, 140, 90] 
    for r in radii:
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{colors["border"]}" stroke-width="1.5" />')

    # 3. 最適解
    draw_circle(svg, cx, cy, 8, colors["white"], colors["accent_red"], stroke_width=3)
    draw_text(svg, cx, cy - 25, "最適解", weight="bold", size=35, color=colors["accent_red"])

    # 4. 粒子と軌跡
    # 数式用ヘルパー (Times New Roman, Italic)
    def mk_math(txt): return f'<tspan font-family="{fonts["math"]}" font-style="italic">{txt}</tspan>'
    def mk_sub(txt): return f'<tspan baseline-shift="sub" font-size="70%">{txt}</tspan>'
    
    # 粒子リスト
    particles = [
        (150, 340, 60, -25, 100, 370),   
        (480, 120, -50, 40, 500, 90),    
        (450, 350, -60, -40, 470, 370), 
        (190, 130, 50, 50, 170, 100),   
        (340, 100, -20, 60, 360, 60)     
    ]

    for i, (px, py, vx, vy, prev_x, prev_y) in enumerate(particles):
        # 過去 (t-1)
        draw_circle(svg, prev_x, prev_y, 9, "none", colors["text_sub"], dash="3 3", stroke_width=2)
        draw_line(svg, prev_x, prev_y, px, py, color=colors["text_sub"], width=2, dash="5 3")

        # 現在 (t)
        draw_circle(svg, px, py, 11, colors["accent_blue"], colors["white"], stroke_width=2)
        draw_line(svg, px, py, px + vx, py + vy, width=3, arrow=True, color=colors["text_main"])

        # --- ラベル配置 (主役の粒子のみ) ---
        if i == 0:
            # "粒子"
            draw_text(svg, px - 15, py + 90, "粒子", weight="bold", size=31, color=colors["accent_blue"])
            draw_line(svg, px - 15, py + 77, px, py + 15, width=1.5, color=colors["accent_blue"])

            # --- 現在位置 x_i(t) ---
            # x_i(t) は数式フォント、(現在位置) は日本語フォント
            label_curr = f"{mk_math('x')}{mk_sub(mk_math('i'))}{mk_math('(t)')}<tspan font-size='28'> (現在位置)</tspan>"
            draw_text(svg, px + 10, py + 25, label_curr, size=32, weight="bold", anchor="start", color=colors["text_main"])
            
            # --- 過去位置 x_i(t-1) ---
            label_prev = f"{mk_math('x')}{mk_sub(mk_math('i'))}{mk_math('(t-1)')}"
            draw_text(svg, prev_x + 20, prev_y + 25, label_prev, size=26, color=colors["text_sub"], anchor="end")

            # --- 速度 v_i ---
            vec_mid_x = px + vx/2
            vec_mid_y = py + vy/2 - 25
            label_v = f"{mk_math('v')}{mk_sub(mk_math('i'))}<tspan font-size='28'> (速度)</tspan>"
            draw_text(svg, vec_mid_x, vec_mid_y, label_v, size=32, weight="bold", color=colors["text_main"])

    save_svg("fig3_6_pso_concept_final.svg", svg)


# =========================================================
# Fig 3.7 モダン・数式フォント版 (項目改善)
# =========================================================
def create_fig3_7_structure_final():
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
        # 項目名をわかりやすく変更
        {"name": "Personal Best ", "width": 140}, # 幅を広げる
        {"name": "Global Best ", "width": 140},   # 幅を広げる
    ]
    # 全体幅再計算
    total_w = sum(c["width"] for c in cols)
    # キャンバス幅調整が必要ならここで行う (1100で足りるか確認 -> 少し足りないので広げる)
    if total_w + 300 > w:
        w = int(total_w + 350)
        svg = get_svg_header(w, h) # ヘッダー再取得

    # 数式用ヘルパー
    def mk_math(txt): return f'<tspan font-family="{fonts["math"]}" font-style="italic">{txt}</tspan>'
    def mk_sub(txt): return f'<tspan baseline-shift="sub" font-size="70%">{txt}</tspan>'

    # --- 1. 世代遷移矢印 (左) ---
    arrow_y = table_y + row_h * 2.5
    draw_text(svg, 90, arrow_y - 30, "世代", size=18, weight="bold", color=colors["text_main"])
    draw_text(svg, 90, arrow_y, f"{mk_math('j')} - 1", size=22, weight="bold", color=colors["text_sub"])
    draw_line(svg, 130, arrow_y, table_x - 10, arrow_y, width=4, arrow=True, color=colors["text_sub"])

    # --- 2. 世代遷移矢印 (右) ---
    end_x = table_x + total_w
    draw_line(svg, end_x + 10, arrow_y, end_x + 60, arrow_y, width=4, arrow=True, color=colors["text_main"])
    draw_text(svg, end_x + 100, arrow_y - 30, "世代", size=18, weight="bold", color=colors["text_main"])
    draw_text(svg, end_x + 100, arrow_y, f"{mk_math('j')} + 1", size=22, weight="bold", color=colors["text_sub"])

    # --- 3. テーブルタイトル ---
    # j も数式フォントに
    title_text = f"世代 {mk_math('j')} における粒子群の情報"
    draw_text(svg, table_x + total_w/2, table_y - 30, title_text, size=20, weight="bold", color=colors["text_main"])

    # --- 4. ヘッダー描画 ---
    cur_x = table_x
    cur_y = table_y
    
    draw_rect(svg, cur_x, cur_y, total_w, row_h, fill=colors["accent_blue"], stroke="none")
    
    for col in cols:
        # ヘッダー内の英数字(pBestなど)もSerifにするか？ -> ここは可読性重視でSans-Serifのままでも良いが、
        # pBest/gBest変数は数式フォントに合わせるのが綺麗。
        # ここでは単純なテキストとして出力し、内部の(pBest)等はそのまま
        draw_text(svg, cur_x + col["width"]/2, cur_y + row_h/2, col["name"], color=colors["white"], weight="bold", size=16)
        cur_x += col["width"]

    # --- 5. データ行描画 ---
    # 数式フォント適用済みのデータ
    # pBest -> pBest_i, gBest -> gBest
    rows_data = [
        ["1", 
         f"{mk_math('x')}{mk_sub(mk_math('j,1'))}", 
         f"{mk_math('v')}{mk_sub(mk_math('j,1'))}", 
         f"{mk_math('f')}({mk_math('x')}{mk_sub(mk_math('j,1'))})", 
         f"{mk_math('pBest')}{mk_sub(mk_math('1'))}", 
         f"{mk_math('gBest')}"],
         
        ["2", 
         f"{mk_math('x')}{mk_sub(mk_math('j,2'))}", 
         f"{mk_math('v')}{mk_sub(mk_math('j,2'))}", 
         f"{mk_math('f')}({mk_math('x')}{mk_sub(mk_math('j,2'))})", 
         f"{mk_math('pBest')}{mk_sub(mk_math('2'))}", 
         f"{mk_math('gBest')}"],
         
        ["...", "...", "...", "...", "...", "..."],
        
        [f"{mk_math('N')}{mk_sub(mk_math('p'))}", 
         f"{mk_math('x')}{mk_sub(mk_math('j,Np'))}", 
         f"{mk_math('v')}{mk_sub(mk_math('j,Np'))}", 
         f"{mk_math('f')}({mk_math('x')}{mk_sub(mk_math('j,Np'))})", 
         f"{mk_math('pBest')}{mk_sub(mk_math('Np'))}", 
         f"{mk_math('gBest')}"]
    ]

    cur_y += row_h
    for r_idx, row in enumerate(rows_data):
        cur_x = table_x
        bg_fill = colors["light_bg"] if r_idx % 2 == 1 else colors["white"]
        draw_rect(svg, cur_x, cur_y, total_w, row_h, fill=bg_fill, stroke="none")
        draw_line(svg, cur_x, cur_y + row_h, cur_x + total_w, cur_y + row_h, color=colors["border"], width=1)

        for c_idx, cell in enumerate(row):
            w = cols[c_idx]["width"]
            
            # 3行目の "..." は太字で中央寄せ
            f_weight = "bold" if r_idx == 2 or c_idx == 0 else "normal"
            f_color = colors["accent_blue"] if c_idx == 0 and r_idx != 2 else colors["text_main"]

            draw_text(svg, cur_x + w/2, cur_y + row_h/2, cell, size=20, weight=f_weight, color=f_color)
            cur_x += w
        cur_y += row_h

    save_svg("fig3_7_pso_structure_final.svg", svg)

if __name__ == "__main__":
    create_fig3_6_final()
    create_fig3_7_structure_final()