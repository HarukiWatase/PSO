import matplotlib.pyplot as plt
import math

# --- デザイン設定 ---
colors = {
    "bg": "#FFFFFF",
    "black": "#000000",       # 主線・文字
    "gray": "#808080",        # 補助線・注釈
    "light_gray": "#D0D0D0",  # 薄い背景（等高線）
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

def draw_circle(svg, x, y, r, fill, stroke, stroke_width=2, dash=None):
    dash_attr = f'stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" {dash_attr} />')

def draw_line(svg, x1, y1, x2, y2, color=colors["black"], width=1, arrow=False, dash=None):
    marker = 'marker-end="url(#arrow)"' if arrow else ""
    dash_attr = f'stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" {marker} {dash_attr} />')

# =========================================================
# Fig 3.6 修正版 (位置調整・等高線拡大)
# =========================================================
def create_fig3_6_concept_adjusted():
    w, h = 600, 500
    svg = get_svg_header(w, h)

    # 1. 座標軸
    origin_x, origin_y = 50, 460
    axis_len_x = 490
    axis_len_y = 450
    
    draw_line(svg, origin_x, origin_y, origin_x, origin_y - axis_len_y, width=2, arrow=True)
    draw_line(svg, origin_x, origin_y, origin_x + axis_len_x, origin_y, width=2, arrow=True)
    draw_text(svg, origin_x + axis_len_x - 10, origin_y + 25, "探索空間", size=16, color=colors["gray"], anchor="end")

    # 2. 等高線 (修正: 中心を左へ、半径を大きくして軸との隙間を埋める)
    cx, cy = 290, 225 # 中心を少し左(320->300)へ
    radii = [220, 190, 140, 90] # 半径を拡大 (220->240等)
    for r in radii:
        svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{colors["light_gray"]}" stroke-width="2" />')

    # 3. 最適解
    draw_circle(svg, cx, cy, 8, colors["white"], colors["black"], stroke_width=3)
    draw_text(svg, cx, cy - 25, "最適解", weight="bold", size=22)

    # 4. 粒子と軌跡
    sub_i = '<tspan baseline-shift="sub" font-size="70%">i</tspan>'
    
    # 粒子リスト
    particles = [
        # (px, py, vx, vy, prev_x, prev_y)
        # 主役粒子: 左上へ移動 (180,380) -> (150, 340)
        (150, 340, 60, -25, 100, 370),   
        
        (480, 120, -50, 40, 500, 90),    # 右上 (変更なし)
        (450, 350, -60, -40, 470, 370),  # 右下 (変更なし)
        (200, 130, 50, 50, 170, 100),    # 左上 (変更なし)
        (320, 410, 0, -60, 320, 440)     # 真下 (変更なし)
    ]

    for i, (px, py, vx, vy, prev_x, prev_y) in enumerate(particles):
        # 過去の位置 (t-1)
        draw_circle(svg, prev_x, prev_y, 9, "none", colors["gray"], dash="3 3", stroke_width=2)
        # 軌跡
        draw_line(svg, prev_x, prev_y, px, py, color=colors["gray"], width=2, dash="5 3")

        # 現在の位置 (t)
        draw_circle(svg, px, py, 11, colors["black"], colors["black"])
        # 速度ベクトル
        draw_line(svg, px, py, px + vx, py + vy, width=3, arrow=True)

        # --- ラベル配置 (主役の粒子のみ) ---
        if i == 0:
            # "粒子" ラベル
            draw_text(svg, px -15 , py + 90, "粒子", weight="bold", size=24)
            draw_line(svg, px -15, py + 77, px, py + 15, width=1.5)

            # --- 現在位置 x_i(t) ---
            # 座標調整
            label_curr = f"x{sub_i}(t)<tspan font-style='normal' font-weight='normal' font-size='16'> (現在位置)</tspan>"
            draw_text(svg, px + 10, py + 25, label_curr, size=22, weight="bold", font_style="italic", anchor="start")
            
            # --- 過去位置 x_i(t-1) ---
            # 座標調整
            label_prev = f"x{sub_i}(t-1)"
            draw_text(svg, prev_x+10 , prev_y+25, label_prev, size=18, color=colors["gray"], font_style="italic", anchor="end")

            # --- 速度 v_i ---
            vec_mid_x = px + vx/2
            vec_mid_y = py + vy/2 - 25
            label_v = f"v{sub_i}<tspan font-style='normal' font-weight='normal' font-size='16'> (速度)</tspan>"
            draw_text(svg, vec_mid_x, vec_mid_y, label_v, size=22, weight="bold", font_style="italic")

    save_svg("fig3_6_pso_concept_adjusted.svg", svg)

if __name__ == "__main__":
    create_fig3_6_concept_adjusted()