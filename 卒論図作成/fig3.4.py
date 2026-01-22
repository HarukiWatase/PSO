import math
import random

# --- モダンデザイン設定 ---
colors = {
    "bg": "#FFFFFF",
    "text_main": "#2C3E50",   # 濃いスレートブルー
    "text_sub": "#575F60",    # グレー
    "accent_blue": "#3498DB", # 一般粒子
    "accent_red": "#E74C3C",  # エリート粒子 (gBest)
    "range_circle": "#ECF0F1", # 背景円
    "border": "#BDC3C7",      # 境界線
    "white": "#FFFFFF",
    "arrow_fill": "#2C3E50"
}

fonts = {
    "main": "'Yu Gothic', 'Meiryo', sans-serif",
}

# SVG生成用ヘルパー
def get_svg_header(w, h):
    return [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="background-color:{colors["bg"]};">',
    ]

def save_svg(filename, content):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(content + ['</svg>']))
    print(f"Generated: {filename}")

# --- 描画パーツ ---
def draw_text(svg, x, y, text, size=24, color=colors["text_main"], weight="normal", font_style="normal", anchor="middle", baseline="middle", font_family=fonts["main"]):
    svg.append(f'<text x="{x}" y="{y}" font-family="{font_family}" font-size="{size}" font-weight="{weight}" font-style="{font_style}" fill="{color}" text-anchor="{anchor}" dominant-baseline="{baseline}">{text}</text>')

def draw_circle(svg, x, y, r, fill, stroke, stroke_width=2, dash=None):
    dash_attr = f'stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" {dash_attr} />')

def draw_rect(svg, x, y, w, h, fill="none", stroke=colors["border"], stroke_width=3):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="15" ry="15" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" />')

def draw_triangle(svg, x, y, size=15, color=colors["text_main"]):
    p1 = f"{x - size},{y - size * 0.8}"
    p2 = f"{x + size},{y}"
    p3 = f"{x - size},{y + size * 0.8}"
    svg.append(f'<polygon points="{p1} {p2} {p3}" fill="{color}" />')

# =========================================================
# (a) Stagnation (停滞) 描画
# =========================================================
def draw_stagnation_panel(svg, x, y, w, h):
    cx, cy = x + w/2, y + h/2
    
    # パネル枠
    draw_rect(svg, x, y, w, h)
    
    # タイトル
    draw_text(svg, cx, y + 45, "探索停滞", weight="lighter", size=48)

    # 【修正】等高線中心位置の調整
    # cy + 25 に設定。
    # 計算: 高さ360の半分180 + 25 = 205 (中心Y)
    # 一番下の円: 205 + 115(半径) = 320
    # 余白: 360(底) - 320 = 40px (タイトル上の余白とほぼ同じ)
    trap_x, trap_y = cx, cy + 28
    
    # 円のサイズ
    radii = [115, 85, 55]
    for r in radii:
        draw_circle(svg, trap_x, trap_y, r, "none", colors["range_circle"], stroke_width=3)

    # 粒子群 (密集)
    num_particles = 12
    random.seed(1)
    for _ in range(num_particles):
        dist = random.uniform(0, 40)
        angle = random.uniform(0, 2*math.pi)
        px = trap_x + dist * math.cos(angle)
        py = trap_y + dist * math.sin(angle)
        
        # 粒子
        node_r = 14
        color = colors["accent_red"] if dist < 15 else colors["accent_blue"] 
        draw_circle(svg, px, py, node_r, color, colors["white"], stroke_width=2)

# =========================================================
# (b) Explosion (再配置) 描画
# =========================================================
def draw_explosion_panel(svg, x, y, w, h):
    cx, cy = x + w/2, y + h/2
    
    # パネル枠
    draw_rect(svg, x, y, w, h)
    
    # タイトル
    draw_text(svg, cx, y + 45, "リスタート戦略", weight="bold", size=48)

    # 等高線中心
    trap_x, trap_y = cx, cy + 28
    
    # 円のサイズ
    radii = [115, 85, 55]
    max_r = radii[0]
    for r in radii:
        draw_circle(svg, trap_x, trap_y, r, "none", colors["range_circle"], stroke_width=3)

    # --- エリート保存 (Keep gBest) ---
    draw_circle(svg, trap_x, trap_y, 20, colors["accent_red"], colors["white"], stroke_width=4)

    # --- 再配置 (Scatter) ---
    num_particles = 11
    node_r = 14
    
    placed_particles = []
    
    random.seed(3) 
    
    for _ in range(num_particles):
        for attempt in range(100):
            dist = random.uniform(35, max_r - 15)
            angle = random.uniform(0, 2*math.pi)
            
            px = trap_x + dist * math.cos(angle)
            py = trap_y + dist * math.sin(angle)
            
            # 重なりチェック
            overlap = False
            for (ex, ey) in placed_particles:
                if math.hypot(px - ex, py - ey) < (node_r * 2 + 2):
                    overlap = True
                    break
            
            if not overlap:
                placed_particles.append((px, py))
                draw_circle(svg, px, py, node_r, colors["accent_blue"], colors["white"], stroke_width=2)
                break

# =========================================================
# メイン生成関数
# =========================================================
def create_fig_restart_strategy_v5():
    # 【修正】全体のサイズ調整
    # パネル幅を狭めた分、全体幅wも小さく (920 -> 800)
    # パネル高さを増やした分、全体高さhも大きく (350 -> 400)
    w, h = 840, 340
    svg = get_svg_header(w, h)
    
    # 【修正】パネルサイズ
    # 幅: 400 -> 340 (スリム化)
    # 高さ: 310 -> 360 (下部余白確保のため拡張)
    panel_w = 370
    panel_h = 320
    
    margin_top = 10
    margin_side = 10
    
    # 左パネル配置
    draw_stagnation_panel(svg, margin_side, margin_top, panel_w, panel_h)
    
    # 中央 矢印部分
    center_x = w / 2
    center_y = margin_top + panel_h / 2
    
    # 三角形
    draw_triangle(svg, center_x, center_y, size=25, color=colors["text_main"])
    
    # 右パネル配置
    draw_explosion_panel(svg, w - margin_side - panel_w, margin_top, panel_w, panel_h)

    # ファイル保存
    save_svg("fig3_4_restart_strategy_fixed_v6.svg", svg)

if __name__ == "__main__":
    create_fig_restart_strategy_v5()