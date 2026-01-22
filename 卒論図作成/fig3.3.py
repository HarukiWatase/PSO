import matplotlib.pyplot as plt
import math
import random

# --- モダンデザイン設定 ---
colors = {
    "bg": "#FFFFFF",
    "text_main": "#2C3E50",   # 濃いスレートブルー (主文字・矢印)
    "text_sub": "#51595A",    # グレー (補助線・注釈)
    "accent_blue": "#3498DB", # 一般粒子
    "accent_red": "#E74C3C",  # リーダー粒子 (gBest, lBest)
    "range_circle": "#ECF0F1", # 近傍範囲の円背景
    "border": "#BDC3C7",      # 薄い境界線
    "white": "#FFFFFF",
    "arrow_fill": "#2C3E50"
}

# フォント設定
fonts = {
    "main": "'Yu Gothic', 'Meiryo', sans-serif",
    "math": "'Times New Roman', 'Times', serif"
}

# SVG生成用ヘルパー
def get_svg_header(w, h):
    return [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="background-color:{colors["bg"]};">',
        '<defs>',
        # 標準矢印 (サイズ: 7x7)
        f'<marker id="arrow" markerWidth="7" markerHeight="7" refX="7" refY="3.5" orient="auto"><polygon points="0 0, 7 3.5, 0 7" fill="{colors["arrow_fill"]}" /></marker>',
        # 認識線用の小さな矢印
        f'<marker id="arrow_small" markerWidth="6" markerHeight="6" refX="6" refY="3" orient="auto"><polygon points="0 0, 6 3, 0 6" fill="{colors["text_sub"]}" /></marker>',
        '</defs>'
    ]

def save_svg(filename, content):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(content + ['</svg>']))
    print(f"Generated: {filename}")

# --- 描画パーツ ---
def draw_text(svg, x, y, text, size=18, color=colors["text_main"], weight="normal", font_style="normal", anchor="middle", baseline="middle", font_family=fonts["main"]):
    svg.append(f'<text x="{x}" y="{y}" font-family="{font_family}" font-size="{size}" font-weight="{weight}" font-style="{font_style}" fill="{color}" text-anchor="{anchor}" dominant-baseline="{baseline}">{text}</text>')

def draw_circle(svg, x, y, r, fill, stroke, stroke_width=2, dash=None):
    dash_attr = f'stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" {dash_attr} />')

def draw_line(svg, x1, y1, x2, y2, color=colors["text_main"], width=1, arrow=False, marker_id="arrow", dash=None):
    marker = f'marker-end="url(#{marker_id})"' if arrow else ""
    dash_attr = f'stroke-dasharray="{dash}"' if dash else ""
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" {marker} {dash_attr} />')

# 数式ヘルパー
def mk_math(txt): return f'<tspan font-family="{fonts["math"]}" font-style="italic">{txt}</tspan>'
def mk_sub(txt): return f'<tspan baseline-shift="sub" font-size="70%">{txt}</tspan>'

# ノード描画ヘルパー (サイズ拡大)
def draw_node(svg, x, y, role, label=None):
    # 半径を大きく修正 (12 -> 20)
    base_r = 24
    
    if role == "leader":
        # リーダーはさらに少し大きく
        draw_circle(svg, x, y, base_r + 4, colors["accent_red"], colors["white"], stroke_width=3)
        if label: 
            # 文字位置調整 (y+1 で中央)
            draw_text(svg, x, y+1, label, size=18, weight="bold", color=colors["white"])
    elif role == "member":
        draw_circle(svg, x, y, base_r, colors["accent_blue"], colors["white"], stroke_width=2)
        if label: 
            # 外部ラベルの位置調整 (y-25 -> y-35)
            draw_text(svg, x, y-35, label, size=18, color=colors["accent_blue"])

# リンク描画ヘルパー (オフセット調整)
def draw_link_node(svg, x1, y1, x2, y2, role="main", node_r=20):
    vec_x, vec_y = x2 - x1, y2 - y1
    dist = math.sqrt(vec_x**2 + vec_y**2)
    if dist == 0: return
    ux, uy = vec_x/dist, vec_y/dist
    
    # 矢印の開始・終了位置をノードサイズに合わせて調整
    start_offset = node_r + 2 
    end_offset = node_r + 12 # 矢印先端がノードに食い込まないように余裕を持つ
    
    sx, sy = x1 + ux*start_offset, y1 + uy*start_offset
    ex, ey = x2 - ux*end_offset, y2 - uy*end_offset
    
    if role == "main":
        draw_line(svg, sx, sy, ex, ey, color=colors["accent_blue"], width=3, arrow=True)
    elif role == "sub":
        draw_line(svg, sx, sy, ex, ey, color=colors["text_sub"], width=1.5, arrow=True, marker_id="arrow_small", dash="3 3")


# =========================================================
# (a) gBest Model (完全円配置)
# =========================================================
def draw_gbest_model_large(svg, offset_x, offset_y, width, height):
    cx, cy = offset_x + width/2, offset_y + height/2
    draw_text(svg, cx, offset_y + height - 30, "(a) gBest Model (Global)", weight="bold", size=20)

    # gBest (中心)
    gx, gy = cx, cy - 20
    
    # メンバー粒子配置 (完全な円)
    members = []
    num_members = 8
    radius = 120 # ノードが大きくなったので半径も少し拡大
    
    for i in range(num_members):
        angle = (2 * math.pi / num_members) * i - math.pi/2
        mx = gx + radius * math.cos(angle)
        my = gy + radius * math.sin(angle)
        members.append((mx, my))
        
    # リンク描画
    for mx, my in members:
        draw_link_node(svg, mx, my, gx, gy, role="main")
        
    # ノード描画
    for i, (mx, my) in enumerate(members):
        label = f"{mk_math('x')}{mk_sub(i+1)}" if i < 2 else None
        draw_node(svg, mx, my, "member", label)
        
    draw_node(svg, gx, gy, "leader", mk_math("gBest"))
    draw_text(svg, cx, offset_y + 80, "全粒子が唯一の最良解を共有", size=18, color=colors["text_sub"])


# =========================================================
# (b) lBest Model (空間的派閥配置・固定版)
# =========================================================
def draw_lbest_model_large(svg, offset_x, offset_y, width, height):
    cx, cy = offset_x + width/2, offset_y + height/2
    draw_text(svg, cx, offset_y + height - 30, "(b) lBest Model (Spatial Groups)", weight="bold", size=20)

    # ノード拡大に伴い、グループ半径も少し拡大
    group_r = 95
    groups = [
        {"cx": offset_x + width*0.25, "cy": cy, "r": group_r, "id": 1},
        {"cx": offset_x + width*0.75, "cy": cy, "r": group_r, "id": 2}
    ]
    
    for g in groups:
        gcx, gcy = g["cx"], g["cy"]
        draw_circle(svg, gcx, gcy, g["r"]+25, colors["range_circle"], colors["border"], dash="5 3")
        
        lx, ly = gcx, gcy
        
        num_members = 5
        start_angle = math.pi/2 if g["id"]==1 else -math.pi/2
        members = []
        for i in range(num_members):
            angle = start_angle + (i - int(num_members/2)) * (math.pi / 3.5)
            dist = g["r"]
            mx = lx + dist * math.cos(angle)
            my = ly + dist * math.sin(angle)
            members.append((mx, my))

        for mx, my in members:
            draw_link_node(svg, mx, my, lx, ly, role="main")

        for i, (mx, my) in enumerate(members):
            label = None
            draw_node(svg, mx, my, "member", label)
            
        draw_node(svg, lx, ly, "leader", f"{mk_math('lBest')}{mk_sub(g['id'])}")

    draw_text(svg, cx, offset_y + 80, "距離が近い集団内で最良解を共有", size=18, color=colors["text_sub"])
    draw_text(svg, cx, offset_y + 105, "(空間的な派閥形成)", size=16, color=colors["text_sub"])


# =========================================================
# メイン生成関数
# =========================================================
def create_fig3_topology_final_large():
    total_w, total_h = 1100, 550
    svg = get_svg_header(total_w, total_h)
    
    draw_gbest_model_large(svg, 0, 0, total_w/2, total_h)
    draw_lbest_model_large(svg, total_w/2, 0, total_w/2, total_h)

    save_svg("fig3_topology_final_large.svg", svg)

if __name__ == "__main__":
    create_fig3_topology_final_large()