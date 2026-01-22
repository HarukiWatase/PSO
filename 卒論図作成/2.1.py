import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- デザイン設定（モノクロ・硬派・学術的） ---
colors = {
    "bg": "#FFFFFF",
    "black": "#000000",       # 線・文字・枠
    "gray": "#808080",        # 補助的な線・注釈
    "white": "#FFFFFF",       # ノードの塗りつぶし
    "arrow_fill": "#000000"   # 矢印の塗りつぶし
}

# フォント設定
plt.rcParams['font.family'] = ['Yu Gothic', 'Meiryo', 'Hiragino Sans', 'sans-serif']
plt.rcParams['text.color'] = colors['black']

# SVG生成用ヘルパー関数
def get_svg_header(w, h):
    return [
        '<?xml version="1.0" encoding="UTF-8" standalone="no"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" style="background-color:{colors["bg"]}; font-family:\'Yu Gothic\', sans-serif;">',
        '<defs>',
        # 黒い矢印マーカー
        f'<marker id="arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><polygon points="0 0, 10 4, 0 8" fill="{colors["arrow_fill"]}" /></marker>',
        '</defs>'
    ]

def save_svg(filename, content):
    with open(filename, "w", encoding="utf-8") as f:
        f.write("\n".join(content + ['</svg>']))
    print(f"Generated: {filename}")

# --- 共通描画関数 ---
def draw_node(svg, x, y, label):
    # 白抜き・黒枠の円
    svg.append(f'<circle cx="{x}" cy="{y}" r="18" fill="{colors["white"]}" stroke="{colors["black"]}" stroke-width="2" />')
    # 黒文字
    svg.append(f'<text x="{x}" y="{y+1}" font-size="14" font-weight="bold" fill="{colors["black"]}" text-anchor="middle" dominant-baseline="middle">{label}</text>')

def draw_link(svg, x1, y1, x2, y2, curved=False, offset=0, label=None):
    # 黒い線
    if not curved:
        # 直線 (終点座標を少し調整して矢印が埋もれないようにする処理は省略し、markerのrefXで調整)
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{colors["black"]}" stroke-width="2" marker-end="url(#arrow)" />')
        if label:
            mx, my = (x1+x2)/2, (y1+y2)/2
            svg.append(f'<text x="{mx}" y="{my-8}" font-size="12" fill="{colors["black"]}" text-anchor="middle">{label}</text>')
    else:
        # 曲線 (2次ベジェ)
        # 制御点を計算 (中点から垂直方向にずらす)
        mx, my = (x1+x2)/2, (y1+y2)/2
        cx = mx + offset # 簡易的な制御点計算（横方向のみ考慮）
        cy = my + (50 if offset==0 else -30) # 縦方向の制御
        
        path_d = f"M {x1} {y1} Q {cx} {cy} {x2} {y2}"
        svg.append(f'<path d="{path_d}" stroke="{colors["black"]}" stroke-width="2" fill="none" marker-end="url(#arrow)" />')

# =========================================================
# 1. 図2.1: 有向グラフの例 (Graph)
# =========================================================
def create_fig2_1():
    w, h = 500, 250
    svg = get_svg_header(w, h)
    
    # 座標
    nodes = {'u': (100, 125), 'v': (400, 125), 'w': (250, 200)}
    
    # リンク (u->v, u->w, w->v)
    draw_link(svg, 100, 125, 380, 125, label="e1") # u->v
    draw_link(svg, 100, 125, 235, 190, label="e2") # u->w
    draw_link(svg, 265, 190, 390, 125, label="e3") # w->v
    
    # 重みラベル (w(e))
    svg.append(f'<text x="250" y="115" font-size="12" fill="{colors["gray"]}" text-anchor="middle">w(e1)=10</text>')

    # ノード
    for lbl, (x, y) in nodes.items():
        draw_node(svg, x, y, lbl)
        
    save_svg("fig2_1_graph.svg", svg)

# =========================================================
# 2. 図2.2: ウォーク (Walk)
# 特徴: ノード・リンクの重複あり（ループ）
# =========================================================
def create_fig2_2():
    w, h = 500, 250
    svg = get_svg_header(w, h)
    
    # 座標: u -> v (ループ) -> w
    nodes = {'u': (100, 125), 'v': (250, 125), 'w': (400, 125)}
    
    # 1. u -> v
    draw_link(svg, 118, 125, 232, 125)
    
    # 2. v -> v (自己ループ)
    # SVGパスで円を描く (vの上側にループ)
    vx, vy = 250, 125
    path_loop = f"M {vx-5} {vy-15} C {vx-30} {vy-60}, {vx+30} {vy-60}, {vx+5} {vy-15}"
    svg.append(f'<path d="{path_loop}" stroke="{colors["black"]}" stroke-width="2" fill="none" marker-end="url(#arrow)" />')
    svg.append(f'<text x="{vx}" y="{vy-55}" font-size="12" fill="{colors["black"]}" text-anchor="middle">Loop</text>')

    # 3. v -> w
    draw_link(svg, 268, 125, 382, 125)
    
    # ノード
    for lbl, (x, y) in nodes.items():
        draw_node(svg, x, y, lbl)

    # 説明
    svg.append(f'<text x="250" y="220" font-size="14" font-weight="bold" fill="{colors["black"]}" text-anchor="middle">ウォーク (Walk)</text>')
    svg.append(f'<text x="250" y="240" font-size="12" fill="{colors["black"]}" text-anchor="middle">ノード・リンクの重複を許容</text>')

    save_svg("fig2_2_walk.svg", svg)

# =========================================================
# 3. 図2.3: トレイル (Trail)
# 特徴: リンク重複なし、ノード重複あり（8の字）
# =========================================================
def create_fig2_3():
    w, h = 500, 250
    svg = get_svg_header(w, h)
    
    # 座標: u -> v -> w -> v
    nodes = {'u': (100, 125), 'v': (250, 125), 'w': (400, 125)}
    
    # 1. u -> v
    draw_link(svg, 118, 125, 232, 125)
    
    # 2. v -> w (上側カーブ)
    path_upper = "M 265 115 Q 325 70 385 115"
    svg.append(f'<path d="{path_upper}" stroke="{colors["black"]}" stroke-width="2" fill="none" marker-end="url(#arrow)" />')
    
    # 3. w -> v (下側カーブ) - 戻ってくる
    path_lower = "M 385 135 Q 325 180 265 135"
    svg.append(f'<path d="{path_lower}" stroke="{colors["black"]}" stroke-width="2" fill="none" marker-end="url(#arrow)" />')
    
    # ノード
    for lbl, (x, y) in nodes.items():
        draw_node(svg, x, y, lbl)

    # 説明
    svg.append(f'<text x="250" y="220" font-size="14" font-weight="bold" fill="{colors["black"]}" text-anchor="middle">トレイル (Trail)</text>')
    svg.append(f'<text x="250" y="240" font-size="12" fill="{colors["black"]}" text-anchor="middle">リンク重複なし / ノード重複あり</text>')

    save_svg("fig2_3_trail.svg", svg)

# =========================================================
# 4. 図2.4: パス (Path)
# 特徴: 重複なし（一筆書き）
# =========================================================
def create_fig2_4():
    w, h = 500, 250
    svg = get_svg_header(w, h)
    
    # 座標: v1 -> v2 -> v3 -> v4
    nodes = {'v1': (80, 125), 'v2': (190, 125), 'v3': (300, 125), 'v4': (410, 125)}
    node_list = list(nodes.values())
    
    # リンク描画
    for i in range(len(node_list)-1):
        x1, y1 = node_list[i]
        x2, y2 = node_list[i+1]
        draw_link(svg, x1+18, y1, x2-18, y2)
    
    # ノード
    for lbl, (x, y) in nodes.items():
        draw_node(svg, x, y, lbl)

    # 説明
    svg.append(f'<text x="250" y="220" font-size="14" font-weight="bold" fill="{colors["black"]}" text-anchor="middle">パス (Path)</text>')
    svg.append(f'<text x="250" y="240" font-size="12" fill="{colors["black"]}" text-anchor="middle">ノード・リンクの重複なし</text>')

    save_svg("fig2_4_path.svg", svg)

# 実行
if __name__ == "__main__":
    create_fig2_1()
    create_fig2_2()
    create_fig2_3()
    create_fig2_4()