"""Render a PNG explaining the six BEVFormer encoder layers."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/assets/bevformer-query-guide/06-six-layer-bev-encoder.png"

W, H = 2000, 1260
BG = "#F7F9FC"
INK = "#102A43"
MUTED = "#486581"
BORDER = "#BCCCDC"
BLUE, BLUE_B = "#E5F1FF", "#2F80ED"
GREEN, GREEN_B = "#E7F7EF", "#26996B"
ORANGE, ORANGE_B = "#FFF0DF", "#E67E22"
PURPLE, PURPLE_B = "#F2EAFE", "#805AD5"
WHITE = "#FFFFFF"


def get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


TITLE = get_font(43, True)
SUBTITLE = get_font(25)
H2 = get_font(30, True)
BOX_TITLE = get_font(25, True)
BODY = get_font(21)
SMALL = get_font(18)


def box(draw, xy, title, lines, fill, outline, title_font=BOX_TITLE, body_font=BODY):
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=3)
    x1, y1, _, _ = xy
    draw.text((x1 + 20, y1 + 14), title, font=title_font, fill=INK)
    y = y1 + 54
    for line in lines:
        draw.text((x1 + 20, y), line, font=body_font, fill=MUTED)
        y += 31


def arrow(draw, start, end, color=MUTED, width=5):
    draw.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    bx, by = x2 - 18 * ux, y2 - 18 * uy
    draw.polygon([(x2, y2), (bx + 9 * px, by + 9 * py), (bx - 9 * px, by - 9 * py)], fill=color)


def pill(draw, xy, text, fill, outline):
    x1, y1, x2, y2 = xy
    draw.rounded_rectangle(xy, radius=14, fill=fill, outline=outline, width=2)
    bbox = draw.textbbox((0, 0), text, font=SMALL)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((x1 + x2 - tw) / 2, (y1 + y2 - th) / 2 - 2), text, font=SMALL, fill=INK)


def main() -> None:
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)

    draw.text((55, 28), "BEVFormer 的 6 层 Encoder 到底在重复什么", font=TITLE, fill=INK)
    draw.text(
        (55, 87),
        "六层保持同一张 200×200 BEV 网格和同一张量形状；每层用独立参数继续融合历史、相机特征和局部非线性变换。",
        font=SUBTITLE,
        fill=MUTED,
    )

    # Left panel: layer stack.
    draw.text((60, 145), "六层怎样串联", font=H2, fill=INK)
    box(draw, (60, 195, 700, 300), "输入 BEV Query", ["q^(0)：[B, 40000, 256]", "40000 = 200 × 200 个固定网格位置"], WHITE, BORDER)

    layer_y = [340, 475, 610, 745, 880, 1015]
    colors = [(BLUE, BLUE_B), (GREEN, GREEN_B), (ORANGE, ORANGE_B), (PURPLE, PURPLE_B), (BLUE, BLUE_B), (GREEN, GREEN_B)]
    for index, y in enumerate(layer_y, start=1):
        fill, outline = colors[index - 1]
        box(
            draw,
            (115, y, 645, y + 100),
            f"Encoder Layer {index}",
            [f"q^({index - 1}) → TSA → SCA → FFN → q^({index})   [B,40000,256]"],
            fill,
            outline,
        )
        if index == 1:
            arrow(draw, (380, 300), (380, y), outline)
        else:
            arrow(draw, (380, layer_y[index - 2] + 100), (380, y), outline)

    # Right panel: one layer internals.
    draw.text((790, 145), "任意一层 l 的内部计算", font=H2, fill=INK)
    box(draw, (790, 195, 1915, 300), "输入状态", ["q^(l)：当前层的 BEV 状态 [B,40000,256]", "BEV 网格位置不变，只有每格的 256 维内容更新"], WHITE, BORDER)

    box(
        draw,
        (790, 350, 1325, 535),
        "① Temporal Self-Attention",
        [
            "输入：q^(l) + 对齐后的 prev_bev",
            "8 Head × 2 个时间队列 × 4 个采样点",
            "t = q^(l) + Dropout(TSA(...))",
            "a = LayerNorm(t)",
        ],
        BLUE,
        BLUE_B,
    )
    box(
        draw,
        (1380, 350, 1915, 535),
        "② Spatial Cross-Attention",
        [
            "输入：a + 六相机四尺度 FPN Feature",
            "8 Head × 4 Level × 8 Point",
            "s = a + Dropout(SCA(...))",
            "b = LayerNorm(s)",
        ],
        GREEN,
        GREEN_B,
    )
    arrow(draw, (1350, 442), (1380, 442), GREEN_B)
    arrow(draw, (1060, 300), (1060, 350), BLUE_B)

    box(
        draw,
        (990, 610, 1720, 795),
        "③ FFN：逐格进行非线性变换",
        [
            "每个格子独立执行：256 → 512 → ReLU → 256",
            "f = b + Dropout(W2 · ReLU(W1 · b))",
            "q^(l+1) = LayerNorm(f)",
            "输出形状仍为 [B,40000,256]",
        ],
        ORANGE,
        ORANGE_B,
    )
    arrow(draw, (1648, 535), (1475, 610), ORANGE_B)

    # Side inputs and reusable geometry.
    box(
        draw,
        (735, 860, 1270, 1035),
        "时序输入与 2D Reference",
        [
            "prev_bev：上一帧 BEV（第一帧可为空）",
            "ref_2d：[B,40000,1,2]",
            "根据自车运动平移/旋转后参与 TSA",
        ],
        BLUE,
        BLUE_B,
    )
    box(
        draw,
        (1325, 860, 1915, 1035),
        "相机输入与 3D Reference",
        [
            "六相机 × 四尺度 Feature",
            "每个 BEV 柱 4 个高度参考点",
            "投影到相机后作为 SCA 几何基准",
        ],
        GREEN,
        GREEN_B,
    )
    arrow(draw, (1005, 860), (1005, 535), BLUE_B)
    arrow(draw, (1620, 860), (1620, 535), GREEN_B)

    draw.rounded_rectangle((735, 1085, 1915, 1205), radius=18, fill=PURPLE, outline=PURPLE_B, width=3)
    draw.text((760, 1100), "六层之间哪些相同，哪些不同", font=BOX_TITLE, fill=INK)
    draw.text((760, 1143), "相同：张量形状、固定 BEV 网格、相机 Feature、几何参考点。", font=BODY, fill=MUTED)
    draw.text((760, 1175), "不同：每层参数独立，输入 Query 已被上一层更新，因此 Offset、Weight 和输出都会变化。", font=BODY, fill=MUTED)

    pill(draw, (65, 1170, 665, 1220), "最终 q^(6) = 当前帧 BEV Feature [B,40000,256]", PURPLE, PURPLE_B)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, "PNG", optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
