"""Render the BEVFormer Query/Feature interaction diagram as a PNG."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/assets/bevformer-query-guide/05-query-feature-head-interaction.png"

WIDTH = 1800
HEIGHT = 1100

BG = "#F7F9FC"
INK = "#102A43"
MUTED = "#486581"
BORDER = "#BCCCDC"
BLUE = "#E5F1FF"
BLUE_BORDER = "#2F80ED"
GREEN = "#E7F7EF"
GREEN_BORDER = "#26996B"
ORANGE = "#FFF0DF"
ORANGE_BORDER = "#E67E22"
PURPLE = "#F2EAFE"
PURPLE_BORDER = "#805AD5"
WHITE = "#FFFFFF"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


F_TITLE = font(42, True)
F_SUBTITLE = font(25)
F_BOX_TITLE = font(27, True)
F_BOX = font(23)
F_SMALL = font(20)
F_CALLOUT = font(28, True)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    fill: str,
    outline: str,
    title: str,
    lines: list[str],
) -> None:
    draw.rounded_rectangle(xy, radius=20, fill=fill, outline=outline, width=3)
    x1, y1, x2, _ = xy
    draw.text((x1 + 24, y1 + 18), title, font=F_BOX_TITLE, fill=INK)
    y = y1 + 62
    for line in lines:
        draw.text((x1 + 24, y), line, font=F_BOX, fill=MUTED)
        y += 35


def arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: str = MUTED,
    width: int = 5,
) -> None:
    draw.line([start, end], fill=color, width=width)
    x1, y1 = start
    x2, y2 = end
    vx, vy = x2 - x1, y2 - y1
    length = max((vx * vx + vy * vy) ** 0.5, 1)
    ux, uy = vx / length, vy / length
    px, py = -uy, ux
    back_x, back_y = x2 - ux * 18, y2 - uy * 18
    p1 = (x2, y2)
    p2 = (back_x + px * 9, back_y + py * 9)
    p3 = (back_x - px * 9, back_y - py * 9)
    draw.polygon([p1, p2, p3], fill=color)


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: str) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=F_SMALL)
    draw.rounded_rectangle(
        (bbox[0] - 10, bbox[1] - 5, bbox[2] + 10, bbox[3] + 5),
        radius=10,
        fill=WHITE,
        outline=color,
        width=2,
    )
    draw.text((x, y), text, font=F_SMALL, fill=color)


def main() -> None:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(image)

    draw.text((60, 34), "Query 参数如何与 Feature 的 8 个 Head 交互", font=F_TITLE, fill=INK)
    draw.text(
        (60, 92),
        "Query 提供读取位置和权重；Feature 提供被读取的 32 维数据。二者在可变形采样算子中相遇。",
        font=F_SUBTITLE,
        fill=MUTED,
    )

    rounded_box(
        draw,
        (60, 160, 510, 320),
        BLUE,
        BLUE_BORDER,
        "Query 路径",
        ["q_p[256]（TSA + Norm 后）", "同一个 Query 控制 8 个 Head"],
    )
    rounded_box(
        draw,
        (650, 160, 1110, 320),
        GREEN,
        GREEN_BORDER,
        "Feature 路径",
        ["六相机 × 四层 FPN", "每个空间位置是 256 维特征"],
    )

    rounded_box(
        draw,
        (60, 390, 510, 565),
        BLUE,
        BLUE_BORDER,
        "Query 产生控制参数",
        ["Offset Δ：[8, 4, 8, 2]", "Weight α：[8, 4, 8]，每个 Head 做 Softmax"],
    )
    rounded_box(
        draw,
        (650, 390, 1110, 565),
        GREEN,
        GREEN_BORDER,
        "Feature 产生 Value Head",
        ["Value Projection：V = W_v F + b_v", "reshape：[256] → [8 Head, 32]"],
    )

    rounded_box(
        draw,
        (1260, 160, 1735, 320),
        ORANGE,
        ORANGE_BORDER,
        "① Offset 决定在哪里读",
        ["s = 几何参考点 ρ + Δ/(W_l,H_l)", "s 是连续坐标，可落在像素之间"],
    )
    rounded_box(
        draw,
        (1260, 390, 1735, 565),
        ORANGE,
        ORANGE_BORDER,
        "② Weight 决定读到多少",
        ["f = BilinearSample(V_h, s) ∈ R^32", "α（标量）× f（32 维向量）"],
    )

    arrow(draw, (285, 320), (285, 390), BLUE_BORDER)
    arrow(draw, (880, 320), (880, 390), GREEN_BORDER)
    arrow(draw, (510, 475), (1260, 235), BLUE_BORDER)
    arrow(draw, (1110, 475), (1260, 475), GREEN_BORDER)
    arrow(draw, (1485, 320), (1485, 390), ORANGE_BORDER)

    label(draw, (635, 344), "Δ 生成采样地址", BLUE_BORDER)
    label(draw, (1115, 437), "V_h 提供数据", GREEN_BORDER)
    label(draw, (1498, 338), "同一 Head h", ORANGE_BORDER)

    rounded_box(
        draw,
        (470, 660, 1330, 835),
        ORANGE,
        ORANGE_BORDER,
        "每个 Head 独立聚合",
        [
            "g_h = sum(l,k) alpha(h,l,k) · BilinearSample(V_h, s(h,l,k))",
            "4 个 Level × 每层 8 个采样点 → 一个 32 维 Head 结果",
        ],
    )

    arrow(draw, (1485, 565), (1210, 660), ORANGE_BORDER)
    arrow(draw, (285, 565), (610, 660), BLUE_BORDER)

    rounded_box(
        draw,
        (145, 925, 1655, 1045),
        PURPLE,
        PURPLE_BORDER,
        "8 个 Head 合并并写回 Query",
        ["Concat：8 × 32 → 256  →  有效相机平均  →  output_proj  →  Dropout  →  + 原 q_p  →  LayerNorm  →  q_p_new"],
    )
    arrow(draw, (900, 835), (900, 925), PURPLE_BORDER)

    draw.rounded_rectangle((1190, 595, 1735, 640), radius=12, fill=WHITE, outline=BORDER, width=2)
    draw.text((1208, 603), "Query：去哪里读、相信多少；Feature：真正的数据", font=F_SMALL, fill=INK)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, format="PNG", optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
