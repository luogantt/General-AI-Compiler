"""Render PNG flowcharts for the step-by-step BEVFormer encoder guide.

The repository documentation intentionally uses PNG rather than SVG so that the
same figures render in Markdown viewers, browsers, and the Codex desktop app.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs/assets/bevformer-query-guide"

BG = "#F7F9FC"
INK = "#102A43"
MUTED = "#486581"
BORDER = "#BCCCDC"
BLUE, BLUE_B = "#E5F1FF", "#2F80ED"
GREEN, GREEN_B = "#E7F7EF", "#26996B"
ORANGE, ORANGE_B = "#FFF0DF", "#E67E22"
PURPLE, PURPLE_B = "#F2EAFE", "#805AD5"
RED, RED_B = "#FDECEC", "#C0392B"
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


TITLE = font(45, True)
SUBTITLE = font(25)
H2 = font(31, True)
BOX_TITLE = font(25, True)
BODY = font(21)
SMALL = font(18)
TINY = font(16)


def rounded_box(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int, int, int],
    title: str,
    lines: Iterable[str],
    fill: str = WHITE,
    outline: str = BORDER,
    *,
    title_font: ImageFont.FreeTypeFont = BOX_TITLE,
    body_font: ImageFont.FreeTypeFont = BODY,
    line_gap: int = 31,
) -> None:
    draw.rounded_rectangle(xy, radius=18, fill=fill, outline=outline, width=3)
    x1, y1, _, _ = xy
    draw.text((x1 + 20, y1 + 14), title, font=title_font, fill=INK)
    y = y1 + 56
    for line in lines:
        draw.text((x1 + 20, y), line, font=body_font, fill=MUTED)
        y += line_gap


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
    dx, dy = x2 - x1, y2 - y1
    length = max((dx * dx + dy * dy) ** 0.5, 1.0)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    bx, by = x2 - 18 * ux, y2 - 18 * uy
    draw.polygon(
        [(x2, y2), (bx + 9 * px, by + 9 * py), (bx - 9 * px, by - 9 * py)],
        fill=color,
    )


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, color: str = MUTED) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=SMALL)
    draw.rounded_rectangle(
        (bbox[0] - 9, bbox[1] - 4, bbox[2] + 9, bbox[3] + 5),
        radius=8,
        fill=BG,
    )
    draw.text((x, y), text, font=SMALL, fill=color)


def render_attention_principle() -> None:
    image = Image.new("RGB", (2200, 1330), BG)
    draw = ImageDraw.Draw(image)
    draw.text((55, 30), "先理解 Attention：Query 不是数据仓库，而是读取控制器", font=TITLE, fill=INK)
    draw.text(
        (55, 90),
        "BEVFormer 使用稀疏可变形 Attention：Query 决定“从 Value 的哪里读、每处占多大权重”。",
        font=SUBTITLE,
        fill=MUTED,
    )

    rounded_box(draw, (60, 190, 480, 390), "① Query q", ["当前任务状态 / 问题", "例如：这个 BEV 格子还缺什么信息？", "形状：256 维"], BLUE, BLUE_B)
    rounded_box(draw, (630, 190, 1110, 390), "② 预测控制参数", ["Offset：基准点周围去哪里采样", "Weight：每个采样点有多重要", "Δ(q), α(q) 都由 Query 线性预测"], PURPLE, PURPLE_B)
    rounded_box(draw, (1260, 190, 1690, 390), "③ Reference r", ["几何基准坐标", "时间分支：BEV 网格中心", "空间分支：投影到相机的 3D 点"], ORANGE, ORANGE_B)
    rounded_box(draw, (1760, 190, 2140, 390), "④ Value V", ["真正被读取的数据", "时间：BEV Feature", "空间：相机 FPN Feature"], GREEN, GREEN_B)
    arrow(draw, (480, 290), (630, 290), BLUE_B)
    arrow(draw, (1110, 290), (1260, 290), PURPLE_B)

    rounded_box(
        draw,
        (480, 520, 1720, 780),
        "⑤ 在 r + Δ(q) 处采样 Value",
        [
            "sampling_location = reference_point + normalized_offset",
            "sample_k = BilinearSample(Value, sampling_location_k)",
            "双线性插值：采样坐标可以落在四个像素 / 网格点之间。",
        ],
        WHITE,
        BORDER,
    )
    arrow(draw, (1475, 390), (1320, 520), ORANGE_B)
    arrow(draw, (1950, 390), (1580, 520), GREEN_B)
    arrow(draw, (870, 390), (920, 520), PURPLE_B)

    rounded_box(
        draw,
        (600, 890, 1600, 1110),
        "⑥ 加权求和，再写回 Query",
        [
            "head_output = Σ_k α_k(q) · sample_k",
            "output = OutputProjection(Concat(all heads))",
            "new_query = LayerNorm(old_query + Dropout(output))",
        ],
        RED,
        RED_B,
    )
    arrow(draw, (1100, 780), (1100, 890), RED_B)

    draw.rounded_rectangle((80, 1190, 2120, 1270), radius=16, fill=BLUE, outline=BLUE_B, width=3)
    draw.text(
        (110, 1212),
        "一条闭合链路：Query → 预测 Offset/Weight → Reference 附近读取 Value → 加权汇总 → 残差更新 Query。",
        font=BOX_TITLE,
        fill=INK,
    )

    path = OUT_DIR / "07-deformable-attention-principle.png"
    image.save(path, "PNG", optimize=True)


def render_single_layer() -> None:
    image = Image.new("RGB", (2400, 3200), BG)
    draw = ImageDraw.Draw(image)
    draw.text((55, 28), "一个 BEVFormer Encoder Layer：从输入到输出的完整数据流", font=TITLE, fill=INK)
    draw.text(
        (55, 88),
        "蓝色是时间融合，绿色是相机空间融合，橙色是逐格 FFN；每一段都以残差 + LayerNorm 结束。",
        font=SUBTITLE,
        fill=MUTED,
    )

    # Inputs.
    rounded_box(draw, (60, 165, 560, 340), "主状态 Q(l)", ["[B, 40000, 256]", "40000 个固定 BEV 格子", "本层要被更新的 Query"], WHITE, BORDER)
    rounded_box(draw, (660, 165, 1160, 340), "时间侧数据", ["上一帧 BEV + 当前 BEV Value", "2D Reference + 自车运动 shift", "都位于 200×200 BEV 平面"], BLUE, BLUE_B)
    rounded_box(draw, (1260, 165, 1830, 340), "空间侧数据", ["六相机 × 四级 FPN Feature", "每格 4 个 3D 高度参考点", "lidar2img 投影 + 可见性 mask"], GREEN, GREEN_B)
    rounded_box(draw, (1930, 165, 2340, 340), "本层参数 θ(l)", ["TSA / SCA / FFN", "LayerNorm", "与其他五层不共享"], PURPLE, PURPLE_B)

    # TSA panel.
    draw.rounded_rectangle((45, 410, 2355, 1245), radius=24, fill="#F4F9FF", outline=BLUE_B, width=4)
    draw.text((75, 438), "A. Temporal Self-Attention：在 BEV 时间轴上读取上一帧和当前帧", font=H2, fill=INK)
    rounded_box(draw, (90, 520, 550, 720), "A1 · 控制向量", ["q_ctrl = Q(l) + BEV_pos", "c = Concat(prev_value, q_ctrl)", "每格：512 维"], BLUE, BLUE_B)
    rounded_box(draw, (660, 520, 1120, 720), "A2 · Query 预测", ["Linear(c) → Offset", "Linear(c) → Weight logits", "8 Head × 2 Queue × 4 Point"], PURPLE, PURPLE_B)
    rounded_box(draw, (1230, 520, 1690, 720), "A3 · Value 分 Head", ["ValueProjection(BEV Value)", "256 = 8 Head × 32 维", "每个 Head 独立采样"], GREEN, GREEN_B)
    rounded_box(draw, (1800, 520, 2300, 720), "A4 · 两组基准点", ["历史：ref2d + ego shift", "当前：ref2d", "坐标均归一化到 [0,1]"], ORANGE, ORANGE_B)
    # A1 drives the two control projections. Value and reference are parallel
    # data branches; they are deliberately not drawn as outputs of A2.
    arrow(draw, (550, 620), (660, 620), BLUE_B)
    arrow(draw, (310, 340), (310, 520), BLUE_B)
    arrow(draw, (910, 340), (1460, 520), GREEN_B)
    arrow(draw, (1010, 340), (2040, 520), ORANGE_B)

    rounded_box(draw, (330, 820, 1020, 1045), "A5 · 稀疏双线性采样", ["location = ref(t) + Offset/(200,200)", "sample = Bilinear(Value(t,head), location)", "每个队列、每个 Head 只读 4 个点"], WHITE, BLUE_B)
    rounded_box(draw, (1170, 820, 2070, 1045), "A6 · 汇总并写回", ["每个队列：Σ Weight × Sample", "历史结果与当前结果取平均 → 8 Head 拼接", "O_time = OutputProjection(256维)"], WHITE, BLUE_B)
    arrow(draw, (870, 720), (690, 820), PURPLE_B)
    arrow(draw, (990, 720), (1500, 820), PURPLE_B)
    arrow(draw, (1460, 720), (1560, 820), GREEN_B)
    arrow(draw, (2040, 720), (1840, 820), ORANGE_B)
    arrow(draw, (1020, 930), (1170, 930), BLUE_B)
    rounded_box(draw, (710, 1090, 1690, 1205), "A7 · 残差与归一化", ["A(l) = LayerNorm(Q(l) + Dropout(O_time))    →    [B,40000,256]"], BLUE, BLUE_B)
    arrow(draw, (1620, 1045), (1380, 1090), BLUE_B)

    # SCA panel.
    draw.rounded_rectangle((45, 1305, 2355, 2370), radius=24, fill="#F2FBF6", outline=GREEN_B, width=4)
    draw.text((75, 1335), "B. Spatial Cross-Attention：把 BEV 柱投影到六相机，并读取对应图像 Feature", font=H2, fill=INK)
    rounded_box(draw, (90, 1425, 565, 1660), "B1 · 几何基准", ["每格建立 4 个高度点", "P3D → lidar2img → (u,v)", "得到 ref_cam 与 bev_mask"], ORANGE, ORANGE_B)
    rounded_box(draw, (670, 1425, 1145, 1660), "B2 · 可见 Query 分组", ["每台相机只保留能看见的格子", "queries_rebatch：按相机重排", "不可见格子不执行采样"], GREEN, GREEN_B)
    rounded_box(draw, (1250, 1425, 1725, 1660), "B3 · Query 预测", ["A(l) → Offset", "A(l) → Weight logits", "8 Head × 4 Level × 8 Point"], PURPLE, PURPLE_B)
    rounded_box(draw, (1830, 1425, 2305, 1660), "B4 · 图像 Value", ["六相机四级 FPN Feature", "ValueProjection", "256 = 8 Head × 32 维"], GREEN, GREEN_B)
    # Geometry produces visibility/ref_cam. Visible queries then drive B3.
    # Camera values are a separate branch supplied by the FPN features.
    arrow(draw, (565, 1545), (670, 1545), ORANGE_B)
    arrow(draw, (1145, 1545), (1250, 1545), GREEN_B)
    # The output of TSA is the controller/residual input of SCA.
    arrow(draw, (900, 1205), (900, 1425), BLUE_B)

    rounded_box(draw, (150, 1770, 940, 2045), "B5 · 每相机稀疏采样", ["4 个 FPN Level", "每 Level 共 8 Point = 4 高度 × 2 偏移", "location = projected_ref + Offset/(W_level,H_level)", "Weight 对 4×8=32 个候选点做 Softmax"], WHITE, GREEN_B)
    rounded_box(draw, (1075, 1770, 2250, 2045), "B6 · Feature 与 Query 在这里真正交互", ["sample = Bilinear(CameraFeature(head), location)", "camera_result = Concat_heads(Σ Weight × sample)", "Query 提供 Offset/Weight；Feature 提供被读取的数值", "并不是把 Query 和整张 Feature 直接相加"], WHITE, GREEN_B)
    arrow(draw, (330, 1660), (600, 1770), ORANGE_B)
    arrow(draw, (1420, 1660), (600, 1770), PURPLE_B)
    arrow(draw, (1550, 1660), (1510, 1770), PURPLE_B)
    arrow(draw, (2070, 1660), (1900, 1770), GREEN_B)
    arrow(draw, (940, 1910), (1075, 1910), GREEN_B)

    rounded_box(draw, (360, 2120, 1030, 2305), "B7 · 六相机合并", ["scatter 回原来的 40000 个格子", "只对可见相机求平均", "slots /= visible_camera_count"], GREEN, GREEN_B)
    rounded_box(draw, (1180, 2120, 2040, 2305), "B8 · 残差与归一化", ["O_space = OutputProjection(slots)", "B(l) = LN(A(l) + Dropout(O_space))", "输出仍为 [B,40000,256]"], GREEN, GREEN_B)
    arrow(draw, (1620, 2045), (700, 2120), GREEN_B)
    arrow(draw, (1030, 2210), (1180, 2210), GREEN_B)

    # FFN.
    draw.rounded_rectangle((45, 2430, 2355, 2810), radius=24, fill="#FFF8EF", outline=ORANGE_B, width=4)
    draw.text((75, 2460), "C. FFN：每个格子内部独立做通道变换", font=H2, fill=INK)
    rounded_box(draw, (180, 2545, 710, 2735), "C1 · 扩维", ["B(l)：256维", "Linear 256→512", "ReLU + Dropout"], ORANGE, ORANGE_B)
    rounded_box(draw, (930, 2545, 1460, 2735), "C2 · 压回", ["Linear 512→256", "得到通道更新量 O_ffn", "格子之间不互相采样"], ORANGE, ORANGE_B)
    rounded_box(draw, (1680, 2545, 2220, 2735), "C3 · 输出本层", ["Q(l+1)=LN(B(l)+Dropout(O_ffn))", "形状：[B,40000,256]", "交给下一层"], ORANGE, ORANGE_B)
    arrow(draw, (710, 2640), (930, 2640), ORANGE_B)
    arrow(draw, (1460, 2640), (1680, 2640), ORANGE_B)

    # Closure.
    draw.rounded_rectangle((45, 2870, 2355, 3130), radius=24, fill=PURPLE, outline=PURPLE_B, width=4)
    draw.text((80, 2900), "闭合关系", font=H2, fill=INK)
    draw.text((80, 2955), "输入 Q(l) 经过 TSA → SCA → FFN 后得到同形状 Q(l+1)。", font=BOX_TITLE, fill=INK)
    draw.text((80, 3000), "下一层只需把 Q(l+1) 当作自己的主状态；相机 Feature 与几何基准继续复用。", font=BODY, fill=MUTED)
    draw.text((80, 3040), "每层参数独立，因此 Query 更新后会预测出新的 Offset、Weight 和采样位置。", font=BODY, fill=MUTED)
    draw.text((80, 3080), "执行六次后：Q(0) → Q(1) → … → Q(6)，Q(6) 就是当前帧最终 BEV Feature。", font=BODY, fill=MUTED)

    path = OUT_DIR / "08-single-bev-encoder-layer-closed-loop.png"
    image.save(path, "PNG", optimize=True)


def render_six_layer_loop() -> None:
    image = Image.new("RGB", (2200, 1320), BG)
    draw = ImageDraw.Draw(image)
    draw.text((55, 30), "六层 Encoder：同一套计算结构，六套独立参数", font=TITLE, fill=INK)
    draw.text((55, 90), "形状和网格不变；Query 内容、Offset、Weight 与采样结果逐层更新。", font=SUBTITLE, fill=MUTED)

    rounded_box(draw, (70, 185, 470, 370), "Q(0)", ["初始 BEV Query", "[B,40000,256]", "固定 200×200 网格"], WHITE, BORDER)
    xs = [560, 830, 1100, 1370, 1640, 1910]
    colors = [(BLUE, BLUE_B), (GREEN, GREEN_B), (ORANGE, ORANGE_B), (PURPLE, PURPLE_B), (BLUE, BLUE_B), (GREEN, GREEN_B)]
    previous_x = 470
    for index, (x, (fill, outline)) in enumerate(zip(xs, colors), start=1):
        rounded_box(draw, (x, 185, x + 220, 370), f"Layer {index}", ["TSA", "SCA", "FFN"], fill, outline, body_font=SMALL)
        arrow(draw, (previous_x, 278), (x, 278), outline)
        previous_x = x + 220

    rounded_box(draw, (1660, 460, 2130, 640), "Q(6)", ["最终 BEV Feature", "[B,40000,256]", "送入 Detection Decoder"], RED, RED_B)
    arrow(draw, (2020, 370), (1880, 460), RED_B)

    draw.text((70, 475), "六层循环的伪代码", font=H2, fill=INK)
    code_box = (70, 530, 1510, 1040)
    draw.rounded_rectangle(code_box, radius=18, fill="#15212F", outline="#324A5F", width=3)
    code_lines = [
        "geometry = build_reference_points_and_camera_projection_once()",
        "Q = Q0",
        "for l in range(6):",
        "    # 每个 layers[l] 都有自己的参数 theta[l]",
        "    T = temporal_deformable_attention(Q, temporal_values, ref2d, theta[l])",
        "    A = layer_norm_1[l](Q + dropout(T))",
        "",
        "    S = spatial_deformable_attention(A, camera_features, geometry, theta[l])",
        "    B = layer_norm_2[l](A + dropout(S))",
        "",
        "    U = linear_2[l](relu(linear_1[l](B)))",
        "    Q = layer_norm_3[l](B + dropout(U))",
        "return Q  # Q6",
    ]
    y = 558
    mono = font(20)
    for line in code_lines:
        draw.text((100, y), line, font=mono, fill="#E7F1FA")
        y += 35

    rounded_box(draw, (1580, 735, 2130, 1040), "循环中什么被复用", ["六相机 FPN Feature", "2D / 3D Reference", "相机投影与 bev_mask", "固定 200×200 网格"], WHITE, BORDER)
    rounded_box(draw, (1580, 1090, 2130, 1260), "每层什么会变化", ["输入 Query、独立层参数", "Offset、Weight、采样位置与输出"], PURPLE, PURPLE_B)
    rounded_box(draw, (70, 1090, 1510, 1260), "不要误解", ["六层 ≠ 六台相机；六层 ≠ 六个时间帧；六层 ≠ 六个不同分辨率。", "它是同一张 BEV 状态在六个独立 Transformer Layer 中连续迭代。"], BLUE, BLUE_B)

    path = OUT_DIR / "09-six-layer-bev-encoder-loop.png"
    image.save(path, "PNG", optimize=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_attention_principle()
    render_single_layer()
    render_six_layer_loop()
    for name in [
        "07-deformable-attention-principle.png",
        "08-single-bev-encoder-layer-closed-loop.png",
        "09-six-layer-bev-encoder-loop.png",
    ]:
        print(OUT_DIR / name)


if __name__ == "__main__":
    main()
