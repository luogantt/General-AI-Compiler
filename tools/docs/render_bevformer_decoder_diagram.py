"""Render the decoder overview and a numerical two-stage sampling example."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs/assets/bevformer-query-guide/10-decoder-to-detections.png"
WIDTH, HEIGHT = 1120, 1690
INK, MUTED = "#172B4D", "#41566F"


def font(size, bold=False):
    path = Path("C:/Windows/Fonts/msyhbd.ttc" if bold else "C:/Windows/Fonts/msyh.ttc")
    return ImageFont.truetype(str(path), size)


def render_sampling_example():
    image = Image.new("RGB", (1120, 1110), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    draw.text((46, 28), "两次加权，各自解决不同的问题", font=font(36, True), fill=INK)
    draw.text((46, 86), "示意值：仅显示一个 Head 的 32 维 Value 中的前两维", font=font(24), fill=MUTED)

    draw.rounded_rectangle((38, 140, 1082, 562), radius=15, fill="#F0F6FF", outline="#A6BEDC", width=2)
    draw.text((62, 160), "① 一次采样：用距离算插值系数 β", font=font(29, True), fill=INK)
    # A geometrically accurate 2x2 cell: x fraction 1/4, y fraction 1/2.
    left, right, top, bottom = 120, 430, 280, 450
    draw.rectangle((left, top, right, bottom), outline="#7B97B8", width=3)
    sx, sy = left + 0.25 * (right-left), top + 0.5 * (bottom-top)
    draw.line((sx, top, sx, bottom), fill="#DF9D47", width=2)
    draw.line((left, sy, right, sy), fill="#DF9D47", width=2)
    for px, py in [(left, top), (right, top), (left, bottom), (right, bottom)]:
        draw.ellipse((px-7, py-7, px+7, py+7), fill="#2866AF")
    draw.ellipse((sx-9, sy-9, sx+9, sy+9), fill="#C86C11")
    for xy, text in [
        ((70, 222), "(10,20)：[2,0]"),
        ((335, 222), "(11,20)：[6,0]"),
        ((70, 475), "(10,21)：[2,4]"),
        ((335, 475), "(11,21)：[6,4]"),
        ((sx+17, sy+12), "(10.25,20.5)"),
    ]:
        draw.text(xy, text, font=font(22), fill=MUTED)

    for y, text in [
        (232, "偏移 Δ = (0.25, 0.5) 格"),
        (278, "β = [0.375, 0.125,"),
        (312, "        0.375, 0.125]"),
        (366, "邻近四格按 β 加权"),
        (460, "β 来自距离，不由 Softmax 生成"),
    ]:
        draw.text((600, y), text, font=font(23), fill=MUTED)
    draw.text((600, 404), "f1 = [3, 2]", font=font(29, True), fill="#205BA2")

    draw.line((560, 572, 560, 598), fill="#536F91", width=3)
    draw.polygon(((554, 592), (566, 592), (560, 600)), fill="#536F91")
    draw.rounded_rectangle((38, 610, 1082, 902), radius=15, fill="#EDF8F2", outline="#9BC9B0", width=2)
    draw.text((62, 628), "② 多次采样：用 Query 预测的 α 做汇总", font=font(29, True), fill=INK)
    cards = [("f1 = [3,2]", "α1 = 0.5"), ("f2 = [6,1]", "α2 = 0.2"),
             ("f3 = [0,4]", "α3 = 0.2"), ("f4 = [2,0]", "α4 = 0.1")]
    for i, (value, weight) in enumerate(cards):
        x = 64 + i * 252
        draw.rounded_rectangle((x, 692, x+235, 796), radius=10, fill="#FFFFFF", outline="#ABCDBB", width=2)
        draw.text((x+18, 706), value, font=font(27, True), fill=INK)
        draw.text((x+18, 754), weight, font=font(24), fill=MUTED)
    draw.text((68, 827), "z = 0.5[3,2] + 0.2[6,1] + 0.2[0,4] + 0.1[2,0] = [2.9,2]", font=font(25, True), fill="#226447")

    draw.line((560, 912, 560, 936), fill="#536F91", width=3)
    draw.polygon(((554, 930), (566, 930), (560, 938)), fill="#536F91")
    draw.rounded_rectangle((38, 950, 1082, 1073), radius=15, fill="#FFF6E9", outline="#D6B98E", width=2)
    draw.text((62, 966), "此时得到的是 Head 汇总特征，不是框坐标", font=font(28, True), fill=INK)
    draw.text((62, 1018), "再完成：8 个 Head 拼接 → 输出投影 → 与原候选相加 → Norm", font=font(24), fill=MUTED)
    output = OUTPUT.with_name("11-decoder-sampling-example.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)
    print(output)


def main():
    image = Image.new("RGB", (WIDTH, HEIGHT), "#FFFFFF")
    draw = ImageDraw.Draw(image)
    draw.text((54, 28), "BEVFormer-base：Decoder 如何得到检测框", font=font(35, True), fill=INK)
    draw.text((54, 84), "从上向下读；中间四步重复六层。公式与实现见正文。", font=font(24), fill=MUTED)

    # The frame marks repeated computation; it is not another data dependency.
    draw.rounded_rectangle((32, 310, 1088, 1115), radius=18, fill="#F6F9FE", outline="#9BB5D7", width=2)
    draw.text((55, 326), "重复 6 层：每层参数独立，使用同一份 BEV 特征 E", font=font(25, True), fill="#205BA2")

    stages = [
        (140, "1—2  准备输入和初始参考点", [
            "BEV：E [40000,256]；目标候选：x [900,256]",
            "位置向量 p → 线性层 + sigmoid → 参考点 r [900,3]",
        ], "#EEF5FF"),
        (380, "3  候选之间交换信息：Self-Attention", [
            "900 × 900 全量读取其他候选 → 残差 + Norm",
            "输出 U [900,256]",
        ], "#FFFFFF"),
        (560, "4  候选读取 BEV：Deformable Cross-Attention", [
            "U + p → Δ、α；r 的 xy + Δ/200 → 采样位置（8 × 4）",
            "读取 E → 每 Head 加权求和 → 残差 + Norm",
        ], "#EAF7F2"),
        (740, "5  加工候选内容：FFN", [
            "256 → 512 → ReLU → 256 → 残差 + Norm",
            "得到新的 x [900,256]",
        ], "#FFFFFF"),
        (920, "6  预测框修正量，更新参考点", [
            "新 x → 回归头 → 修正中心 r，并保留本层结果",
            "新 x、新 r 进入下一层；第 6 层结束后向下继续",
        ], "#FFFFFF"),
        (1160, "7  分类和框编码", [
            "每层输出类别 logits 与框编码，各 [900,10]",
            "中心已换算成米；推理选择最后一层结果",
        ], "#FFF6E9"),
        (1340, "8  筛选、还原物理量", [
            "900 × 10 个类别分数 → sigmoid → Top-300",
            "尺寸 exp、角度 atan2 → 范围过滤 → 中心 z 转底中心",
        ], "#FFF6E9"),
        (1520, "最终输出：K ≤ 300", [
            "3D 框 [K,9]  +  类别 [K]  +  分数 [K]",
        ], "#EAF7F2"),
    ]

    x0, x1, box_h = 62, 1058, 140
    for index, (top, title, lines, fill) in enumerate(stages):
        bottom = top + (110 if index == len(stages) - 1 else box_h)
        draw.rounded_rectangle((x0, top, x1, bottom), radius=12, fill=fill, outline="#B4C5D9", width=2)
        draw.text((x0 + 22, top + 15), title, font=font(27, True), fill=INK)
        for line_index, line in enumerate(lines):
            draw.text((x0 + 22, top + 61 + 34 * line_index), line, font=font(23), fill=MUTED)
        if index + 1 < len(stages):
            next_top = stages[index + 1][0]
            start = bottom + 6
            # Do not run an arrow through the repeated-layer frame's heading.
            end = min(next_top - 9, start + 24)
            draw.line((560, start, 560, end), fill="#536F91", width=3)
            draw.polygon(((554, end - 6), (566, end - 6), (560, end + 2)), fill="#536F91")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT)
    print(OUTPUT)
    render_sampling_example()


if __name__ == "__main__":
    main()
