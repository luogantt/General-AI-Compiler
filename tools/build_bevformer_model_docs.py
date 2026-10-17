"""Build the clickable PyTorch BEVFormer-base model map, without importing torch.

python tools/build_bevformer_model_docs.py --source-root /path/to/BEVFormer-master

Local excerpts are extracted from the supplied checkout. Framework excerpts use
the versions in BEVFormer's installation guide, cached under build/ for offline
rebuilds. The generated HTML embeds its data and needs no server or dependencies.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
PLUGIN = "projects/mmdet3d_plugin/"
MODEL = PLUGIN + "bevformer/"
CONFIG = "projects/configs/bevformer/bevformer_base.py"
DETECTOR = MODEL + "detectors/bevformer.py"
HEAD = MODEL + "dense_heads/bevformer_head.py"
TRANSFORMER = MODEL + "modules/transformer.py"
ENCODER = MODEL + "modules/encoder.py"
DECODER = MODEL + "modules/decoder.py"
TSA = MODEL + "modules/temporal_self_attention.py"
SCA = MODEL + "modules/spatial_cross_attention.py"
BASE_LAYER = MODEL + "modules/custom_base_transformer_layer.py"
CODER = PLUGIN + "core/bbox/coders/nms_free_coder.py"
RESNET = "mmdet/models/backbones/resnet.py"
FPN = "mmdet/models/necks/fpn.py"
MMCV_TRANSFORMER = "mmcv/cnn/bricks/transformer.py"
DCN = "mmcv/ops/modulated_deform_conv.py"


class SourceCatalog:
    def __init__(self, source_root: Path):
        self.root = source_root.resolve()
        self.sources = {}
        self.external = {}

    def dependency(self, path: str, repository: str, tag: str):
        cache = ROOT / "build/bevformer-architecture-sources" / tag / path
        if not cache.exists():
            url = f"https://raw.githubusercontent.com/open-mmlab/{repository}/{tag}/{path}"
            with urlopen(url, timeout=45) as response:
                content = response.read()
            ast.parse(content.decode("utf-8"))
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(content)
        self.external[path] = (cache, f"{repository} {tag}",
                               f"https://github.com/open-mmlab/{repository}/blob/{tag}/{path}")

    def add(self, key, path, symbol=None, *, anchor=None, count=25, before=0, note=""):
        if path in self.external:
            source, owner, url = self.external[path]
            local_url = None
        else:
            source, owner = self.root / path, "本地 BEVFormer-master"
            url = f"https://github.com/fundamentalvision/BEVFormer/blob/master/{path}"
            local_url = source.as_uri()
        raw = source.read_bytes()
        text = raw.decode("utf-8-sig")
        lines = text.splitlines()
        low, high = 1, len(lines)
        if symbol:
            node = ast.parse(text)
            for name in symbol.split("."):
                node = next(n for n in node.body if isinstance(
                    n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
            low, high = node.lineno, node.end_lineno
        if anchor:
            matches = [i + 1 for i in range(low - 1, high) if anchor in lines[i]]
            if len(matches) != 1:
                raise ValueError(f"{path}: anchor {anchor!r} has {len(matches)} matches")
            start = max(low, matches[0] - before)
        else:
            start = low
        end = min(high, start + count - 1)
        self.sources[key] = dict(file=path, owner=owner, symbol=symbol or anchor,
                                 start=start, end=end, sha256=hashlib.sha256(raw).hexdigest(),
                                 code="\n".join(lines[start-1:end]), localUrl=local_url,
                                 localPath=str(source.resolve()) if local_url else None,
                                 url=url + (f"#L{start}-L{end}" if path in self.external else ""),
                                 note=note or ("按安装指南版本引用的外部依赖；不表示本机已安装此版本。"
                                               if path in self.external else
                                               "代码及行号来自本地文件快照；上游 master 仅供对照，可能与本地不同。"))
        return key


def build_catalog(source_root: Path):
    catalog = SourceCatalog(source_root)
    for path, repository, tag in [
        (RESNET, "mmdetection", "v2.14.0"), (FPN, "mmdetection", "v2.14.0"),
        (MMCV_TRANSFORMER, "mmcv", "v1.4.0"), (DCN, "mmcv", "v1.4.0"),
    ]:
        catalog.dependency(path, repository, tag)
    add = catalog.add
    add("config", CONFIG, anchor="_dim_ = 256", count=30)
    add("input-config", CONFIG, anchor="test_pipeline =", count=20)
    add("normalize", PLUGIN + "datasets/pipelines/transform_3d.py", "NormalizeMultiviewImage.__call__", count=14)
    add("pad", PLUGIN + "datasets/pipelines/transform_3d.py", "PadMultiViewImage._pad_img", count=16)
    add("extract", DETECTOR, "BEVFormer.extract_img_feat", anchor="B = img.size(0)", count=34)
    add("entry", DETECTOR, "BEVFormer.simple_test", count=20)
    add("entry-head", DETECTOR, "BEVFormer.simple_test_pts", count=14)
    add("history", DETECTOR, "BEVFormer.forward_test", anchor="if img_metas[0][0]['scene_token']", count=29)
    add("backbone-config", CONFIG, anchor="img_backbone=dict(", count=12)
    add("resnet-stages", RESNET, "ResNet", anchor="arch_settings =", count=8)
    add("resnet-forward", RESNET, "ResNet.forward", count=24)
    add("resnet-stem", RESNET, "ResNet._make_stem_layer", count=65)
    add("bottleneck", RESNET, "Bottleneck.forward", anchor="def _inner_forward", count=40)
    add("dcn-choice", RESNET, "Bottleneck.__init__", anchor="fallback_on_stride = False", count=24)
    add("dcn-forward", DCN, "ModulatedDeformConv2dPack.forward", count=18)
    add("fpn-config", CONFIG, anchor="img_neck=dict(", count=8)
    add("fpn-forward", FPN, "FPN.forward", anchor="laterals =", count=40)
    add("head-query", HEAD, "BEVFormerHead._init_layers", anchor="if not self.as_two_stage:", count=6)
    add("head-forward", HEAD, "BEVFormerHead.forward", anchor="bs, num_cam", count=37)
    add("motion", TRANSFORMER, "PerceptionTransformer.get_bev_features", anchor="delta_x =", count=24)
    add("rotate", TRANSFORMER, "PerceptionTransformer.get_bev_features", anchor="if prev_bev is not None:", count=24)
    add("features", TRANSFORMER, "PerceptionTransformer.get_bev_features", anchor="feat_flatten = []", count=40)
    add("transformer", TRANSFORMER, "PerceptionTransformer.forward", anchor="bev_embed = self.get_bev_features(", count=39)
    add("object-queries", TRANSFORMER, "PerceptionTransformer.forward", anchor="query_pos, query = torch.split(", count=14)
    add("encoder-config", CONFIG, anchor="encoder=dict(", count=28)
    add("encoder-ref", ENCODER, "BEVFormerEncoder.forward", anchor="ref_3d = self.get_reference_points(", count=12)
    add("project", ENCODER, "BEVFormerEncoder.point_sampling", anchor="reference_points_cam = torch.matmul", count=21)
    add("encoder-loop", ENCODER, "BEVFormerEncoder.forward", anchor="for lid, layer in enumerate(self.layers):", count=29)
    add("encoder-layer", ENCODER, "BEVFormerLayer.forward", anchor="for layer in self.operation_order:", count=59)
    add("tsa", TSA, "TemporalSelfAttention.forward", anchor="if value is None:", count=33)
    add("tsa-combine", TSA, "TemporalSelfAttention.forward", anchor="output = output.view(num_query", count=17)
    add("sca", SCA, "SpatialCrossAttention.forward", anchor="D = reference_points_cam.size(3)", count=37)
    add("sca-aggregate", SCA, "SpatialCrossAttention.forward", anchor="slots[j, index_query_per_img] +=", before=2, count=13)
    add("norm-build", BASE_LAYER, "MyCustomBaseTransformerLayer.__init__", anchor="self.norms = ModuleList()", count=4)
    add("ffn", MMCV_TRANSFORMER, "FFN.__init__", anchor="layers = []", count=15)
    add("ffn-forward", MMCV_TRANSFORMER, "FFN.forward", count=11)
    add("decoder-config", CONFIG, anchor="decoder=dict(", count=22)
    add("decoder-loop", DECODER, "DetectionTransformerDecoder.forward", anchor="for lid, layer in enumerate(self.layers):", count=15)
    add("decoder-refine", DECODER, "DetectionTransformerDecoder.forward", anchor="if reg_branches is not None:", count=24)
    add("mha", MMCV_TRANSFORMER, "MultiheadAttention.forward", anchor="out = self.attn(", count=12)
    add("decoder-cross", DECODER, "CustomMSDeformableAttention.forward", anchor="value = self.value_proj(value)", count=39)
    add("heads-init", HEAD, "BEVFormerHead._init_layers", count=26)
    add("heads", HEAD, "BEVFormerHead.forward", anchor="for lvl in range(hs.shape[0]):", count=38)
    add("raw-output", HEAD, "BEVFormerHead.forward", anchor="outs = {", count=10)
    add("decode", CODER, "NMSFreeCoder.decode_single", anchor="max_num = self.max_num", count=13)
    add("decode-last", CODER, "NMSFreeCoder.decode", anchor="all_cls_scores =", count=10)
    add("decode-range", CODER, "NMSFreeCoder.decode_single", anchor="if self.post_center_range is not None:", count=22)
    add("decode-box", PLUGIN + "core/bbox/util.py", "denormalize_bbox", count=35)
    add("get-bboxes", HEAD, "BEVFormerHead.get_bboxes", anchor="preds_dicts = self.bbox_coder.decode", count=22)
    nodes = {}

    def node(key, title, subtitle, summary, inp, action, out, sources, *, notes=(), related=(),
             kind="compute", view=None, formula=None):
        nodes[key] = dict(title=title, subtitle=subtitle, kind=kind, summary=summary,
                          input=inp, action=action, output=out, sources=sources,
                          notes=list(notes), related=list(related), view=view, formula=formula)

    node("config", "模型配置", "bevformer_base.py", "本图采用完整 BEVFormer-base 的相机 3D 检测配置。", "模型结构参数与训练得到的权重。",
         "配置文件组装 ResNet-101、DCNv2、FPN、6 层 Encoder、6 层 Decoder 和检测头。",
         "BEV 网格 200×200；隐藏维度 256；目标 Query 900；类别 10。", ["config", "encoder-config", "decoder-config"],
         kind="config", notes=["Base 与 Tiny 共用许多类，但网络深度、图像尺度与 BEV 网格不同；本图不混用两套尺寸。",
         "图中 B=批量大小，N=6 路相机，C=256；具体尺寸示例按 B=1、输入已 padding 到 928×1600。"])
    node("model", "完整模型入口", "BEVFormer.simple_test()", "模型组织类负责提取特征、调用检测头，并维护推理结果。", "img、img_metas、可选 prev_bev。",
         "extract_feat() → simple_test_pts() → pts_bbox_head() → get_bboxes()。",
         "当前帧检测结果与供下一帧使用的 BEV 特征。", ["entry", "entry-head"], kind="config", related=["input", "head", "history", "output"])
    node("head", "检测头总入口", "BEVFormerHead.forward()", "这个类管理 Query、Transformer、分类分支和回归分支；并非一个单独卷积层。",
         "FPN 多尺度特征 list[Tensor]、img_metas、prev_bev。", "读取 learned embeddings，调用 PerceptionTransformer，再对每一层 Decoder 的输出进行预测。",
         "字典：bev_embed、all_cls_scores、all_bbox_preds；两个 enc_* 字段在该配置为 None。", ["head-forward", "heads", "raw-output"],
         kind="config", related=["bev-queries", "object-queries", "transformer", "predict"])
    node("transformer", "Transformer 总入口", "PerceptionTransformer", "先用 Encoder 把相机特征转换为 BEV，再用 Decoder 查询 BEV 中的目标。",
         "多尺度图像特征、BEV Query、Object Query、标定与历史特征。", "get_bev_features() → Encoder；生成目标参考点 → Decoder。",
         "bev_embed、各层目标特征 inter_states、初始与逐层更新的 reference points。", ["transformer", "features"],
         kind="config", related=["encoder", "bev", "decoder"])
    node("input", "六相机图像输入", "img · 当前帧", "六张图像覆盖车辆周围；输入是图像张量，而非点云或已生成的 BEV 图。",
         "nuScenes 原始图像通常每路 900×1600，附带相机标定和自车状态。",
         "按配置做 BGR 通道归一化、padding 到 32 的倍数；相机维与 batch 合并后送入同一个 Backbone。",
         "img: [B,6,3,Hpad,Wpad]；示例 [1,6,3,928,1600] → Backbone 输入 [6,3,928,1600]。",
         ["input-config", "normalize", "pad", "extract"], kind="input", related=["backbone", "geometry"],
         notes=["forward_test 最外层还可能有测试增强的 list 包装；这里展示送到特征提取函数的张量。",
                "模型实际输入尺寸由数据管线决定，928×1600 是本图示例，不是 PyTorch 模型唯一允许的尺寸。"])
    node("backbone", "图像 Backbone", "ResNet-101 + DCNv2", "每个相机使用同一套权重提取二维图像特征，六路图像尚未在这里融合成 BEV。",
         "[B×6,3,Hpad,Wpad]。", "Stem → C2 → C3 → C4 → C5；C4、C5 的 Bottleneck 中用 DCNv2 替换中间 3×3 卷积。",
         "C3/C4/C5 三个张量，通道分别 512/1024/2048，stride 分别 8/16/32。", ["backbone-config", "resnet-forward", "resnet-stages"],
         view="backbone", related=["dcn", "fpn"], notes=["ResNet-101 的四个 stage 含 3、4、23、3 个 Bottleneck；不是只有主图这一个方框那么简单。"])
    node("stem", "Stem", "7×7 Conv → BN → ReLU → MaxPool", "先降低图像分辨率，再进入残差模块。", "[6,3,928,1600]。",
         "7×7 卷积 stride=2，然后 3×3 MaxPool stride=2。", "[6,64,232,400]，总 stride=4。", ["resnet-stem", "resnet-forward"])
    for key, title, depth, channels, size, src, text in [
        ("c2", "C2 / layer1", 3, 256, "232,400", ["resnet-stages", "bottleneck"], "普通 Bottleneck；继续传给 C3，不作为该配置的 FPN 直接输入。"),
        ("c3", "C3 / layer2", 4, 512, "116,200", ["resnet-stages", "bottleneck"], "普通 Bottleneck；此级输出同时送给 FPN。"),
        ("c4", "C4 / layer3", 23, 1024, "58,100", ["backbone-config", "dcn-choice", "bottleneck"], "23 个 Bottleneck 的 conv2 为 DCNv2；此级输出同时送给 FPN。"),
        ("c5", "C5 / layer4", 3, 2048, "29,50", ["backbone-config", "dcn-choice", "bottleneck"], "3 个 Bottleneck 的 conv2 为 DCNv2；此级输出送给 FPN。"),
    ]:
        node(key, title, f"{depth} 个残差 Bottleneck", text, "上一个 ResNet stage 的输出。",
             f"每块：1×1 Conv → {'3×3 DCNv2' if key in ('c4', 'c5') else '3×3 Conv'} → 1×1 Conv，加 shortcut 后激活；相应位置带 BN/ReLU。",
             f"示例 [6,{channels},{size}]。", src, related=["dcn"] if key in ("c4", "c5") else [],
             notes=["配置为 caffe style，发生下采样时 stride 放在首个 1×1 卷积，而不是 3×3 卷积。"])
    node("dcn", "DCNv2 可变形卷积", "C4 / C5 内部的 conv2", "根据局部图像内容调整采样位置和采样权重。", "Bottleneck 的中间特征；C4/C5 内部分别为 256/512 通道。",
         "额外卷积预测 offset 和 mask：3×3、deform_groups=1 时为 18+9=27 个通道。偏移后以双线性插值采样，再加权求和。",
         "与对应普通 3×3 卷积相同的输出尺寸；卷积核权重形状固定。", ["dcn-choice", "dcn-forward"],
         notes=["DCNv2 是 Backbone 中的卷积；Encoder 的 MSDeformableAttention 是另一类模块。"], related=["c4", "c5", "sca"])
    node("fpn", "多尺度特征金字塔", "FPN · 4 个尺度", "把不同深度的图像特征融合，并统一为 256 个通道。", "C3、C4、C5，通道 512/1024/2048。",
         "横向 1×1 投影 → 自顶向下上采样与相加 → 3×3 卷积；再从输出额外生成一个更低分辨率尺度。",
         "四个 [B,6,256,Hi,Wi]：示例 116×200、58×100、29×50、15×25。", ["fpn-config", "fpn-forward", "extract"], related=["features", "encoder"],
         notes=["FPN 输出仍是相机视角的特征图；到 Spatial Cross-Attention 才采集到统一 BEV 网格。"])
    node("features", "图像特征准备", "Flatten + 相机 / 尺度编码", "为 Attention 准备多相机、多尺度 Key/Value 及其索引信息。", "FPN 的四个尺度 [B,6,256,Hi,Wi]。",
         "展平每个尺度的 H×W，加入 camera/level embeddings，拼接空间位置。",
         "feat_flatten: [6,30825,B,256]（本图尺寸）；spatial_shapes: [4,2]；level_start_index: [4]。", ["features"], related=["fpn", "sca"],
         notes=["30825 = 116×200 + 58×100 + 29×50 + 15×25；图像尺寸改变后该数值也变化。"])
    node("geometry", "标定与自车状态", "img_metas · 几何信息", "空间融合必须知道 BEV 位置在每个相机中投影到哪里；时间融合必须知道车辆移动了多少。",
         "每样本的 lidar2img（6 个 4×4 矩阵）、img_shape、can_bus（18 维）、scene_token 等。",
         "将每个 BEV 网格柱上的 4 个高度参考点投影到图像，生成可见性 mask；CAN bus 用于平移、旋转和 Query 嵌入。",
         "reference_points_cam: [6,B,40000,4,2]；bev_mask: [6,B,40000,4]；自车运动 shift。", ["encoder-ref", "project", "motion"],
         kind="input", related=["project", "motion", "sca"], notes=["lidar2img 这里是坐标变换矩阵的名字，不意味着该相机模型输入了 LiDAR 点云。"])
    node("project", "BEV → 相机投影", "point_sampling()", "把三维参考位置变成各相机图像上的二维采样参考。",
         "网格参考点、point_cloud_range 与 6 路 lidar2img。", "反归一化到三维坐标 → 4×4 变换 → 除深度 → 按图像宽高归一化 → 检查前方与图像内可见性。",
         "二维参考点与可见性 mask，供 SCA 挑选每个相机真正可见的 Query。", ["project", "encoder-ref"], related=["sca", "geometry"])
    node("history", "上一帧 BEV", "prev_bev · 可选状态", "推理时保存上一帧 BEV，下一帧复用；不需要重复运行上一帧图像 Backbone。",
         "上一帧缓存 [40000,B,256]；新场景的首帧为 None。", "forward_test 检查 scene_token，计算自车运动增量，把 prev_bev 交给 Transformer；完成本帧后更新缓存。",
         "经旋转与平移参考调整的历史信息，进入 Temporal Self-Attention。", ["history", "rotate"],
         kind="state", related=["motion", "tsa", "cache"], notes=["训练的 queue_length=4 与单次推理的缓存输入不是同一个维度；图中展示逐帧推理。"])
    node("motion", "历史对齐与自车编码", "旋转 / shift / CAN bus MLP", "车动了，上一帧的地图坐标也要相应调整。", "prev_bev、CAN bus、自车运动增量、BEV Query。",
         "旋转历史 BEV；把平移换成归一化参考点 shift；CAN bus 经 MLP 后加到 BEV Query。",
         "对齐后的历史状态、时间采样参考点，以及带自车编码的当前 Query。", ["motion", "rotate"], related=["history", "bev-queries", "tsa"])
    node("bev-queries", "BEV 网格 Query", "固定 200×200 网格的可学习 Query", "每个 Query 对应俯视图中的一个固定网格；学习的是 Query 特征，不是网格坐标。",
         "可学习 bev_embedding: [40000,256] 与二维位置编码。", "扩展 batch 维，准备位置编码，结合自车信息进入 Encoder。",
         "Encoder 内部 Query: [B,40000,256]，每一层逐渐融合时间和图像信息。", ["head-query", "head-forward", "rotate"],
         kind="input", related=["encoder", "object-queries"], notes=["BEV 范围 x/y 均为 [-51.2,51.2] 米，200 格对应约 0.512 米/格。"])
    node("encoder", "BEV Encoder", "6 层 · 时序融合 + 空间融合", "将六相机当前图像信息与历史 BEV 融合成当前帧俯视特征。",
         "BEV Query、位置编码、FPN 特征、几何参考点、可选 prev_bev。", "每层：TSA + Residual → LN → SCA + Residual → LN → FFN + Residual → LN。重复 6 次。",
         "Encoder 返回 [B,40000,256]，对应 200×200 网格、每格 256 个特征。", ["encoder-config", "encoder-loop", "encoder-layer"],
         view="encoder", related=["tsa", "sca", "ffn", "bev"], notes=["六层参数分别训练；重复的是结构，不是共用同一层参数。"])
    node("tsa", "时序自注意力 TSA", "TemporalSelfAttention + Residual", "从历史与当前 BEV 状态中采样，融合时间信息。",
         "当前 Query [B,40000,256]、历史 BEV、二维参考点。", "预测采样偏移和注意力权重，对两份 BEV value 做 deformable attention，合并时间结果并加 residual。",
         "[B,40000,256]，随后做 LayerNorm。", ["tsa", "tsa-combine", "encoder-layer"], related=["history", "motion"],
         notes=["首帧 prev_bev=None 时实现会用当前 Query 构造两份 value；不是把这一整层跳过。",
                "这里不是把所有 40000 个网格两两构造完整注意力矩阵。"])
    node("sca", "空间交叉注意力 SCA", "SpatialCrossAttention + Residual", "让 BEV 网格从看得见它的相机特征中取信息，实现图像视角到俯视视角的融合。",
         "当前 BEV Query、四尺度图像 Key/Value、投影参考点和可见性 mask。", "按相机挑选可见 Query 并 rebatch → MSDeformableAttention3D 预测偏移/权重并采样 → 聚合可见相机结果 → 输出投影与 residual。",
         "[B,40000,256]，随后做 LayerNorm。", ["sca", "sca-aggregate", "encoder-config"], related=["project", "features"],
         notes=["实际每个相机的可见 Query 数不同；源码按 max_len 组织批次，不能把每个相机都算 40000 个 Query 当作必需工作量。",
                "配置 num_points=8、num_points_in_pillar=4；不要误解为每个高度点都另有 8 个采样点。"])
    node("ln", "LayerNorm", "沿 256 维通道归一化", "对每个 Query 自己的通道做归一化，稳定特征分布。", "Encoder: [B,40000,256]；Decoder: [900,B,256]。",
         "在最后一维求均值与方差，再乘 learned scale、加 bias。每层有 3 个独立 LayerNorm。", "形状保持不变。",
         ["norm-build", "encoder-layer", "decoder-config"], notes=["Residual 加法在 Attention / FFN 模块内部完成；图里的 LN 是后续的归一化，不再额外重复加 residual。"])
    node("ffn", "前馈网络 FFN", "Linear 256→512→256 + Residual", "对每个位置独立做通道变换；它不负责跨相机或跨网格取数据。",
         "Encoder 或 Decoder 的 Query 特征，最后一维 256。", "Linear(256,512) → ReLU → Linear(512,256) → Residual；配置中含 Dropout，eval 模式关闭随机丢弃。",
         "Query 数和 256 维输出保持不变。", ["ffn", "ffn-forward", "encoder-config"], formula="y = x + W₂ · ReLU(W₁ · x + b₁) + b₂")
    node("bev", "当前帧 BEV 特征", "200×200×256 的俯视特征图", "这是场景的隐式特征地图，还不是最终目标框或可视化 RGB 图片。",
         "第 6 层 Encoder 输出 [B,40000,256]。", "PerceptionTransformer 为 Decoder 调整维度为 [40000,B,256]；同一结果也作为历史缓存保存。",
         "BEV value 供检测 Decoder 查询；bev_embed 写入模型原始输出字典。", ["transformer", "raw-output"],
         kind="output", related=["decoder", "cache"], notes=["逻辑形状可以写成 [B,200,200,256]，但此处源码实际存储为展平序列并发生 permute。"])
    node("cache", "保存到下一帧", "prev_frame_info['prev_bev']", "把当前帧的 BEV 输出保存为下一帧状态。", "new_prev_bev: [40000,B,256]。",
         "forward_test 更新 BEV、位置和角度缓存；切换 scene 时清空历史。", "下一帧的 prev_bev。", ["history", "entry-head"],
         kind="state", related=["history"], notes=["此箭头跨越帧边界 t→t+1，不是当前帧计算图内的无限循环。"])
    node("object-queries", "目标 Object Query", "900 个目标候选槽位", "这些 Query 从 BEV 中找目标；与 40000 个 BEV 网格 Query 不同。",
         "query_embedding: [900,512]。", "拆成两份 [900,256]：query_pos 与 query；由 query_pos 预测初始 3D 参考点并 sigmoid。",
         "query/query_pos: [900,B,256]；reference_points: [B,900,3]。", ["head-query", "object-queries"], kind="input", related=["decoder", "bev-queries"])
    node("decoder", "检测 Decoder", "6 层 · 查询 BEV 中的目标", "以 900 个目标 Query 从当前 BEV 特征中提取目标相关信息，并逐层修正参考位置。",
         "900 个目标 Query、BEV value [40000,B,256]、参考点。", "每层：目标 self-attention → LN → BEV deformable cross-attention → LN → FFN → LN；再由回归分支更新下一层参考点。",
         "inter_states: [6,900,B,256]；inter_references: [6,B,900,3]。", ["decoder-config", "decoder-loop", "decoder-refine"],
         view="decoder", related=["mha", "decoder-cross", "refine", "predict"])
    node("mha", "目标自注意力", "MultiheadAttention · 8 heads", "让 900 个目标候选之间交换信息。", "query 和 query_pos: [900,B,256]。",
         "目标 Query 之间做多头注意力，随后输出投影、residual，再进入 LN。", "[900,B,256]。", ["decoder-config", "mha"],
         notes=["这一层是普通多头自注意力，与 Encoder 的稀疏采样式 TSA 不是同一种计算。"])
    node("decoder-cross", "从 BEV 采样目标特征", "CustomMSDeformableAttention", "每个目标候选在自己的参考位置附近，从 BEV 中取特征。", "目标 Query、参考点 x/y、BEV [40000,B,256]。",
         "对唯一一个 200×200 BEV level 预测采样偏移和权重，进行 deformable attention，加 residual，再进入 LN。",
         "[900,B,256]。", ["decoder-loop", "decoder-cross"], related=["bev"], notes=["Decoder 这里读的是 BEV，不是再直接读六路相机图像。"])
    node("refine", "逐层参考点更新", "reg_branches[lid]", "用本层目标特征预测位置修正，指导下一层到更合适的位置采样。",
         "本层 Query 特征 [B,900,256] 与旧 reference_points [B,900,3]。", "回归分支产生偏移，与旧参考点的 inverse_sigmoid 相加，再 sigmoid，作为下一层参考点。",
         "新 reference_points: [B,900,3]；各层特征保留用于分类和框预测。", ["decoder-refine"], related=["predict", "decoder-cross"],
         notes=["第 l 层框预测使用本层特征与本层输入参考点：第 0 层用初始参考点，之后用上一层更新的参考点。本层新参考点指导下一层采样。"])
    node("predict", "分类与 3D 框预测", "cls_branches / reg_branches", "把 Decoder 的目标特征变成类别分数和框参数；每层都有一份预测。",
         "hs 经 permute 后 [6,B,900,256]，以及初始/中间 reference points。", "分类 MLP 输出 10 个类别 logits；回归 MLP 输出 10 维编码，结合参考点调整中心坐标。",
         "all_cls_scores: [6,B,900,10]；all_bbox_preds: [6,B,900,10]。", ["heads-init", "heads", "raw-output"], related=["decode"],
         notes=["两种张量最后一维都为 10，但意义不同：类别数与框编码长度刚好相同。",
                "框编码为 cx,cy,log(w),log(l),cz,log(h),sin(yaw),cos(yaw),vx,vy；中心已在 head 中映射到物理范围。",
                "源码某些旧 docstring 写 box 为 9 维；本图按实际 code_size=10 与 sin/cos 实现说明。"])
    node("decode", "检测结果解码", "NMSFreeCoder + get_bboxes()", "把网络原始预测变成可以使用的三维目标列表。", "取第 6 层的 cls_scores [B,900,10] 和 bbox_preds [B,900,10]。",
         "类别 logits 经 sigmoid；在 Query×类别上取 Top-300；框的尺寸取 exp、角度由 atan2 恢复；过滤中心超出范围的候选，再转换为 box_type_3d。",
         "每样本最多 300 个框；boxes_3d 是框对象，boxes_3d.tensor 为 [K,9]，scores_3d / labels_3d 为 [K]。", ["decode-last", "decode", "decode-range", "decode-box", "get-bboxes"],
         notes=["配置使用 NMSFreeCoder，不执行传统 NMS；900 个候选不等于最终一定输出 900 个目标。",
                "get_bboxes 将重心 z 转成框底部中心 z：z ← z − h/2。"], related=["output"])
    node("output", "最终输出", "3D 检测框 + 分数 + 类别", "提供当前帧周围目标的几何位置、尺寸、方向、速度和类别。", "解码后每样本 K≤300 个候选。",
         "打包为 bbox_results[b]['pts_bbox']；BEV 缓存作为另一条内部状态路径维护。",
         "boxes_3d.tensor: [K,9] → x,y,z,w,l,h,yaw,vx,vy；scores_3d: [K]；labels_3d: [K]（10 类）。", ["get-bboxes", "entry-head", "entry"],
         kind="output", related=["predict", "cache"], notes=["这是官方相机 3D 检测路径；不包含规划控制输出。原始 head 导出可能止于 predict，并未执行这里的解码。"])

    def placement(id, row, col=2, label=None, caption=None):
        return dict(id=id, row=row, col=col, **({"label": label} if label else {}), **({"caption": caption} if caption else {}))

    def edge(a, b, label=None, kind="data", route=None):
        return dict(from_=a, to=b, **({"label": label} if label else {}), kind=kind, **({"route": route} if route else {}))

    def chain(*ids):
        return [edge(a, b) for a, b in zip(ids, ids[1:])]

    # Repeated LayerNorm/FFN occurrences have individual IDs for unambiguous
    # geometry, while sharing a source-backed description rather than copies.
    for prefix in ("enc", "dec"):
        for i in range(1, 4):
            nodes[f"{prefix}-ln{i}"] = {**nodes["ln"], "title": f"LayerNorm {i}"}
        nodes[f"{prefix}-ffn"] = {**nodes["ffn"]}
    nodes["enc-result"] = {**nodes["bev"], "title": "本层输出 → 下一层", "subtitle": "结构重复 6 层",
                           "summary": "本层归一化结果作为下一层 BEV Query；第 6 层之后得到最终 BEV。",
                           "input": "本层最后一次 LayerNorm 的输出 [B,40000,256]。",
                           "action": "赋值 bev_query = output；继续下一层，或在第 6 层完成后返回。",
                           "output": "下一层 Query [B,40000,256]；最终 Encoder 输出保持同一形状。",
                           "sources": ["encoder-loop"], "notes": [], "related": ["encoder", "bev"]}
    nodes["dec-result"] = {**nodes["refine"], "title": "下一层 / 第 6 层完成", "subtitle": "特征 + 更新的参考点",
                           "summary": "收集本层特征和参考点；继续下一层，或在第 6 层完成后返回。",
                           "input": "本层特征 [900,B,256] 与已更新的参考点 [B,900,3]。",
                           "action": "保留本层结果；下一层直接使用特征与新参考点，这里不再做一次回归更新。",
                           "output": "完成六层后 stack：inter_states [6,900,B,256]；inter_references [6,B,900,3]。",
                           "sources": ["decoder-refine", "decoder-loop"], "notes": [], "related": ["predict", "decoder"]}

    views = [
        dict(id="overview", title="完整模型", subtitle="沿中间主线向下读；两侧是额外输入与跨帧状态。", rows=10,
             nodes=[placement("input",0,caption="img · [B,6,3,Hpad,Wpad]"),
                    placement("backbone",1,caption="ResNet-101 + DCNv2\nC3 / C4 / C5"),
                    placement("fpn",2,caption="4 个尺度 · [B,6,256,Hi,Wi]"),
                    placement("geometry",3,1), placement("history",4,3), placement("bev-queries",3,3),
                    placement("encoder",4), placement("bev",5,caption="逻辑网格 [B,200,200,256]"), placement("cache",5,3),
                    placement("decoder",6,caption="6 层 · 900 Object Queries"), placement("object-queries",6,1),
                    placement("predict",7,caption="分类 / 框编码\n各 [6,B,900,10]"),
                    placement("decode",8,caption="最后一层 → Top-300 → 框解码"),
                    placement("output",9,caption="K≤300 · 3D 框 / 分数 / 类别")],
             edges=chain("input","backbone","fpn","encoder","bev","decoder","predict","decode","output") +
                   [edge("geometry","encoder"),edge("history","encoder",kind="state"),edge("bev-queries","encoder"),
                    edge("object-queries","decoder"),edge("bev","cache",kind="state")],
             footnote="缓存箭头在此结束：本帧 BEV 存为下一帧 prev_bev。模型主线是单帧前向，不把跨帧状态画成当前帧死循环。"),
        dict(id="backbone", title="展开 Backbone", subtitle="六相机共用 ResNet 权重；C3/C4/C5 三路特征送入 FPN。", rows=8,
             nodes=[placement("input",0),placement("stem",1),placement("c2",2),placement("c3",3),placement("c4",4),
                    placement("c5",5),placement("fpn",6,3),placement("features",7,3)],
             edges=chain("input","stem","c2","c3","c4","c5") +
                   [edge("c3","fpn",route="right"),edge("c4","fpn",route="right"),edge("c5","fpn"),edge("fpn","features")],
             footnote="C4/C5 的 DCNv2 位于每个 Bottleneck 的中间 3×3 卷积中；点击相应 stage 后可继续打开 DCNv2 代码。"),
        dict(id="encoder", title="展开 Encoder", subtitle="展示一层；按同样结构执行 6 层。残差加法已经包含在 TSA、SCA、FFN 内。", rows=9,
             nodes=[placement("bev-queries",0),placement("motion",0,1),placement("tsa",1),placement("enc-ln1",2),
                    placement("features",3,3),placement("project",4,1),placement("sca",4),placement("enc-ln2",5),
                    placement("enc-ffn",6),placement("enc-ln3",7),placement("enc-result",8)],
             edges=chain("bev-queries","tsa","enc-ln1","sca","enc-ln2","enc-ffn","enc-ln3","enc-result") +
                   [edge("motion","tsa",kind="state"),edge("features","sca"),edge("project","sca")],
             footnote="shape 沿主线保持 [B,40000,256]。历史旋转、CAN bus 嵌入和参考点准备在 Encoder 层循环之前完成。"),
        dict(id="decoder", title="展开 Decoder", subtitle="展示一层；目标 Query 与 BEV 网格 Query 的数量和用途不同。", rows=10,
             nodes=[placement("object-queries",0),placement("mha",1),placement("dec-ln1",2),placement("bev",3,1),
                    placement("decoder-cross",3),placement("dec-ln2",4),placement("dec-ffn",5),placement("dec-ln3",6),
                    placement("refine",7),placement("dec-result",8),placement("predict",9)],
             edges=chain("object-queries","mha","dec-ln1","decoder-cross","dec-ln2","dec-ffn","dec-ln3","refine","dec-result","predict") +
                   [edge("bev","decoder-cross")],
             footnote="主线特征为 [900,B,256]。参考点更新用于下一层；6 层的特征都被收集作预测，最终解码选最后一层。"),
    ]
    for view in views:
        for e in view["edges"]:
            e["from"] = e.pop("from_")
    result = dict(title="BEVFormer-base · 模型架构与 PyTorch 源码", sourceRoot=str(catalog.root),
                  generatedNote="按本地 BEVFormer-base 配置与 forward 实现生成。示例输入 B=1、6 路相机、padding 后 928×1600；形状为结构推导，不是本页运行模型的结果。",
                  sources=catalog.sources, nodes=nodes, views=views,
                  quickNodes=["config","model","head","transformer"])
    validate(result)
    return result


def validate(data):
    for key, node in data["nodes"].items():
        assert node["sources"], f"No source for {key}"
        for source in node["sources"]:
            assert source in data["sources"], (key, source)
        for related in node["related"]:
            assert related in data["nodes"], (key, related)
    for view in data["views"]:
        ids = [n["id"] for n in view["nodes"]]
        assert len(set(ids)) == len(ids), view["id"]
        for item in view["nodes"]:
            assert item["id"] in data["nodes"]
            assert 0 <= item["row"] < view["rows"] and 1 <= item["col"] <= 3
        for e in view["edges"]:
            assert e["from"] in ids and e["to"] in ids, e


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT.parent.parent / "project/BEVFormer-master")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/bevformer-model-architecture.html")
    parser.add_argument("--check", action="store_true", help="Verify that the generated page is up to date")
    args = parser.parse_args()
    data = build_catalog(args.source_root)
    template = (ROOT / "tools/templates/bevformer-model-architecture.html").read_text(encoding="utf-8")
    placeholder = "__BEVFORMER_CATALOG_JSON__"
    assert template.count(placeholder) == 1
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    output = template.replace(placeholder, payload)
    if args.check:
        if not args.output.exists() or args.output.read_text(encoding="utf-8") != output:
            raise SystemExit("Generated HTML is stale; rebuild it first.")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(f"{'Verified' if args.check else 'Built'} {args.output}")
    print(f"{len(data['nodes'])} modules; {len(data['sources'])} source excerpts; {len(data['views'])} views")


if __name__ == "__main__":
    main()
