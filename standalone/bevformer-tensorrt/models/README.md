# 模型资产

配套资产包解压到项目根目录后，得到 `models/bevformer-base/`。
源码包不携带完整模型、TensorRT engine 或真实数据集。

- `adapted.onnx`：当前 BEVFormer-base 已适配图，权重内嵌。换 GPU 后用本项目重新构建 engine。
- `images.npy` / `reference.npz`：同一固定合成输入和原始 PyTorch 模块的参考输出。
- `ranges.json`：此前 32 帧真实 mini_train 图像测得的 backbone 范围，程序检查模型与策略绑定。
- `export-report.json` / `fixed-metadata.json`：原始导出来源与固定几何条件。
- `manifest.json`：每个交付文件的 SHA-256、大小和适用范围。

运行 `python scripts/verify_assets.py --manifest models/bevformer-base/manifest.json` 核对资产。
合成输入只能做接口与数值回归，不能代替真实图像校准或检测精度评估。

当前图包含六相机 ResNet-101/DCNv2、FPN、六层 Encoder、六层 Decoder 和检测头。
输入为 `[1,6,3,928,1600]` FP32。几何投影固定为 synthetic all-visible、
CAN-bus 为零、无历史 BEV，输出为原始检测头张量。它没有实现真实场景的连续帧
输入、最终框解码或 mAP/NDS 验证。

代码许可证不包含模型权重和数据集许可。交付资产用于复现已有实验；如要公开上传
权重，保留 BEVFormer / 权重来源许可并遵循其条款。真实 nuScenes 图像没有打包。
