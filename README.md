# Hunyuan3D Ascend：本机几何生成与验证

在 **Ascend / CANN / torch_npu** 环境运行 Hunyuan3D 几何生成，支持：

- **Hunyuan3D-2mv**：带 `front / left / back / right` 标签的多视图，共同约束**同一个网格**。
- **Hunyuan3D 2.1**：单图几何生成，作为质量对照或单视图方案。
- 保留内部白色区域的去白底处理、严格推理权重检查、闭合网格检查与简化。
- **不随机删除三角面**的 CPU 深度缓冲渲染，输出正、背、侧、斜视 PNG。
- Blender 友好的 GLB / OBJ 输出。

本仓库是独立的**部署与验证脚本**，不是腾讯官方 Ascend 发行版，也不包含上游模型源码、权重或用户图像。只实现几何生成；**Hunyuan3D-Paint / PBR 贴图 CUDA 光栅器尚未适配**。

English: local geometry-inference integration for Ascend NPUs, with true labeled multi-view conditioning, checked checkpoint loading, topology reports and complete-triangle CPU previews. Texture synthesis, rigging and full numerical parity with CUDA are outside the verified scope.

## 已验证环境

| 项目 | 实测版本 / 条件 |
| --- | --- |
| 系统 | Linux aarch64，Ascend 910；不是 NVIDIA CUDA |
| Python | 3.11.15 |
| CANN | 9.0.0 |
| PyTorch / torch_npu | `2.10.0+cpu` / `2.10.0`；版本字符串有 `+cpu` 不代表未使用 NPU |
| transformers | 4.46.3 |
| diffusers | 0.31.0 |
| huggingface-hub | 0.26.2 |
| torchvision | 0.25.0 |
| trimesh / pymeshlab | 5.2.0 / 2025.7.post1 |
| scikit-image / numba | 0.26.0 / 0.68.0 |

几何网络通过 `torch_npu` 使用 CANN，网格提取、修复和预览包含本机 CPU 运算。2.1 实测还出现 `ArgSort` AiCPU 与 `aten::scatter_reduce.two_out` CPU 回退警告，因此不宣称完整计算图全部运行在 NPU。**这条路径不要求额外编写 Triton kernel**；不能据此宣称所有 CUDA/Triton 扩展已经支持 Ascend。基础 Triton 加法测试也不能证明整条模型数值等价。

## 快速开始

### 1. 准备独立环境

先按华为对应版本的安装指南准备 CANN、PyTorch 和 torch_npu。不要把上游 CUDA 依赖列表直接安装到已有 ComfyUI 环境。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 使用能 import torch/torch_npu 的基础 Python 创建独立环境。
python3.11 -m venv --system-site-packages .venv
source .venv/bin/activate
python -c 'import torch, torch_npu; print(torch.__version__, torch.npu.is_available())'
python -m pip install -r requirements.txt

# pymeshlab 在部分 Ubuntu aarch64 环境需要这些系统库。
sudo apt-get install libharfbuzz0b libgl1 libglib2.0-0 libopengl0 libglx0
```

如果 `torch` 不可见，检查 venv 使用的 Python 和其基础环境，而不是随手用 pip 换掉已匹配 CANN 的 torch。`libhccl.so` 等加载失败时，先检查是否 source 了 `set_env.sh`。

### 2. 获取匹配的上游代码

上游代码和模型继续遵守各自的腾讯许可证。下列是发布时记录的源码修订；模型下载器会另行保存权重仓库修订与 SHA256。

```bash
mkdir -p external

git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.git external/Hunyuan3D-2
git -C external/Hunyuan3D-2 checkout f8db63096c8282cb27354314d896feba5ba6ff8a

# 仅在使用单视图 2.1 时需要：
git clone https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git external/Hunyuan3D-2.1
git -C external/Hunyuan3D-2.1 checkout 82920d643c0dc2f7bfd7255f45f62d386edfe60c
```

本仓库不替你安装上游全套训练、贴图、CUDA rasterizer 或 DeepSpeed 依赖。

### 3. 下载几何权重

```bash
python scripts/download_models.py --family 2mv --output models
# 单视图 2.1（可选）：
python scripts/download_models.py --family 2.1 --output models
```

网络需要镜像时显式指定：

```bash
python scripts/download_models.py --family 2mv --output models \
  --endpoint https://hf-mirror.com
export HF_ENDPOINT=https://hf-mirror.com
```

下载器按 API 返回的**精确字节数**和 LFS SHA256 检查文件；先保存 `.part`，验证后再改名；外层重试配合 `curl -C -` 断点续传。没有使用可能截断文件的 `curl --retry -o` 组合。已存在的损坏文件会报错，不会静默覆盖。

首次模型构造还可能访问 Hugging Face 获取 DINO 配置；准备好缓存后可设置 `HF_HUB_OFFLINE=1` 验证离线运行。整个生成过程不发送图片到在线推理 API。

2mv 标准权重：

```text
repo:   tencent/Hunyuan3D-2mv
file:   hunyuan3d-dit-v2-mv/model.fp16.safetensors
bytes:  4928151562
sha256: d36f5881bcdc56726b73e517cd444c13c60732431622da7268145355c8d38e9c
```

它已包含几何生成所需的 DiT、推理 VAE 和 conditioner 参数，不应按“1.1B×2字节”估算整个下载大小。2.1 此处使用 `hunyuan3d-dit-v2-1/model.fp16.ckpt`，同样由原版 loader 读取模型、VAE 与 conditioner。

### 4. 准备输入

已有干净的透明 RGBA 图片可以直接使用。白色背景图片可显式启用边缘泛洪去底：

```bash
python scripts/prepare_images.py --input front-white.png \
  --output inputs-front --white-background
```

左正面、右背面的**分离式双视图设定图**：

```bash
python scripts/prepare_images.py --input reference-sheet.png \
  --output inputs --white-background --split-two
```

请先检查 `front_preview.png / back_preview.png` 和 mask。

- `--split-two` 将两个最大的非白连通区域按左右顺序解释为 front/back；不是通用设定图识别器。
- 不用 HSV 饱和度直接删像素：那会把白毛、浅色身体、黑色轮廓一并删除。
- 泛洪仅删除与画面边缘连通的近白区域，保留封闭在角色内部的白色。
- 背景不是白色、角色与白底连通或有复杂阴影时，先人工修 mask。单纯 `convert('RGBA')` 不等于抠图。
- 输入姿势、比例和方向应该一致。正背图不能确定所有侧面厚度，模型仍会推断未见区域。

### 5. 真正的正背联合建模

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python scripts/infer.py \
  --family 2mv --source external/Hunyuan3D-2 --models models \
  --front inputs/front.png --back inputs/back.png \
  --output outputs/multiview --device npu:0 \
  --seed 12345 --steps 50 --resolution 320
```

可以增加 `--left`、`--right`。多视图语义来自专用模型和带方向标签的字典：

```python
image = {'front': front_image, 'back': back_image}
# image tensor: [1, 2, 3, 512, 512]
# view_idxs: [(0, 2)]
```

脚本检查 batch 与 view 维度，确保是 **1 个对象的 2 个视图**。分别跑两张图生成两份网格，或向单视图模型传普通列表，都不是这里的多视图融合。

单视图对照：

```bash
python scripts/infer.py \
  --family 2.1 --source external/Hunyuan3D-2.1 --models models \
  --front inputs/front.png --output outputs/single21 \
  --device npu:0 --seed 12345 --steps 50 --resolution 320
```

结果目录包含 `raw.glb`、`raw.obj`、`report.json`。输出目录必须是新的或空目录，防止覆盖旧实验。

### 6. 网格修复、简化和预览

```bash
python scripts/repair_mesh.py --mesh outputs/multiview/raw.glb \
  --output outputs/multiview-repaired --faces 80000

python scripts/render_cpu.py \
  --mesh outputs/multiview-repaired/mesh_editable.obj \
  --output outputs/multiview-preview
```

修复工具默认只允许丢弃极小碎片（每个次级部件不超过主体面数的0.1%，至少允许2面碎片）。遇到更大的独立部件会报错，不生成清理结果；只有确认这些部件应被删除时才显式加 `--keep-largest`，这会丢弃所有次级部件。随后去重/退化面、尝试非流形与小孔修复，再保形简化。高细节版、编辑版和删除部件的面数记录在 `repair_report.json`。**闭合性、法线一致性、有限坐标和正体积不全通过时会返回失败**；导出文件存在不代表验收通过。

CPU 预览使用全部面片和深度缓冲，模型 **Y-up，正面从 +Z 观察**。不会靠随机删面加速，也不会猜“最长轴就是上方”。输出为透明 RGBA；图像算法检查和模型视觉检查都应保留。

在 Blender 中通过 glTF 导入 `mesh_editable.glb`，或通过 Wavefront 导入 OBJ。GLB 会按 glTF 的 Y-up 约定转换。实测 Blender 3.0.1 可导入 80,000 面 GLB 并保存工程。服务器无头 Blender 发行包曾产生全黑渲染，所以仓库使用独立 CPU 预览，**不把无效黑图当作模型故障证据**。

## 权重兼容策略

`scripts/weight_checks.py` 包含限定范围的加载适配：

1. 新旧 transformers 的 DINO attention projection 名称只有在目标模型存在对应 key 时才重映射，并拒绝 key 冲突。
2. 2mv 推理权重允许缺少 ShapeVAE 的 **`encoder.* / pre_kl.*`**，这些属于未执行的训练/网格编码路径。
3. `decoder`、`post_kl`、DiT、conditioner 的推理参数缺失或任何多余参数都报错，不用 warning-only `strict=False` 掩盖问题。
4. 加载 hook 在 `finally` 中恢复；仅用于本脚本的单进程构造期，不适合在共享多线程服务里并发加载模型。

推荐使用已实测的依赖版本。key 重映射不是“任意 transformers 版本通用兼容”的承诺。

## 实测结果与边界

2026-10-27，同一角色参考输入的本机测试，非通用基准；用户图像与产物未上传：

| 几何路线 | 条件 | 正面轮廓 IoU | 背面投影轮廓 IoU |
| --- | --- | --- | --- |
| 2.1 修正单视图输入 | front；50步，res320，seed12345 | 0.9316 | 0.9328 |
| 2mv | front+back；50步，res320，seed12345 | 0.8645 | 0.8801 |
| 2mv | front+back；50步，res384，seed42 | 0.8600 | 0.8729 |
| 2mv | front+back；50步，res384，seed7 | 0.8296 | 0.8418 |

轮廓先按 bbox 裁切、保持比例缩放和中心对齐后计算交并比，不是原位相机配准。**背面投影轮廓分数高，不代表背面细节已还原**；单视图和多视图也不能仅按这个分数定优劣。没有侧视图时，侧面纵深是推断。这里没有提供 furry 特定的公开质量排名。

首个 2mv 实验约25.2秒完成50步和res320几何解码（不含加载）。原始主体523,748面，另有2个2面碎片；清理后主体闭合。80,000面版39,990顶点，重新读取GLB仍闭合且法线一致。2.1编辑版39,992顶点/80,000面，同样通过检查。这些是**本次设备观测**，不是性能保证。

粗模可以继续在 Blender 雕刻，但没有证明精确四指、爪部解剖、吻部或尾根完全满足角色设定；也没有UV纹理、骨骼、表情或动画级拓扑。工程通过不等于最终资产通过。

## 验证

```bash
python -m unittest discover -s tests -v
python -m compileall -q scripts tests
```

真实设备已验证本仓库的2mv和2.1发布入口、网格修复与CPU预览；2mv还验证了预先缓存后的 `HF_HUB_OFFLINE=1` 运行。回归检查共8项通过，详情与限制见 [发布前验证记录](docs/verification.md)。不把单元测试或低步数smoke当作最终造型质量验证。

## 仓库内容

```text
scripts/
  infer.py             单视图/真正多视图生成、GLB/OBJ和实验报告
  weight_checks.py     严格推理参数检查与DINO key兼容
  download_models.py   固定模型修订、断点续传、精确大小与SHA256检查
  prepare_images.py    显式去白底/双视图分割、mask与预览
  repair_mesh.py       主体清理、闭合性检查、保形简化
  render_cpu.py        完整面片CPU深度缓冲预览
 tests/test_safety.py   有针对性的回归验证
```

## 许可与上游

本仓库原创部署/验证脚本采用 [MIT](LICENSE)。**这不改变上游代码、权重或第三方依赖的许可**，也不是把 Hunyuan3D 重新授权为 MIT。

- [Hunyuan3D-2 官方代码及许可](https://github.com/Tencent-Hunyuan/Hunyuan3D-2)
- [Hunyuan3D-2mv 官方模型](https://huggingface.co/tencent/Hunyuan3D-2mv)
- [官方多视图调用示例](https://github.com/Tencent-Hunyuan/Hunyuan3D-2/blob/main/examples/shape_gen_multiview.py)
- [Hunyuan3D-2.1 官方代码及许可](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)

下载和使用模型前请自行阅读对应版本的腾讯自定义许可证，包括地域、商业使用和分发条件。本仓库不分发权重、完整上游源码、用户设定图或生成模型。
