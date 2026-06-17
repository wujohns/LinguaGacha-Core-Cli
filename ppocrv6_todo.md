# PP-OCRv6 Test Script TODO

本文记录 `ppocrv6/test-ppocrv6-det.mjs` 当前为了快速跑通而做出的妥协，以及后续如果要提高一致性、精度或工程可用性时需要补齐的事项。

## 当前定位

当前脚本是一个最小验证脚本，用于确认：

- 本地 `ppocrv6/model/inference.onnx` 可以在 Node.js 中被加载。
- `onnxruntime-web` WASM backend 可以完成推理。
- 输入图片经过简化预处理后，模型能输出文本概率图。
- 概率图可以通过简化后处理得到大致文本框。

它不是 PaddleOCR 官方推理流程的完整复刻，也不是 transformers.js pipeline 适配结果。

## 主要妥协

### 1. 未验证 transformers.js pipeline

当前脚本直接使用 `onnxruntime-web` 加载 ONNX，不再检查：

```js
pipeline("image-to-text", "PaddlePaddle/PP-OCRv6_medium_det_onnx")
```

因此它只能说明“JS 可以跑这个 ONNX”，不能说明“transformers.js pipeline 原生支持这个模型仓库”。

后续如果需要验证 transformers.js 适配，应单独恢复或新增 pipeline compatibility probe。

### 2. 使用 WASM backend 而非 native ONNX Runtime

当前推理使用：

```js
executionProviders: ["wasm"]
```

这是为了避开 `onnxruntime-node` native 包安装不稳定的问题。代价是：

- 性能可能弱于 native ONNX Runtime。
- 不验证 CPU native、CUDA、TensorRT 等生产推理后端。
- 更适合跨平台可运行性验证，不适合作为最终性能评估。

后续如果关注性能或生产部署，应重新评估 `onnxruntime-node` 或其他后端。

### 3. 未解析 `inference.yml`

当前脚本没有读取和解析 `ppocrv6/model/inference.yml`，而是将关键参数手写在脚本中：

```js
const NORMALIZE_MEAN = [0.485, 0.456, 0.406];
const NORMALIZE_STD = [0.229, 0.224, 0.225];
const THRESH = 0.2;
const BOX_THRESH = 0.45;
const MAX_CANDIDATES = 3000;
const UNCLIP_RATIO = 1.4;
```

代价是：

- 如果更换模型，脚本不会自动同步模型配置。
- `inference.yml` 中的流程没有被完整解释执行。
- 当前脚本实际只适配当前这份 PP-OCRv6 det 模型。

后续可以考虑读取 `inference.yml`，至少自动同步 normalize 和 postprocess 参数。

### 4. BGR/RGB 通道顺序未严格对齐

`inference.yml` 指定：

```yaml
DecodeImage:
  img_mode: BGR
```

当前脚本用 `sharp().raw()` 获取 RGB 数据，并直接按 RGB 顺序归一化。

代价是：

- 如果模型严格期望 BGR，当前输入通道顺序存在偏差。
- 检测结果可能能跑出来，但与 PaddleOCR 官方推理不完全一致。
- 对颜色敏感场景可能影响检测框质量。

后续需要确认 ONNX 导出模型的真实输入通道约定；如确认为 BGR，应在写入 CHW tensor 时交换 R/B 通道。

### 5. `DetResizeForTest` 未完整复刻

官方配置中有：

```yaml
DetResizeForTest: null
```

当前脚本只做了简化 resize：

- 长边超过 960 时缩到 960。
- 宽高 round 到 32 的倍数。
- 宽高最小为 32。

代价是：

- 没有覆盖 PaddleOCR `DetResizeForTest` 的全部分支和边界策略。
- 坐标缩放与官方实现可能有像素级差异。
- 特殊长宽比、大图、小图场景可能偏差更明显。

后续应对照 PaddleOCR 官方实现补齐 resize 逻辑，并加入对照样例。

### 6. 后处理不是完整 DBPostProcess

官方配置中使用：

```yaml
PostProcess:
  name: DBPostProcess
  thresh: 0.2
  box_thresh: 0.45
  max_candidates: 3000
  unclip_ratio: 1.4
```

当前脚本采用纯 JS 简化后处理：

- 概率图二值化。
- 8 邻域连通域 flood fill。
- 每个连通域取 AABB 外接矩形。
- 连通域平均分过滤。
- AABB 简单外扩。
- 输出轴对齐四边形。

缺少官方 DBPostProcess 的关键步骤：

- OpenCV contour 提取。
- `minAreaRect` 最小外接旋转矩形。
- polygon 内平均分 score。
- polygon unclip / offset。
- 旋转四点框输出。

这是当前最大精度妥协。后续如果要接近 PaddleOCR 官方结果，应优先补齐完整 DBPostProcess。

### 7. unclip 是 AABB 近似

当前外扩距离使用：

```js
const distance = (width * height * UNCLIP_RATIO) / (2 * (width + height));
```

它保留了“面积 * ratio / 周长”的思路，但只对轴对齐矩形四边外扩。

代价是：

- 无法处理任意 polygon。
- 无法处理旋转文本框。
- 倾斜文本或不规则文本区域会被粗糙框住。
- 框可能比官方 DBPostProcess 结果更大或更松。

后续应改为真正的 polygon offset/unclip。

### 8. 只做文本检测，不做完整 OCR

当前输出只有文本区域框：

```json
{
  "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "score": 0.96
}
```

没有：

- 文本裁剪。
- 方向分类。
- recognition 模型。
- 字符串解码。
- 检测与识别置信度合并。

因此当前脚本不是完整 OCR，只是 text detection spike。

后续如需完整 OCR，需要串接 PP-OCR recognition 模型。

### 9. 顶部常量配置，不做 CLI 或批处理

当前脚本通过顶部常量输入：

```js
const INPUT_IMAGE_PATH = "";
const OUTPUT_JSON_PATH = "";
const DEBUG_IMAGE_PATH = "";
```

代价是：

- 每次换图片需要改源码。
- 不适合批处理。
- 不适合被其他程序调用。
- 不适合作为正式 CLI。

这是刻意简化后的结果。若后续进入工具化阶段，再恢复参数解析或封装为函数。

### 10. 错误处理较轻

当前只做基础错误处理：

- 输入/输出路径是否填写。
- 图片尺寸能否读取。
- 模型输出 tensor 是否存在。

没有做：

- 模型文件校验。
- 输出 shape 严格校验。
- 图片格式细分诊断。
- 大图内存保护。
- 空检测结果诊断。
- 阈值调参提示。

后续如需稳定工具，应补齐这些错误边界。

## 建议优先级

如果后续继续推进，建议按以下顺序处理：

1. 确认并修正 BGR/RGB 通道顺序。
2. 对齐 PaddleOCR 官方 `DetResizeForTest`。
3. 实现完整 DBPostProcess，替换 connected-components AABB 近似。
4. 引入官方 PaddleOCR/Python 结果作为 golden 对照。
5. 再评估是否需要 `onnxruntime-node`、WebGPU、CUDA 或 TensorRT 后端。
6. 如需完整 OCR，接入 PP-OCR recognition 模型。
7. 如需工具化，再恢复 CLI 参数或封装为可调用模块。

## 保留当前脚本的价值

尽管有上述妥协，当前脚本仍然有价值：

- 依赖少。
- 逻辑直观。
- 可以快速验证 ONNX 模型是否能在 JS 中跑通。
- 可以产出大致文本框和 debug 图片。
- 适合阅读、定位推理链路、做后续改造起点。

