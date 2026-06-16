# PP-OCRv6 Detection Script Notes

本文说明 `ppocrv6/test-ppocrv6-det.mjs` 的运行逻辑和当前实现策略。

## 目标

`ppocrv6/test-ppocrv6-det.mjs` 是一个独立评估脚本，用 Node.js 跑
`PaddlePaddle/PP-OCRv6_medium_det_onnx` 的 ONNX 文本检测模型。

它只做文本区域检测，不做文字识别。输出结果是文本框坐标和置信度，不包含 OCR 文本内容。

当前默认模型文件在：

- `ppocrv6/model/inference.onnx`
- `ppocrv6/model/inference.yml`

运行示例：

```bash
cd ppocrv6
node test-ppocrv6-det.mjs \
  --image ./sample.png \
  --out ./boxes.json \
  --debug-image ./debug.png \
  --skip-pipeline-check
```

## 总体流程

脚本执行链路如下：

1. 解析命令行参数。
2. 解析模型来源，默认使用本地 `ppocrv6/model`。
3. 可选检查 transformers.js pipeline 是否能直接加载该模型。
4. 读取 `inference.onnx`，用 `onnxruntime-web` 的 WASM backend 创建推理会话。
5. 用 `sharp` 读取输入图片并做 PaddleOCR 风格预处理。
6. 构造 `float32[1, 3, H, W]` 输入 tensor。
7. 执行 ONNX 推理，得到 `float32[1, 1, H, W]` 文本概率图。
8. 对概率图做后处理，得到文本检测框。
9. 写出 JSON 结果。
10. 如果传入 `--debug-image`，在原图上画框并输出调试图。

## 参数

脚本支持这些参数：

- `--image <path>`：必填，输入图片。
- `--out <path>`：必填，输出 JSON。
- `--debug-image <path>`：可选，输出带检测框的调试图片。
- `--model <path-or-hf-id>`：可选，模型目录、本地 `.onnx` 文件，或 Hugging Face model id。
- `--cache-dir <path>`：可选，远程模型下载缓存目录。
- `--skip-pipeline-check`：可选，跳过 transformers.js pipeline 兼容性检查。
- `--help`：显示帮助。

默认模型路径是脚本所在目录下的 `model` 目录，也就是 `ppocrv6/model`。

## 模型解析策略

`resolveModelFiles()` 按下面顺序处理 `--model`：

1. 如果 `--model` 指向一个本地目录：
   - 查找目录内的 `inference.onnx`。
   - 如果存在 `inference.yml`，也记录下来。
   - 当前默认路径就是这种情况：`ppocrv6/model`。
2. 如果 `--model` 指向一个本地 `.onnx` 文件：
   - 直接使用该文件。
   - `inference.yml` 记为 `null`。
3. 如果 `--model` 不是本地路径：
   - 当作 Hugging Face model id。
   - 下载：
     - `https://huggingface.co/<model>/resolve/main/inference.onnx`
     - `https://huggingface.co/<model>/raw/main/inference.yml`
   - 文件写入 `--cache-dir` 下的模型缓存目录。

下载使用 `undici`，并通过 `getProxyAgent()` 读取这些环境变量：

- `HTTPS_PROXY`
- `https_proxy`
- `HTTP_PROXY`
- `http_proxy`

所以如果需要代理，可以这样运行：

```bash
HTTPS_PROXY=http://127.0.0.1:7990 HTTP_PROXY=http://127.0.0.1:7990 \
  node test-ppocrv6-det.mjs --image ./sample.png --out ./boxes.json
```

## transformers.js 兼容性检查

`checkTransformersPipeline()` 的目的不是完成检测，而是记录“这个 Hugging Face 仓库能不能被 transformers.js pipeline 直接使用”。

当前判断策略：

- 如果传入的是本地模型路径，pipeline 检查时会改用原始 Hugging Face id：
  `PaddlePaddle/PP-OCRv6_medium_det_onnx`。
- 动态 import `@huggingface/transformers`。
- 尝试：

```js
pipeline("image-to-text", "PaddlePaddle/PP-OCRv6_medium_det_onnx")
```

同时脚本会记录本地模型目录是否具备 transformers.js 常见结构：

- `inference.onnx`
- `inference.yml`
- `config.json`
- `onnx/model.onnx`

当前模型目录只有 `inference.onnx` 和 `inference.yml`。它不是 transformers.js pipeline 常见的模型布局，因此脚本预期 pipeline 直接加载不可用。

`@huggingface/transformers` 没有放入默认依赖，因为近期版本会拉取 `onnxruntime-node`，它的 native 安装脚本在本环境里遇到重定向时曾失败。脚本本体仍保留动态检测逻辑，需要时可手动安装。

## ONNX Runtime 策略

脚本使用：

```js
import * as ort from "onnxruntime-web";
```

推理 backend：

```js
executionProviders: ["wasm"]
```

采用 `onnxruntime-web` 的原因：

- 可以在 Node 中通过 WASM 跑 ONNX。
- 避免 `onnxruntime-node` native 包安装失败的问题。
- 足够验证这个 ONNX 图能否被 JavaScript 运行。

模型加载方式：

```js
const modelBytes = await fs.readFile(modelFiles.onnxPath);
const session = await ort.InferenceSession.create(modelBytes, {
  executionProviders: ["wasm"],
});
```

读取模型输入输出名：

```js
const input = session.inputNames[0] ?? "x";
const output = session.outputNames[0] ?? "fetch_name_0";
```

对于当前模型，已知输入是 `x`，输出是 `fetch_name_0`。

## 图像预处理

预处理由 `preprocessImage()` 完成，对齐 `inference.yml` 中的主要参数：

```yaml
NormalizeImage:
  scale: 1./255.
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
ToCHWImage: null
```

当前脚本流程：

1. 用 `sharp` 读取图片尺寸。
2. 用 `resizeForDet()` 计算检测输入尺寸。
3. 用 `sharp` resize 到目标尺寸。
4. 移除 alpha 通道。
5. 读取 raw RGB 像素。
6. 每个通道做：

```js
value = (pixel / 255 - mean[channel]) / std[channel]
```

7. 从 HWC 排布转成 CHW 排布。
8. 返回 `Float32Array`。

### resize 策略

当前 resize 参数：

```js
const DET_LIMIT_SIDE_LEN = 960;
const DET_LIMIT_TYPE = "max";
```

策略：

- 如果图片长边大于 960，将长边压到 960，短边等比例缩放。
- 然后宽高都 round 到 32 的倍数。
- 宽高最低为 32。

最终模型输入 shape：

```js
[1, 3, resizedHeight, resizedWidth]
```

脚本也会记录原图尺寸、resize 后尺寸，以及 `ratioH` / `ratioW`。

### BGR/RGB 说明

`inference.yml` 写的是 `img_mode: BGR`，但当前脚本用 `sharp` 读取 raw RGB，并按 RGB 的 mean/std 做归一化。

这对本次 spike 足够验证模型可以跑通，但如果要严格对齐 PaddleOCR 官方结果，应进一步确认 PaddleOCR 导出模型实际期望的通道顺序，并在必要时改成 BGR 排布。

## 推理

预处理后构造 tensor：

```js
const tensor = new ort.Tensor("float32", preprocessed.data, [
  1,
  3,
  preprocessed.height,
  preprocessed.width,
]);
```

然后执行：

```js
const results = await session.run({ [input]: tensor });
```

输出 tensor 预期是文本概率图：

```js
[1, 1, outputHeight, outputWidth]
```

脚本从 `results[output]` 读取输出；如果名字不匹配，则 fallback 到第一个输出名。

## 后处理策略

后处理由 `postprocessDetection()` 完成。

`inference.yml` 中官方配置是：

```yaml
PostProcess:
  name: DBPostProcess
  thresh: 0.2
  box_thresh: 0.45
  max_candidates: 3000
  unclip_ratio: 1.4
```

当前脚本保留这些核心阈值：

```js
const POSTPROCESS_DEFAULTS = {
  thresh: 0.2,
  boxThresh: 0.45,
  maxCandidates: 3000,
  unclipRatio: 1.4,
  minSize: 3,
  mode: "connected-components-aabb",
};
```

但当前实现不是完整 PaddleOCR `DBPostProcess`。它采用纯 JS 的 connected-components AABB 近似方案。

原因：

- 完整 DBPostProcess 依赖 OpenCV 的轮廓提取、最小外接旋转矩形和 polygon unclip。
- 之前 OpenCV.js 在当前环境初始化开销/稳定性不理想。
- 评估目标优先是验证 ONNX 模型能在 JS 中跑通，并能从概率图得到可视化文本框。

### 当前后处理步骤

1. 从输出 tensor 读取 `predH`、`predW` 和概率数组。
2. 二值化：

```js
bitmap[i] = pred[i] > 0.2 ? 1 : 0;
```

3. 用 8 邻域 flood fill 找连通域。
4. 每个连通域记录：
   - `xmin`
   - `xmax`
   - `ymin`
   - `ymax`
   - `pixelCount`
   - `scoreSum`
5. 连通域按像素数量从大到小排序。
6. 最多取 `maxCandidates = 3000` 个。
7. 丢弃太小的区域：

```js
min(width, height) < 3
```

8. 计算区域平均分：

```js
score = scoreSum / pixelCount
```

9. 丢弃低置信度区域：

```js
score < 0.45
```

10. 用连通域外接矩形生成基础框：

```js
[
  [xmin, ymin],
  [xmax, ymin],
  [xmax, ymax],
  [xmin, ymax],
]
```

11. 用 `unclipAabb()` 扩张轴对齐框。
12. 将模型输出坐标映射回原图坐标：

```js
x_original = round(x / predW * srcW)
y_original = round(y / predH * srcH)
```

13. 输出按从上到下、从左到右排序。

### unclipAabb 策略

官方 DBPostProcess 的 unclip 基于 polygon 面积和周长：

```text
distance = area * unclip_ratio / perimeter
```

当前脚本保留这个距离公式，但只对 AABB 框做四边扩张：

```js
xmin -= distance
xmax += distance
ymin -= distance
ymax += distance
```

这不是旋转 polygon unclip，但对当前轴对齐近似框足够简单稳定。

## 输出 JSON

输出 JSON 的主要结构：

```json
{
  "image": "/abs/path/image.png",
  "model": "/abs/path/ppocrv6/model",
  "files": {
    "onnx": "/abs/path/ppocrv6/model/inference.onnx",
    "yml": "/abs/path/ppocrv6/model/inference.yml"
  },
  "transformersPipeline": {
    "attempted": false,
    "ok": false,
    "error": "Skipped by --skip-pipeline-check"
  },
  "runtime": {
    "engine": "onnxruntime-web",
    "backend": "wasm",
    "inputName": "x",
    "outputName": "fetch_name_0",
    "inputShape": [1, 3, 256, 640],
    "outputShape": [1, 1, 256, 640]
  },
  "preprocessing": {
    "resize": {
      "originalWidth": 640,
      "originalHeight": 240,
      "resizedWidth": 640,
      "resizedHeight": 256,
      "ratioW": 1,
      "ratioH": 1.0666666666666667
    },
    "normalize": {
      "mean": [0.485, 0.456, 0.406],
      "std": [0.229, 0.224, 0.225]
    }
  },
  "postprocess": {
    "thresh": 0.2,
    "boxThresh": 0.45,
    "maxCandidates": 3000,
    "unclipRatio": 1.4,
    "minSize": 3,
    "mode": "connected-components-aabb"
  },
  "boxes": [
    {
      "points": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
      "score": 0.96
    }
  ]
}
```

## Debug 图片

如果传入 `--debug-image`，脚本会调用 `drawDebugImage()`：

1. 读取原图尺寸。
2. 生成 SVG overlay。
3. 为每个 box 画 polygon。
4. 给 box 标号。
5. 用 `sharp().composite()` 将 SVG 覆盖到原图上。
6. 写出 debug 图片。

颜色策略：

- 偶数框：`#00D084`
- 奇数框：`#FF4D4F`

## 当前限制

当前脚本是评估实现，不是完整 OCR 产品实现。主要限制如下：

- 不做文字识别，只做文本检测。
- 后处理是 AABB 近似，不是完整 DBPostProcess。
- 不输出旋转框，只输出轴对齐四边形。
- 通道顺序当前按 sharp RGB 处理，未严格复刻 `img_mode: BGR`。
- `DetResizeForTest` 的完整 PaddleOCR 细节没有全部复刻，只实现了长边限制和 32 倍数对齐。
- transformers.js pipeline 检查是诊断性质，不参与实际检测。

## 后续改进方向

如果要从 spike 走向更接近 PaddleOCR 官方行为，优先级建议如下：

1. 严格确认并对齐 BGR/RGB 通道顺序。
2. 将后处理替换为完整 DBPostProcess：
   - contour 提取
   - `minAreaRect`
   - polygon score
   - pyclipper/clipper 风格 unclip
   - 旋转四点框输出
3. 对齐 PaddleOCR `DetResizeForTest` 的全部 resize 策略。
4. 引入官方 PaddleOCR/Python 输出作为 golden 对照。
5. 如需完整 OCR，再串接 PP-OCR recognition 模型。

