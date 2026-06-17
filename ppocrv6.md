# PP-OCRv6 Detection Script Notes

本文说明 `ppocrv6/test-ppocrv6-det.mjs` 的当前运行逻辑。

这个脚本已经简化为“直接编辑脚本顶部参数再运行”的评估脚本，不再作为 CLI 工具使用：

- 不解析命令行参数。
- 不下载模型。
- 不检查 transformers.js pipeline。
- 只使用本工程内的本地模型文件。
- 只保留图片预处理、ONNX 推理、概率图后处理、JSON/debug 输出。

## 顶部配置

运行前只需要改脚本顶部这三个常量：

```js
const INPUT_IMAGE_PATH = "";
const OUTPUT_JSON_PATH = "";
const DEBUG_IMAGE_PATH = "";
```

含义：

- `INPUT_IMAGE_PATH`：输入图片路径，必填。
- `OUTPUT_JSON_PATH`：检测结果 JSON 输出路径，必填。
- `DEBUG_IMAGE_PATH`：调试图片输出路径，可留空；留空时不输出画框图片。

示例：

```js
const INPUT_IMAGE_PATH = "/tmp/ppocrv6-transformers-eval/sample.png";
const OUTPUT_JSON_PATH = "./boxes.json";
const DEBUG_IMAGE_PATH = "./debug.png";
```

然后运行：

```bash
cd ppocrv6
node test-ppocrv6-det.mjs
```

如果 `INPUT_IMAGE_PATH` 或 `OUTPUT_JSON_PATH` 留空，脚本会直接报错提醒填写。

## 本地模型

脚本固定使用本地模型：

```js
const MODEL_DIR = path.join(SCRIPT_DIR, "model");
const MODEL_ONNX_PATH = path.join(MODEL_DIR, "inference.onnx");
const MODEL_YML_PATH = path.join(MODEL_DIR, "inference.yml");
```

对应文件：

- `ppocrv6/model/inference.onnx`
- `ppocrv6/model/inference.yml`

`MODEL_YML_PATH` 目前只写入结果 JSON 方便追踪，脚本没有解析 YAML；预处理和后处理参数直接写在脚本常量里。

## 总体流程

脚本执行链路：

1. 检查顶部输入/输出路径是否已填写。
2. 读取 `ppocrv6/model/inference.onnx`。
3. 用 `onnxruntime-web` 的 WASM backend 创建 ONNX session。
4. 读取 session 的输入/输出名，默认 fallback 为 `x` 和 `fetch_name_0`。
5. 用 `sharp` 读取图片并做预处理。
6. 构造 `float32[1, 3, H, W]` 输入 tensor。
7. 执行模型推理，得到文本概率图。
8. 用简化后处理从概率图提取文本框。
9. 写出 JSON。
10. 如果 `DEBUG_IMAGE_PATH` 非空，额外输出画框调试图。

## ONNX 推理策略

脚本使用：

```js
import * as ort from "onnxruntime-web";
```

推理 backend 固定为 WASM：

```js
const session = await ort.InferenceSession.create(modelBytes, {
  executionProviders: ["wasm"],
});
```

这里没有使用 `onnxruntime-node`，原因是之前 native 包安装过程在当前环境里不稳定；`onnxruntime-web` 的 WASM backend 足够验证 ONNX 模型在 JS 中跑通。

当前模型输入输出预期：

- 输入名：`x`
- 输出名：`fetch_name_0`
- 输入 shape：`[1, 3, resizedHeight, resizedWidth]`
- 输出 shape：`[1, 1, mapHeight, mapWidth]`

脚本会从 session 动态读取输入输出名：

```js
const inputName = session.inputNames[0] ?? "x";
const outputName = session.outputNames[0] ?? "fetch_name_0";
```

## 图像预处理

预处理在 `preprocessImage()` 中完成。

核心参数：

```js
const DET_LIMIT_SIDE_LEN = 960;
const NORMALIZE_MEAN = [0.485, 0.456, 0.406];
const NORMALIZE_STD = [0.229, 0.224, 0.225];
```

处理步骤：

1. 用 `sharp(imagePath).metadata()` 读取原始宽高。
2. 如果长边超过 `960`，按比例缩放到长边 `960`。
3. 将宽高 round 到 32 的倍数，且最小为 32。
4. 用 `sharp` resize 到目标宽高。
5. 移除 alpha 通道。
6. 读取 raw RGB 像素。
7. 对每个通道做归一化：

```js
value = (pixel / 255 - mean[channel]) / std[channel]
```

8. 将 HWC 排布转成 CHW 排布。
9. 返回 `Float32Array` 以及原始/resize 后尺寸。

注意：`inference.yml` 中写的是 `img_mode: BGR`，当前脚本用 `sharp` 的 RGB raw 数据直接归一化。此处是评估实现，若要严格对齐 PaddleOCR 官方推理，需要进一步确认并可能调整通道顺序。

## 后处理策略

当前后处理是简化版本，不是完整 PaddleOCR `DBPostProcess`。

常量：

```js
const THRESH = 0.2;
const BOX_THRESH = 0.45;
const MAX_CANDIDATES = 3000;
const UNCLIP_RATIO = 1.4;
const MIN_SIZE = 3;
```

执行步骤：

1. 读取模型输出概率图。
2. 用 `THRESH = 0.2` 二值化：

```js
bitmap[index] = scores[index] > THRESH ? 1 : 0;
```

3. 用 8 邻域 flood fill 找连通域。
4. 每个连通域记录：
   - `xmin`
   - `xmax`
   - `ymin`
   - `ymax`
   - `pixelCount`
   - `score`
5. 连通域按像素数从大到小排序。
6. 最多处理 `MAX_CANDIDATES = 3000` 个。
7. 丢弃过小区域：

```js
Math.min(width, height) < MIN_SIZE
```

8. 丢弃低分区域：

```js
component.score < BOX_THRESH
```

9. 用连通域 AABB 生成轴对齐矩形。
10. 按 `UNCLIP_RATIO` 做 AABB 外扩。
11. 将概率图坐标映射回原图坐标。
12. 按从上到下、从左到右排序输出。

### AABB unclip

官方 DBPostProcess 的 unclip 是 polygon offset。当前脚本只对轴对齐框做近似扩张：

```js
const distance = (width * height * UNCLIP_RATIO) / (2 * (width + height));
```

然后：

```js
xmin -= distance;
xmax += distance;
ymin -= distance;
ymax += distance;
```

这保留了 PaddleOCR 中“面积 * ratio / 周长”的基本思路，但不是完整 polygon unclip。

## 输出 JSON

输出结构大致如下：

```json
{
  "image": "/abs/path/input.png",
  "model": {
    "onnx": "/abs/path/ppocrv6/model/inference.onnx",
    "yml": "/abs/path/ppocrv6/model/inference.yml"
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
    "originalWidth": 640,
    "originalHeight": 240,
    "resizedWidth": 640,
    "resizedHeight": 256,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
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

如果 `DEBUG_IMAGE_PATH` 非空，`drawDebugImage()` 会：

1. 读取原图尺寸。
2. 生成 SVG overlay。
3. 给每个检测框画 polygon。
4. 给检测框标号。
5. 用 `sharp().composite()` 把 SVG 覆盖到原图。
6. 写出 debug 图片。

框颜色交替使用：

- `#00D084`
- `#FF4D4F`

## 当前取舍

当前脚本刻意保持简单：

- 顶部常量配置，不做 CLI。
- 固定本地模型，不做下载。
- 不做 transformers.js pipeline 检查。
- 不解析 `inference.yml`。
- 后处理使用纯 JS connected-components AABB 近似。
- 不引入 OpenCV.js / clipper / onnxruntime-node。

因此它适合作为最小验证脚本：确认模型能在 JS 中加载、推理，并输出可观察的检测框。

如果要进一步逼近 PaddleOCR 官方结果，后续再考虑：

1. 严格处理 BGR/RGB 通道顺序。
2. 完整复刻 `DetResizeForTest`。
3. 用 OpenCV 或其他库实现完整 DBPostProcess：
   - contour 提取
   - minAreaRect
   - polygon score
   - polygon unclip
   - 旋转框输出
4. 增加官方 PaddleOCR/Python 结果作为对照。

