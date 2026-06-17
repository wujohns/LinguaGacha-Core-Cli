# PP-OCRv6 Detection Script Notes

本文说明 `ppocrv6/test-ppocrv6-det.cjs` 的当前运行逻辑。

脚本定位仍然是“直接编辑顶部参数再运行”的单张图片检测验证脚本：

- 不解析命令行参数。
- 不下载模型。
- 不检查 transformers.js pipeline。
- 不做文本识别，只做 text detection。
- 使用本工程内的 `ppocrv6/model/inference.onnx` 和 `ppocrv6/model/inference.yml`。

脚本使用 CommonJS，是因为 `@techstark/opencv-js` 在当前 Node.js 后端脚本中通过 `require()` 加载更稳定。

## 运行方式

编辑脚本顶部三个常量：

```js
const INPUT_IMAGE_PATH = "./test.png";
const OUTPUT_JSON_PATH = "./boxes.json";
const DEBUG_IMAGE_PATH = "./debug.png";
```

然后运行：

```bash
cd ppocrv6
npm run detect
```

或直接：

```bash
node test-ppocrv6-det.cjs
```

## 依赖

核心依赖：

- `onnxruntime-web`：使用 WASM backend 运行本地 ONNX。
- `sharp`：读取图片、resize、raw RGB 像素、debug 图片输出。
- `js-yaml`：读取 `inference.yml`。
- `@techstark/opencv-js`：执行 DBPostProcess 中的 contour、minAreaRect、fillPoly、mean 等 OpenCV 操作。
- `clipper-lib`：执行 polygon unclip / offset。

## 总体流程

1. 等待 OpenCV.js runtime 可用。
2. 读取并解析 `ppocrv6/model/inference.yml`。
3. 读取 `ppocrv6/model/inference.onnx` 并创建 ONNX session。
4. 按 `DetResizeForTest` 计算 resize 尺寸。
5. 用 `sharp` resize 图片并读取 raw RGB 数据。
6. 按 `DecodeImage.img_mode` 写入 RGB 或 BGR 通道顺序。
7. 按 `NormalizeImage.scale/mean/std` 归一化。
8. 构造 `float32[1, 3, H, W]` tensor。
9. 使用 `onnxruntime-web` WASM backend 推理。
10. 使用 OpenCV.js + clipper-lib 做检测后处理。
11. 写出 JSON。
12. 如果 `DEBUG_IMAGE_PATH` 非空，写出画框 debug 图片。

## 预处理

脚本从 `inference.yml` 同步：

- `DecodeImage.img_mode`
- `DetResizeForTest`
- `NormalizeImage.scale`
- `NormalizeImage.mean`
- `NormalizeImage.std`

当前模型配置为 `img_mode: BGR`，因此 `sharp().raw()` 读到的 RGB 数据会在写入 CHW tensor 时变成 BGR。

`DetResizeForTest: null` 时使用 PaddleOCR 推理默认值：

- `limit_side_len = 736`
- `limit_type = "min"`
- resize 后宽高对齐到 32 的倍数

## 后处理

脚本从 `inference.yml` 同步：

- `PostProcess.thresh`
- `PostProcess.box_thresh`
- `PostProcess.max_candidates`
- `PostProcess.unclip_ratio`

当前后处理流程：

1. 将模型输出概率图按 `thresh` 二值化。
2. 使用 OpenCV.js `findContours` 提取 contour。
3. 使用 `minAreaRect` 得到候选旋转框。
4. 使用 polygon mask 计算候选框内平均 score。
5. 按 `box_thresh` 过滤低分框。
6. 使用 `clipper-lib` 按 `area * unclip_ratio / perimeter` 做 polygon offset。
7. 对 unclip 后 polygon 再次取 `minAreaRect`。
8. 映射回原图坐标。
9. 按从上到下、从左到右排序输出。

输出 JSON 中 `postprocess.mode` 为：

```json
"opencv-dbpostprocess-quad"
```

## 输出

JSON 输出包含：

- 输入图片路径。
- 模型文件路径。
- ONNX Runtime 输入/输出名和 shape。
- 预处理配置和 resize 信息。
- 后处理阈值和模式。
- 检测框 `points` 和 `score`。

如果 `DEBUG_IMAGE_PATH` 非空，脚本会用 `sharp().composite()` 把检测框画到原图上。

## 剩余验证

当前脚本已经使用 Node.js 依赖尽量贴近 PaddleOCR 检测流程，但还没有和 PaddleOCR/Python 官方输出做 golden 对照。

严格一致性仍需要检查：

- OpenCV.js 和 Python OpenCV 的 contour 顺序差异。
- `RotatedRect.points()` 和 `cv2.boxPoints()` 的点序差异。
- `clipper-lib` 和 Python `pyclipper` 的像素级 offset 差异。
- `sharp` resize 和 OpenCV resize 的插值差异。
