# PP-OCRv6 Detection TODO

本文只记录 `ppocrv6/test-ppocrv6-det.cjs` 后续仍需要处理或验证的事项。已完成内容不再放在 TODO 中。

## 1. PaddleOCR golden 对照

需要用 PaddleOCR/Python 官方流程生成 golden，确认当前 Node.js 检测结果是否接近官方结果。

建议覆盖：

- 普通横排文本。
- 倾斜文本。
- 竖排文本。
- 小图。
- 大图。
- 高宽比极端的图片。

需要对比：

- 检测框数量。
- 四点坐标。
- 框排序。
- score。

## 2. 细节一致性复核

当前实现已经使用 OpenCV.js 和 clipper-lib 复刻主要 DBPostProcess 流程，但仍需要和 PaddleOCR/Python 输出逐项复核：

- `findContours` 在 OpenCV.js 和 Python OpenCV 中的 contour 顺序是否一致。
- `RotatedRect.points()` 和 Python `cv2.boxPoints()` 的点序是否一致。
- `clipper-lib` 的 offset 结果与 Python `pyclipper` 是否存在像素级差异。
- `sharp` resize 与 OpenCV resize 的插值结果是否存在像素级差异。

## 3. golden 回归脚本

建立一个简单回归入口，用当前脚本输出对比 golden JSON，至少报告：

- box count 差异。
- 坐标误差。
- score 误差。
- 排序差异。
