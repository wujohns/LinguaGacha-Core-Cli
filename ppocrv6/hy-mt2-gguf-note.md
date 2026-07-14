# Hy-MT2 1.25bit GGUF Note

The model file is kept locally for a future retry:

- `model/hy-mt2/Hy-MT2-1.8B-1.25Bit.gguf`
- Source: `https://huggingface.co/AngelSlim/Hy-MT2-1.8B-1.25Bit-GGUF`

Current status: paused until upstream runtime support lands.

Why paused:

- The GGUF uses tensor `ggmlType 42`.
- `node-llama-cpp@3.18.1` only recognizes tensor types up to `40:NVFP4`, so model loading fails before inference with `Invalid type or block size`.
- The official `llama.cpp` release `b9785` also rejects the file with `invalid ggml type 42. should be in [0, 42)`.
- The model card says this GGUF depends on the STQ kernel from `llama.cpp` PR `#22836`; at the time of this note, that support is not in the official release used here.

Removed for now:

- `node-llama-cpp` dependency
- local `llama.cpp` runtime under `.runtime`
- runnable Node translation script

Revisit when official `llama.cpp` and `node-llama-cpp` releases support STQ/type 42.

需要等待这个特性就绪: https://github.com/ggml-org/llama.cpp/pull/22836

可以考虑不使用 llama，而是使用如下的更适合的格式:
https://huggingface.co/onnx-community/HY-MT1.5-1.8B-ONNX

onnx 的 1.25bit 特性可以考虑关注 https://github.com/microsoft/onnxruntime/issues/28549
