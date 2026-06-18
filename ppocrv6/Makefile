SHELL := /bin/sh

HF_BASE := https://huggingface.co/PaddlePaddle
COMIC_HF_BASE := https://huggingface.co/ogkalu/comic-text-and-bubble-detector
CURL ?= curl
CURL_PROXY ?=
CURL_FLAGS ?= -L --fail --retry 3 --retry-delay 2

MODEL_DIR := model
MEDIUM_DET_DIR := $(MODEL_DIR)/medium-det
MEDIUM_REC_DIR := $(MODEL_DIR)/medium-rec
TINY_DET_DIR := $(MODEL_DIR)/tiny-det
TINY_REC_DIR := $(MODEL_DIR)/tiny-rec
COMIC_BUBBLE_DIR := $(MODEL_DIR)/comic-bubble

.PHONY: models det rec medium-det medium-rec tiny-det tiny-rec comic-bubble comic-models

models: medium-det medium-rec tiny-det tiny-rec comic-bubble

comic-models: comic-bubble

det: medium-det tiny-det

rec: medium-rec tiny-rec

medium-det: $(MEDIUM_DET_DIR)/inference.onnx $(MEDIUM_DET_DIR)/inference.yml

medium-rec: $(MEDIUM_REC_DIR)/inference.onnx $(MEDIUM_REC_DIR)/inference.yml

tiny-det: $(TINY_DET_DIR)/inference.onnx $(TINY_DET_DIR)/inference.yml

tiny-rec: $(TINY_REC_DIR)/inference.onnx $(TINY_REC_DIR)/inference.yml

comic-bubble: $(COMIC_BUBBLE_DIR)/detector-v4-s_int8.onnx $(COMIC_BUBBLE_DIR)/config.json $(COMIC_BUBBLE_DIR)/preprocessor_config.json

$(MEDIUM_DET_DIR)/inference.onnx:
	@mkdir -p $(dir $@)
	$(CURL) $(CURL_PROXY) $(CURL_FLAGS) -o $@.tmp $(HF_BASE)/PP-OCRv6_medium_det_onnx/resolve/main/inference.onnx
	@mv $@.tmp $@

$(MEDIUM_DET_DIR)/inference.yml:
	@mkdir -p $(dir $@)
	$(CURL) $(CURL_PROXY) $(CURL_FLAGS) -o $@.tmp $(HF_BASE)/PP-OCRv6_medium_det_onnx/resolve/main/inference.yml
	@mv $@.tmp $@

$(MEDIUM_REC_DIR)/inference.onnx:
	@mkdir -p $(dir $@)
	$(CURL) $(CURL_PROXY) $(CURL_FLAGS) -o $@.tmp $(HF_BASE)/PP-OCRv6_medium_rec_onnx/resolve/main/inference.onnx
	@mv $@.tmp $@

$(MEDIUM_REC_DIR)/inference.yml:
	@mkdir -p $(dir $@)
	$(CURL) $(CURL_PROXY) $(CURL_FLAGS) -o $@.tmp $(HF_BASE)/PP-OCRv6_medium_rec_onnx/resolve/main/inference.yml
	@mv $@.tmp $@

$(TINY_DET_DIR)/inference.onnx:
	@mkdir -p $(dir $@)
	$(CURL) $(CURL_PROXY) $(CURL_FLAGS) -o $@.tmp $(HF_BASE)/PP-OCRv6_tiny_det_onnx/resolve/main/inference.onnx
	@mv $@.tmp $@

$(TINY_DET_DIR)/inference.yml:
	@mkdir -p $(dir $@)
	$(CURL) $(CURL_PROXY) $(CURL_FLAGS) -o $@.tmp $(HF_BASE)/PP-OCRv6_tiny_det_onnx/resolve/main/inference.yml
	@mv $@.tmp $@

$(TINY_REC_DIR)/inference.onnx:
	@mkdir -p $(dir $@)
	$(CURL) $(CURL_PROXY) $(CURL_FLAGS) -o $@.tmp $(HF_BASE)/PP-OCRv6_tiny_rec_onnx/resolve/main/inference.onnx
	@mv $@.tmp $@

$(TINY_REC_DIR)/inference.yml:
	@mkdir -p $(dir $@)
	$(CURL) $(CURL_PROXY) $(CURL_FLAGS) -o $@.tmp $(HF_BASE)/PP-OCRv6_tiny_rec_onnx/resolve/main/inference.yml
	@mv $@.tmp $@

$(COMIC_BUBBLE_DIR)/detector-v4-s_int8.onnx:
	@mkdir -p $(dir $@)
	$(CURL) $(CURL_PROXY) $(CURL_FLAGS) -o $@.tmp $(COMIC_HF_BASE)/resolve/main/detector-v4-s_int8.onnx
	@mv $@.tmp $@

$(COMIC_BUBBLE_DIR)/config.json:
	@mkdir -p $(dir $@)
	$(CURL) $(CURL_PROXY) $(CURL_FLAGS) -o $@.tmp $(COMIC_HF_BASE)/resolve/main/config.json
	@mv $@.tmp $@

$(COMIC_BUBBLE_DIR)/preprocessor_config.json:
	@mkdir -p $(dir $@)
	$(CURL) $(CURL_PROXY) $(CURL_FLAGS) -o $@.tmp $(COMIC_HF_BASE)/resolve/main/preprocessor_config.json
	@mv $@.tmp $@
