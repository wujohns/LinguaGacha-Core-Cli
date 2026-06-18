SHELL := /bin/sh

HF_BASE := https://huggingface.co/PaddlePaddle
CURL ?= curl
CURL_PROXY ?=
CURL_FLAGS ?= -L --fail --retry 3 --retry-delay 2

MODEL_DIR := model
MEDIUM_DET_DIR := $(MODEL_DIR)/medium-det
MEDIUM_REC_DIR := $(MODEL_DIR)/medium-rec
TINY_DET_DIR := $(MODEL_DIR)/tiny-det
TINY_REC_DIR := $(MODEL_DIR)/tiny-rec

.PHONY: models det rec medium-det medium-rec tiny-det tiny-rec

models: medium-det medium-rec tiny-det tiny-rec

det: medium-det tiny-det

rec: medium-rec tiny-rec

medium-det: $(MEDIUM_DET_DIR)/inference.onnx $(MEDIUM_DET_DIR)/inference.yml

medium-rec: $(MEDIUM_REC_DIR)/inference.onnx $(MEDIUM_REC_DIR)/inference.yml

tiny-det: $(TINY_DET_DIR)/inference.onnx $(TINY_DET_DIR)/inference.yml

tiny-rec: $(TINY_REC_DIR)/inference.onnx $(TINY_REC_DIR)/inference.yml

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
