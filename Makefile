# Strata Apex — Native Apple Silicon Metal 4 LLM & Swarm Inference Engine Build System

VERSION     := 0.1.0
PREFIX      ?= /opt/homebrew
CXX         := clang++
CXXFLAGS    := -std=c++20 -O3 -Wall -Wextra -Werror -Iinclude -ffile-prefix-map=$(CURDIR)=.
OBJCXXFLAGS := $(CXXFLAGS) -fobjc-arc

FRAMEWORKS  := -framework Foundation -framework Metal -framework CoreML -framework IOSurface -framework Accelerate -framework ImageIO -framework CoreGraphics

BUILD_DIR   := build
BENCH_DIR   := bench_data
DIST_DIR    := dist

GPU_SRCS    := $(CURDIR)/src/gpu/strata_codec.cpp $(CURDIR)/src/gpu/strata_gpu.mm
HW_SRCS     := $(CURDIR)/src/hw/strata_sme2_io.mm $(CURDIR)/src/hw/strata_ane_probe.mm
MODEL_SRCS  := $(CURDIR)/src/model/strata_pack.cpp $(CURDIR)/src/model/strata_sampler.cpp

EMBED_SHADER_ASSETS := \
	src/gpu/strata_kernels.metal include/strata_abi.h \
	src/gpu/strata_qwen38_kernels.metal include/strata_qwen38.h \
	src/gpu/strata_qwen_flash_kernels.metal include/strata_qwen_flash.h \
	src/gpu/strata_glm53_kernels.metal include/strata_glm53.h \
	src/gpu/strata_inkling2_kernels.metal include/strata_inkling2.h \
	src/gpu/strata_inkling2_vision_kernels.metal include/strata_inkling2_vision.h

CORE_OBJS   := \
	$(BUILD_DIR)/strata_paths.o \
	$(BUILD_DIR)/strata_embedded_shaders.o \
	$(BUILD_DIR)/strata_codec.o \
	$(BUILD_DIR)/strata_gpu.o \
	$(BUILD_DIR)/strata_metal_cache.o \
	$(BUILD_DIR)/strata_sme2_io.o \
	$(BUILD_DIR)/strata_ane_probe.o \
	$(BUILD_DIR)/strata_pack.o \
	$(BUILD_DIR)/strata_sampler.o \
	$(BUILD_DIR)/strata_inkling2_vision.o

CLI_OBJS    := \
	$(CORE_OBJS) \
	$(BUILD_DIR)/strata_convert.o \
	$(BUILD_DIR)/strata_infer.o \
	$(BUILD_DIR)/strata_qwen38_convert.o \
	$(BUILD_DIR)/strata_qwen38_gpu.o \
	$(BUILD_DIR)/strata_qwen38_bench.o \
	$(BUILD_DIR)/strata_qwen_flash_convert.o \
	$(BUILD_DIR)/strata_qwen_flash_gpu.o \
	$(BUILD_DIR)/strata_qwen_flash_bench.o \
	$(BUILD_DIR)/strata_glm53_convert.o \
	$(BUILD_DIR)/strata_glm53_gpu.o \
	$(BUILD_DIR)/strata_glm53_bench.o \
	$(BUILD_DIR)/strata_inkling2_convert.o \
	$(BUILD_DIR)/strata_inkling2_gpu.o \
	$(BUILD_DIR)/strata_inkling2_bench.o \
	$(BUILD_DIR)/strata_chat_repl.o \
	$(BUILD_DIR)/strata_openai_server.o

.PHONY: all poc cli test_gpu test_hw test_model test_cli test install uninstall dist clean clean_bench

all: poc cli

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(BENCH_DIR):
	@mkdir -p $(BENCH_DIR)

$(BUILD_DIR)/strata_paths.o: src/model/strata_paths.cpp include/strata_paths.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(CURDIR)/src/model/strata_paths.cpp -o $@

$(BUILD_DIR)/strata_embedded_shaders.cpp: scripts/embed_shaders.py $(EMBED_SHADER_ASSETS) | $(BUILD_DIR)
	python3 scripts/embed_shaders.py $@

$(BUILD_DIR)/strata_embedded_shaders.o: $(BUILD_DIR)/strata_embedded_shaders.cpp include/strata_embedded_shaders.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/strata_codec.o: src/gpu/strata_codec.cpp include/strata_abi.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(CURDIR)/src/gpu/strata_codec.cpp -o $@

$(BUILD_DIR)/strata_gpu.o: src/gpu/strata_gpu.mm include/strata_abi.h include/strata_checkpoint.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/gpu/strata_gpu.mm -o $@

$(BUILD_DIR)/strata_metal_cache.o: src/gpu/strata_metal_cache.mm include/strata_metal_cache.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/gpu/strata_metal_cache.mm -o $@

$(BUILD_DIR)/strata_sme2_io.o: src/hw/strata_sme2_io.mm include/strata_abi.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/hw/strata_sme2_io.mm -o $@

$(BUILD_DIR)/strata_ane_probe.o: src/hw/strata_ane_probe.mm include/strata_abi.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/hw/strata_ane_probe.mm -o $@

$(BUILD_DIR)/strata_pack.o: src/model/strata_pack.cpp include/strata_abi.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(CURDIR)/src/model/strata_pack.cpp -o $@

$(BUILD_DIR)/strata_sampler.o: src/model/strata_sampler.cpp include/strata_abi.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(CURDIR)/src/model/strata_sampler.cpp -o $@

$(BUILD_DIR)/strata_convert.o: src/model/strata_convert.cpp include/strata_abi.h include/strata_checkpoint.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(CURDIR)/src/model/strata_convert.cpp -o $@

$(BUILD_DIR)/strata_infer.o: src/cli/strata_infer.mm include/strata_abi.h include/strata_checkpoint.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/cli/strata_infer.mm -o $@

$(BUILD_DIR)/strata_qwen38_convert.o: src/model/strata_qwen38_convert.cpp include/strata_qwen38.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(CURDIR)/src/model/strata_qwen38_convert.cpp -o $@

$(BUILD_DIR)/strata_qwen38_gpu.o: src/gpu/strata_qwen38_gpu.mm src/gpu/strata_qwen38_kernels.metal include/strata_qwen38.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/gpu/strata_qwen38_gpu.mm -o $@

$(BUILD_DIR)/strata_qwen38_bench.o: src/cli/strata_qwen38_bench.mm include/strata_qwen38.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/cli/strata_qwen38_bench.mm -o $@

$(BUILD_DIR)/strata_qwen_flash_convert.o: src/model/strata_qwen_flash_convert.cpp include/strata_qwen_flash.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(CURDIR)/src/model/strata_qwen_flash_convert.cpp -o $@

$(BUILD_DIR)/strata_qwen_flash_gpu.o: src/gpu/strata_qwen_flash_gpu.mm src/gpu/strata_qwen_flash_kernels.metal include/strata_qwen_flash.h include/strata_metal_cache.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/gpu/strata_qwen_flash_gpu.mm -o $@

$(BUILD_DIR)/strata_qwen_flash_bench.o: src/cli/strata_qwen_flash_bench.mm include/strata_qwen_flash.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/cli/strata_qwen_flash_bench.mm -o $@

$(BUILD_DIR)/strata_glm53_convert.o: src/model/strata_glm53_convert.cpp include/strata_glm53.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(CURDIR)/src/model/strata_glm53_convert.cpp -o $@

$(BUILD_DIR)/strata_glm53_gpu.o: src/gpu/strata_glm53_gpu.mm src/gpu/strata_glm53_kernels.metal include/strata_glm53.h include/strata_metal_cache.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/gpu/strata_glm53_gpu.mm -o $@

$(BUILD_DIR)/strata_glm53_bench.o: src/cli/strata_glm53_bench.mm include/strata_glm53.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/cli/strata_glm53_bench.mm -o $@

$(BUILD_DIR)/strata_inkling2_convert.o: src/model/strata_inkling2_convert.cpp include/strata_inkling2.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $(CURDIR)/src/model/strata_inkling2_convert.cpp -o $@

$(BUILD_DIR)/strata_inkling2_gpu.o: src/gpu/strata_inkling2_gpu.mm src/gpu/strata_inkling2_kernels.metal include/strata_inkling2.h include/strata_metal_cache.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/gpu/strata_inkling2_gpu.mm -o $@

$(BUILD_DIR)/strata_inkling2_vision.o: src/model/strata_inkling2_vision.mm src/gpu/strata_inkling2_vision_kernels.metal include/strata_inkling2_vision.h include/strata_metal_cache.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/model/strata_inkling2_vision.mm -o $@

$(BUILD_DIR)/strata_inkling2_bench.o: src/cli/strata_inkling2_bench.mm include/strata_inkling2.h include/strata_inkling2_vision.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/cli/strata_inkling2_bench.mm -o $@

$(BUILD_DIR)/strata_chat_repl.o: src/cli/strata_chat_repl.mm include/strata_qwen38.h include/strata_qwen_flash.h include/strata_glm53.h include/strata_inkling2.h include/strata_inkling2_vision.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/cli/strata_chat_repl.mm -o $@

$(BUILD_DIR)/strata_openai_server.o: src/cli/strata_openai_server.mm include/strata_qwen38.h include/strata_qwen_flash.h include/strata_glm53.h include/strata_inkling2.h include/strata_inkling2_vision.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/cli/strata_openai_server.mm -o $@

$(BUILD_DIR)/strata_engine.o: src/strata_engine.mm include/strata_abi.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/strata_engine.mm -o $@

$(BUILD_DIR)/strata_cli.o: src/cli/strata_cli.mm include/strata_abi.h include/strata_checkpoint.h include/strata_inkling2_vision.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/cli/strata_cli.mm -o $@

$(BUILD_DIR)/test_cli.o: src/cli/test_cli.mm include/strata_abi.h include/strata_checkpoint.h include/strata_qwen_flash.h include/strata_glm53.h include/strata_inkling2.h include/strata_inkling2_vision.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) -c $(CURDIR)/src/cli/test_cli.mm -o $@

test_gpu: $(BUILD_DIR)/test_gpu
$(BUILD_DIR)/test_gpu: $(GPU_SRCS) src/gpu/test_gpu.mm $(BUILD_DIR)/strata_metal_cache.o $(BUILD_DIR)/strata_paths.o $(BUILD_DIR)/strata_embedded_shaders.o include/strata_abi.h include/strata_checkpoint.h include/strata_metal_cache.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) $(GPU_SRCS) $(CURDIR)/src/gpu/test_gpu.mm $(BUILD_DIR)/strata_metal_cache.o $(BUILD_DIR)/strata_paths.o $(BUILD_DIR)/strata_embedded_shaders.o $(FRAMEWORKS) -o $@

test_hw: $(BUILD_DIR)/test_hw
$(BUILD_DIR)/test_hw: $(HW_SRCS) src/hw/test_hw.mm include/strata_abi.h | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) $(HW_SRCS) $(CURDIR)/src/hw/test_hw.mm $(FRAMEWORKS) -o $@

test_model: $(BUILD_DIR)/test_model
$(BUILD_DIR)/test_model: $(MODEL_SRCS) src/model/test_model.cpp $(BUILD_DIR)/strata_paths.o include/strata_abi.h | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(MODEL_SRCS) $(CURDIR)/src/model/test_model.cpp $(BUILD_DIR)/strata_paths.o -o $@

poc: $(BUILD_DIR)/strata_poc
$(BUILD_DIR)/strata_poc: $(BUILD_DIR)/strata_engine.o $(CORE_OBJS) | $(BUILD_DIR) $(BENCH_DIR)
	$(CXX) $(OBJCXXFLAGS) $(BUILD_DIR)/strata_engine.o $(CORE_OBJS) $(FRAMEWORKS) -o $@

cli: $(BUILD_DIR)/strata-apex $(BUILD_DIR)/strata_cli
$(BUILD_DIR)/strata-apex: $(BUILD_DIR)/strata_cli.o $(CLI_OBJS) | $(BUILD_DIR)
	$(CXX) $(OBJCXXFLAGS) $(BUILD_DIR)/strata_cli.o $(CLI_OBJS) $(FRAMEWORKS) -o $@

$(BUILD_DIR)/strata_cli: $(BUILD_DIR)/strata-apex
	ln -f $(BUILD_DIR)/strata-apex $@

test_cli: $(BUILD_DIR)/test_cli
$(BUILD_DIR)/test_cli: $(BUILD_DIR)/test_cli.o $(CLI_OBJS) | $(BUILD_DIR) $(BENCH_DIR)
	$(CXX) $(OBJCXXFLAGS) $(BUILD_DIR)/test_cli.o $(CLI_OBJS) $(FRAMEWORKS) -o $@

test: $(BUILD_DIR)/test_model $(BUILD_DIR)/test_gpu $(BUILD_DIR)/test_hw $(BUILD_DIR)/test_cli $(BUILD_DIR)/strata_poc
	./$(BUILD_DIR)/test_model
	./$(BUILD_DIR)/test_gpu
	./$(BUILD_DIR)/test_hw
	./$(BUILD_DIR)/test_cli
	./$(BUILD_DIR)/strata_poc

install: cli
	install -d "$(DESTDIR)$(PREFIX)/bin"
	install -m 0755 $(BUILD_DIR)/strata-apex "$(DESTDIR)$(PREFIX)/bin/strata-apex"
	install -m 0755 scripts/bench_showcase.sh "$(DESTDIR)$(PREFIX)/bin/strata-apex-bench"

uninstall:
	rm -f "$(DESTDIR)$(PREFIX)/bin/strata-apex" "$(DESTDIR)$(PREFIX)/bin/strata-apex-bench"

dist: cli
	@mkdir -p $(DIST_DIR)/stage/strata-apex-$(VERSION)
	install -m 0755 $(BUILD_DIR)/strata-apex $(DIST_DIR)/stage/strata-apex-$(VERSION)/strata-apex
	install -m 0755 scripts/bench_showcase.sh $(DIST_DIR)/stage/strata-apex-$(VERSION)/strata-apex-bench
	install -m 0644 README.md $(DIST_DIR)/stage/strata-apex-$(VERSION)/README.md
	COPYFILE_DISABLE=1 tar --no-xattrs --no-mac-metadata --uid 0 --gid 0 --uname root --gname wheel \
		-czf $(DIST_DIR)/strata-apex-$(VERSION)-macos-arm64.tar.gz -C $(DIST_DIR)/stage strata-apex-$(VERSION)
	@if gunzip -c $(DIST_DIR)/strata-apex-$(VERSION)-macos-arm64.tar.gz | grep -a -q -F -e "$$HOME" -e "$(CURDIR)"; then \
		echo "[dist] ERROR: release archive contains this machine's home or build path" >&2; exit 1; fi
	@if tar -tzvf $(DIST_DIR)/strata-apex-$(VERSION)-macos-arm64.tar.gz | awk '$$3 != "root" || $$4 != "wheel" || $$NF ~ /(^|\/)\._/ { bad = 1 } END { exit !bad }'; then \
		echo "[dist] ERROR: release archive has a local owner name or macOS metadata entries" >&2; exit 1; fi
	cd $(DIST_DIR) && shasum -a 256 strata-apex-$(VERSION)-macos-arm64.tar.gz > strata-apex-$(VERSION)-macos-arm64.tar.gz.sha256
	rm -rf $(DIST_DIR)/stage
	@echo "[dist] Built $(DIST_DIR)/strata-apex-$(VERSION)-macos-arm64.tar.gz"

clean_bench:
	rm -rf $(BENCH_DIR)/*.strata $(BENCH_DIR)/*.bin $(BENCH_DIR)/*.ngram $(BENCH_DIR)/*.safetensors $(BENCH_DIR)/*.json $(BENCH_DIR)/*.txt $(BENCH_DIR)/cli_*

clean: clean_bench
	rm -rf $(BUILD_DIR) $(DIST_DIR)
