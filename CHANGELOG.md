# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.0] - 2025-10-16

### Breaking Changes
- **Removed FP8 Model Support**: FP8 quantized models are no longer supported due to numerical instability in the language model. Use FP16 or BF16 models instead. After comprehensive investigation and debugging, it was discovered that FP8 weight-only quantization using torchao's int8_weight_only produces NaN values in the language model's [SEG] token embeddings, breaking scene-aware tracking during MLLM inference.

- The root cause is fundamental numerical instability of the LLM under int8 quantization, not a software bug. Quantizing only the vision model provides minimal VRAM savings (~6%) and isn't worth the complexity. FP8 support has been removed.

  - Quantizing only vision model saves only ~300MB (6%) - not worth the complexity

### Added
- Memory optimization: Pre-allocated output tensor to eliminate VRAM spike at end of propagation
  - Reduces peak VRAM usage by ~600-800MB during segmentation
- Scene change detection resolution optimization
  - Reduced from 1024x1024 to 512x512 for HSV histogram comparison
  - Saves additional 200-400MB peak VRAM during propagation
  - No quality impact (HSV histogram comparison robust to resolution)
- Comprehensive debug logging for MLLM inference analysis
  - `[MLLM-DEBUG]`, `[MLLM-PREDICT]` for troubleshooting
  - Useful for exploring alternative quantization methods

### Changed
- Output mask creation optimized to use pre-allocation instead of list stacking
  - Prevents memory duplication during `torch.stack()` operation
  - More efficient memory profile for large frame counts
- FP8 quantization code disabled with migration guidance (use FP16/BF16 instead)
- **Debug logging disabled by default** - Set `SEC_DEBUG=true` environment variable to enable
  - Eliminates verbose console output during normal operation
  - Debug logs (`[FP8-DEBUG]`, `[MLLM-DEBUG]`, `[MLLM-PREDICT]`) available for troubleshooting
  - Example: `SEC_DEBUG=true python -m comfyui.main` (or equivalent for your environment)

### Fixed
- **Scene Detection Bug #1**: Fixed color space conversion (PIL images are RGB, not BGR)
- **Scene Detection Bug #2**: Fixed inverted scene detection logic (MLLM now called on actual scene changes)
- **Scene Detection Bug #3**: Relaxed object score threshold for better reliability
- VRAM spike at completion of video segmentation (now uses pre-allocated tensors)

### Future Work
Exploring alternative quantization methods for VRAM savings:
- **bitsandbytes 4-bit** (NF4 format, ~6GB savings, better stability than int8)
- **GPTQ quantization** (4-8GB savings with calibration)
- **Kernel optimizations** (Flash Attention improvements)

---

## [1.1.0] - 2025-10-13

### Added
- **Single-file model formats**: Download just one file instead of sharded 4-file format
  - FP16 (7.35GB) - Recommended default
  - FP8 (3.97GB) - VRAM-constrained systems (RTX 30+ required)
  - BF16 (7.35GB) - Alternative to FP16
  - FP32 (14.14GB) - Full precision
- **FP8 quantization support**: Automatic weight-only quantization (W8A16) using torchao + Marlin kernels
  - Saves 1.5-2GB VRAM in real-world usage
  - Requires RTX 30 series or newer (Ampere+ architecture)
  - Automatic fallback to FP16 on older GPUs

### Changed
- Model loader now supports multiple precision formats with auto-detection
- Retains compatibility with sharded model format
- Added `torchao>=0.1.0` to requirements.txt for FP8 support
- Automatic GPU capability detection for FP8 compatibility
- Node package added to ComfyUI-Manager for easy install

---

## [1.0.0] - 2025-09-15

### Added
- Initial release of ComfyUI SeC Nodes
- **SeC Model Loader** node with device selection and Flash Attention support
- **SeC Video Segmentation** node with:
  - Multiple prompt types: points, bbox, mask
  - Bidirectional tracking support
  - MLLM memory size configuration
  - Video frame offloading to CPU
  - Auto model unloading
- **Coordinate Plotter** node for visualization
- Support for SeC-4B (Qwen2.5-3B) model
- Self-contained inference code (no external repo dependencies)
- Comprehensive error handling and validation
- BBOX type compatibility with KJNodes

### Documentation
- Comprehensive README with installation instructions
- Detailed node reference documentation
- GPU VRAM recommendations table
- Example workflows
