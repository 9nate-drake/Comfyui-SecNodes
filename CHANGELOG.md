# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Memory optimization: Pre-allocated output tensor to eliminate VRAM spike at end of propagation
  - Reduces peak VRAM usage by ~600-800MB during segmentation
  - Particularly beneficial for 8GB GPU users
- Scene change detection resolution optimization
  - Reduced from 1024x1024 to 512x512 for HSV histogram comparison
  - Saves additional 200-400MB peak VRAM during propagation
  - No quality impact (HSV histogram comparison robust to resolution)

### Changed
- Output mask creation optimized to use pre-allocation instead of list stacking
  - Prevents memory duplication during `torch.stack()` operation
  - More efficient memory profile for large frame counts

### Fixed
- VRAM spike at completion of video segmentation (now uses pre-allocated tensors)

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
