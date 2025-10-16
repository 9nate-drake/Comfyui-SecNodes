# ComfyUI SeC Nodes

ComfyUI custom nodes for **SeC (Segment Concept)** - State-of-the-art video object segmentation that outperforms SAM 2.1, utilizing the SeC-4B model developed by OpenIXCLab.

## Changelog

### v1.2 (2025-10-16) - FP8 Removal & Performance Optimizations

⚠️ **IMPORTANT BREAKING CHANGE**: FP8 support has been removed due to fundamental numerical instability issues. **Use FP16 or BF16 models instead.**

**What Changed:**
- FP8 quantization disabled - produces NaN values in language model embeddings during scene detection
- All users should migrate to FP16 or BF16 models (same segmentation quality, fully reliable)
- Memory optimization: Pre-allocated output tensor (saves 600-800MB VRAM spike)
- Scene detection resolution optimization: 1024x1024 → 512x512 (saves 200-400MB, no quality impact)

**Full Technical Details:** See [CHANGELOG.md](CHANGELOG.md) for comprehensive investigation and FP8 failure analysis.

### v1.1 (2025-10-13) - Single-File Models & FP8 Support

**Features:**
- **Single-file model formats**: Download just one file instead of sharded 4-file format
  - FP16 (7.35GB) - Recommended default
  - BF16 (7.35GB) - Alternative to FP16
  - FP32 (14.14GB) - Full precision
- Model loader supports multiple precision formats with auto-detection
- Automatic GPU capability detection
- Node package added to ComfyUI-Manager for easy install

**Download:** Single-file models available at [https://huggingface.co/VeryAladeen/Sec-4B](https://huggingface.co/VeryAladeen/Sec-4B)

## What is SeC?

**SeC (Segment Concept)** is a breakthrough in video object segmentation that shifts from simple feature matching to **high-level conceptual understanding**. Unlike SAM 2.1 which relies primarily on visual similarity, SeC uses a **Large Vision-Language Model (LVLM)** to understand *what* an object is conceptually, enabling robust tracking through:

- **Semantic Understanding**: Recognizes objects by concept, not just appearance
- **Scene Complexity Adaptation**: Automatically balances semantic reasoning vs feature matching
- **Superior Robustness**: Handles occlusions, appearance changes, and complex scenes better than SAM 2.1
- **SOTA Performance**: +11.8 points over SAM 2.1 on SeCVOS benchmark

### How SeC Works

1. **Visual Grounding**: You provide initial prompts (points/bbox/mask) on one frame
2. **Concept Extraction**: SeC's LVLM analyzes the object to build a semantic understanding
3. **Smart Tracking**: Dynamically uses both semantic reasoning and visual features
4. **Keyframe Bank**: Maintains diverse views of the object for robust concept understanding

The result? SeC tracks objects more reliably through challenging scenarios like rapid appearance changes, occlusions, and complex multi-object scenes.

## Demo

https://github.com/user-attachments/assets/5cc6677e-4a9d-4e55-801d-b92305a37725

*Example: SeC tracking an object through scene changes and dynamic movement*



https://github.com/user-attachments/assets/9e99d55c-ba8e-4041-985e-b95cbd6dd066

*Example: SAM fails to track correct dog for some scenes*

## Features

- **SeC Model Loader**: Load SeC models with simple settings
- **SeC Video Segmentation**: SOTA video segmentation with visual prompts
- **Coordinate Plotter**: Visualize coordinate points before segmentation
- **Self-Contained**: All inference code bundled - no external repos needed
- **Bidirectional Tracking**: Track from any frame in any direction

## Installation

### Option 1: ComfyUI-Manager (Recommended - Easiest)

1. **Install ComfyUI-Manager** (if you don't have it already):
   - Get it from: https://github.com/ltdrdata/ComfyUI-Manager

2. **Download a model** (see Model Download section below)

3. **Install SeC Nodes**:
   - Open ComfyUI Manager in ComfyUI
   - Search for **"SeC"** or **"SecNodes"**
   - Click **Install**
   - Click **Restart** when prompted

4. **Done!** The SeC nodes will appear in the "SeC" category

### Option 2: Manual Installation

#### Step 1: Install Custom Node
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/9nate-drake/Comfyui-SecNodes
```

#### Step 2: Install Dependencies

**ComfyUI Portable (Windows):**
```bash
cd ComfyUI/custom_nodes/Comfyui-SecNodes
../../python_embeded/python.exe -m pip install -r requirements.txt
```

**Standard Python Installation (Linux/Mac):**
```bash
cd ComfyUI/custom_nodes/Comfyui-SecNodes
pip install -r requirements.txt
```

#### Step 3: Restart ComfyUI
The nodes will appear in the "SeC" category.

---

## Model Download

**Download ONE of the following model formats:**

The SeC Model Loader will automatically detect and let you select which model to use. Download from [https://huggingface.co/VeryAladeen/Sec-4B](https://huggingface.co/VeryAladeen/Sec-4B) and place in your `ComfyUI/models/sams/` folder:

- **SeC-4B-fp16.safetensors** (Recommended) - 7.35 GB
  - Best balance of quality and size
  - Works on all CUDA GPUs
  - **Recommended for all systems**
- **SeC-4B-bf16.safetensors** (Alternative) - 7.35 GB
  - Alternative to FP16, better for some GPUs
- **SeC-4B-fp32.safetensors** (Full Precision) - 14.14 GB
  - Maximum precision, highest VRAM usage
  - Better compatibility on some older GPUs

⚠️ **FP8 Support Removed (v1.2)**
- FP8 quantization has been removed due to numerical instability issues
- All users should use FP16 or BF16 models instead (same quality, fully reliable)
- See [CHANGELOG.md](CHANGELOG.md) for full technical investigation

#### Alternative: Original Sharded Model

**For users who prefer the original OpenIXCLab format:**

```bash
cd ComfyUI/models/sams

# Download using huggingface-cli (recommended)
huggingface-cli download OpenIXCLab/SeC-4B --local-dir SeC-4B

# Or using git lfs
git lfs clone https://huggingface.co/OpenIXCLab/SeC-4B
```

**Details:**
- Size: ~14.14 GB (sharded into 4 files)
- Precision: FP32
- Includes all config files in the download

## Requirements

- **Python**: 3.10-3.12 (3.12 recommended)
  - Python 3.13: Not recommended - experimental support with known dependency installation issues
- **PyTorch**: 2.6.0+ (included with ComfyUI)
- **CUDA**: 11.8+ for GPU acceleration
- **CUDA GPU**: Recommended (CPU supported but significantly slower)
- **VRAM**: See GPU VRAM recommendations below
  - Can reduce significantly by enabling `offload_video_to_cpu` (~3% speed penalty)

**Note on CPU Mode:**
- CPU inference automatically uses float32 precision (bfloat16/float16 not supported on CPU)
- Expect significantly slower performance compared to GPU (~10-20x slower depending on hardware)
- Not recommended for production use, mainly for testing or systems without GPUs

**Flash Attention 2 (Optional):**
- Provides ~2x speedup but requires specific hardware
- **GPU Requirements**: Ampere/Ada/Hopper architecture only (RTX 30/40 series, A100, H100)
  - Does NOT work on RTX 20 series (Turing) or older GPUs
- **CUDA**: 12.0+ required
- **Windows + Python 3.12**: Use pre-compiled wheels or disable flash attention
- The node automatically falls back to standard attention if Flash Attention is unavailable

## GPU VRAM Recommendations

| VRAM | Resolution | Frames | Model | Key Settings | Performance |
|------|-----------|--------|-------|--------------|-------------|
| **10-12GB** | 256x256 - 512x384 | Up to 100 | **FP16**<br>with CPU offload | `offload_video_to_cpu: True`<br>`mllm_memory_size: 5-10`<br>**Minimum viable config** | 6-10 it/s |
| **16-20GB** | 512x384 - 720p | 100-500 | **FP16 recommended** | Default settings work well<br>`mllm_memory_size: 12-15` | 5-6 it/s |
| **24GB+** | 720p - 1080p+ | 500+ | **FP16 recommended** | `mllm_memory_size: 15-20` for max quality<br>Can use BF16 if preferred | 4-5 it/s |

**Quick Tips:**
- **FP16 is the standard recommendation** - best compatibility and reliability across all systems
- **Low on VRAM (10-12GB)?** Enable `offload_video_to_cpu` (saves 2-3GB, only ~3% slower)
- **Lower `mllm_memory_size`** (5-10) if you need to squeeze into limited VRAM
- **FP8 has been removed** (v1.2) - use FP16 or BF16 instead for reliable performance

### FP8 Quantization (Removed in v1.2)

**Status: No longer supported**

FP8 quantization has been removed due to fundamental numerical instability in the language model. During testing, FP8 produces NaN values in embeddings during scene change detection, breaking semantic tracking completely.

**Real-world FP8 analysis:**
- Would save 1.5-2GB VRAM (not 50% as model size suggests)
- Quantizing vision-only saves only ~300MB (3% total, not worth complexity)
- Full LLM quantization produces NaN - not viable

**Why FP16/BF16 instead:**
- Same memory footprint as FP8 files (7.35GB) - only ~300MB more than FP8
- Fully reliable performance with no numerical issues
- Better GPU compatibility across all CUDA hardware

**For detailed investigation:** See [CHANGELOG.md](CHANGELOG.md) for comprehensive technical analysis of FP8 failure and future quantization alternatives being explored.


## Nodes Reference

### 1. SeC Model Loader
Load and configure the SeC model for inference. Automatically detects available models in `ComfyUI/models/sams/`.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **model_file** | CHOICE | First available | Select which model to load:<br>• FP32 (Full Precision - ~14.5GB)<br>• FP16 (Half Precision - 7.35GB) - **Recommended**<br>• BF16 (Brain Float - ~7GB)<br>• SeC-4B (Sharded/Original - ~14GB)<br>**Note:** Each model uses its native precision automatically. FP8 is no longer supported (v1.2). |
| **device** | CHOICE | `auto` | Device selection (dynamically detects available GPUs):<br>• `auto`: gpu0 if available, else CPU (recommended)<br>• `cpu`: Force CPU (automatically converts to float32)<br>• `gpu0`, `gpu1`, etc.: Specific GPU |
| *use_flash_attn* | BOOLEAN | True | Enable Flash Attention 2 for faster inference.<br>**Note:** Automatically disabled for FP32 precision (requires FP16/BF16) |
| *allow_mask_overlap* | BOOLEAN | True | Allow objects to overlap (disable for strict separation) |

**Outputs:** `model`

**Notes:**
- **Model Selection**: Dynamically shows available models in `ComfyUI/models/sams/` directory
  - Download at least one model format (see Model Download section above)
  - Models are loaded in their **native precision** (preserves memory benefits of model format)
  - FP16 and BF16 have identical sizes (7.35GB) - choose based on GPU preference
- **Config files**: Bundled in this repo - no separate download needed for single-file models
- **Device options dynamically adapt** to your system:
  - 1 GPU system: Shows `auto`, `cpu`, `gpu0`
  - 2 GPU system: Shows `auto`, `cpu`, `gpu0`, `gpu1`
  - 3+ GPU system: Shows all available GPUs
  - No GPU: Shows only `auto` and `cpu`
- **CPU mode**: Automatically converts model to float32 precision (CPU limitation). CPU inference is significantly slower than GPU (~10-20x).
- **Flash Attention**: Automatically disabled for FP32 models (requires FP16/BF16). Standard attention will be used instead.

---

### 2. SeC Video Segmentation
Segment and track objects across video frames.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **model** | MODEL | - | SeC model from loader |
| **frames** | IMAGE | - | Video frames as IMAGE batch |
| *positive_points* | STRING | "" | JSON: `'[{"x": 100, "y": 200}]'` |
| *negative_points* | STRING | "" | JSON: `'[{"x": 50, "y": 50}]'` |
| *bbox* | STRING | "" | Bounding box: `"x1,y1,x2,y2"` |
| *input_mask* | MASK | - | Binary mask input |
| *tracking_direction* | CHOICE | `forward` | forward / backward / bidirectional |
| *annotation_frame_idx* | INT | 0 | Frame where prompt is applied |
| *object_id* | INT | 1 | Unique ID for multi-object tracking |
| *max_frames_to_track* | INT | -1 | Max frames (-1 = all) |
| *mllm_memory_size* | INT | 12 | Number of keyframes for semantic understanding (affects compute on scene changes, not VRAM). Original paper used 7. |
| *offload_video_to_cpu* | BOOLEAN | False | Offload video frames to CPU (saves significant GPU memory, ~3% slower) |
| *auto_unload_model* | BOOLEAN | True | Unload the model from VRAM and RAM after segmentation. Set false if doing multiple segmentations in succession. |

**Outputs:** `masks` (MASK), `object_ids` (INT)

**Important Notes:**
- Provide at least one visual prompt (points, bbox, or mask)
- **Output always matches input frame count**: If you input 100 frames, you get 100 masks
- Frames before/after the tracked range will have empty (blank) masks
  - Example: 100 frames, annotation_frame_idx=50, direction=forward → frames 0-49 are blank, 50-99 are tracked
  - Example: 100 frames, annotation_frame_idx=50, direction=backward → frames 0-50 are tracked, 51-99 are blank
  - Example: 100 frames, annotation_frame_idx=50, direction=bidirectional → all frames 0-99 are tracked

**Input Combination Behavior:**

You can combine different input types for powerful segmentation control:

| Input Combination | Behavior |
|-------------------|----------|
| **Points only** | Standard point-based segmentation |
| **Bbox only** | Segment the most prominent object within bounding box |
| **Mask only** | Track the masked region |
| **Bbox + Points** | **Two-stage refinement**: Bbox establishes initial region, then points refine the segmentation within that region |
| **Mask + Positive points** | Only positive points **inside the mask** are used to refine which part of the masked region to segment |
| **Mask + Negative points** | All negative points are used to exclude regions from the mask |
| **Mask + Positive + Negative** | Positive points inside mask refine the region, negative points exclude areas |

**Example Use Cases:**
- **Bbox + point refinement**: Draw bbox around a person, add point on their shirt to segment just the shirt instead of the whole person
- **Rough mask + precise points**: Draw a rough mask around a person, then add positive points on their face to focus the segmentation
- **Mask + negative exclusions**: Mask an object, add negative points on unwanted parts (e.g., exclude a hand from a person mask)
- **Point filtering**: Positive points outside the mask boundary are automatically ignored, preventing accidental selections

**⚠ Important Note on Negative Points with Masks:**
- Negative points work best when placed **inside or near** the masked region
- Negative points far outside the mask (>50 pixels away) may cause unexpected results or empty segmentation
- You'll receive a warning in the console if negative points are too far from the mask

---

### 3. Coordinate Plotter
Visualize coordinate points on images for debugging.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **coordinates** | STRING | `'[{"x": 100, "y": 100}]'` | JSON coordinates to plot |
| *image* | IMAGE | - | Optional image (overrides width/height) |
| *point_shape* | CHOICE | `circle` | circle / square / triangle |
| *point_size* | INT | 10 | Point size in pixels (1-100) |
| *point_color* | STRING | `#00FF00` | Hex `#FF0000` or RGB `255,0,0` |
| *width* | INT | 512 | Canvas width if no image |
| *height* | INT | 512 | Canvas height if no image |

**Outputs:** `image` (IMAGE)


## Tracking Directions

| Direction | Best For | Behavior |
|-----------|----------|----------|
| **forward** | Standard videos, object appears at start | Frame N → end |
| **backward** | Object appears later, reverse analysis | Frame N → start |
| **bidirectional** | Object clearest in middle, complex scenes | Frame N → both directions |


### Understanding mllm_memory_size

The `mllm_memory_size` parameter controls how many historical keyframes SeC's Large Vision-Language Model uses for semantic understanding:

- **What it does**: Stores frame references (first frame + last N-1 frames) for the LVLM to analyze when scene changes occur
- **VRAM impact**: None - testing shows values 3-20 use similar VRAM (~11-13GB for typical videos)
- **Compute impact**: Higher values mean more frames processed through the vision encoder on scene changes
- **Quality trade-off**: More keyframes = better object concept understanding in complex scenes, but diminishing returns after ~10-12 frames
- **Original research**: SeC paper used 7 and achieved SOTA performance (+11.8 over SAM 2.1), emphasizing "quality over quantity" of keyframes

**Recommended Values:**
- **Default (12)**: Balanced approach - higher than paper's 7 for extra context, but not excessive
- **Low (5-7)**: Faster inference on simple videos, matches original research setup
- **High (15-20)**: Maximum semantic context for very complex videos (no VRAM penalty)

**Why doesn't it affect VRAM?** The parameter stores lightweight frame indices and mask arrays, not full frame tensors. When scene changes occur, frames are loaded from disk on-demand for LVLM processing. The underlying SAM2 architecture supports up to 22 frames.

## Attribution

This node implements the **SeC-4B** model developed by OpenIXCLab.

- **Model Repository**: [OpenIXCLab/SeC-4B](https://huggingface.co/OpenIXCLab/SeC-4B)
- **Paper**: [arXiv:2507.15852](https://arxiv.org/abs/2507.15852)
- **Official Implementation**: [github.com/OpenIXCLab/SeC](https://github.com/OpenIXCLab/SeC)
- **License**: Apache 2.0

**Dataset**: The original work includes the [SeCVOS Benchmark](https://huggingface.co/datasets/OpenIXCLab/SeCVOS) dataset.

## Known Limitations

**Mask-Only Inputs**: Using only a mask or bounding box may result in less stable tracking. This is due to how the underlying SAM2 and MLLM components process mask and bbox inputs. For best results, combine masks/bboxes with coordinate points for more precise control.

## Troubleshooting


**CUDA out of memory**:
- Enable `offload_video_to_cpu` (saves 2-3GB VRAM, only ~3% slower)
- Lower `mllm_memory_size` (5-10) to reduce computation at scene changes
- Process fewer frames at once (split video into smaller batches)
- Use FP16 or BF16 models (FP8 is no longer supported)
- See GPU VRAM recommendations above for your hardware tier

**Slow inference**:
- Enable `use_flash_attn` in model loader (requires Flash Attention 2)
- Disable `offload_video_to_cpu` if you have sufficient VRAM

---

*Self-contained ComfyUI nodes - just install and segment!* 🎉
