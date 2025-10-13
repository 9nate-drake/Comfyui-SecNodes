# Uses SeC-4B model from OpenIXCLab
# Model: https://huggingface.co/OpenIXCLab/SeC-4B
# Licensed under Apache 2.0

import torch
import numpy as np
from PIL import Image
import folder_paths
import os
import sys
from safetensors.torch import load_file

from .inference.configuration_sec import SeCConfig
from .inference.modeling_sec import SeCModel
from transformers import AutoTokenizer
from pathlib import Path


def get_gpu_compute_capability():
    """
    Get CUDA compute capability of the current GPU.
    Returns (major, minor) tuple or None if CUDA not available.
    """
    if not torch.cuda.is_available():
        return None

    try:
        device = torch.cuda.current_device()
        major, minor = torch.cuda.get_device_capability(device)
        return (major, minor)
    except Exception as e:
        print(f"Warning: Could not detect GPU compute capability: {e}")
        return None


def supports_fp8_quantization():
    """
    Check if current GPU supports FP8 quantization with Marlin kernels.
    Requires compute capability >= 8.6 (Ampere or newer: RTX 30/40 series, A100, H100).
    """
    capability = get_gpu_compute_capability()
    if capability is None:
        return False

    major, minor = capability
    # Ampere (8.6) and newer support weight-only FP8 via Marlin kernels
    return (major > 8) or (major == 8 and minor >= 6)


def apply_fp8_weight_only_quantization(model, precision_str):
    """
    Apply weight-only FP8 quantization to Linear layers in the model using torchao.
    Only quantizes Language Model and Vision Model (transformer components).
    Skips Grounding Encoder (contains Conv2d layers incompatible with FP8).

    Args:
        model: The SeCModel instance
        precision_str: Model precision string (e.g., "fp8", "fp16")

    Returns:
        bool: True if quantization was applied, False otherwise
    """
    # Only quantize if:
    # 1. Model file is FP8 (to benefit from storage savings)
    # 2. GPU supports FP8 quantization (Ampere+)
    # 3. Model is on CUDA device
    if precision_str != "fp8":
        return False

    if not supports_fp8_quantization():
        capability = get_gpu_compute_capability()
        if capability:
            major, minor = capability
            print(f"  GPU compute capability {major}.{minor} does not support FP8 quantization (requires >= 8.6)")
        else:
            print(f"  No CUDA GPU detected - FP8 quantization requires Ampere or newer (RTX 30/40 series)")
        print(f"  Falling back to FP16 inference (you still save 46% on download size!)")
        return False

    try:
        # Import torchao quantization functions
        from torchao.quantization import quantize_, int8_weight_only

        quantized_components = []

        # Quantize Language Model and Vision Model (transformer components)
        if hasattr(model, 'language_model'):
            try:
                quantize_(model.language_model, int8_weight_only())
                quantized_components.append("LLM")
            except Exception as e:
                print(f"  Warning: Could not quantize Language Model: {e}")

        if hasattr(model, 'vision_encoder'):
            try:
                quantize_(model.vision_encoder, int8_weight_only())
                quantized_components.append("Vision")
            except Exception as e:
                print(f"  Warning: Could not quantize Vision Model: {e}")

        if quantized_components:
            print(f"✓ FP8 quantization applied ({', '.join(quantized_components)})")
            return True
        else:
            print("  Warning: No components were quantized")
            return False

    except ImportError:
        print("  Warning: torchao not installed - install with: pip install torchao")
        print("  Falling back to FP16 inference")
        return False
    except Exception as e:
        print(f"  Warning: FP8 quantization failed: {e}")
        print("  Falling back to FP16 inference")
        return False


def get_repo_config_path():
    """Get path to model config files stored in the repo."""
    repo_root = Path(__file__).parent
    config_dir = repo_root / "model_config"
    return str(config_dir)


def get_available_sec_models():
    """
    Scan for all available SeC-4B models in registered 'sams' folder paths.
    Returns a list of dicts with model info: {'name': str, 'path': str, 'is_single_file': bool, 'config_path': str, 'precision': str}
    """
    available_models = []

    try:
        sams_paths = folder_paths.get_folder_paths("sams")
    except KeyError:
        return available_models

    for sams_dir in sams_paths:
        # Check for single-file models with different precisions
        single_file_models = [
            ("SeC-4B-fp32.safetensors", "fp32"),
            ("SeC-4B-fp16.safetensors", "fp16"),
            ("SeC-4B-bf16.safetensors", "bf16"),
            ("SeC-4B-fp8.safetensors", "fp8"),
        ]

        for filename, precision in single_file_models:
            model_path = os.path.join(sams_dir, filename)
            if os.path.exists(model_path) and os.path.isfile(model_path):
                config_path = get_repo_config_path()
                available_models.append({
                    'name': filename,  # Display actual filename
                    'path': model_path,
                    'is_single_file': True,
                    'config_path': config_path,
                    'precision': precision
                })

        # Check for sharded model directory (original format)
        model_dir = os.path.join(sams_dir, "SeC-4B")
        if os.path.exists(model_dir) and os.path.isdir(model_dir):
            # Verify required files exist
            config_exists = os.path.exists(os.path.join(model_dir, "config.json"))
            model_exists = (
                os.path.exists(os.path.join(model_dir, "model.safetensors")) or
                os.path.exists(os.path.join(model_dir, "model.safetensors.index.json")) or
                os.path.exists(os.path.join(model_dir, "pytorch_model.bin")) or
                os.path.exists(os.path.join(model_dir, "pytorch_model.bin.index.json"))
            )
            tokenizer_exists = os.path.exists(os.path.join(model_dir, "tokenizer_config.json"))

            if config_exists and model_exists and tokenizer_exists:
                available_models.append({
                    'name': "SeC-4B (sharded)",
                    'path': model_dir,
                    'is_single_file': False,
                    'config_path': model_dir,
                    'precision': 'fp16'  # Original format is typically fp16
                })

    return available_models


def find_sec_model():
    """
    Find SeC-4B model in registered 'sams' folder paths (legacy function for backward compatibility).
    Returns the highest priority model found.
    Returns a tuple: (model_path, is_single_file, config_path)
    """
    available_models = get_available_sec_models()

    if not available_models:
        return None, False, None

    # Return the first model (highest priority)
    model = available_models[0]
    return model['path'], model['is_single_file'], model['config_path']


# Auto-download functionality removed per user request
# Users should manually download models using huggingface-cli or git lfs
# See README.md for download instructions


class SeCModelLoader:
    """
    ComfyUI node for loading SeC (Segment Concept) models
    """

    @classmethod
    def INPUT_TYPES(cls):
        # Dynamically build device list based on available GPUs
        device_choices = ["auto", "cpu"]

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                device_choices.append(f"gpu{i}")

        # Dynamically scan for available models
        available_models = get_available_sec_models()
        if available_models:
            model_choices = [model['name'] for model in available_models]
        else:
            model_choices = ["(No models found - see README for download instructions)"]

        return {
            "required": {
                "model_file": (model_choices, {
                    "default": model_choices[0],
                    "tooltip": "Select SeC model file. Each file has a native precision that will be used automatically."
                }),
                "device": (device_choices, {
                    "default": "auto",
                    "tooltip": "Device: auto (gpu0 if available, else CPU), cpu, gpu0/gpu1/etc (specific GPU)"
                })
            },
            "optional": {
                "use_flash_attn": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Flash Attention 2 for faster inference. Automatically disabled for float32 precision."
                }),
                "allow_mask_overlap": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Allow tracked objects to overlap. Disable for strictly separate objects."
                })
            }
        }
    
    RETURN_TYPES = ("SEC_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "SeC"
    TITLE = "SeC Model Loader"

    def load_model(self, model_file, device, use_flash_attn=True, allow_mask_overlap=True):
        """Load SeC model"""

        # Get available models and find the selected one
        available_models = get_available_sec_models()

        selected_model = None
        for model in available_models:
            if model['name'] == model_file:
                selected_model = model
                break

        # Error if no model found
        if selected_model is None or "(No models found" in model_file:
            raise RuntimeError(
                "No SeC model found in ComfyUI/models/sams/\n\n"
                "Please download a model manually:\n"
                "1. Choose a model format (FP8/FP16/BF16/FP32 or original sharded)\n"
                "2. Follow download instructions in README.md\n"
                "3. Restart ComfyUI or refresh the node to detect the model"
            )

        model_path = selected_model['path']
        is_single_file = selected_model['is_single_file']
        config_path = selected_model['config_path']
        precision_str = selected_model['precision']

        print(f"Loading SeC model: {os.path.basename(model_path) if is_single_file else 'SeC-4B (sharded)'} [{precision_str.upper()}]")

        # Handle device selection
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device.startswith("gpu"):
            # Map gpu0 -> cuda:0, gpu1 -> cuda:1, etc.
            try:
                gpu_num = int(device[3:])  # Extract number after "gpu"
                if torch.cuda.is_available():
                    available_gpus = torch.cuda.device_count()
                    if gpu_num >= available_gpus:
                        raise ValueError(f"GPU {gpu_num} not available. System has {available_gpus} GPU(s) (0-{available_gpus-1})")
                else:
                    raise ValueError(f"CUDA not available but GPU device '{device}' was selected")
                device = f"cuda:{gpu_num}"
            except (ValueError, IndexError) as e:
                if "invalid literal" in str(e):
                    raise ValueError(f"Invalid GPU device format: '{device}'. Expected format: 'gpu0', 'gpu1', etc.")
                raise

        # Map model precision to torch dtype
        # Note: FP8 models are stored as FP8 but must be converted to FP16 for inference
        # due to PyTorch/cuDNN limitations with FP8 convolutions
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "fp8": torch.float16  # FP8 files converted to FP16 for inference (cuDNN limitation)
        }

        torch_dtype = dtype_map.get(precision_str, torch.float16)

        # Special message for FP8 models
        if precision_str == "fp8":
            if supports_fp8_quantization():
                print(f"FP8 model: converting to FP16, then applying weight-only quantization")
            else:
                print(f"FP8 model: converting to FP16 for inference (GPU doesn't support FP8 quantization)")

        # Force float32 for CPU mode to avoid dtype mismatches
        if device == "cpu" and torch_dtype != torch.float32:
            print(f"⚠ CPU mode requires float32 precision. Model will be converted from {precision_str.upper()} -> FP32 on load")
            torch_dtype = torch.float32

        # Flash Attention requires float16/bfloat16 - auto-disable for float32
        if torch_dtype == torch.float32 and use_flash_attn:
            print(f"⚠ Flash Attention not compatible with FP32. Disabling Flash Attention.")
            print("  Note: Inference will use standard attention")
            use_flash_attn = False

        hydra_overrides_extra = []
        overlap_value = "false" if allow_mask_overlap else "true"
        hydra_overrides_extra.append(f"++model.non_overlap_masks={overlap_value}")

        try:
            # Load config from appropriate location (config_path for single files, model_path for directories)
            config = SeCConfig.from_pretrained(config_path)
            config.hydra_overrides_extra = hydra_overrides_extra

            # Prepare for GPU loading
            if device.startswith("cuda:"):
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            # Load model differently based on format
            if is_single_file:
                # Manual instantiation and weight loading for single-file models
                model = SeCModel(config, use_flash_attn=use_flash_attn)
                state_dict = load_file(model_path)

                # Convert FP8 weights to FP16 if needed
                if precision_str == "fp8":
                    converted_state_dict = {}
                    for key, tensor in state_dict.items():
                        if tensor.dtype == torch.float8_e4m3fn:
                            converted_state_dict[key] = tensor.to(torch.float16)
                        else:
                            converted_state_dict[key] = tensor
                    state_dict = converted_state_dict

                # Load state dict into model
                model.load_state_dict(state_dict, strict=True)
                model = model.eval()

                # Move to target device and convert dtype if needed
                if device.startswith("cuda:"):
                    model = model.to(device=device, dtype=torch_dtype)
                else:
                    model = model.to(device="cpu", dtype=torch_dtype)

            else:
                # Directory-based loading for sharded models (original approach)
                load_kwargs = {
                    "config": config,
                    "torch_dtype": torch_dtype,
                    "use_flash_attn": use_flash_attn,
                }

                if device.startswith("cuda:"):
                    load_kwargs["device_map"] = {"": device}  # Force to specific GPU
                    load_kwargs["low_cpu_mem_usage"] = True
                else:
                    load_kwargs["low_cpu_mem_usage"] = True

                model = SeCModel.from_pretrained(model_path, **load_kwargs).eval()

            # Load tokenizer from config location (not weights location for single files)
            tokenizer = AutoTokenizer.from_pretrained(config_path, trust_remote_code=True)
            model.preparing_for_generation(tokenizer=tokenizer, torch_dtype=torch_dtype)

            if device.startswith("cuda") and torch_dtype != torch.float32:
                #print(f"Installing dtype conversion hooks...")

                def dtype_conversion_hook(module, args, kwargs):
                    try:
                        module_dtype = None
                        for param in module.parameters():
                            module_dtype = param.dtype
                            break

                        if module_dtype is None:
                            return args, kwargs

                        if isinstance(module, torch.nn.Embedding):
                            return args, kwargs

                        new_args = []
                        for arg in args:
                            if isinstance(arg, torch.Tensor):
                                if arg.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                                    new_args.append(arg)
                                elif arg.dtype != module_dtype:
                                    new_args.append(arg.to(dtype=module_dtype))
                                else:
                                    new_args.append(arg)
                            else:
                                new_args.append(arg)

                        new_kwargs = {}
                        for k, v in kwargs.items():
                            if isinstance(v, torch.Tensor):
                                if v.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                                    new_kwargs[k] = v
                                elif v.dtype != module_dtype:
                                    new_kwargs[k] = v.to(dtype=module_dtype)
                                else:
                                    new_kwargs[k] = v
                            else:
                                new_kwargs[k] = v

                        return tuple(new_args), new_kwargs
                    except Exception:
                        return args, kwargs

                for module in model.modules():
                    if len(list(module.parameters(recurse=False))) > 0:
                        module.register_forward_pre_hook(dtype_conversion_hook, with_kwargs=True)

            # Apply FP8 weight-only quantization if applicable (after model is on device)
            if device.startswith("cuda:"):
                apply_fp8_weight_only_quantization(model, precision_str)

            print(f"✓ Model loaded on {device}")

            # Store loading metadata for potential auto-reload
            model._sec_loading_metadata = {
                'model_path': model_path,
                'is_single_file': is_single_file,
                'config_path': config_path,
                'torch_dtype': torch_dtype,
                'device': device,
                'use_flash_attn': use_flash_attn,
                'allow_mask_overlap': allow_mask_overlap,
                'config': config,
                'hydra_overrides_extra': hydra_overrides_extra
            }

            return (model,)

        except Exception as e:
            raise RuntimeError(f"Failed to load SeC model: {str(e)}")


class SeCVideoSegmentation:
    """
    SeC Video Object Segmentation - Concept-driven video segmentation using multimodal understanding.
    
    Performs intelligent video object segmentation by combining visual features with semantic reasoning.
    Supports multiple prompt types and adapts computational effort based on scene complexity.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SEC_MODEL", {
                    "tooltip": "SeC model loaded from SeCModelLoader node"
                }),
                "frames": ("IMAGE", {
                    "tooltip": "Sequential video frames as IMAGE tensor batch"
                })
            },
            "optional": {
                "positive_points": ("STRING", {
                    "default": "",
                    "tooltip": "Positive click coordinates as JSON: '[{\"x\": 63, \"y\": 782}]'"
                }),
                "negative_points": ("STRING", {
                    "default": "",
                    "tooltip": "Negative click coordinates as JSON: '[{\"x\": 100, \"y\": 200}]'"
                }),
                "bbox": ("BBOX", {
                    "tooltip": "Bounding box as (x_min, y_min, x_max, y_max) or (x, y, width, height) tuple. Compatible with KJNodes Points Editor bbox output."
                }),
                "input_mask": ("MASK", {
                    "tooltip": "Binary mask for object initialization"
                }),
                "tracking_direction": (["forward", "backward", "bidirectional"], {
                    "default": "forward",
                    "tooltip": "Tracking direction from annotation frame"
                }),
                "annotation_frame_idx": ("INT", {
                    "default": 0,
                    "min": 0,
                    "tooltip": "Frame where initial prompt is applied"
                }),
                "object_id": ("INT", {
                    "default": 1,
                    "min": 1,
                    "tooltip": "Unique ID for multi-object tracking"
                }),
                "max_frames_to_track": ("INT", {
                    "default": -1,
                    "min": -1,
                    "tooltip": "Advanced: Max frames to process (-1 for all)"
                }),
                "mllm_memory_size": ("INT", {
                    "default": 12,
                    "min": 1,
                    "max": 20,
                    "tooltip": "Number of keyframes for semantic understanding (no VRAM impact). Original paper used 7, we default to 12 for balance."
                }),
                "offload_video_to_cpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Offload video frames to CPU (saves significant GPU memory, ~3% slower)"
                }),
                "auto_unload_model": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Automatically unload model from memory after segmentation to free GPU and RAM. Model will auto-reload if needed for subsequent runs."
                })
            }
        }
    
    RETURN_TYPES = ("MASK", "INT")
    RETURN_NAMES = ("masks", "object_ids") 
    FUNCTION = "segment_video"
    CATEGORY = "SeC"
    TITLE = "SeC Video Segmentation"
    DESCRIPTION = ("Concept-driven video object segmentation using Large Vision-Language Models for visual concept extraction. "
                   "Provide visual prompts (points/bbox/mask) and SeC automatically understands the object concept for robust tracking.")
    
    def parse_points(self, points_str, image_shape=None):
        """Parse point coordinates from JSON string and validate bounds.

        Returns:
            tuple: (points_array, labels_array, validation_errors) where validation_errors
                   is a list of error messages, or (None, None, errors) if all points invalid
        """
        import json

        if not points_str or not points_str.strip():
            return None, None, []

        try:
            points_list = json.loads(points_str)

            if not isinstance(points_list, list):
                raise ValueError(f"Points must be a JSON array, got {type(points_list).__name__}")

            if len(points_list) == 0:
                return None, None, []

            points = []
            validation_errors = []

            for i, point_dict in enumerate(points_list):
                if not isinstance(point_dict, dict):
                    err = f"Point {i} is not a dictionary"
                    print(f"Warning: {err}, skipping")
                    validation_errors.append(err)
                    continue

                if 'x' not in point_dict or 'y' not in point_dict:
                    err = f"Point {i} missing 'x' or 'y' key"
                    print(f"Warning: {err}, skipping")
                    validation_errors.append(err)
                    continue

                try:
                    x = float(point_dict['x'])
                    y = float(point_dict['y'])

                    # Validate coordinates are non-negative
                    if x < 0 or y < 0:
                        err = f"Point {i} has negative coordinates ({x}, {y})"
                        print(f"Warning: {err}, skipping")
                        validation_errors.append(err)
                        continue

                    # Validate within image bounds if provided
                    if image_shape is not None:
                        height, width = image_shape[1], image_shape[2]  # [batch, height, width, channels]
                        if x >= width or y >= height:
                            err = f"Point {i} ({x}, {y}) outside image bounds ({width}x{height})"
                            print(f"Warning: {err}, skipping")
                            validation_errors.append(err)
                            continue

                    points.append([x, y])

                except (ValueError, TypeError) as e:
                    err = f"Could not convert point {i} coordinates to float: {e}"
                    print(f"Warning: {err}, skipping")
                    validation_errors.append(err)
                    continue

            if not points:
                return None, None, validation_errors

            return np.array(points, dtype=np.float32), np.ones(len(points), dtype=np.int32), validation_errors

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in points: {str(e)}")
        except Exception as e:
            print(f"Error parsing points: {e}")
            return None, None, [str(e)]
    
    def parse_bbox(self, bbox):
        """Parse bounding box from BBOX type (tuple/list/dict) and validate

        Supports multiple formats:
        - KJNodes: [{'startX': x, 'startY': y, 'endX': x2, 'endY': y2}]
        - Tuple/list: (x1, y1, x2, y2) or (x, y, width, height)
        - Dict: {'startX': x, 'startY': y, 'endX': x2, 'endY': y2}
        """
        if bbox is None:
            return None

        try:
            coords = None

            # Try to extract coordinates regardless of type checks
            # This handles cases where ComfyUI wraps data in unexpected ways
            if hasattr(bbox, '__iter__') and not isinstance(bbox, (str, bytes)):
                # It's some kind of sequence
                try:
                    bbox_list = list(bbox)

                    if len(bbox_list) == 0:
                        return None

                    first_elem = bbox_list[0]

                    # Try to access as dict-like (KJNodes format)
                    try:
                        # Use getitem access instead of isinstance check
                        if hasattr(first_elem, '__getitem__'):
                            try:
                                x1 = float(first_elem['startX'])
                                y1 = float(first_elem['startY'])
                                x2 = float(first_elem['endX'])
                                y2 = float(first_elem['endY'])
                                coords = [x1, y1, x2, y2]
                            except (KeyError, TypeError):
                                # Not dict format, might be numeric sequence
                                pass

                        # If still no coords, try as numeric sequence
                        if coords is None:
                            if len(bbox_list) == 4:
                                # Direct 4-element tuple/list: (x1, y1, x2, y2)
                                coords = [float(x) for x in bbox_list]
                            elif hasattr(first_elem, '__iter__') and not isinstance(first_elem, (str, bytes)):
                                # Nested sequence: [[x1, y1, x2, y2]]
                                inner = list(first_elem)
                                if len(inner) == 4:
                                    coords = [float(x) for x in inner]

                    except Exception as e:
                        raise ValueError(f"Could not extract coordinates from sequence: {e}")

                except Exception as e:
                    raise ValueError(f"Failed to process bbox as sequence: {e}")

            # Try single dict format
            elif hasattr(bbox, '__getitem__'):
                try:
                    x1 = float(bbox['startX'])
                    y1 = float(bbox['startY'])
                    x2 = float(bbox['endX'])
                    y2 = float(bbox['endY'])
                    coords = [x1, y1, x2, y2]
                except (KeyError, TypeError) as e:
                    raise ValueError(f"Dictionary bbox missing required keys: {e}")

            else:
                raise ValueError(f"Unsupported bbox type: {type(bbox)}")

            if coords is None:
                raise ValueError(f"Could not extract coordinates from bbox. Type: {type(bbox)}, Content: {repr(bbox)[:200]}")

            # Handle xywh format (convert to xyxy)
            x1, y1, x2, y2 = coords
            if x2 < x1 or y2 < y1:
                # Assume xywh format: (x, y, width, height)
                width, height = x2, y2
                x2 = x1 + width
                y2 = y1 + height
                coords = [x1, y1, x2, y2]

            # Validate coordinates
            if coords[0] >= coords[2]:
                raise ValueError(f"Invalid bbox: x1 ({coords[0]}) must be < x2 ({coords[2]})")
            if coords[1] >= coords[3]:
                raise ValueError(f"Invalid bbox: y1 ({coords[1]}) must be < y2 ({coords[3]})")
            if coords[0] < 0 or coords[1] < 0:
                raise ValueError(f"Bounding box coordinates must be non-negative, got x1={coords[0]}, y1={coords[1]}")

            return np.array(coords, dtype=np.float32)

        except (ValueError, TypeError) as e:
            error_msg = f"Invalid bbox: {str(e)}\n"
            error_msg += f"Input type: {type(bbox)}\n"
            error_msg += f"Input content: {repr(bbox)[:500]}"
            raise ValueError(error_msg)
    
    def tensor_to_pil_images(self, tensor):
        """Convert tensor to list of PIL images"""
        images = []
        for i in range(tensor.shape[0]):
            img_array = (tensor[i] * 255).clamp(0, 255).byte().cpu().numpy()
            pil_img = Image.fromarray(img_array, mode='RGB')
            images.append(pil_img)
        return images

    def pil_images_to_tensor(self, pil_images):
        """Convert list of PIL images to tensor"""
        if not pil_images:
            return torch.empty(0)

        arrays = []
        for img in pil_images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            arr = np.array(img, dtype=np.float32) / 255.0
            arrays.append(arr)

        tensor = torch.from_numpy(np.stack(arrays))
        return tensor

    def mask_to_tensor(self, mask_array):
        """Convert numpy mask to ComfyUI MASK tensor (2D grayscale)"""
        if mask_array.ndim > 2:
            mask_array = mask_array[..., 0] if mask_array.shape[-1] <= 4 else mask_array[0]

        mask_tensor = torch.from_numpy(mask_array.astype(np.float32))
        return mask_tensor
    
    def save_frames_temporarily(self, pil_images, temp_dir=None):
        """Save frames temporarily for video processing"""
        import tempfile

        # Use system temp directory for cross-platform compatibility
        if temp_dir is None:
            temp_base = tempfile.gettempdir()
            temp_dir = os.path.join(temp_base, "sec_frames")

        os.makedirs(temp_dir, exist_ok=True)

        # Clean up old files safely
        try:
            for file in os.listdir(temp_dir):
                if file.endswith(('.jpg', '.png')):
                    try:
                        os.remove(os.path.join(temp_dir, file))
                    except (PermissionError, OSError) as e:
                        print(f"Warning: Could not remove old frame file {file}: {e}")
        except Exception as e:
            print(f"Warning: Error during temp directory cleanup: {e}")

        frame_paths = []
        for i, img in enumerate(pil_images):
            frame_path = os.path.join(temp_dir, f"{i:05d}.jpg")
            img.save(frame_path, 'JPEG', quality=95)
            frame_paths.append(frame_path)

        return temp_dir, frame_paths

    def _cleanup_model_memory(self, model):
        """
        Completely unload model from memory while preserving loading metadata for auto-reload.
        Frees all GPU and RAM memory by deleting model components entirely.
        """
        try:
            original_device = None
            if hasattr(model, 'device'):
                original_device = model.device

            # Preserve loading metadata before cleanup
            loading_metadata = getattr(model, '_sec_loading_metadata', None)

            # Clear SAM2 inference states first
            if hasattr(model, 'grounding_encoder'):
                try:
                    if hasattr(model.grounding_encoder, '_states'):
                        model.grounding_encoder._states.clear()
                except:
                    pass

            # Step 2: Delete all model components to free memory
            components_deleted = []

            # Main components that take the most memory
            main_components = ['vision_encoder', 'language_model', 'grounding_encoder', 'tokenizer']
            for component in main_components:
                if hasattr(model, component):
                    try:
                        # Move to CPU first if it's a model component
                        comp = getattr(model, component)
                        if hasattr(comp, 'cpu'):
                            comp.cpu()
                        delattr(model, component)
                        components_deleted.append(component)
                    except Exception as e:
                        print(f"  Warning: Could not delete {component}: {e}")

            # Step 3: Delete any other heavy attributes
            other_attrs = ['vision_config', 'llm_config', 'llm', 'vision_model', 'config']
            for attr_name in other_attrs:
                if hasattr(model, attr_name) and not attr_name.startswith('_sec_'):
                    try:
                        delattr(model, attr_name)
                        components_deleted.append(attr_name)
                    except Exception as e:
                        print(f"  Warning: Could not delete {attr_name}: {e}")

            # Step 4: Force cleanup of any remaining model parameters
            try:
                # Clear any remaining parameters that might exist
                for name, param in list(model.named_parameters()):
                    if param.is_cuda:
                        param.data = torch.empty(0, device='cpu')
            except:
                pass

            # Step 4: Clear PyTorch caches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            import gc
            gc.collect()

            # Step 5: Restore metadata and mark as unloaded
            if loading_metadata:
                model._sec_loading_metadata = loading_metadata
            model._sec_unloaded = True

            # if original_device and str(original_device).startswith('cuda'):
            #     print(f"Model completely unloaded from {original_device} (GPU and RAM freed)")
            # else:
            #     print(f"Model completely unloaded from memory (RAM freed)")

            # if components_deleted:
            #     print(f"Deleted components: {', '.join(components_deleted)}")

        except Exception as e:
            print(f"Warning: Model cleanup encountered an issue: {e}")
            # Don't raise - cleanup failures shouldn't break the workflow

    def _reload_model(self, model):
        """
        Reload model components using stored metadata.
        Returns True if successful, False if metadata is missing.
        """
        try:
            if not hasattr(model, '_sec_loading_metadata'):
                return False

            metadata = model._sec_loading_metadata

            # Extract metadata
            torch_dtype = metadata['torch_dtype']
            device = metadata['device']
            use_flash_attn = metadata['use_flash_attn']
            allow_mask_overlap = metadata['allow_mask_overlap']
            model_path = metadata['model_path']
            config_path = metadata.get('config_path', model_path)  # Fallback for old metadata
            is_single_file = metadata.get('is_single_file', False)  # Fallback for old metadata
            hydra_overrides_extra = metadata['hydra_overrides_extra']

            # Recreate config fresh to avoid any stored state issues
            # Use config_path which points to repo config for single files
            from .inference.configuration_sec import SeCConfig
            config = SeCConfig.from_pretrained(config_path)
            config.hydra_overrides_extra = hydra_overrides_extra

            # Prepare for loading
            from .inference.modeling_sec import SeCModel
            from transformers import AutoTokenizer

            if device.startswith("cuda:"):
                import gc
                gc.collect()
                torch.cuda.empty_cache()

            # Detect precision from model path to handle FP8 conversion
            precision_str = "fp16"  # default
            if "fp8" in model_path.lower():
                precision_str = "fp8"
            elif "fp32" in model_path.lower():
                precision_str = "fp32"
            elif "bf16" in model_path.lower():
                precision_str = "bf16"

            # Load model based on format
            if is_single_file:
                # Manual instantiation for single-file models
                fresh_model = SeCModel(config, use_flash_attn=use_flash_attn)
                state_dict = load_file(model_path)

                # Convert FP8 weights to FP16 if needed (same as initial load)
                if precision_str == "fp8":
                    converted_state_dict = {}
                    for key, tensor in state_dict.items():
                        if tensor.dtype == torch.float8_e4m3fn:
                            converted_state_dict[key] = tensor.to(torch.float16)
                        else:
                            converted_state_dict[key] = tensor
                    state_dict = converted_state_dict

                fresh_model.load_state_dict(state_dict, strict=True)
                fresh_model = fresh_model.eval()

                # Move to device and convert dtype
                if device.startswith("cuda:"):
                    fresh_model = fresh_model.to(device=device, dtype=torch_dtype)
                else:
                    fresh_model = fresh_model.to(device="cpu", dtype=torch_dtype)

            else:
                # Directory-based loading for sharded models
                load_kwargs = {
                    "config": config,
                    "torch_dtype": torch_dtype,
                    "use_flash_attn": use_flash_attn,
                }

                if device.startswith("cuda:"):
                    load_kwargs["device_map"] = {"": device}
                    load_kwargs["low_cpu_mem_usage"] = True
                else:
                    load_kwargs["low_cpu_mem_usage"] = True

                fresh_model = SeCModel.from_pretrained(model_path, **load_kwargs).eval()

            # Convert to target dtype
            if device.startswith("cuda") and torch_dtype != torch.float32:
                fresh_model = fresh_model.to(dtype=torch_dtype)
            elif device == "cpu":
                fresh_model = fresh_model.to(dtype=torch_dtype)

            # Set up tokenizer (use config_path for single files)
            tokenizer = AutoTokenizer.from_pretrained(config_path, trust_remote_code=True)
            fresh_model.preparing_for_generation(tokenizer=tokenizer, torch_dtype=torch_dtype)

            # Reinstall dtype conversion hooks if needed
            if device.startswith("cuda") and torch_dtype != torch.float32:
                def dtype_conversion_hook(module, args, kwargs):
                    try:
                        module_dtype = None
                        for param in module.parameters():
                            module_dtype = param.dtype
                            break

                        if module_dtype is None:
                            return args, kwargs

                        if isinstance(module, torch.nn.Embedding):
                            return args, kwargs

                        new_args = []
                        for arg in args:
                            if isinstance(arg, torch.Tensor):
                                if arg.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                                    new_args.append(arg)
                                elif arg.dtype != module_dtype:
                                    new_args.append(arg.to(dtype=module_dtype))
                                else:
                                    new_args.append(arg)
                            else:
                                new_args.append(arg)

                        new_kwargs = {}
                        for k, v in kwargs.items():
                            if isinstance(v, torch.Tensor):
                                if v.dtype in [torch.long, torch.int, torch.int32, torch.int64]:
                                    new_kwargs[k] = v
                                elif v.dtype != module_dtype:
                                    new_kwargs[k] = v.to(dtype=module_dtype)
                                else:
                                    new_kwargs[k] = v
                            else:
                                new_kwargs[k] = v

                        return tuple(new_args), new_kwargs
                    except Exception:
                        return args, kwargs

                for module in fresh_model.modules():
                    if len(list(module.parameters(recurse=False))) > 0:
                        module.register_forward_pre_hook(dtype_conversion_hook, with_kwargs=True)

            # Apply FP8 weight-only quantization if applicable (after model is on device)
            if device.startswith("cuda:"):
                apply_fp8_weight_only_quantization(fresh_model, precision_str)

            # Copy all attributes from fresh model to original model
            for attr_name in dir(fresh_model):
                if not attr_name.startswith('_sec_') and not attr_name.startswith('__'):
                    try:
                        setattr(model, attr_name, getattr(fresh_model, attr_name))
                    except:
                        pass

            # Restore metadata and clear unloaded flag
            model._sec_loading_metadata = metadata
            model._sec_unloaded = False

            # Clean up the temporary fresh_model to free its memory
            del fresh_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            import gc
            gc.collect()

            return True

        except Exception as e:
            print(f"Failed to reload model: {e}")
            return False

    def segment_video(self, model, frames, positive_points="", negative_points="",
                     bbox=None, input_mask=None, tracking_direction="forward",
                     annotation_frame_idx=0, object_id=1, max_frames_to_track=-1, mllm_memory_size=12,
                     offload_video_to_cpu=False, auto_unload_model=True):
        """Perform video object segmentation"""

        # === Model State Validation and Auto-Reload ===
        # Check if model has been unloaded and reload if necessary
        if hasattr(model, '_sec_unloaded') and model._sec_unloaded:
            print("Detected unloaded model, attempting auto-reload...")
            if not self._reload_model(model):
                raise RuntimeError(
                    "Model has been unloaded and auto-reload failed. "
                    "This may happen if the model was loaded with an older version. "
                    "Please use a fresh Model Loader node."
                )

        # === Input Validation ===
        # Validate frames tensor
        if frames is None or frames.numel() == 0:
            raise ValueError("Frames tensor is empty. Please provide at least one frame.")

        if frames.ndim != 4:
            raise ValueError(f"Frames tensor must be 4D [batch, height, width, channels], got shape {frames.shape}")

        num_frames = frames.shape[0]
        if num_frames == 0:
            raise ValueError("No frames provided. Frames tensor has 0 frames.")

        # Validate annotation_frame_idx bounds
        if annotation_frame_idx < 0:
            raise ValueError(f"annotation_frame_idx must be >= 0, got {annotation_frame_idx}")

        if annotation_frame_idx >= num_frames:
            raise ValueError(
                f"annotation_frame_idx ({annotation_frame_idx}) is out of bounds. "
                f"Video has {num_frames} frame(s), valid range is 0-{num_frames-1}"
            )

        # Validate at least one input provided
        has_input = (
            (positive_points and positive_points.strip()) or
            (negative_points and negative_points.strip()) or
            (bbox is not None) or
            (input_mask is not None)
        )
        if not has_input:
            raise ValueError(
                "At least one visual prompt must be provided: "
                "positive_points, negative_points, bbox, or input_mask"
            )

        video_dir = None  # Track for cleanup
        try:
            pil_images = self.tensor_to_pil_images(frames)
            video_dir, frame_paths = self.save_frames_temporarily(pil_images)

            # Automatically set offload_state_to_cpu based on model device
            try:
                offload_state_to_cpu = str(model.device) == "cpu"
            except AttributeError:
                # Fallback if model doesn't have device attribute
                offload_state_to_cpu = False

            inference_state = model.grounding_encoder.init_state(
                video_path=video_dir,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu
            )
            model.grounding_encoder.reset_state(inference_state)

            # Parse inputs with bounds checking
            pos_points, pos_labels, pos_errors = self.parse_points(positive_points, frames.shape)
            neg_points, neg_labels, neg_errors = self.parse_points(negative_points, frames.shape)
            bbox_coords = self.parse_bbox(bbox)

            # Collect validation errors for better error messages
            all_validation_errors = []
            if pos_errors:
                all_validation_errors.extend([f"Positive {err}" for err in pos_errors])
            if neg_errors:
                all_validation_errors.extend([f"Negative {err}" for err in neg_errors])

            init_mask = None

            # Step 1: Add mask if provided (establishes initial region)
            if input_mask is not None:
                # Handle both [H, W] and [B, H, W] mask formats
                if input_mask.dim() == 2:
                    mask_array = input_mask.cpu().numpy()
                elif input_mask.dim() == 3:
                    mask_array = input_mask[0].cpu().numpy()
                else:
                    raise ValueError(f"Unexpected mask dimensions: {input_mask.dim()}. Expected 2D [H,W] or 3D [B,H,W]")

                init_mask = (mask_array > 0.5).astype(np.bool_)

                _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_mask(
                    inference_state=inference_state,
                    frame_idx=annotation_frame_idx,
                    obj_id=object_id,
                    mask=init_mask,
                )

            # Step 2: Filter positive points if mask was provided
            # Only keep positive points that fall inside the mask boundary
            if init_mask is not None and pos_points is not None:
                filtered_pos_points = []
                filtered_pos_labels = []
                for i, point in enumerate(pos_points):
                    x, y = int(point[0]), int(point[1])
                    # Check if point is within mask bounds and inside the mask
                    if 0 <= y < init_mask.shape[0] and 0 <= x < init_mask.shape[1]:
                        if init_mask[y, x]:  # Point is inside mask
                            filtered_pos_points.append(point)
                            filtered_pos_labels.append(pos_labels[i])

                # Replace pos_points with filtered version
                if filtered_pos_points:
                    pos_points = np.array(filtered_pos_points)
                    pos_labels = np.array(filtered_pos_labels, dtype=np.int32)
                else:
                    # No positive points inside mask - clear them
                    pos_points = None
                    pos_labels = None

            # Step 2b: Warn about negative points when mask is provided
            # Negative points should ideally be inside or near the mask to refine segmentation
            if init_mask is not None and neg_points is not None:
                # Find pixels in the mask
                mask_pixels = np.argwhere(init_mask)
                if len(mask_pixels) > 0:
                    points_outside = []
                    for i, point in enumerate(neg_points):
                        x, y = int(point[0]), int(point[1])
                        # Calculate minimum distance to any mask pixel
                        distances = np.sqrt(((mask_pixels[:, 0] - y) ** 2) + ((mask_pixels[:, 1] - x) ** 2))
                        min_dist = distances.min()

                        # If point is >50 pixels away from mask, warn
                        if min_dist > 50:
                            points_outside.append((i, min_dist))

                    if points_outside:
                        print(f"  Warning: {len(points_outside)} negative point(s) are far from the mask region.")
                        print(f"  Negative points work best inside or near the masked object to refine segmentation.")
                        print(f"  Points far outside the mask may cause unexpected results or empty segmentation.")

            # Step 3: Combine points for refinement
            points = None
            labels = None
            if pos_points is not None and neg_points is not None:
                points = np.concatenate([pos_points, neg_points], axis=0)
                labels = np.concatenate([pos_labels, np.zeros(len(neg_points), dtype=np.int32)], axis=0)
            elif pos_points is not None:
                points = pos_points
                labels = pos_labels
            elif neg_points is not None:
                points = neg_points
                labels = np.zeros(len(neg_points), dtype=np.int32)

            # Step 4: Add points/bbox to refine the segmentation
            if points is not None or bbox_coords is not None:
                _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=annotation_frame_idx,
                    obj_id=object_id,
                    points=points,
                    labels=labels if points is not None else None,
                    box=bbox_coords,
                )
                init_mask = (out_mask_logits[0] > 0.0).cpu().numpy()

            # Ensure at least one input was provided
            if init_mask is None:
                error_msg = "At least one visual prompt (points, bbox, or mask) must be provided."
                if all_validation_errors:
                    error_msg += f" Point validation failures: {'; '.join(all_validation_errors)}"
                raise ValueError(error_msg)

            if max_frames_to_track == -1:
                max_frames_to_track = len(pil_images)

            video_segments = {}
            
            if tracking_direction == "bidirectional":
                for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(
                    inference_state,
                    start_frame_idx=annotation_frame_idx,
                    max_frame_num_to_track=max_frames_to_track,
                    reverse=False,
                    init_mask=init_mask,
                    tokenizer=None,
                    mllm_memory_size=mllm_memory_size,
                ):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }

                model.grounding_encoder.reset_state(inference_state)

                if points is not None or bbox_coords is not None:
                    _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=annotation_frame_idx,
                        obj_id=object_id,
                        points=points,
                        labels=labels if points is not None else None,
                        box=bbox_coords,
                    )
                elif input_mask is not None:
                    _, out_obj_ids, out_mask_logits = model.grounding_encoder.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=annotation_frame_idx,
                        obj_id=object_id,
                        mask=init_mask,
                    )

                for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(
                    inference_state,
                    start_frame_idx=annotation_frame_idx,
                    max_frame_num_to_track=max_frames_to_track,
                    reverse=True,
                    init_mask=init_mask,
                    tokenizer=None,
                    mllm_memory_size=mllm_memory_size,
                ):
                    if out_frame_idx not in video_segments:
                        video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
            else:
                reverse = (tracking_direction == "backward")
                for out_frame_idx, out_obj_ids, out_mask_logits in model.propagate_in_video(
                    inference_state,
                    start_frame_idx=annotation_frame_idx,
                    max_frame_num_to_track=max_frames_to_track,
                    reverse=reverse,
                    init_mask=init_mask,
                    tokenizer=None,
                    mllm_memory_size=mllm_memory_size,
                ):
                    video_segments[out_frame_idx] = {
                        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                        for i, out_obj_id in enumerate(out_obj_ids)
                    }
            
            # Create output masks for all input frames
            # Frames not in video_segments get empty masks
            num_frames = len(pil_images)
            output_masks = []
            output_obj_ids = []

            for frame_idx in range(num_frames):
                if frame_idx in video_segments:
                    # Frame was tracked - use real mask
                    for obj_id, mask in video_segments[frame_idx].items():
                        mask_tensor = self.mask_to_tensor(mask)
                        output_masks.append(mask_tensor)
                        output_obj_ids.append(obj_id)
                else:
                    # Frame not tracked - use empty mask
                    empty_mask = torch.zeros(frames.shape[1], frames.shape[2])
                    output_masks.append(empty_mask)
                    output_obj_ids.append(0)

            masks_tensor = torch.stack(output_masks)
            obj_ids_tensor = torch.tensor(output_obj_ids, dtype=torch.int32)

            return (masks_tensor, obj_ids_tensor)

        except Exception as e:
            raise RuntimeError(f"SeC video segmentation failed: {str(e)}")

        finally:
            # Cleanup: Always remove temp directory and clear cache
            import shutil
            import gc

            if video_dir is not None and os.path.exists(video_dir):
                try:
                    shutil.rmtree(video_dir)
                except Exception as e:
                    print(f"Warning: Failed to clean up temp directory {video_dir}: {e}")

            # Model cleanup - move to CPU if requested
            if auto_unload_model:
                self._cleanup_model_memory(model)
            else:
                # Still clear GPU cache even if keeping model loaded
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            gc.collect()


class CoordinatePlotter:
    """
    ComfyUI node for visualizing coordinates on images
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "coordinates": ("STRING", {
                    "default": '[{"x": 100, "y": 100}]',
                    "tooltip": "JSON coordinates to plot: '[{\"x\": 100, \"y\": 200}]'"
                })
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Optional image to plot on. If provided, overrides width/height."
                }),
                "point_shape": (["circle", "square", "triangle"], {
                    "default": "circle",
                    "tooltip": "Shape to draw for each coordinate point"
                }),
                "point_size": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Size of points in pixels"
                }),
                "point_color": ("STRING", {
                    "default": "#00FF00",
                    "tooltip": "Point color as hex '#FF0000' or RGB '255,0,0'"
                }),
                "width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "tooltip": "Canvas width (ignored if image provided)"
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "tooltip": "Canvas height (ignored if image provided)"
                })
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "plot_coordinates"
    CATEGORY = "SeC"
    TITLE = "Coordinate Plotter"
    DESCRIPTION = "Visualize coordinate points on an image or blank canvas. Useful for previewing point selections."

    def parse_color(self, color_str):
        """Parse hex or RGB color string to RGB tuple for PIL"""
        import re

        color_str = color_str.strip()

        if color_str.startswith('#'):
            color_str = color_str[1:]

        if re.match(r'^[0-9A-Fa-f]{6}$', color_str):
            r = int(color_str[0:2], 16)
            g = int(color_str[2:4], 16)
            b = int(color_str[4:6], 16)
            return (r, g, b)

        if ',' in color_str:
            parts = [int(x.strip()) for x in color_str.split(',')]
            if len(parts) == 3:
                r, g, b = parts
                return (r, g, b)

        return (0, 255, 0)

    def draw_shape(self, draw, x, y, shape, size, color):
        """Draw a shape at the specified coordinates using PIL"""
        x, y = int(x), int(y)

        if shape == "circle":
            # Draw filled circle with white outline
            draw.ellipse([x - size, y - size, x + size, y + size], fill=color, outline=(255, 255, 255), width=2)

        elif shape == "square":
            half_size = size
            # Draw filled square with white outline
            draw.rectangle([x - half_size, y - half_size, x + half_size, y + half_size],
                          fill=color, outline=(255, 255, 255), width=2)

        elif shape == "triangle":
            height = int(size * 1.732)
            half_base = size

            pts = [
                (x, y - height),
                (x - half_base, y + size),
                (x + half_base, y + size)
            ]

            # Draw filled triangle with white outline
            draw.polygon(pts, fill=color, outline=(255, 255, 255))

    def plot_coordinates(self, coordinates, image=None, point_shape="circle",
                        point_size=10, point_color="#00FF00", width=512, height=512):
        """Plot coordinates on image or blank canvas"""
        import json
        from PIL import ImageDraw

        try:
            if not coordinates or not coordinates.strip():
                coords_list = []
            else:
                coords_list = json.loads(coordinates)
                if not isinstance(coords_list, list):
                    raise ValueError("Coordinates must be a JSON array")

            # Create PIL Image canvas
            if image is not None:
                # Convert ComfyUI tensor to PIL Image
                img_array = (image[0].cpu().numpy() * 255).astype(np.uint8)
                canvas = Image.fromarray(img_array, mode='RGB')
            else:
                # Create blank canvas
                canvas = Image.new('RGB', (width, height), color=(0, 0, 0))

            # Create drawing context
            draw = ImageDraw.Draw(canvas)
            color = self.parse_color(point_color)

            # Draw each coordinate point
            for coord in coords_list:
                if isinstance(coord, dict) and 'x' in coord and 'y' in coord:
                    x = float(coord['x'])
                    y = float(coord['y'])
                    self.draw_shape(draw, x, y, point_shape, point_size, color)

            # Convert PIL Image back to ComfyUI tensor
            canvas_array = np.array(canvas, dtype=np.float32) / 255.0
            output = torch.from_numpy(canvas_array).unsqueeze(0)

            return (output,)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON coordinates: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Coordinate plotting failed: {str(e)}")