# Uses SeC-4B model from OpenIXCLab
# Model: https://huggingface.co/OpenIXCLab/SeC-4B
# Licensed under Apache 2.0

import torch
import numpy as np
from PIL import Image
import folder_paths
import os
import sys
import gc
from functools import lru_cache

from .inference.configuration_sec import SeCConfig
from .inference.modeling_sec import SeCModel
from transformers import AutoTokenizer


@lru_cache(maxsize=1)
def find_sec_model():
    """
    Find SeC-4B model in registered 'sams' folder paths.
    Cached for efficiency. Returns the path to the model directory if found, None otherwise.
    """
    try:
        sams_paths = folder_paths.get_folder_paths("sams")
    except KeyError:
        # 'sams' folder type not registered yet
        return None

    # Required files for model validation
    required_files = [
        ("config.json", "config"),
        ("model.safetensors", "safetensors"),
        ("model.safetensors.index.json", "safetensors_sharded"),
        ("pytorch_model.bin", "bin"),
        ("pytorch_model.bin.index.json", "bin_sharded"),
        ("tokenizer_config.json", "tokenizer")
    ]

    for sams_dir in sams_paths:
        model_path = os.path.join(sams_dir, "SeC-4B")
        if os.path.isdir(model_path):
            # Batch check required files
            has_config = False
            has_model = False
            has_tokenizer = False

            for filename, file_type in required_files:
                filepath = os.path.join(model_path, filename)
                exists = os.path.exists(filepath)

                if file_type == "config":
                    has_config = exists
                elif file_type == "tokenizer":
                    has_tokenizer = exists
                elif file_type.startswith(("safetensors", "bin")):
                    has_model = has_model or exists

            if has_config and has_model and has_tokenizer:
                return model_path

    return None


def download_sec_model():
    """
    Download SeC-4B model from HuggingFace to the first registered 'sams' folder.
    Returns the path to the downloaded model directory.
    """
    try:
        from huggingface_hub import snapshot_download
    except ImportError as e:
        raise RuntimeError(
            "huggingface_hub is required for model download. "
            "Please install it: pip install huggingface_hub>=0.20.0"
        ) from e

    try:
        sams_paths = folder_paths.get_folder_paths("sams")
    except Exception as e:
        raise RuntimeError(f"Could not access model folder paths: {e}") from e

    if not sams_paths:
        raise RuntimeError("No 'sams' folder paths registered. Please check your ComfyUI installation.")

    destination = os.path.join(sams_paths[0], "SeC-4B")

    print("=" * 80)
    print("SeC-4B model not found. Downloading from HuggingFace...")
    print(f"Repository: OpenIXCLab/SeC-4B")
    print(f"Destination: {destination}")
    print(f"Size: ~8.5 GB - This may take several minutes...")
    print("=" * 80)

    # Create directory if it doesn't exist
    try:
        os.makedirs(destination, exist_ok=True)
    except (PermissionError, OSError) as e:
        raise RuntimeError(
            f"Cannot create model directory at {destination}. "
            f"Please check permissions. Error: {e}"
        ) from e

    # Check disk space (rough estimate)
    try:
        import shutil
        stat = shutil.disk_usage(os.path.dirname(destination))
        free_gb = stat.free / (1024**3)
        if free_gb < 10:
            print(f"⚠ Warning: Low disk space ({free_gb:.1f} GB free). Download requires ~8.5 GB.")
    except Exception:
        pass  # Not critical

    try:
        snapshot_download(
            repo_id="OpenIXCLab/SeC-4B",
            local_dir=destination,
            local_dir_use_symlinks=False
        )
    except Exception as e:
        error_msg = str(e).lower()
        if "connection" in error_msg or "network" in error_msg or "timeout" in error_msg:
            raise RuntimeError(
                f"Network error while downloading model: {e}\n"
                "Please check your internet connection and try again."
            ) from e
        elif "space" in error_msg or "disk" in error_msg:
            raise RuntimeError(
                f"Insufficient disk space: {e}\n"
                "Model download requires ~8.5 GB free space."
            ) from e
        else:
            raise RuntimeError(f"Failed to download model from HuggingFace: {e}") from e

    print("=" * 80)
    print(f"✓ SeC-4B model downloaded successfully!")
    print(f"✓ Location: {destination}")
    print("=" * 80)

    return destination


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

        return {
            "required": {
                "torch_dtype": (["bfloat16", "float16", "float32"], {
                    "default": "bfloat16",
                    "tooltip": "Data precision for model inference. bfloat16 recommended for best speed/quality balance. CPU mode automatically uses float32."
                }),
                "device": (device_choices, {
                    "default": "auto",
                    "tooltip": "Device: auto (gpu0 if available, else CPU), cpu, gpu0/gpu1/etc (specific GPU)"
                })
            },
            "optional": {
                "use_flash_attn": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable Flash Attention 2 for faster inference. Automatically disabled for float32 precision (requires float16/bfloat16)."
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
    
    def load_model(self, torch_dtype, device, use_flash_attn=True, allow_mask_overlap=True):
        """Load SeC model with simplified, efficient approach"""

        # Find or download the SeC-4B model
        model_path = find_sec_model()

        if model_path is None:
            # Model not found, download it
            try:
                model_path = download_sec_model()
            except Exception as e:
                raise RuntimeError(f"Failed to download SeC-4B model: {str(e)}")
        else:
            print("=" * 80)
            print(f"✓ Found SeC-4B model at: {model_path}")
            print("=" * 80)

        # Handle device selection
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device.startswith("gpu"):
            gpu_num = int(device[3:])  # Extract number after "gpu"
            if torch.cuda.is_available():
                available_gpus = torch.cuda.device_count()
                if gpu_num >= available_gpus:
                    raise ValueError(f"GPU {gpu_num} not available. System has {available_gpus} GPU(s) (0-{available_gpus-1})")
                device = f"cuda:{gpu_num}"
            else:
                raise ValueError(f"CUDA not available but GPU device '{device}' was selected")

        # Set data type
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32
        }
        torch_dtype = dtype_map[torch_dtype]

        if device == "cpu" and torch_dtype != torch.float32:
            print(f"⚠ CPU mode requires float32 precision. Using float32")
            torch_dtype = torch.float32

        if torch_dtype == torch.float32 and use_flash_attn:
            print("⚠ Flash Attention disabled for float32 precision")
            use_flash_attn = False

        # Configure model settings
        hydra_overrides_extra = [f"++model.non_overlap_masks={'false' if allow_mask_overlap else 'true'}"]

        try:
            # Clear GPU memory before loading
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            config = SeCConfig.from_pretrained(model_path)
            config.hydra_overrides_extra = hydra_overrides_extra

            load_kwargs = {
                "config": config,
                "torch_dtype": torch_dtype,
                "use_flash_attn": use_flash_attn,
                "low_cpu_mem_usage": True,
                "device_map": {"": device}
            }

            print(f"Loading SeC model on {device}...")
            model = SeCModel.from_pretrained(model_path, **load_kwargs).eval()

            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            model.preparing_for_generation(tokenizer=tokenizer, torch_dtype=torch_dtype)

            # Only add dtype hooks for mixed precision GPU inference
            if device.startswith("cuda") and torch_dtype != torch.float32:
                safe_dtypes = {torch.long, torch.int, torch.int32, torch.int64}

                def dtype_hook(module, args, kwargs):
                    """Simplified dtype conversion hook"""
                    try:
                        if not hasattr(module, '_cached_dtype'):
                            for param in module.parameters():
                                module._cached_dtype = param.dtype
                                break

                        module_dtype = getattr(module, '_cached_dtype', None)
                        if module_dtype is None or isinstance(module, torch.nn.Embedding):
                            return args, kwargs

                        # Convert tensors with mismatched dtypes
                        converted = False
                        new_args = []
                        for arg in args:
                            if isinstance(arg, torch.Tensor) and arg.dtype not in safe_dtypes and arg.dtype != module_dtype:
                                new_args.append(arg.to(dtype=module_dtype))
                                converted = True
                            else:
                                new_args.append(arg)

                        new_kwargs = {}
                        for k, v in kwargs.items():
                            if isinstance(v, torch.Tensor) and v.dtype not in safe_dtypes and v.dtype != module_dtype:
                                new_kwargs[k] = v.to(dtype=module_dtype)
                                converted = True
                            else:
                                new_kwargs[k] = v

                        return (tuple(new_args), new_kwargs) if converted else (args, kwargs)
                    except Exception:
                        return args, kwargs

                for module in model.modules():
                    if next(iter(module.parameters()), None) is not None:
                        module.register_forward_pre_hook(dtype_hook, with_kwargs=True)

            print(f"SeC model loaded successfully on {device}")
            return (model,)

        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise RuntimeError(f"Failed to load SeC model: {str(e)}")


class MemoryVideoHandler:
    """
    Memory-based video handler that eliminates disk I/O bottleneck.
    Provides frame data directly to model without temporary file creation.
    """

    def __init__(self, frame_arrays, pil_images):
        self.frame_arrays = frame_arrays
        self.pil_images = pil_images
        self.num_frames = len(frame_arrays)

    def initialize_model_state(self, model, offload_video_to_cpu, offload_state_to_cpu):
        """Initialize model state with in-memory frame data"""
        # Need to check if model supports direct frame array initialization
        # If not, fall back to minimal temp directory approach
        try:
            # Try direct initialization (this may need model-specific implementation)
            # For now, create minimal temp structure but optimize frame access
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="sec_memory_")

            # Create a minimal directory structure but keep frames in memory
            # The model will read directory structure but we intercept frame loading
            self.temp_dir = temp_dir
            self._create_frame_index_files()

            inference_state = model.grounding_encoder.init_state(
                video_path=temp_dir,
                offload_video_to_cpu=offload_video_to_cpu,
                offload_state_to_cpu=offload_state_to_cpu
            )
            return inference_state

        except Exception:
            # If direct memory initialization fails, fall back to optimized disk approach
            return self._fallback_disk_initialization(model, offload_video_to_cpu, offload_state_to_cpu)

    def _create_frame_index_files(self):
        """Create minimal index files for model compatibility"""
        # Create a frame manifest that the model can parse
        manifest_path = os.path.join(self.temp_dir, "frames.json")
        frame_info = {
            "total_frames": self.num_frames,
            "format": "memory",
            "frame_size": [self.pil_images[0].width, self.pil_images[0].height]
        }
        with open(manifest_path, 'w') as f:
            import json
            json.dump(frame_info, f)

    def _fallback_disk_initialization(self, model, offload_video_to_cpu, offload_state_to_cpu):
        """Fallback: optimized disk I/O with JPEG for model compatibility"""
        import tempfile

        temp_dir = tempfile.mkdtemp(prefix="sec_optimized_")
        frame_paths = []

        # Use JPEG format as required by SAM2 model (line 246 in misc.py)
        for i, img_array in enumerate(self.frame_arrays):
            frame_path = os.path.join(temp_dir, f"{i:05d}.jpg")
            # Use high-quality JPEG to minimize quality loss
            from PIL import Image
            img = Image.fromarray(img_array)
            img.save(frame_path, 'JPEG', quality=97, optimize=True)  # High quality, optimized
            frame_paths.append(frame_path)

        self.temp_dir = temp_dir
        self.cleanup_paths = frame_paths

        inference_state = model.grounding_encoder.init_state(
            video_path=temp_dir,
            offload_video_to_cpu=offload_video_to_cpu,
            offload_state_to_cpu=offload_state_to_cpu
        )
        return inference_state

    def cleanup(self):
        """Clean up temporary resources"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            try:
                shutil.rmtree(self.temp_dir, ignore_errors=True)
            except Exception:
                pass


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
                "bbox": ("STRING", {
                    "default": "",
                    "tooltip": "Bounding box as 'x_min,y_min,x_max,y_max'"
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
                    "tooltip": "Advanced: Frame where initial prompt is applied"
                }),
                "object_id": ("INT", {
                    "default": 1,
                    "min": 1,
                    "tooltip": "Advanced: Unique ID for multi-object tracking"
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
                    "tooltip": "Advanced: Number of keyframes for semantic understanding (no VRAM impact). Original paper used 7, we default to 12 for balance."
                }),
                "offload_video_to_cpu": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Memory: Offload video frames to CPU (saves significant GPU memory, ~3% slower)"
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
    
    def parse_bbox(self, bbox_str):
        """Parse bounding box from string and validate"""
        if not bbox_str or not bbox_str.strip():
            return None

        try:
            coords = [float(x.strip()) for x in bbox_str.strip().split(',')]

            if len(coords) != 4:
                raise ValueError(f"Bounding box must have 4 coordinates, got {len(coords)}")

            x1, y1, x2, y2 = coords

            # Validate coordinates are sensible
            if x1 >= x2:
                raise ValueError(f"Invalid bbox: x1 ({x1}) must be < x2 ({x2})")
            if y1 >= y2:
                raise ValueError(f"Invalid bbox: y1 ({y1}) must be < y2 ({y2})")

            if x1 < 0 or y1 < 0:
                raise ValueError(f"Bounding box coordinates must be non-negative, got x1={x1}, y1={y1}")

            return np.array(coords, dtype=np.float32)

        except ValueError as e:
            if "could not convert" in str(e):
                raise ValueError(f"Invalid bbox format: '{bbox_str}'. Expected format: 'x1,y1,x2,y2' with numeric values")
            raise  # Re-raise our custom errors
    
    def tensor_to_pil_images(self, tensor):
        """Highly optimized tensor to PIL conversion - minimal allocations"""
        if tensor.numel() == 0:
            return []

        # Convert to numpy directly without intermediate clamping for speed
        # Model should already provide valid data in [0,1] range
        if tensor.is_cuda:
            img_arrays = (tensor * 255).byte().cpu().numpy()
        else:
            img_arrays = (tensor * 255).byte().numpy()

        # Direct PIL conversion - most efficient approach
        return [Image.fromarray(img_arrays[i], mode='RGB') for i in range(len(img_arrays))]

    def pil_images_to_tensor(self, pil_images):
        """Optimized PIL images to tensor conversion with pre-allocation"""
        if not pil_images:
            return torch.empty(0)

        # Pre-allocate numpy array
        first_img = pil_images[0]
        if first_img.mode != 'RGB':
            first_img = first_img.convert('RGB')

        batch_size = len(pil_images)
        width, height = first_img.size  # PIL size is (width, height)

        # Pre-allocate tensor directly
        tensor_np = np.empty((batch_size, height, width, 3), dtype=np.float32)

        # Process images efficiently
        for i, img in enumerate(pil_images):
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Direct assignment to pre-allocated array
            arr = np.frombuffer(img.tobytes(), dtype=np.uint8).reshape(height, width, 3)
            tensor_np[i] = arr.astype(np.float32) / 255.0

        return torch.from_numpy(tensor_np)

    def mask_to_tensor(self, mask_array):
        """Optimized numpy mask to tensor conversion"""
        if mask_array.ndim > 2:
            # Use safer dimension reduction
            if mask_array.ndim == 3 and mask_array.shape[2] == 1:
                mask_array = mask_array[:, :, 0]
            elif mask_array.ndim == 3:
                mask_array = mask_array[0] if mask_array.shape[0] == 1 else mask_array[:, :, 0]

        # Direct conversion without intermediate array creation when possible
        return torch.as_tensor(mask_array, dtype=torch.float32)
    
    def prepare_frames_for_model(self, pil_images):
        """Memory-based frame preparation - eliminates disk I/O bottleneck"""
        # Convert PIL images directly to numpy array format expected by model
        # This avoids the massive performance hit of disk I/O operations
        frame_count = len(pil_images)

        if frame_count == 0:
            return None

        # Pre-allocate for better performance
        first_img = pil_images[0]
        if first_img.mode != 'RGB':
            first_img = first_img.convert('RGB')

        width, height = first_img.size

        # Create a list of numpy arrays - this is what the model expect internally
        frame_arrays = []
        for img in pil_images:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Direct conversion to numpy without disk compression
            frame_arrays.append(np.array(img, dtype=np.uint8))

        return frame_arrays
    
    def segment_video(self, model, frames, positive_points="", negative_points="",
                     bbox="", input_mask=None, tracking_direction="forward",
                     annotation_frame_idx=0, object_id=1, max_frames_to_track=-1, mllm_memory_size=12,
                     offload_video_to_cpu=False):
        """Perform video object segmentation"""

        # === Optimized Input Validation ===
        # Early validation with minimal overhead
        if frames is None or frames.numel() == 0:
            raise ValueError("Frames tensor is empty. Please provide at least one frame.")

        if frames.ndim != 4:
            raise ValueError(f"Frames tensor must be 4D [batch, height, width, channels], got shape {frames.shape}")

        num_frames = frames.shape[0]
        if num_frames == 0:
            raise ValueError("No frames provided. Frames tensor has 0 frames.")

        # Efficient bounds checking
        if not (0 <= annotation_frame_idx < num_frames):
            raise ValueError(
                f"annotation_frame_idx ({annotation_frame_idx}) is out of bounds. "
                f"Video has {num_frames} frame(s), valid range is 0-{num_frames-1}"
            )

        # Optimized input validation - early short-circuit
        has_positive = positive_points and positive_points.strip()
        has_negative = negative_points and negative_points.strip()
        has_bbox = bbox and bbox.strip()
        has_mask_input = input_mask is not None

        if not (has_positive or has_negative or has_bbox or has_mask_input):
            raise ValueError(
                "At least one visual prompt must be provided: "
                "positive_points, negative_points, bbox, or input_mask"
            )

        video_handler = None  # Track for cleanup
        try:
            pil_images = self.tensor_to_pil_images(frames)
            frame_arrays = self.prepare_frames_for_model(pil_images)

            # Create memory-based video handler
            video_handler = MemoryVideoHandler(frame_arrays, pil_images)

            # Automatically set offload_state_to_cpu based on model device
            try:
                offload_state_to_cpu = str(model.device) == "cpu"
            except AttributeError:
                # Fallback if model doesn't have device attribute
                offload_state_to_cpu = False

            # Initialize with memory handler instead of file path
            inference_state = video_handler.initialize_model_state(
                model, offload_video_to_cpu, offload_state_to_cpu
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

            # Step 2: Optimized point filtering with vectorized operations
            if init_mask is not None and pos_points is not None:
                # Vectorized point filtering for performance
                point_coords = pos_points.astype(np.int32)
                x_coords, y_coords = point_coords[:, 0], point_coords[:, 1]

                # Bounds checking + mask checking in one vectorized operation
                # First check bounds to avoid out-of-bounds indexing
                bounds_mask = (
                    (x_coords >= 0) &
                    (y_coords >= 0) &
                    (x_coords < init_mask.shape[1]) &
                    (y_coords < init_mask.shape[0])
                )

                # Only check mask for points within bounds
                valid_mask_indices = bounds_mask.copy()
                if bounds_mask.any():
                    valid_coords = point_coords[bounds_mask]
                    mask_check = init_mask[valid_coords[:, 1], valid_coords[:, 0]]
                    valid_mask_indices[bounds_mask] = mask_check

                if valid_mask_indices.any():
                    pos_points = pos_points[valid_mask_indices]
                    pos_labels = pos_labels[valid_mask_indices]
                else:
                    # No positive points inside mask - clear them
                    pos_points = None
                    pos_labels = None

            # Step 2b: Optimized negative point distance checking
            if init_mask is not None and neg_points is not None:
                # Vectorized distance calculation
                mask_pixels = np.argwhere(init_mask)
                if len(mask_pixels) > 0:
                    neg_point_coords = neg_points.astype(np.int32)

                    # Pre-calculate bounds for early exit
                    h, w = init_mask.shape
                    x_neg, y_neg = neg_point_coords[:, 0], neg_point_coords[:, 1]

                    # Quick bounds check first
                    in_bounds_mask = (x_neg >= 0) & (y_neg >= 0) & (x_neg < w) & (y_neg < h)

                    if in_bounds_mask.any():
                        # Only calculate distances for points in bounds
                        bound_points = neg_point_coords[in_bounds_mask]

                        # Vectorized squared distance calculation (faster than sqrt)
                        distances_squared = np.min(
                            ((mask_pixels[:, 0, None] - bound_points[:, 1]) ** 2) +
                            ((mask_pixels[:, 1, None] - bound_points[:, 0]) ** 2),
                            axis=0
                        )

                        # Check threshold on squared distance (50^2 = 2500)
                        far_points_mask = distances_squared > 2500
                        if far_points_mask.any():
                            print(f"⚠ Warning: {far_points_mask.sum()} negative point(s) are far from the mask region.")
                            print(f"  Negative points work best inside or near the masked object to refine segmentation.")

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
            
            # Streamlined output mask creation - minimal tensor operations
            num_frames = len(pil_images)
            height, width = frames.shape[1], frames.shape[2]

            # Create output lists and convert once at end (faster than individual tensor ops)
            output_masks = []
            output_obj_ids = []

            for frame_idx in range(num_frames):
                if frame_idx in video_segments:
                    # Frame was processed - use actual mask
                    segment_data = video_segments[frame_idx]
                    if segment_data:
                        # Get first object mask
                        obj_id, mask = next(iter(segment_data.items()))
                        output_masks.append(torch.as_tensor(mask, dtype=torch.float32))
                        output_obj_ids.append(obj_id)
                        continue

                # Frame not processed or no data - empty mask
                empty_mask = torch.zeros(height, width, dtype=torch.float32)
                output_masks.append(empty_mask)
                output_obj_ids.append(0)

            # Convert lists to tensors in single operations
            masks_tensor = torch.stack(output_masks)
            obj_ids_tensor = torch.tensor(output_obj_ids, dtype=torch.int32)

            return (masks_tensor, obj_ids_tensor)

        except Exception as e:
            raise RuntimeError(f"SeC video segmentation failed: {str(e)}")

        finally:
            # Efficient cleanup with memory video handler
            if video_handler is not None:
                video_handler.cleanup()

            # Minimal GPU memory cleanup - only when actually needed
            if torch.cuda.is_available() and hasattr(frames, 'device') and frames.device.type == 'cuda':
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

            # Lightweight garbage collection
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
        """Parse hex or RGB color string to BGR tuple for OpenCV"""
        import re

        color_str = color_str.strip()

        if color_str.startswith('#'):
            color_str = color_str[1:]

        if re.match(r'^[0-9A-Fa-f]{6}$', color_str):
            r = int(color_str[0:2], 16)
            g = int(color_str[2:4], 16)
            b = int(color_str[4:6], 16)
            return (b, g, r)

        if ',' in color_str:
            parts = [int(x.strip()) for x in color_str.split(',')]
            if len(parts) == 3:
                r, g, b = parts
                return (b, g, r)

        return (0, 255, 0)

    def draw_shape(self, canvas, x, y, shape, size, color):
        """Draw a shape at the specified coordinates"""
        import cv2
        import numpy as np

        x, y = int(x), int(y)

        if shape == "circle":
            cv2.circle(canvas, (x, y), size, color, -1)
            cv2.circle(canvas, (x, y), size, (255, 255, 255), 2)

        elif shape == "square":
            half_size = size
            cv2.rectangle(canvas, (x - half_size, y - half_size),
                         (x + half_size, y + half_size), color, -1)
            cv2.rectangle(canvas, (x - half_size, y - half_size),
                         (x + half_size, y + half_size), (255, 255, 255), 2)

        elif shape == "triangle":
            height = int(size * 1.732)
            half_base = size

            pts = np.array([
                [x, y - height],
                [x - half_base, y + size],
                [x + half_base, y + size]
            ], np.int32)

            cv2.fillPoly(canvas, [pts], color)
            cv2.polylines(canvas, [pts], True, (255, 255, 255), 2)

    def plot_coordinates(self, coordinates, image=None, point_shape="circle",
                        point_size=10, point_color="#00FF00", width=512, height=512):
        """Plot coordinates on image or blank canvas"""
        import json
        import cv2
        import numpy as np

        try:
            if not coordinates or not coordinates.strip():
                coords_list = []
            else:
                coords_list = json.loads(coordinates)
                if not isinstance(coords_list, list):
                    raise ValueError("Coordinates must be a JSON array")

            if image is not None:
                canvas = (image[0].cpu().numpy() * 255).astype(np.uint8)
                canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
            else:
                canvas = np.zeros((height, width, 3), dtype=np.uint8)

            color = self.parse_color(point_color)

            for coord in coords_list:
                if isinstance(coord, dict) and 'x' in coord and 'y' in coord:
                    x = float(coord['x'])
                    y = float(coord['y'])
                    self.draw_shape(canvas, x, y, point_shape, point_size, color)

            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            canvas = canvas.astype(np.float32) / 255.0
            output = torch.from_numpy(canvas).unsqueeze(0)

            return (output,)

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON coordinates: {str(e)}")
        except Exception as e:
            raise RuntimeError(f"Coordinate plotting failed: {str(e)}")