import warnings
from collections import OrderedDict
import os
import torch
import torch.distributed
from torch import nn
import torch.nn.functional as F

from torch.nn.init import trunc_normal_
from tqdm import tqdm

from omegaconf import OmegaConf

# Note: SAM2 components are now imported lazily inside build_sam2_video_predictor()
# This prevents eager loading during ComfyUI startup and avoids Hydra conflicts

def _import_sam2_components():
    """Import all SAM2 components lazily when needed."""
    # Import SAM2 Hydra initialization first
    from . import init_sam2_hydra
    init_sam2_hydra()

    # Import SAM2 components
    from .sam2.sam2_video_predictor import SAM2VideoPredictor as _SAM2VideoPredictor
    from .sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base
    from .sam2.utils.misc import concat_points, fill_holes_in_mask_scores, load_video_frames
    from .sam2.modeling.sam2_utils import get_1d_sine_pe, MLP, select_closest_cond_frames

    # Import all required classes for local instantiation - complete isolation from global imports
    from .sam2.modeling.backbones.hieradet import Hiera
    from .sam2.modeling.backbones.image_encoder import ImageEncoder, FpnNeck
    from .sam2.modeling.position_encoding import PositionEmbeddingSine
    from .sam2.modeling.memory_attention import MemoryAttention, MemoryAttentionLayer
    from .sam2.modeling.sam.transformer import RoPEAttention
    from .sam2.modeling.memory_encoder import MemoryEncoder, MaskDownSampler, Fuser, CXBlock

    return {
        "SAM2VideoPredictor": _SAM2VideoPredictor,
        "SAM2Base": SAM2Base,
        "NO_OBJ_SCORE": NO_OBJ_SCORE,
        "concat_points": concat_points,
        "fill_holes_in_mask_scores": fill_holes_in_mask_scores,
        "load_video_frames": load_video_frames,
        "get_1d_sine_pe": get_1d_sine_pe,
        "MLP": MLP,
        "select_closest_cond_frames": select_closest_cond_frames,
        "Hiera": Hiera,
        "ImageEncoder": ImageEncoder,
        "FpnNeck": FpnNeck,
        "PositionEmbeddingSine": PositionEmbeddingSine,
        "MemoryAttention": MemoryAttention,
        "MemoryAttentionLayer": MemoryAttentionLayer,
        "RoPEAttention": RoPEAttention,
        "MemoryEncoder": MemoryEncoder,
        "MaskDownSampler": MaskDownSampler,
        "Fuser": Fuser,
        "CXBlock": CXBlock,
    }

# Local class registry will be populated when components are imported
LOCAL_CLASS_REGISTRY = {}

def _get_local_class_registry():
    """Get the local class registry, importing components if needed."""
    if not LOCAL_CLASS_REGISTRY:
        components = _import_sam2_components()
        LOCAL_CLASS_REGISTRY.update({
            "inference.sam2.modeling.sam2_base.SAM2Base": components["SAM2Base"],
            "inference.sam2.modeling.backbones.hieradet.Hiera": components["Hiera"],
            "inference.sam2.modeling.backbones.image_encoder.ImageEncoder": components["ImageEncoder"],
            "inference.sam2.modeling.backbones.image_encoder.FpnNeck": components["FpnNeck"],
            "inference.sam2.modeling.position_encoding.PositionEmbeddingSine": components["PositionEmbeddingSine"],
            "inference.sam2.modeling.memory_attention.MemoryAttention": components["MemoryAttention"],
            "inference.sam2.modeling.memory_attention.MemoryAttentionLayer": components["MemoryAttentionLayer"],
            "inference.sam2.modeling.sam.transformer.RoPEAttention": components["RoPEAttention"],
            "inference.sam2.modeling.memory_encoder.MemoryEncoder": components["MemoryEncoder"],
            "inference.sam2.modeling.memory_encoder.MaskDownSampler": components["MaskDownSampler"],
            "inference.sam2.modeling.memory_encoder.Fuser": components["Fuser"],
            "inference.sam2.modeling.memory_encoder.CXBlock": components["CXBlock"],
            "inference.sam2_video_predictor.SAM2VideoPredictor": get_sam2_video_predictor_class(),
        })
    return LOCAL_CLASS_REGISTRY

def build_sam2_video_predictor(
    config_file,
    num_maskmem=7,
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
    # Import SAM2 components only when actually building the predictor
    components = _import_sam2_components()
    registry = _get_local_class_registry()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(current_dir, "..", "configs")
    config_path = os.path.join(config_dir, config_file)

    cfg = OmegaConf.load(config_path)

    hydra_overrides = [
        "++model._target_=inference.sam2_video_predictor.SAM2VideoPredictor",
    ]
    if apply_postprocessing:
        hydra_overrides_extra = hydra_overrides_extra.copy()
        hydra_overrides_extra += [
            # dynamically fall back to multi-mask if the single mask is not stable
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_via_stability=true",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_delta=0.05",
            "++model.sam_mask_decoder_extra_args.dynamic_multimask_stability_thresh=0.98",
            # the sigmoid mask logits on interacted frames with clicks in the memory encoder so that the encoded masks are exactly as what users see from clicking
            "++model.binarize_mask_from_pts_for_mem_enc=true",
            # fill small holes in the low-res masks up to `fill_hole_area` (before resizing them to the original video resolution)
            "++model.fill_hole_area=8",
        ]
    hydra_overrides_extra.append(
        f"model.num_maskmem={num_maskmem}"
    )
    hydra_overrides.extend(hydra_overrides_extra)

    for override in hydra_overrides:
        if override.startswith("++"):
            key_path = override[2:].split("=")[0]
            value = "=".join(override[2:].split("=")[1:])
            if value.lower() in ["true", "false"]:
                value = value.lower() == "true"
            elif value.replace(".", "").replace("-", "").isdigit():
                if "." in value:
                    value = float(value)
                else:
                    value = int(value)

            keys = key_path.split(".")
            current = cfg
            for key in keys[:-1]:
                if key not in current:
                    current[key] = OmegaConf.create({})
                current = current[key]
            current[keys[-1]] = value
        else:
            if "=" in override:
                key_path = override.split("=")[0]
                value = "=".join(override.split("=")[1:])
                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                elif value.replace(".", "").replace("-", "").isdigit():
                    if "." in value:
                        value = float(value)
                    else:
                        value = int(value)

                keys = key_path.split(".")
                current = cfg
                for key in keys[:-1]:
                    if key not in current:
                        current[key] = OmegaConf.create({})
                    current = current[key]
                current[keys[-1]] = value

    OmegaConf.resolve(cfg)

    def create_component(config_node):
        """Recursively create components from config, handling all OmegaConf types properly"""
        if OmegaConf.is_list(config_node):
            return [create_component(item) for item in config_node]

        elif OmegaConf.is_dict(config_node):
            if '_target_' in config_node:
                target = config_node._target_
                kwargs = {k: create_component(v) for k, v in config_node.items() if k != '_target_'}

                module_path, class_name = target.rsplit('.', 1)

                if target in registry:
                    cls = registry[target]
                    return cls(**kwargs)
                else:
                    raise RuntimeError(f"Unknown local target: {target}. Available targets: {list(registry.keys())}")
            else:
                return {k: create_component(v) for k, v in config_node.items()}

        else:
            return config_node

    model = create_component(cfg.model)

    return model

def get_sam2_video_predictor_class():
    """Get the SAM2VideoPredictor class dynamically."""
    components = _import_sam2_components()
    base_class = components["SAM2VideoPredictor"]

    class SAM2VideoPredictor(base_class):
        def init_state(self, video_path, **kwargs):
            inference_state = super().init_state(video_path=video_path, **kwargs)
            frame_names = [
                os.path.splitext(p)[0]
                for p in os.listdir(video_path)
                if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
            ]
            frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
            inference_state["video_paths"] = [
                os.path.join(video_path, f"{frame_name}.jpg")
                for frame_name in frame_names
            ]
            return inference_state

    return SAM2VideoPredictor