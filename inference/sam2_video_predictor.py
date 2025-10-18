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

LOCAL_CLASS_REGISTRY = {
    "inference.sam2.modeling.sam2_base.SAM2Base": SAM2Base,
    "inference.sam2.modeling.backbones.hieradet.Hiera": Hiera,
    "inference.sam2.modeling.backbones.image_encoder.ImageEncoder": ImageEncoder,
    "inference.sam2.modeling.backbones.image_encoder.FpnNeck": FpnNeck,
    "inference.sam2.modeling.position_encoding.PositionEmbeddingSine": PositionEmbeddingSine,
    "inference.sam2.modeling.memory_attention.MemoryAttention": MemoryAttention,
    "inference.sam2.modeling.memory_attention.MemoryAttentionLayer": MemoryAttentionLayer,
    "inference.sam2.modeling.sam.transformer.RoPEAttention": RoPEAttention,
    "inference.sam2.modeling.memory_encoder.MemoryEncoder": MemoryEncoder,
    "inference.sam2.modeling.memory_encoder.MaskDownSampler": MaskDownSampler,
    "inference.sam2.modeling.memory_encoder.Fuser": Fuser,
    "inference.sam2.modeling.memory_encoder.CXBlock": CXBlock,
}

def build_sam2_video_predictor(
    config_file,
    num_maskmem=7,
    hydra_overrides_extra=[],
    apply_postprocessing=True,
    **kwargs,
):
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

                if target in LOCAL_CLASS_REGISTRY:
                    cls = LOCAL_CLASS_REGISTRY[target]
                    return cls(**kwargs)
                else:
                    raise RuntimeError(f"Unknown local target: {target}. Available targets: {list(LOCAL_CLASS_REGISTRY.keys())}")
            else:
                return {k: create_component(v) for k, v in config_node.items()}

        else:
            return config_node

    model = create_component(cfg.model)
    
    return model

class SAM2VideoPredictor(_SAM2VideoPredictor):

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
    
    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        start_frame_idx=0,
        iou_thre=0.3,
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].size(1)  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        device = current_vision_feats[-1].device
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        if self.num_maskmem == 0:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)
            return pix_feat

        num_obj_ptr_tokens = 0
        tpos_sign_mul = -1 if track_in_reverse else 1
        if not is_init_cond_frame:
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with stride>1), in which case
            # we take (self.num_maskmem - 2) frames among every stride-th frames plus the last frame.
            stride = 1 if self.training else self.memory_temporal_stride_for_eval
            
            _memory_frames = max(self.max_obj_ptrs_in_encoder, self.num_maskmem)
            max_obj_ptrs_in_encoder = min(num_frames, _memory_frames)
            
            valid_indices = [
                i for i in range(frame_idx - 1, start_frame_idx, -1)
                if (output_dict["non_cond_frame_outputs"][i]['ious'].max().item() > iou_thre
                    and output_dict["non_cond_frame_outputs"][i]['object_score_logits'].item() > 0)
            ][:max_obj_ptrs_in_encoder - 1]
            valid_indices.sort()
            if frame_idx - 1 not in valid_indices and (frame_idx - 1) != start_frame_idx:
                valid_indices.append(frame_idx - 1)

            for t_pos in range(1, self.num_maskmem):
                t_rel = t_pos - self.num_maskmem    # how many frames before current frame
                if t_rel < -len(valid_indices):
                    continue
                prev_frame_idx = valid_indices[t_rel]
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue
                feats = prev["maskmem_features"].to(device, non_blocking=True)
                to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                maskmem_enc = prev["maskmem_pos_enc"][-1].to(device)
                maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    (
                        (
                            (frame_idx - t) * tpos_sign_mul
                            if self.use_signed_tpos_enc_to_obj_ptrs
                            else abs(frame_idx - t)
                        ),
                        out["obj_ptr"].to(device, non_blocking=True),
                    )
                    for t, out in ptr_cond_outputs.items()
                ]
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    if -t_diff <= -len(valid_indices):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        valid_indices[-t_diff], unselected_cond_outputs.get(valid_indices[-t_diff], None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"].to(device, non_blocking=True)))
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    obj_ptrs = torch.stack(ptrs_list, dim=0)
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = torch.tensor(pos_list).to(
                            device=device, non_blocking=True
                        )
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        obj_pos = self.obj_ptr_tpos_proj(obj_pos)
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = obj_ptrs.new_zeros(len(pos_list), B, self.mem_dim)
                    if self.mem_dim < C:
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
                        obj_pos = obj_pos.repeat_interleave(C // self.mem_dim, dim=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            if self.directly_add_no_mem_embed:
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
                return pix_feat_with_mem

            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        memory = torch.cat(to_cat_memory, dim=0)
        memory_pos_embed = torch.cat(to_cat_memory_pos_embed, dim=0)

        pix_feat_with_mem = self.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)
        return pix_feat_with_mem
    
    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse,
        prev_sam_mask_logits,
        ## Extension: LLM prompt
        start_frame_idx=0,
        language_embd=None,
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        if len(current_vision_feats) > 1:
            high_res_features = [
                x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            pix_feat = current_vision_feats[-1].permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feat_sizes[-1])
            sam_outputs = self._use_mask_as_output(
                pix_feat, high_res_features, mask_inputs
            )
        else:
            pix_feat = self._prepare_memory_conditioned_features(
                frame_idx=frame_idx,
                is_init_cond_frame=is_init_cond_frame,
                current_vision_feats=current_vision_feats[-1:],
                current_vision_pos_embeds=current_vision_pos_embeds[-1:],
                feat_sizes=feat_sizes[-1:],
                output_dict=output_dict,
                num_frames=num_frames,
                track_in_reverse=track_in_reverse,
                start_frame_idx=start_frame_idx,
            )
 
            if language_embd is not None:
                _language_embd = language_embd.reshape(-1, 1, 256 // self.mem_dim, self.mem_dim)
                _language_embd = _language_embd.permute(0, 2, 1, 3).flatten(0, 1)
                _language_embd_pos = _language_embd.new_zeros(_language_embd.size(0), 1, self.mem_dim)
                pix_feat_with_language = self.token_attn(
                    curr=current_vision_feats[-1:],
                    curr_pos=current_vision_pos_embeds[-1:],
                    memory=_language_embd,
                    memory_pos=_language_embd_pos,
                    num_obj_ptr_tokens=_language_embd.shape[0],
                )
                pix_feat_with_language = pix_feat_with_language.permute(1, 2, 0).view(1, 256, 64, 64)
                pix_feat = (pix_feat_with_language + pix_feat) / 2

            if prev_sam_mask_logits is not None:
                assert point_inputs is not None and mask_inputs is None
                mask_inputs = prev_sam_mask_logits
            multimask_output = self._use_multimask(is_init_cond_frame, point_inputs)
            sam_outputs = self._forward_sam_heads(
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=multimask_output,
            )
        return current_out, sam_outputs, high_res_features, pix_feat
    
    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
        # Whether to run the memory encoder on the predicted masks. Sometimes we might want
        # to skip the memory encoder with `run_mem_encoder=False`. For example,
        # in demo we might call `track_step` multiple times for each user click,
        # and only encode the memory when the user finalizes their clicks. And in ablation
        # settings like SAM training on static images, we don't need the memory encoder.
        run_mem_encoder=True,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
        prev_sam_mask_logits=None,
        ## Extension: LLM prompt
        start_frame_idx=0,
        language_embd=None,
    ):
        current_out, sam_outputs, _, _ = self._track_step(
            frame_idx,
            is_init_cond_frame,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            mask_inputs,
            output_dict,
            num_frames,
            track_in_reverse,
            prev_sam_mask_logits,
            start_frame_idx,
            language_embd,
        )

        (
            low_res_multimasks,
            _,
            ious,
            low_res_masks,
            high_res_masks,
            obj_ptr,
            object_score_logits,
        ) = sam_outputs

        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr
        current_out["low_res_multimasks"] = low_res_multimasks
        if not self.training:
            current_out["object_score_logits"] = object_score_logits
            current_out["ious"] = ious

        self._encode_memory_in_output(
            current_vision_feats,
            feat_sizes,
            point_inputs,
            run_mem_encoder,
            high_res_masks,
            object_score_logits,
            current_out,
        )
        return current_out
    
    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
        ## Extension: LLM prompt
        start_frame_idx=0,
        language_embd=None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx, batch_size)

        assert point_inputs is None or mask_inputs is None
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            run_mem_encoder=run_mem_encoder,
            prev_sam_mask_logits=prev_sam_mask_logits,
            start_frame_idx=start_frame_idx,
            language_embd=language_embd,
        )

        storage_device = inference_state["storage_device"]
        maskmem_features = current_out["maskmem_features"]
        if maskmem_features is not None:
            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(storage_device, non_blocking=True)
        pred_masks_gpu = current_out["pred_masks"]
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )
        pred_masks = pred_masks_gpu.to(storage_device, non_blocking=True)
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        obj_ptr = current_out["obj_ptr"]
        ious = current_out["ious"]
        object_score_logits = current_out["object_score_logits"]
        low_res_multimasks = current_out["low_res_multimasks"]

        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
            "ious": ious,
            "object_score_logits": object_score_logits,
            "low_res_multimasks": low_res_multimasks,
        }
        return compact_current_out, pred_masks_gpu


LOCAL_CLASS_REGISTRY["inference.sam2_video_predictor.SAM2VideoPredictor"] = SAM2VideoPredictor


