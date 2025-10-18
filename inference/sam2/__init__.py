# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from hydra import initialize_config_module
from hydra.core.global_hydra import GlobalHydra

# Global flag to track if SAM2 Hydra has been initialized
_sam2_hydra_initialized = False

def init_sam2_hydra():
    """
    Initialize SAM2 Hydra configuration lazily.
    Only initializes when actually needed (when building SAM2 models),
    not during module import to prevent global Hydra conflicts.
    """
    global _sam2_hydra_initialized

    if _sam2_hydra_initialized:
        return

    try:
        # Check if Hydra is already initialized (potential conflict)
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # Initialize SAM2 configuration module
        initialize_config_module("inference.sam2.configs", version_base="1.2")
        _sam2_hydra_initialized = True

    except Exception as e:
        raise RuntimeError(f"Failed to initialize SAM2 Hydra configuration: {e}")

# Note: We no longer initialize Hydra at module import time
# This prevents global Hydra conflicts during ComfyUI startup
# Hydra will be initialized only when SAM2 is actually used