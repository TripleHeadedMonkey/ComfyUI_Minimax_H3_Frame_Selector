"""Scoped MiniMax H3 offset anchor for continuation workflows.

Unlike the public Keyframe Offset extension, the PackedLayout compatibility
wrapper below leaves every ordinary MiniMax H3 conditioning path untouched.
It activates only when this node's private marker is present.
"""

import importlib
import logging

import comfy.ldm.minimax.model as minimax_model
import comfy.utils


_MARKER = "_ifs_scoped_offset_anchor"
_LOG = logging.getLogger(__name__)
_FALLBACK_ACTIVE = False


def _native_guides_available():
    """Detect the supported core Add Guide node without version guessing."""
    for module_name in ("comfy_extras.nodes_minimax", "comfy_extras.nodes_minimax_h3"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        mappings = getattr(module, "NODE_CLASS_MAPPINGS", {})
        if "MiniMaxH3AddGuide" in mappings or hasattr(module, "MiniMaxH3AddGuide"):
            return True
    return False


def _install_scoped_layout_wrapper():
    global _FALLBACK_ACTIVE
    if _native_guides_available():
        _LOG.info("MiniMax H3 Scoped Offset Anchor: native Add Guide detected; core remains unmodified")
        return
    original_init = minimax_model.PackedLayout.__init__
    if getattr(original_init, "_ifs_scoped_offset_patch", False):
        _FALLBACK_ACTIVE = True
        return
    original_module = str(getattr(original_init, "__module__", "") or "")
    core_module = str(getattr(minimax_model.PackedLayout, "__module__", "") or "")
    if hasattr(original_init, "__wrapped__") or (original_module and core_module and original_module != core_module):
        raise RuntimeError(
            "another extension already owns MiniMax H3 PackedLayout; refusing to stack the scoped fallback. "
            "Update ComfyUI for native Add Guide support or remove the conflicting H3 patch."
        )

    def scoped_init(self, *args, **kwargs):
        # PackedLayout currently receives keyframes as its sixth positional
        # argument, but accepting both forms keeps this compatible with core
        # revisions that call it by keyword.
        supplied = kwargs.get("keyframes")
        positional = False
        if supplied is None and len(args) > 5:
            supplied = args[5]
            positional = True

        if not supplied or not any(kf.get(_MARKER, False) for kf in supplied):
            return original_init(self, *args, **kwargs)

        original_indices = [int(kf.get("resolved_frame_index", 0)) for kf in supplied]
        safe_keyframes = []
        for keyframe in supplied:
            copied = dict(keyframe)
            copied.pop(_MARKER, None)
            copied["resolved_frame_index"] = 0
            safe_keyframes.append(copied)

        call_args = list(args)
        call_kwargs = dict(kwargs)
        if positional:
            call_args[5] = safe_keyframes
        else:
            call_kwargs["keyframes"] = safe_keyframes

        result = original_init(self, *call_args, **call_kwargs)

        text_len = call_kwargs.get("text_len", call_args[0] if call_args else 0)
        cond_segments = [(a, b) for a, b, kind in self.segments if kind == "cond"]
        if len(cond_segments) < len(original_indices):
            raise RuntimeError(
                "MiniMax H3 layout produced fewer conditioning segments than offset anchors; "
                "this ComfyUI version is not compatible with the scoped offset node."
            )
        for (start, end), frame_index in zip(cond_segments, original_indices):
            cond_t = float(text_len) + minimax_model.FRAME_RESCALE * float(frame_index)
            self.position_ids[start:end, 0] = cond_t
        return result

    scoped_init._ifs_scoped_offset_patch = True
    scoped_init._ifs_original_init = original_init
    minimax_model.PackedLayout.__init__ = scoped_init
    _FALLBACK_ACTIVE = True
    _LOG.info("MiniMax H3 Scoped Offset Anchor: compatibility wrapper installed")


_install_scoped_layout_wrapper()


def _video_tensor(latent):
    samples = latent.get("samples") if isinstance(latent, dict) else None
    if hasattr(samples, "tensors") and samples.tensors:
        return samples.tensors[0]
    raise TypeError("Expected a MiniMax H3 audiovisual LATENT from MiniMaxH3ReferenceToVideo.")


def _frame_count(positive, video):
    for item in positive:
        if len(item) > 1 and "minimax_frame_count" in item[1]:
            return int(item[1]["minimax_frame_count"])
    latent_t = int(video.shape[2])
    return 5 if latent_t <= 2 else ((latent_t - 2) // 5) * 17 + 5


class MiniMaxH3ScopedOffsetAnchor:
    """Add one movable image anchor before chaining native AddGuide nodes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "vae": ("VAE",),
                "image": ("IMAGE",),
                "frame_index": ("INT", {
                    "default": 12,
                    "min": 0,
                    "max": 3599,
                    "step": 1,
                    "tooltip": "Exact frame where the continuation image is anchored (0-based).",
                }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "FLOAT")
    RETURN_NAMES = ("positive", "latent", "offset_frames", "offset_seconds")
    FUNCTION = "add_anchor"
    CATEGORY = "model/conditioning/minimax"
    DESCRIPTION = "Adds a safely scoped offset continuation anchor, then chains into native MiniMaxH3AddGuide nodes."

    def add_anchor(self, positive, latent, vae, image, frame_index):
        video = _video_tensor(latent)
        frame_count = _frame_count(positive, video)
        requested = int(frame_index)
        resolved = max(0, min(requested, frame_count - 1))
        if requested != resolved:
            print(
                f"[MiniMax H3 Scoped Offset Anchor] frame_index clamped "
                f"{requested}->{resolved} for {frame_count} frames."
            )

        height = int(video.shape[-2]) * 16
        width = int(video.shape[-1]) * 16
        pixels = image[:1, ..., :3].movedim(-1, 1)
        pixels = comfy.utils.common_upscale(pixels, width, height, "lanczos", "disabled")
        pixels = pixels.movedim(1, -1)
        encoded = vae.encode(pixels)

        result = []
        for tensor, metadata in positive:
            updated = dict(metadata)
            keyframes = [dict(kf) for kf in updated.get("minimax_keyframes", [])]
            keyframes.append({
                "resolved_frame_index": resolved,
                "latent": encoded,
                **({_MARKER: True} if _FALLBACK_ACTIVE else {}),
            })
            updated["minimax_keyframes"] = keyframes
            updated["minimax_frame_count"] = frame_count
            result.append([tensor, updated])

        print(
            f"[MiniMax H3 Scoped Offset Anchor] Added continuation anchor at "
            f"frame {resolved}/{frame_count - 1}."
        )
        return (result, latent, resolved, resolved / 24.0)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ScopedOffsetAnchor": MiniMaxH3ScopedOffsetAnchor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ScopedOffsetAnchor": "MiniMax H3 Scoped Offset Anchor",
}
