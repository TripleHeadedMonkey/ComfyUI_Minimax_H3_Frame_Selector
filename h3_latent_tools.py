"""MiniMax H3 joint audiovisual latent continuation helpers.

These nodes deliberately operate on sampler-output H3 latents. They do not
derive temporal boundaries from decoded image dimensions, so spatially
upscaling VAEs (including custom 2x decoders) remain compatible.
"""

import json
import os
import threading
import uuid
from datetime import datetime
from pathlib import Path

import torch

import folder_paths


FPS = 24
AUDIO_LATENT_FPS = 40
_chain_lock = threading.Lock()


class AnyType(str):
    def __ne__(self, _other):
        return False


ANY = AnyType("*")


def _clean_project_name(value):
    import re
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value or "Untitled Project")).strip(" ._")
    return name[:100] or "Untitled Project"


def _nested_parts(latent):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise TypeError("Expected a MiniMax H3 LATENT dictionary containing samples.")
    samples = latent["samples"]
    if not getattr(samples, "is_nested", False):
        raise TypeError("Expected a joint MiniMax H3 video+audio NestedTensor latent.")
    parts = samples.unbind() if hasattr(samples, "unbind") else getattr(samples, "tensors", ())
    parts = list(parts)
    if len(parts) != 2:
        raise TypeError(f"Expected two H3 latent streams (video and audio), got {len(parts)}.")
    return parts[0], parts[1]


def _make_nested(video, audio):
    try:
        import comfy.nested_tensor
        return comfy.nested_tensor.NestedTensor((video, audio))
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("This ComfyUI build does not provide the H3 NestedTensor implementation.") from exc


def _frames_from_video_t(video_t):
    video_t = int(video_t)
    if video_t < 2 or (video_t - 2) % 5:
        raise ValueError(f"Video latent T={video_t} is not on MiniMax H3's 17k+5 temporal grid.")
    return 5 + ((video_t - 2) // 5) * 17


def _video_t_from_frames(frames):
    frames = int(frames)
    if frames < 5 or (frames - 5) % 17:
        raise ValueError(f"H3 frame count must be 17k+5; got {frames}.")
    return 2 + 5 * ((frames - 5) // 17)


def _expected_audio_t(frames):
    return round(int(frames) * AUDIO_LATENT_FPS / FPS)


def _exact_boundaries(max_frames):
    return [
        frames for frames in range(5, int(max_frames) + 1, 17)
        if (frames * AUDIO_LATENT_FPS) % FPS == 0
    ]


def _resolve_handover(requested, max_frames):
    boundaries = _exact_boundaries(max_frames)
    if not boundaries:
        raise ValueError(
            f"The source is only {max_frames} frames; an exact shared H3 AV handover requires at least 39 frames."
        )
    requested = max(1, int(requested))
    return min(boundaries, key=lambda value: (abs(value - requested), value))


def _validate(latent):
    video, audio = _nested_parts(latent)
    if video.ndim != 5 or int(video.shape[1]) != 24:
        raise ValueError(f"H3 video latent must be [B,24,T,H,W], got {tuple(video.shape)}.")
    if audio.ndim != 4 or int(audio.shape[1]) != 32 or int(audio.shape[2]) != 2:
        raise ValueError(f"H3 audio latent must be [B,32,2,T40], got {tuple(audio.shape)}.")
    if int(video.shape[0]) != 1 or int(audio.shape[0]) != 1:
        raise ValueError("H3 latent continuation currently requires batch size 1.")
    frames = _frames_from_video_t(video.shape[2])
    expected = _expected_audio_t(frames)
    if int(audio.shape[-1]) != expected:
        raise ValueError(
            f"H3 AV duration mismatch: {frames} video frames require {expected} audio ticks, "
            f"but the latent contains {audio.shape[-1]}."
        )
    return video, audio, frames


def _decode_video(vae, video):
    images = vae.decode(video)
    if images.ndim == 5:
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


def _decode_audio(audio_vae, audio, sample_rate=None):
    waveform = audio_vae.decode(audio).movedim(-1, 1)
    # Match ComfyUI's H3 audio decode convention while avoiding amplification.
    deviation = torch.std(waveform, dim=[1, 2], keepdim=True) * 5.0
    waveform = waveform / torch.clamp(deviation, min=1.0)
    rate = sample_rate or getattr(
        audio_vae, "audio_sample_rate_output", getattr(audio_vae, "audio_sample_rate", 32000)
    )
    return {"waveform": waveform, "sample_rate": int(rate)}


def _chain_directory(project_name, create=False):
    folder = (
        Path(folder_paths.get_output_directory())
        / "frame_selector_projects"
        / _clean_project_name(project_name)
        / "latent_chain"
    )
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    return folder


def _atomic_json(path, value):
    temporary = path.with_name(f"{path.stem}_{uuid.uuid4().hex}.tmp.json")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _atomic_torch_save(path, value):
    temporary = path.with_name(f"{path.stem}_{uuid.uuid4().hex}.tmp.pt")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _load_saved_chain(project_name):
    path = _chain_directory(project_name) / "assembled.pt"
    if not path.is_file():
        return None
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_saved_context(project_name):
    """Prefer the last accepted segment over the recursively assembled chain."""
    folder = _chain_directory(project_name)
    for filename in ("latest_segment.pt", "assembled.pt"):
        path = folder / filename
        if not path.is_file():
            continue
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")
    return None


def _read_chain_manifest(project_name):
    path = _chain_directory(project_name) / "chain.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) else None


def _write_active_generation_context(
    project_name, frames, handover_frames=None, requested_output_frames=None
):
    folder = _chain_directory(project_name, create=True)
    data = {
        "context_frames_24fps": max(0, int(frames)),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if handover_frames is not None:
        data["latent_handover_frames_24fps"] = max(0, int(handover_frames))
    if requested_output_frames is not None:
        data["requested_output_frames_24fps"] = max(0, int(requested_output_frames))
    _atomic_json(folder / "active_generation.json", data)


def _read_generation_plan(project_name):
    path = _chain_directory(project_name) / "generation_plan.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    return data if isinstance(data, dict) and int(data.get("schema_version", 0)) == 1 else None


def _read_project_selection(project_name):
    project = _clean_project_name(project_name)
    path = Path(folder_paths.get_output_directory()) / "frame_selector_projects" / project / "project_state.json"
    if not path.is_file():
        return None
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    selection = state.get("latest_selection")
    return selection if isinstance(selection, dict) and selection.get("decision") == "accepted" else None


def _exact_video_context(project_name, selection, vae, target_video, context_frames):
    """Re-encode an off-grid accepted endpoint from its exact saved RGB tail."""
    revision = int((selection or {}).get("selection_revision", -1))
    residual = int((selection or {}).get("residual_decoded_frames", 0))
    # Project 043's initial 1f residual produced a perfect first extension;
    # preserve that fast path. Larger residuals visibly desynchronise the next
    # latent tail from the accepted RGB endpoint and require exact re-encoding.
    if revision < 0 or residual <= 1:
        return None

    folder = _chain_directory(project_name, create=True)
    cache_path = folder / "exact_video_context.pt"
    if cache_path.is_file():
        try:
            try:
                cached = torch.load(cache_path, map_location="cpu", weights_only=True)
            except TypeError:
                cached = torch.load(cache_path, map_location="cpu")
            video = cached.get("video") if isinstance(cached, dict) else None
            if (
                int(cached.get("selection_revision", -2)) == revision
                and torch.is_tensor(video)
                and _frames_from_video_t(video.shape[2]) == int(context_frames)
                and tuple(video.shape[-2:]) == tuple(target_video.shape[-2:])
            ):
                print(
                    f"[MiniMax H3 Continuation Builder] Using cached exact RGB-reencoded "
                    f"tail for off-grid selection revision {revision}."
                )
                return video
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            pass

    relative = str((selection or {}).get("cut_video_file") or "")
    if not relative:
        return None
    project_root = folder.parent.resolve()
    cut_path = (project_root / relative.replace("\\", os.sep)).resolve()
    try:
        cut_path.relative_to(project_root)
    except ValueError:
        return None
    if not cut_path.is_file():
        return None

    try:
        import cv2
        import numpy as np
        import comfy.utils

        capture = cv2.VideoCapture(str(cut_path))
        if not capture.isOpened():
            return None
        count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if count < int(context_frames):
            capture.release()
            return None
        capture.set(cv2.CAP_PROP_POS_FRAMES, count - int(context_frames))
        frames = []
        try:
            for _ in range(int(context_frames)):
                ok, frame = capture.read()
                if not ok:
                    return None
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            capture.release()

        pixels = torch.from_numpy(np.stack(frames)).to(dtype=torch.float32).div_(255.0)
        height = int(target_video.shape[-2]) * 16
        width = int(target_video.shape[-1]) * 16
        pixels = comfy.utils.common_upscale(
            pixels.movedim(-1, 1), width, height, "lanczos", "disabled"
        ).movedim(1, -1)
        encoded = vae.encode(pixels)
        if not torch.is_tensor(encoded) or encoded.ndim != 5 or int(encoded.shape[1]) != 24:
            raise ValueError(f"video VAE returned unexpected shape {getattr(encoded, 'shape', None)}")
        encoded_frames = _frames_from_video_t(encoded.shape[2])
        if encoded_frames != int(context_frames):
            raise ValueError(
                f"video VAE encoded {encoded_frames} frames instead of {int(context_frames)}"
            )
        if tuple(encoded.shape[-2:]) != tuple(target_video.shape[-2:]):
            raise ValueError(
                f"video VAE encoded latent canvas {tuple(encoded.shape[-2:])}, "
                f"expected {tuple(target_video.shape[-2:])}"
            )
        encoded = encoded.detach().cpu()
        _atomic_torch_save(cache_path, {
            "selection_revision": revision,
            "context_frames": int(context_frames),
            "video": encoded,
        })
        print(
            f"[MiniMax H3 Continuation Builder] Re-encoded exact {context_frames}f RGB tail "
            f"for off-grid selection revision {revision} (residual={residual}f)."
        )
        return encoded
    except Exception as error:
        print(
            "[MiniMax H3 Continuation Builder] WARNING: exact off-grid tail re-encode failed; "
            f"using the rounded latent tail instead: {error}"
        )
        return None


def _anchor_conditioning(positive, vae, anchor_path, target_video, frame_index):
    import numpy as np
    from PIL import Image
    import comfy.utils

    with Image.open(anchor_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    pixels = torch.from_numpy(rgb).unsqueeze(0).movedim(-1, 1)
    height = int(target_video.shape[-2]) * 16
    width = int(target_video.shape[-1]) * 16
    pixels = comfy.utils.common_upscale(pixels, width, height, "lanczos", "disabled").movedim(1, -1)
    encoded = vae.encode(pixels)
    try:
        from .minimax_offset import _FALLBACK_ACTIVE, _MARKER
    except Exception:
        _FALLBACK_ACTIVE, _MARKER = False, "_ifs_scoped_offset_anchor"

    result = []
    for tensor, metadata in positive:
        updated = dict(metadata)
        keyframes = [dict(item) for item in updated.get("minimax_keyframes", [])]
        keyframe = {"resolved_frame_index": int(frame_index), "latent": encoded}
        if _FALLBACK_ACTIVE:
            keyframe[_MARKER] = True

        # Preserve the original two-anchor continuation conditioning. H3 is a
        # joint audiovisual model, and removing/relocating the existing frame-0
        # guide changed the audio solution even though the supplied audio slice
        # itself was correct. The additional hidden-pre-roll endpoint anchor is
        # appended without rewriting the user's existing conditioning.
        keyframes.append(keyframe)
        updated["minimax_keyframes"] = keyframes
        result.append([tensor, updated])
    return result


def _noise_parts(latent, video, audio):
    mask = latent.get("noise_mask") if isinstance(latent, dict) else None
    if mask is None:
        return torch.ones_like(video), torch.ones_like(audio)
    if not getattr(mask, "is_nested", False):
        raise TypeError("H3 target noise_mask must be a joint video+audio NestedTensor.")
    parts = mask.unbind() if hasattr(mask, "unbind") else getattr(mask, "tensors", ())
    parts = list(parts)
    if len(parts) != 2:
        raise TypeError("H3 target noise_mask must contain exactly two streams.")
    return parts[0].clone(), parts[1].clone()


class MiniMaxH3ContinuationTail:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "requested_context_frames": ("INT", {"default": 39, "min": 5, "max": 3599, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE", "AUDIO", "IMAGE", "INT", "FLOAT")
    RETURN_NAMES = (
        "context_latent", "context_frames", "context_audio", "last_frame",
        "resolved_context_frames", "context_seconds",
    )
    FUNCTION = "extract"
    CATEGORY = "MiniMax H3/continuation"
    DESCRIPTION = "Extracts an exact-grid joint AV latent tail and decodes references at the VAE's native output size."

    def extract(self, samples, video_vae, audio_vae, requested_context_frames=39):
        video, audio, total_frames = _validate(samples)
        count = _resolve_handover(requested_context_frames, total_frames)
        video_t = _video_t_from_frames(count)
        audio_t = count * AUDIO_LATENT_FPS // FPS
        tail_video = video[:, :, -video_t:].clone()
        tail_audio = audio[..., -audio_t:].clone()
        context = {"samples": _make_nested(tail_video, tail_audio)}
        images = _decode_video(video_vae, tail_video)
        decoded_audio = _decode_audio(audio_vae, tail_audio, samples.get("sample_rate"))
        print(
            f"[MiniMax H3 Continuation Tail] Requested {int(requested_context_frames)}f; "
            f"resolved to exact AV boundary {count}f ({count / FPS:.3f}s)."
        )
        return (context, images, decoded_audio, images[-1:].clone(), count, count / float(FPS))


class MiniMaxH3ContinuationBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "target_latent": ("LATENT",),
                "video_vae": ("VAE",),
                "project_name": ("STRING", {"default": "My Project", "multiline": False}),
                "source_mode": (["project_chain", "connected_context"],),
                "requested_context_frames": ("INT", {"default": 39, "min": 5, "max": 3599, "step": 1}),
            },
            "optional": {
                "context_latent": ("LATENT",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "LATENT", "INT", "FLOAT", "INT", "BOOLEAN")
    RETURN_NAMES = (
        "positive", "sampler_latent", "context_frames", "context_seconds",
        "anchor_frame_index", "using_context",
    )
    FUNCTION = "build"
    CATEGORY = "MiniMax H3/continuation"
    DESCRIPTION = "Builds a sampler-ready H3 target using accepted visual latent history while leaving fresh audio unchanged."

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # A newly accepted project chain must be observed on the next queued run.
        return float("nan")

    def build(
        self,
        positive,
        target_latent,
        video_vae,
        project_name="My Project",
        source_mode="project_chain",
        requested_context_frames=39,
        context_latent=None,
    ):
        target_video, target_audio, target_frames = _validate(target_latent)

        if source_mode == "connected_context":
            if context_latent is None:
                raise ValueError("connected_context mode requires the context_latent input.")
            source_video, source_audio, source_frames = _validate(context_latent)
        else:
            saved = _load_saved_context(project_name)
            if saved is None:
                first_plan = _read_generation_plan(project_name)
                first_output_frames = int(
                    (first_plan or {}).get("requested_output_frames_24fps", target_frames)
                )
                _write_active_generation_context(
                    project_name,
                    0,
                    handover_frames=0,
                    requested_output_frames=first_output_frames,
                )
                print(
                    "[MiniMax H3 Continuation Builder] No accepted project latent exists yet; "
                    "first run passes the target latent through unchanged."
                )
                return (positive, target_latent, 0, 0.0, -1, False)
            selection = _read_project_selection(project_name)
            manifest = _read_chain_manifest(project_name)
            selected_revision = int((selection or {}).get("selection_revision", -1))
            committed_revision = int((manifest or {}).get("selection_revision", -2))
            if selected_revision < 0 or committed_revision != selected_revision:
                # Older workflows allowed the output node to race ahead of the
                # interactive selector.  In that case the missing accepted
                # latent cannot be reconstructed from the encoded MP4.  Drop
                # only the stale latent cache and allow one image-guided fresh
                # segment to reseed it; accepted clips and project state stay.
                folder = _chain_directory(project_name)
                for filename in (
                    "assembled.pt", "latest_segment.pt", "exact_video_context.pt",
                    "chain.json", "active_generation.json"
                ):
                    try:
                        (folder / filename).unlink()
                    except FileNotFoundError:
                        pass
                recovery_plan = _read_generation_plan(project_name) or {}
                _write_active_generation_context(
                    project_name,
                    int(recovery_plan.get("cut_prefix_frames_24fps", 0)),
                    handover_frames=0,
                    requested_output_frames=int(
                        recovery_plan.get(
                            "requested_output_frames_24fps", target_frames
                        )
                    ),
                )
                print(
                    "[MiniMax H3 Continuation Builder] WARNING: stale latent cache detected "
                    f"(project selection revision {selected_revision}, latent revision "
                    f"{committed_revision}). Cleared latent cache and using one image-guided "
                    "fresh segment to reseed it. Connect the selector's selected_frame output "
                    "to Latent Chain's selector_commit_signal input."
                )
                return (positive, target_latent, 0, 0.0, -1, False)
            source_video = saved["video"]
            source_audio = saved["audio"]
            source_frames = _frames_from_video_t(source_video.shape[2])

        if tuple(source_video.shape[-2:]) != tuple(target_video.shape[-2:]):
            raise ValueError(
                "Continuation latent canvas differs from the new target. Use the same latent resolution; "
                "standard versus 2x VAE decoding does not change this latent canvas."
            )
        if source_video.dtype != target_video.dtype:
            source_video = source_video.to(dtype=target_video.dtype)
        if source_audio.dtype != target_audio.dtype:
            source_audio = source_audio.to(dtype=target_audio.dtype)
        maximum = min(source_frames, target_frames)
        plan = _read_generation_plan(project_name) if source_mode == "project_chain" else None
        if plan is None:
            raise ValueError(
                "Continuation requires a current Project Audio Start Time generation plan. "
                "Connect its internal_generation_duration output to the audio loader's duration input."
            )
        planned_internal = int(plan.get("internal_generation_frames_24fps", 0))
        planned_output = int(plan.get("requested_output_frames_24fps", 0))
        prefix_frames = int(plan.get("cut_prefix_frames_24fps", 0))
        planned_handover = int(plan.get("latent_handover_frames_24fps", 0))
        if prefix_frames <= 0 or planned_handover <= 0:
            _write_active_generation_context(
                project_name,
                0,
                handover_frames=0,
                requested_output_frames=planned_output,
            )
            print(
                "[MiniMax H3 Continuation Builder] Generation plan is direct; "
                "passing target through without latent continuation."
            )
            return (positive, target_latent, 0, 0.0, -1, False)
        if target_frames != planned_internal:
            raise ValueError(
                f"Generation plan requires {planned_internal} H3 frames "
                f"({planned_internal / FPS:.3f}s), but the audio-selected latent contains {target_frames}f. "
                "Connect internal_generation_duration to the audio loader duration."
            )

        context_frames = _resolve_handover(planned_handover, maximum)
        if source_mode == "project_chain":
            exact_video = _exact_video_context(
                project_name,
                _read_project_selection(project_name),
                video_vae,
                target_video,
                context_frames,
            )
            if exact_video is not None:
                source_video = exact_video
                source_frames = context_frames
        video_t = _video_t_from_frames(context_frames)

        output_video = target_video.clone()
        output_video[:, :, :video_t] = source_video[:, :, -video_t:].to(output_video.device)

        output_audio = target_audio.clone()
        output_video_mask, output_audio_mask = _noise_parts(
            target_latent, output_video, output_audio
        )
        output_video_mask[:, :, :video_t] = 0.0

        if source_mode == "project_chain":
            _write_active_generation_context(
                project_name,
                prefix_frames,
                handover_frames=context_frames,
                requested_output_frames=planned_output,
            )

        output = dict(target_latent)
        output["samples"] = _make_nested(output_video, output_audio)
        output["noise_mask"] = _make_nested(output_video_mask, output_audio_mask)
        anchor_index = -1
        if source_mode == "project_chain":
            selection = _read_project_selection(project_name)
            selection_residual = int((selection or {}).get("residual_decoded_frames", 0))
            anchor_file = str((selection or {}).get("anchor_image_file") or "")
            if anchor_file:
                project_root = _chain_directory(project_name).parent
                anchor_path = (project_root / anchor_file).resolve()
                try:
                    anchor_path.relative_to(project_root.resolve())
                except ValueError as exc:
                    raise ValueError("Project selection anchor path escapes the project folder.") from exc
                if anchor_path.is_file():
                    # A latent-grid selection (or the harmless initial 1f
                    # residual) remains anchored at the end of hidden context.
                    # After an off-grid accepted cut, however, the latent tail
                    # is re-encoded from RGB and the first newly generated
                    # frame must itself equal the selected endpoint.  Anchoring
                    # only frame prefix-1 allowed frame prefix to jump to a
                    # different view before the requested camera motion began
                    # (Project 044's second extension).  Repeating the endpoint
                    # for one frame is preferable to exposing that correction.
                    visible_endpoint_anchor = selection_residual > 1
                    anchor_index = min(
                        target_frames - 1,
                        prefix_frames if visible_endpoint_anchor else prefix_frames - 1,
                    )
                    positive = _anchor_conditioning(positive, video_vae, anchor_path, target_video, anchor_index)
        print(
            f"[MiniMax H3 Continuation Builder] Preserving {context_frames}f "
            f"({context_frames / FPS:.3f}s) visual context inside the native {target_frames}f plan; "
            f"cut pre-roll={prefix_frames}f, requested output={planned_output}f; "
            f"anchor={anchor_index}."
        )
        return (positive, output, context_frames, context_frames / float(FPS), anchor_index, True)


class MiniMaxH3LatentChain:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment": ("LATENT",),
                "selector_commit_signal": ("IMAGE",),
                "project_name": ("STRING", {"default": "My Project", "multiline": False}),
                "handover_frames": ("INT", {"default": 39, "min": 5, "max": 3599, "step": 1}),
                "control": (["append_or_start", "reset_and_start"],),
            },
        }

    RETURN_TYPES = ("LATENT", "LATENT", "INT", "FLOAT", "INT")
    RETURN_NAMES = (
        "current_segment", "full_visual_project_latent", "total_frames",
        "total_seconds", "segment_count",
    )
    FUNCTION = "append"
    CATEGORY = "MiniMax H3/continuation"
    DESCRIPTION = "Stores a project latent chain and removes each continuation's repeated AV prefix."
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def append(
        self,
        segment,
        selector_commit_signal,
        project_name="My Project",
        handover_frames=39,
        control="append_or_start",
    ):
        current_video, current_audio, current_frames = _validate(segment)
        folder = _chain_directory(project_name)
        tensor_path = folder / "assembled.pt"
        manifest_path = folder / "chain.json"
        selection = _read_project_selection(project_name)
        selection_revision = int((selection or {}).get("selection_revision", -1))

        if selection is None:
            # Never turn a merely generated sample into project history.  This
            # is the backend safety net for Comfy execution orders that invoke
            # this output node after the selector has returned a discard
            # blocker.  It also cleans files produced by older node versions.
            if folder.is_dir():
                for filename in (
                    "assembled.pt", "latest_segment.pt", "exact_video_context.pt",
                    "chain.json", "active_generation.json"
                ):
                    try:
                        (folder / filename).unlink()
                    except FileNotFoundError:
                        pass
            print(
                "[MiniMax H3 Latent Chain] No accepted frame selection exists; "
                "the generated latent was not committed."
            )
            return (segment, segment, current_frames, current_frames / float(FPS), 0)

        folder.mkdir(parents=True, exist_ok=True)

        accepted_frames = int(selection.get("latent_boundary_frames", 0))
        if accepted_frames < 5:
            raise ValueError(
                "The selected cut is earlier than H3's first sliceable latent boundary (5 frames)."
            )
        accepted_frames = min(accepted_frames, current_frames)
        accepted_frames = 5 + 17 * ((accepted_frames - 5) // 17)
        accepted_video_t = _video_t_from_frames(accepted_frames)
        current_video = current_video[:, :, :accepted_video_t]
        current_audio = current_audio[..., :_expected_audio_t(accepted_frames)]
        current_frames = accepted_frames

        with _chain_lock:
            existing_manifest = {}
            if manifest_path.is_file():
                try:
                    existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    existing_manifest = {}
            if (
                control != "reset_and_start"
                and tensor_path.is_file()
                and selection_revision >= 0
                and int(existing_manifest.get("selection_revision", -2)) == selection_revision
            ):
                saved = _load_saved_chain(project_name)
                output_video, output_audio = saved["video"], saved["audio"]
                output_frames = _frames_from_video_t(output_video.shape[2])
                segment_count = int(existing_manifest.get("segment_count", 1))
                assembled = {"samples": _make_nested(output_video, output_audio)}
                print(
                    f"[MiniMax H3 Latent Chain] Selection revision {selection_revision} was already committed; "
                    "returning the existing chain unchanged."
                )
                return (segment, assembled, output_frames, output_frames / float(FPS), segment_count)
            starts_new = control == "reset_and_start" or not tensor_path.is_file()
            if starts_new:
                output_video = current_video.detach().cpu().clone()
                output_audio = current_audio.detach().cpu().clone()
                output_frames = current_frames
                segment_count = 1
                resolved = 0
            else:
                try:
                    saved = torch.load(tensor_path, map_location="cpu", weights_only=True)
                except TypeError:
                    saved = torch.load(tensor_path, map_location="cpu")
                previous_video = saved["video"]
                previous_audio = saved["audio"]
                previous_frames = _frames_from_video_t(previous_video.shape[2])
                if tuple(previous_video.shape[-2:]) != tuple(current_video.shape[-2:]):
                    raise ValueError("Latent chain canvas changed; use reset_and_start for a new scene/resolution.")
                if previous_video.dtype != current_video.dtype or previous_audio.dtype != current_audio.dtype:
                    raise ValueError("Latent chain dtype changed; keep the same sampler precision or reset the chain.")
                resolved = _resolve_handover(handover_frames, current_frames)
                trim_video_t = _video_t_from_frames(resolved)
                trim_audio_t = resolved * AUDIO_LATENT_FPS // FPS
                video_tail = current_video[:, :, trim_video_t:].detach().cpu()
                audio_tail = current_audio[..., trim_audio_t:].detach().cpu()
                output_frames = previous_frames + current_frames - resolved
                target_audio_t = _expected_audio_t(output_frames)
                needed_audio_t = target_audio_t - int(previous_audio.shape[-1])
                if needed_audio_t < 1 or needed_audio_t > int(audio_tail.shape[-1]):
                    raise ValueError(
                        f"Audio seam cannot satisfy the absolute 24/40 Hz boundary: need {needed_audio_t} ticks, "
                        f"continuation provides {audio_tail.shape[-1]}."
                    )
                output_video = torch.cat((previous_video, video_tail), dim=2)
                output_audio = torch.cat((previous_audio, audio_tail[..., :needed_audio_t]), dim=-1)
                segment_count = int(existing_manifest.get("segment_count", 1)) + 1

            verified_frames = _frames_from_video_t(output_video.shape[2])
            if verified_frames != output_frames or int(output_audio.shape[-1]) != _expected_audio_t(output_frames):
                raise RuntimeError("Latent-chain reconstruction failed its final H3 AV duration check.")
            _atomic_torch_save(tensor_path, {"video": output_video, "audio": output_audio})
            _atomic_torch_save(
                folder / "latest_segment.pt",
                {
                    "video": current_video.detach().cpu().clone(),
                    "audio": current_audio.detach().cpu().clone(),
                },
            )
            manifest = {
                "schema_version": 1,
                "project": _clean_project_name(project_name),
                "segment_count": segment_count,
                "total_frames": output_frames,
                "total_seconds": output_frames / float(FPS),
                "last_source_frames": current_frames,
                "last_handover_frames": resolved,
                "selection_revision": selection_revision,
                "accepted_latent_frames": current_frames,
                "residual_decoded_frames": int((selection or {}).get("residual_decoded_frames", 0)),
                "video_latent_shape": list(output_video.shape),
                "audio_latent_shape": list(output_audio.shape),
                "updated_at": datetime.now().isoformat(timespec="seconds"),
            }
            _atomic_json(manifest_path, manifest)

        assembled = {"samples": _make_nested(output_video, output_audio)}
        print(
            f"[MiniMax H3 Latent Chain] {segment_count} segment(s), {output_frames}f "
            f"({output_frames / FPS:.3f}s); removed {resolved} repeated frame(s)."
        )
        return (segment, assembled, output_frames, output_frames / float(FPS), segment_count)


def commit_accepted_segment(segment, project_name):
    """Commit a sampler result synchronously from the selector's accept path.

    Nothing writes a provisional sampler result to disk.  Consequently a
    crash or discard leaves latest_segment.pt pointing at the last selection
    that actually reached the selector's accepted state.
    """
    plan = _read_generation_plan(project_name) or {}
    handover = int(plan.get("latent_handover_frames_24fps", 39)) or 39
    return MiniMaxH3LatentChain().append(
        segment=segment,
        selector_commit_signal=None,
        project_name=project_name,
        handover_frames=handover,
        control="append_or_start",
    )


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ContinuationTail": MiniMaxH3ContinuationTail,
    "MiniMaxH3ContinuationBuilder": MiniMaxH3ContinuationBuilder,
    "MiniMaxH3LatentChain": MiniMaxH3LatentChain,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ContinuationTail": "MiniMax H3 Continuation Tail",
    "MiniMaxH3ContinuationBuilder": "MiniMax H3 Continuation Builder",
    "MiniMaxH3LatentChain": "MiniMax H3 Latent Chain",
}
