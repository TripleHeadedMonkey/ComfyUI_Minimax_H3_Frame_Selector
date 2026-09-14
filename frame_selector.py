import hashlib
import io
import importlib
import json
import math
import os
import re
import shutil
import subprocess
import threading
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import folder_paths
from server import PromptServer
from aiohttp import web
from comfy_execution.graph_utils import ExecutionBlocker


class AnyType(str):
    """A socket type which can connect to IMAGE, VIDEO, or filename bundles."""

    def __ne__(self, _other):
        return False


ANY = AnyType("*")
_pending = {}
_pending_lock = threading.Lock()
_last_selection = {}
_running_totals = {}
_running_totals_lock = threading.Lock()
_start_frames_lock = threading.Lock()
_project_state_lock = threading.Lock()
_ffmpeg_executable = None
PROJECT_TIMELINE_FPS = 24


def _safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bundle_path(value):
    """Resolve common VHS / video-combine filename bundles to a local path."""
    if isinstance(value, (str, os.PathLike)):
        path = Path(value)
        if path.is_file():
            return str(path.resolve())

    if not isinstance(value, dict):
        return None

    # VHS_FILENAMES normally stores: (filename, subfolder, type, format)
    candidates = []
    for key in ("gifs", "videos", "files", "filenames"):
        item = value.get(key)
        if isinstance(item, (list, tuple)):
            candidates.extend(item)
    for item in candidates:
        if isinstance(item, (list, tuple)) and item:
            filename = str(item[0])
            subfolder = str(item[1]) if len(item) > 1 and item[1] else ""
            kind = str(item[2]) if len(item) > 2 and item[2] else "output"
            base = folder_paths.get_directory_by_type(kind) or folder_paths.get_output_directory()
            path = Path(base) / subfolder / filename
            if path.is_file():
                return str(path.resolve())
        elif isinstance(item, (str, os.PathLike)) and Path(item).is_file():
            return str(Path(item).resolve())

    for key in ("path", "filename", "file"):
        raw = value.get(key)
        if not raw:
            continue
        path = Path(str(raw))
        if path.is_file():
            return str(path.resolve())
        subfolder = str(value.get("subfolder", ""))
        kind = str(value.get("type", "output"))
        base = folder_paths.get_directory_by_type(kind) or folder_paths.get_output_directory()
        path = Path(base) / subfolder / str(raw)
        if path.is_file():
            return str(path.resolve())
    return None


def _native_video_path(value):
    """Best-effort support for newer ComfyUI VIDEO wrapper objects."""
    direct = _bundle_path(value)
    if direct:
        return direct
    for attr in ("path", "filename", "file_path"):
        raw = getattr(value, attr, None)
        if raw and Path(str(raw)).is_file():
            return str(Path(str(raw)).resolve())
    # Native ComfyUI VideoInput. File-backed videos return a path; component-
    # backed videos return a BytesIO encoded by ComfyUI itself.
    stream_getter = getattr(value, "get_stream_source", None)
    if callable(stream_getter):
        source = stream_getter()
        if isinstance(source, (str, os.PathLike)) and Path(source).is_file():
            return str(Path(source).resolve())
        if hasattr(source, "read"):
            target_dir = Path(folder_paths.get_temp_directory()) / "interactive_frame_selector"
            target_dir.mkdir(parents=True, exist_ok=True)
            target = target_dir / f"native_video_{uuid.uuid4().hex}.mp4"
            try:
                source.seek(0)
            except Exception:
                pass
            with open(target, "wb") as handle:
                shutil.copyfileobj(source, handle)
            return str(target.resolve())
    return None


def _native_video_from_file(path):
    """Return ComfyUI's native file-backed VIDEO object when the API exists."""
    for module_name in (
        "comfy_api.latest._input_impl.video_types",
        "comfy_api.latest.input_impl.video_types",
    ):
        try:
            module = importlib.import_module(module_name)
            video_class = getattr(module, "VideoFromFile")
            return video_class(str(Path(path).resolve()))
        except (ImportError, AttributeError, TypeError):
            continue
    return None


def _open_video(path):
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("Video-file input requires opencv-python (cv2). IMAGE batches work without it.") from exc
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open rendered video: {path}")
    return cv2, cap


def _copy_preview(path, token):
    suffix = Path(path).suffix.lower() or ".mp4"
    name = f"frame_selector_{token}{suffix}"
    subfolder = "interactive_frame_selector"
    target_dir = Path(folder_paths.get_temp_directory()) / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name
    shutil.copy2(path, target)
    return {"filename": name, "subfolder": subfolder, "type": "temp"}


def _tensor_preview(frames, fps, token):
    import cv2

    name = f"frame_selector_{token}.mp4"
    subfolder = "interactive_frame_selector"
    target_dir = Path(folder_paths.get_temp_directory()) / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / name
    h, w = int(frames.shape[1]), int(frames.shape[2])
    writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not create the temporary MP4 preview.")
    try:
        for frame in frames:
            rgb = np.clip(frame.detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            if rgb.shape[-1] == 4:
                rgb = rgb[..., :3]
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
    return {"filename": name, "subfolder": subfolder, "type": "temp"}


def _preview_path(preview):
    return str((Path(folder_paths.get_temp_directory()) / preview["subfolder"] / preview["filename"]).resolve())


def _source_info(video, requested_fps, token):
    if torch.is_tensor(video):
        frames = video
        if frames.ndim == 3:
            frames = frames.unsqueeze(0)
        if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
            raise ValueError("IMAGE input must be [frames, height, width, RGB/RGBA].")
        count = int(frames.shape[0])
        fps = float(requested_fps)
        preview = _tensor_preview(frames, fps, token)
        fingerprint = f"tensor:{tuple(frames.shape)}:{float(frames.flatten()[0]) if frames.numel() else 0}"
        return {"kind": "tensor", "frames": frames, "path": _preview_path(preview), "count": count, "fps": fps,
                "preview": preview, "fingerprint": fingerprint}

    path = _native_video_path(video)
    if not path:
        raise TypeError(
            "Unsupported video value. Connect a rendered VIDEO/filename output or an IMAGE frame batch. "
            "If your node pack uses a private video wrapper, connect its filename output."
        )
    cv2, cap = _open_video(path)
    try:
        count = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        detected = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    fps = detected if detected > 0.001 else float(requested_fps)
    stat = os.stat(path)
    fingerprint = hashlib.sha1(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode()).hexdigest()
    return {"kind": "file", "path": path, "count": count, "fps": fps,
            "preview": _copy_preview(path, token), "fingerprint": fingerprint}


def _extract_frame(source, index):
    index = max(0, min(int(index), source["count"] - 1))
    if source["kind"] == "tensor":
        frame = source["frames"][index:index + 1]
        return frame[..., :3].detach().cpu().float()

    # OpenCV commonly decodes HD YUV video with a BT.601 matrix even when the
    # stream is tagged BT.709. That made selected/start PNGs several RGB levels
    # darker than the exact same frame in the accepted MP4, producing a sudden
    # colour shift at the next join. FFmpeg respects the stream's colour tags
    # and matches both the browser preview and our cut encoder.
    command = [
        _find_ffmpeg(), "-hide_banner", "-loglevel", "error",
        "-i", source["path"],
        "-vf", f"select=eq(n\\,{index})",
        "-vsync", "0", "-frames:v", "1",
        "-f", "image2pipe", "-vcodec", "png", "pipe:1",
    ]
    completed = subprocess.run(command, capture_output=True)
    if completed.returncode != 0 or not completed.stdout:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"Could not decode frame {index} with FFmpeg: {detail}")
    from PIL import Image
    with Image.open(io.BytesIO(completed.stdout)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(rgb.copy()).unsqueeze(0)


def _clean_project_name(value):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(value or "Untitled Project")).strip(" ._")
    return name[:100] or "Untitled Project"


def _find_ffmpeg():
    global _ffmpeg_executable
    if _ffmpeg_executable:
        return _ffmpeg_executable
    candidates = []
    executable = shutil.which("ffmpeg")
    if executable:
        candidates.append(executable)
    try:
        import imageio_ffmpeg
        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        pass
    failures = []
    for candidate in dict.fromkeys(candidates):
        try:
            completed = subprocess.run(
                [candidate, "-version"], capture_output=True, text=True, timeout=8
            )
            if completed.returncode == 0:
                _ffmpeg_executable = candidate
                return candidate
            failures.append(f"{candidate}: exit {completed.returncode}")
        except Exception as exc:
            failures.append(f"{candidate}: {exc}")
    detail = "; ".join(failures) if failures else "no executable was found"
    raise RuntimeError(
        "A working FFmpeg executable is required for exact video cuts and sequence previews "
        f"({detail}). Install FFmpeg or run: pip install imageio-ffmpeg"
    )


def _guidance_audio_raw(guidance_audio, project_name):
    """Write Comfy AUDIO samples as temporary interleaved float32 PCM."""
    if not isinstance(guidance_audio, dict) or "waveform" not in guidance_audio:
        raise TypeError("guidance_audio must be a ComfyUI AUDIO value containing waveform and sample_rate.")
    waveform = guidance_audio["waveform"]
    if not torch.is_tensor(waveform):
        raise TypeError("guidance_audio waveform must be a torch tensor.")
    if waveform.ndim == 3:
        waveform = waveform[0]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.ndim != 2:
        raise ValueError(f"guidance_audio waveform must resolve to [channels,samples], got {tuple(waveform.shape)}.")
    sample_rate = int(guidance_audio.get("sample_rate", 0))
    if sample_rate < 1000:
        raise ValueError(f"Invalid guidance_audio sample rate: {sample_rate}.")
    channels = int(waveform.shape[0])
    if channels < 1:
        raise ValueError("guidance_audio contains no channels.")
    pcm = waveform.detach().cpu().float().clamp(-1.0, 1.0).transpose(0, 1).contiguous().numpy()
    folder = Path(folder_paths.get_temp_directory()) / "interactive_frame_selector" / "guidance_audio"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{_clean_project_name(project_name)}_{uuid.uuid4().hex}.f32le"
    with open(path, "wb") as handle:
        handle.write(pcm.astype("<f4", copy=False).tobytes())
    return path, sample_rate, channels


def _audio_value_fingerprint(audio):
    """Small deterministic signature used to invalidate original-audio previews."""
    waveform = audio.get("waveform") if isinstance(audio, dict) else None
    if not torch.is_tensor(waveform):
        return "invalid"
    flat = waveform.detach().cpu().float().reshape(-1)
    if flat.numel() > 4096:
        indices = torch.linspace(0, flat.numel() - 1, 4096).long()
        flat = flat[indices]
    digest = hashlib.sha1()
    digest.update(str(tuple(waveform.shape)).encode())
    digest.update(str(audio.get("sample_rate", 0)).encode())
    digest.update(flat.contiguous().numpy().astype("<f4", copy=False).tobytes())
    return digest.hexdigest()[:12]


def _save_cut(
    source,
    selected,
    project_name,
    video_start_frame=0,
    audio_start_frame=None,
    guidance_audio=None,
):
    """Remove visual overlap while retaining matching fresh audio from time zero."""
    project = _clean_project_name(project_name)
    output_dir = Path(folder_paths.get_output_directory()) / "frame_selector_projects" / project
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_start_frame = max(0, min(int(video_start_frame), int(selected)))
    audio_start_frame = video_start_frame if audio_start_frame is None else max(
        0, min(int(audio_start_frame), int(selected))
    )
    output = output_dir / (
        f"cut_{stamp}_frames_{video_start_frame:06d}-{selected:06d}_{uuid.uuid4().hex[:6]}.mp4"
    )
    video_frame_count = selected - video_start_frame + 1
    output_frame_count = video_frame_count
    duration = output_frame_count / source["fps"]
    raw_audio = None
    guidance_sidecar = output.with_suffix(".guidance.wav")
    command = [_find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", "-i", source["path"]]
    if guidance_audio is not None:
        raw_audio, sample_rate, channels = _guidance_audio_raw(guidance_audio, project_name)
        command.extend((
            "-f", "f32le", "-ar", str(sample_rate), "-ac", str(channels), "-i", str(raw_audio)
        ))
        audio_map = "1:a:0"
    else:
        audio_map = "0:a?"
        print(
            "[Interactive Frame Selector] WARNING: guidance_audio is not connected; "
            "the saved clip will use the model-rendered audio."
        )
    command.extend((
        "-map", "0:v:0", "-map", audio_map,
        "-vf", (
            f"trim=start_frame={video_start_frame}:end_frame={selected + 1},"
            f"setpts=N/({source['fps']:.9f}*TB),"
            f"tpad=stop_mode=clone:stop_duration={1 / source['fps']:.9f},"
            f"trim=end_frame={output_frame_count},"
            f"setpts=N/({source['fps']:.9f}*TB)"
        ),
        "-af", (
            f"atrim=start={audio_start_frame / source['fps']:.9f}:"
            f"end={(audio_start_frame / source['fps']) + duration:.9f},"
            f"asetpts=PTS-STARTPTS,apad,atrim=duration={duration:.9f}"
        ),
        "-frames:v", str(output_frame_count), "-r", f"{source['fps']:.9f}",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
        str(output),
    ))
    try:
        completed = subprocess.run(command, capture_output=True, text=True)
        if completed.returncode == 0 and raw_audio is not None:
            # Keep the accepted source samples losslessly. Project Preview joins
            # these PCM ranges first and performs the only lossy audio encode.
            sidecar_command = [
                _find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "f32le", "-ar", str(sample_rate), "-ac", str(channels),
                "-i", str(raw_audio),
                "-af", (
                    f"atrim=start={audio_start_frame / source['fps']:.9f}:"
                    f"end={(audio_start_frame / source['fps']) + duration:.9f},"
                    "asetpts=PTS-STARTPTS"
                ),
                "-t", f"{duration:.9f}", "-c:a", "pcm_f32le", str(guidance_sidecar),
            ]
            sidecar_completed = subprocess.run(sidecar_command, capture_output=True, text=True)
            if sidecar_completed.returncode != 0:
                raise RuntimeError(
                    "FFmpeg saved the cut but could not preserve its lossless guidance audio: "
                    f"{sidecar_completed.stderr.strip()}"
                )
    finally:
        if raw_audio is not None:
            try:
                raw_audio.unlink(missing_ok=True)
            except OSError:
                pass
    if completed.returncode != 0:
        raise RuntimeError(f"FFmpeg could not create the cut copy: {completed.stderr.strip()}")
    return output


def _selector_display_preview(
    source, start_frame, token, end_frame_exclusive=None,
    guidance_audio=None, project_name="My Project",
):
    """Create a view-only clip whose frame zero is the first selectable frame."""
    start_frame = max(0, min(int(start_frame), int(source["count"]) - 1))
    end_frame_exclusive = int(source["count"]) if end_frame_exclusive is None else max(
        start_frame + 1, min(int(end_frame_exclusive), int(source["count"]))
    )
    display_count = end_frame_exclusive - start_frame
    duration = display_count / float(source["fps"])
    subfolder = "interactive_frame_selector/selection_ranges"
    folder = Path(folder_paths.get_temp_directory()) / subfolder
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"selection_{token}.mp4"
    output = folder / filename
    raw_audio = None
    base = [
        _find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", source["path"],
    ]
    if guidance_audio is not None:
        raw_audio, sample_rate, channels = _guidance_audio_raw(guidance_audio, project_name)
        base.extend((
            "-f", "f32le", "-ar", str(sample_rate), "-ac", str(channels),
            "-i", str(raw_audio),
        ))
        audio_map = "1:a:0"
    else:
        audio_map = "0:a:0"
    base.extend([
        "-map", "0:v:0",
        "-vf", (
            f"trim=start_frame={start_frame}:end_frame={end_frame_exclusive},"
            f"setpts=N/({source['fps']:.9f}*TB),"
            f"tpad=stop_mode=clone:stop_duration={1 / source['fps']:.9f},"
            f"trim=end_frame={display_count},"
            f"setpts=N/({source['fps']:.9f}*TB)"
        ),
        "-frames:v", str(display_count), "-r", f"{source['fps']:.9f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
    ])
    audio_start = start_frame / float(source["fps"])
    with_audio = base + [
        "-map", audio_map,
        "-af", (
            f"atrim=start={audio_start:.9f}:end={audio_start + duration:.9f},"
            f"asetpts=PTS-STARTPTS,apad,atrim=duration={duration:.9f}"
        ),
        "-c:a", "aac", "-movflags", "+faststart", str(output),
    ]
    try:
        completed = subprocess.run(with_audio, capture_output=True, text=True)
        if completed.returncode != 0:
            video_only = base + ["-an", "-movflags", "+faststart", str(output)]
            completed = subprocess.run(video_only, capture_output=True, text=True)
    finally:
        if raw_audio is not None:
            try:
                raw_audio.unlink(missing_ok=True)
            except OSError:
                pass
    if completed.returncode != 0:
        raise RuntimeError(f"FFmpeg could not create the selector display range: {completed.stderr.strip()}")
    return {"filename": filename, "subfolder": subfolder, "type": "temp"}, display_count, start_frame


def _audio_position_path(project_name, create=False):
    project = _clean_project_name(project_name)
    folder = Path(folder_paths.get_output_directory()) / "frame_selector_projects" / project
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    return folder / "audio_position.json"


def _project_state_path(project_name, create=False):
    project = _clean_project_name(project_name)
    folder = Path(folder_paths.get_output_directory()) / "frame_selector_projects" / project
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    return folder / "project_state.json"


def _generation_plan_path(project_name, create=False):
    folder = _project_state_path(project_name, create=create).parent / "latent_chain"
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    return folder / "generation_plan.json"


def _write_generation_plan(project_name, data):
    path = _generation_plan_path(project_name, create=True)
    temporary = path.with_name(f"generation_plan_{uuid.uuid4().hex}.tmp.json")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _active_generation_cut_frames(project_name, source_fps, frame_count):
    path = _project_state_path(project_name).parent / "latent_chain" / "active_generation.json"
    if not path.is_file():
        return 0, 0, int(frame_count)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        audio_24 = max(0, int(data.get("context_frames_24fps", 0)))
        video_24 = max(audio_24, int(data.get("latent_handover_frames_24fps", audio_24)))
        requested_24 = max(0, int(data.get("requested_output_frames_24fps", 0)))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return 0, 0, int(frame_count)
    maximum = max(0, int(frame_count) - 1)
    audio = max(0, min(int(round(audio_24 * float(source_fps) / PROJECT_TIMELINE_FPS)), maximum))
    video = max(0, min(int(round(video_24 * float(source_fps) / PROJECT_TIMELINE_FPS)), maximum))
    # For an off-grid endpoint the Builder anchors the selected image on the
    # first saveable frame. The previous accepted clip already contains that
    # exact endpoint, so retaining it again creates a one-frame join stutter.
    # Drop that duplicate picture only; keep audio at the original handover so
    # the guidance track remains sample-continuous.
    project_state = _read_project_state(project_name) or {}
    residual = int(project_state.get("latest_selection", {}).get("residual_decoded_frames", 0))
    if video > 0 and residual > 1:
        video = min(maximum, video + max(1, int(round(float(source_fps) / PROJECT_TIMELINE_FPS))))
    if requested_24 > 0:
        requested_source = max(
            1, int(round(requested_24 * float(source_fps) / PROJECT_TIMELINE_FPS))
        )
        selection_end = min(int(frame_count), video + requested_source)
    else:
        selection_end = int(frame_count)
    return video, audio, selection_end


def _read_project_state(project_name):
    path = _project_state_path(project_name)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or int(data.get("schema_version", 0)) != 1:
            return None
        return data
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _write_project_state(project_name, state):
    path = _project_state_path(project_name, create=True)
    temporary = path.with_name(f"project_state_{uuid.uuid4().hex}.tmp.json")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def _legacy_audio_seconds(project_name, default=0.0):
    path = _audio_position_path(project_name)
    if not path.is_file():
        return float(default)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return float(data.get("position_seconds", default))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return float(default)


def _read_audio_position(project_name, default=0.0):
    state = _read_project_state(project_name)
    if state is not None and "timeline_frames" in state:
        fps = int(state.get("timeline_fps", PROJECT_TIMELINE_FPS)) or PROJECT_TIMELINE_FPS
        return int(state["timeline_frames"]) / float(fps)
    return _legacy_audio_seconds(project_name, default)


def _write_audio_position(project_name, value):
    with _project_state_lock:
        state = _read_project_state(project_name) or {
            "schema_version": 1,
            "project": _clean_project_name(project_name),
            "timeline_fps": PROJECT_TIMELINE_FPS,
            "timeline_frames": 0,
            "revision": 0,
            "accepted_clips": [],
        }
        state["timeline_fps"] = PROJECT_TIMELINE_FPS
        base_frames = max(0, int(round(float(value) * PROJECT_TIMELINE_FPS)))
        state["timeline_frames"] = base_frames
        state["audio_base_frames"] = base_frames
        state["revision"] = int(state.get("revision", 0)) + 1
        state["updated_at"] = datetime.now().isoformat(timespec="seconds")
        _write_project_state(project_name, state)
        return _write_legacy_audio_position(project_name, state["timeline_frames"] / PROJECT_TIMELINE_FPS)


def _write_legacy_audio_position(project_name, value):
    path = _audio_position_path(project_name, create=True)
    temporary = path.with_name(f"audio_position_{uuid.uuid4().hex}.tmp.json")
    temporary.write_text(json.dumps({"position_seconds": float(value)}, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def _record_accepted_cut(
    project_name, saved_path, selected, source_fps, start_frame=0, video_start_frame=None
):
    """Commit an accepted cut using an integer 24 fps project timeline."""
    file_hash = _sha256_file(saved_path)
    with _project_state_lock:
        state = _read_project_state(project_name)
        if state is None:
            legacy = _legacy_audio_seconds(project_name, 0.0)
            state = {
                "schema_version": 1,
                "project": _clean_project_name(project_name),
                "timeline_fps": PROJECT_TIMELINE_FPS,
                "timeline_frames": max(0, int(round(legacy * PROJECT_TIMELINE_FPS))),
                "audio_base_frames": max(0, int(round(legacy * PROJECT_TIMELINE_FPS))),
                "revision": 0,
                "accepted_clips": [],
            }
        previous_frames = int(state.get("timeline_frames", 0))
        start_frame = max(0, min(int(start_frame), int(selected)))
        video_start_frame = start_frame if video_start_frame is None else max(
            start_frame, min(int(video_start_frame), int(selected))
        )
        # `selected` is an inclusive frame index. Keep project/audio advancement
        # identical to the frame count written by _save_cut (end_frame=selected+1).
        accepted_source_frames = int(selected) - video_start_frame + 1
        advance_frames = max(
            0,
            int(round(accepted_source_frames * PROJECT_TIMELINE_FPS / float(source_fps))),
        )
        next_frames = previous_frames + advance_frames
        revision = int(state.get("revision", 0)) + 1
        try:
            relative_file = str(Path(saved_path).resolve().relative_to(_project_state_path(project_name, True).parent.resolve()))
        except ValueError:
            relative_file = str(Path(saved_path).resolve())
        clips = list(state.get("accepted_clips", []))
        selected_frame_count_24 = max(1, int(round(int(selected) * PROJECT_TIMELINE_FPS / float(source_fps))) + 1)
        if selected_frame_count_24 >= 5:
            latent_boundary_frames = 5 + 17 * ((selected_frame_count_24 - 5) // 17)
        else:
            latent_boundary_frames = 0
        residual_frames = selected_frame_count_24 - latent_boundary_frames
        guidance_source_start = None
        plan_path = _generation_plan_path(project_name)
        if plan_path.is_file():
            try:
                plan = json.loads(plan_path.read_text(encoding="utf-8"))
                guidance_source_start = float(plan["desired_audio_start_seconds"])
            except (OSError, KeyError, ValueError, TypeError, json.JSONDecodeError):
                pass
        clip_record = {
            "revision": revision,
            "file": relative_file,
            "sha256": file_hash,
            "selected_frame": int(selected),
            "saved_start_frame": video_start_frame,
            "audio_start_frame": start_frame,
            "saved_frame_count": int(selected) - video_start_frame + 1,
            "source_fps": float(source_fps),
            "timeline_start_frame": previous_frames,
            "timeline_end_frame": next_frames,
            "accepted_at": datetime.now().isoformat(timespec="seconds"),
        }
        if guidance_source_start is not None:
            clip_record["guidance_source_start_seconds"] = guidance_source_start
        clips.append(clip_record)
        state.update({
            "schema_version": 1,
            "timeline_fps": PROJECT_TIMELINE_FPS,
            "timeline_frames": next_frames,
            "revision": revision,
            "accepted_clips": clips,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "latest_selection": {
                "selection_revision": revision,
                "decision": "accepted",
                "selected_frame_index": int(selected),
                "selected_frame_count_24fps": selected_frame_count_24,
                "selected_time_seconds": int(selected) / float(source_fps),
                "accepted_duration_seconds": accepted_source_frames / float(source_fps),
                "source_fps": float(source_fps),
                "discarded_context_start_frame": start_frame,
                "discarded_visual_context_start_frame": video_start_frame,
                "latent_boundary_frames": latent_boundary_frames,
                "residual_decoded_frames": residual_frames,
                "cut_video_file": relative_file,
            },
        })
        _write_project_state(project_name, state)
        previous = previous_frames / float(PROJECT_TIMELINE_FPS)
        updated = next_frames / float(PROJECT_TIMELINE_FPS)
        _write_legacy_audio_position(project_name, updated)
        return previous, updated


def _record_discard(project_name):
    """Record a rejection and remove latent state that has never been accepted.

    ExecutionBlocker normally prevents the latent-chain node from committing a
    rejected sampler result.  Some Comfy execution orders can nevertheless run
    an output node early, so an empty project's chain is treated as provisional
    until the selector accepts its first clip.
    """
    with _project_state_lock:
        state = _read_project_state(project_name) or {
            "schema_version": 1,
            "project": _clean_project_name(project_name),
            "timeline_fps": PROJECT_TIMELINE_FPS,
            "timeline_frames": 0,
            "revision": 0,
            "accepted_clips": [],
        }
        revision = int(state.get("revision", 0)) + 1
        accepted_clips = list(state.get("accepted_clips", []))
        state.update({
            "schema_version": 1,
            "revision": revision,
            "accepted_clips": accepted_clips,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "last_decision": {
                "decision": "discarded",
                "decision_revision": revision,
                "decided_at": datetime.now().isoformat(timespec="seconds"),
            },
        })
        # With no accepted clip there must be no usable continuation anchor.
        # Preserve latest_selection on established projects so their last
        # accepted visual handover remains available after rejecting a retry.
        if not accepted_clips:
            state.pop("latest_selection", None)
        _write_project_state(project_name, state)

    chain = _project_state_path(project_name, True).parent / "latent_chain"
    if chain.is_dir():
        # This describes only the generation that was just rejected.  The next
        # Builder execution will create a fresh value from accepted history.
        try:
            (chain / "active_generation.json").unlink()
        except FileNotFoundError:
            pass

        if not accepted_clips:
            # These files can only represent the rejected first generation.
            for filename in ("assembled.pt", "latest_segment.pt", "exact_video_context.pt", "chain.json"):
                path = chain / filename
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
    return revision


def _save_selection_anchor(project_name, image, selection_revision):
    from PIL import Image
    folder = _project_state_path(project_name, True).parent / "latent_selections"
    folder.mkdir(parents=True, exist_ok=True)
    filename = f"selection_{int(selection_revision):06d}.png"
    path = folder / filename
    temporary = folder / f"selection_{int(selection_revision):06d}_{uuid.uuid4().hex}.tmp.png"
    rgb = np.clip(image[0, ..., :3].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(rgb, mode="RGB").save(temporary, format="PNG")
    os.replace(temporary, path)
    with _project_state_lock:
        state = _read_project_state(project_name)
        if state and int(state.get("latest_selection", {}).get("selection_revision", -1)) == int(selection_revision):
            state["latest_selection"]["anchor_image_file"] = str(
                path.resolve().relative_to(_project_state_path(project_name, True).parent.resolve())
            )
            _write_project_state(project_name, state)
    return path


@PromptServer.instance.routes.post("/interactive_frame_selector/select")
async def frame_selector_select(request):
    data = await request.json()
    token = str(data.get("token", ""))
    with _pending_lock:
        state = _pending.get(token)
    if state is None:
        return web.json_response({"ok": False, "error": "Selection is no longer pending."}, status=404)
    with state["condition"]:
        state["action"] = str(data.get("action", "select"))
        display_frame = max(0, min(_safe_int(data.get("frame")), state["display_count"] - 1))
        state["frame"] = display_frame + int(state.get("source_frame_offset", 0))
        state["condition"].notify_all()
    return web.json_response({"ok": True})


@PromptServer.instance.routes.get("/interactive_frame_selector/pending")
async def frame_selector_pending(_request):
    with _pending_lock:
        pending = [dict(state["detail"]) for state in _pending.values() if state.get("action") is None]
    return web.json_response({"ok": True, "pending": pending})


class InteractiveFrameSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": (ANY,),
                "fallback_fps": ("FLOAT", {"default": 24.0, "min": 0.01, "max": 240.0, "step": 0.001}),
                "project_name": ("STRING", {"default": "My Project", "multiline": False}),
                "selection_behavior": (["always_pause", "reuse_last_selection"],),
            },
            "optional": {
                "guidance_audio": ("AUDIO",),
                "latent_segment": ("LATENT",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "FLOAT", "FLOAT")
    RETURN_NAMES = ("selected_frame", "selected_time_seconds", "source_selection_time_seconds")
    FUNCTION = "select"
    CATEGORY = "video/interactive"
    DESCRIPTION = "Pauses execution and returns one manually selected frame from a rendered video."
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # The backend handles optional selection reuse; Comfy must execute this node to ask.
        return float("nan")

    def select(
        self,
        video,
        fallback_fps=24.0,
        project_name="My Project",
        selection_behavior="always_pause",
        guidance_audio=None,
        latent_segment=None,
        unique_id=None,
    ):
        token = uuid.uuid4().hex
        source = _source_info(video, fallback_fps, token)
        video_start_frame, audio_start_frame, selection_end_frame = _active_generation_cut_frames(
            project_name, source["fps"], source["count"]
        )
        cache_key = f"{unique_id}:{source['fingerprint']}"
        if selection_behavior == "reuse_last_selection" and cache_key in _last_selection:
            selected = max(
                video_start_frame,
                min(int(_last_selection[cache_key]), selection_end_frame - 1),
            )
            selected_image = _extract_frame(source, selected)
            saved = _save_cut(
                source, selected, project_name, video_start_frame, audio_start_frame, guidance_audio
            )
            previous_audio, next_audio = _record_accepted_cut(
                project_name, saved, selected, source["fps"], audio_start_frame, video_start_frame
            )
            selection_revision = int((_read_project_state(project_name) or {}).get("revision", 0))
            _save_selection_anchor(project_name, selected_image, selection_revision)
            _save_project_start_frame(project_name, selected_image, selection_revision)
            if latent_segment is not None:
                from .h3_latent_tools import commit_accepted_segment
                commit_accepted_segment(latent_segment, project_name)
            print(f"[Interactive Frame Selector] Saved cut copy: {saved}")
            print(f"[Interactive Frame Selector] Audio position: {previous_audio:.3f}s -> {next_audio:.3f}s")
            accepted_seconds = (selected - video_start_frame + 1) / source["fps"]
            return (selected_image, accepted_seconds, selected / source["fps"])

        display_preview, display_count, source_frame_offset = _selector_display_preview(
            source, video_start_frame, token, selection_end_frame,
            guidance_audio, project_name,
        )
        detail = {
            "token": token,
            "node_id": str(unique_id),
            "frame_count": display_count,
            "fps": source["fps"],
            "preview": display_preview,
            "minimum_frame": 0,
        }
        condition = threading.Condition()
        state = {
            "condition": condition,
            "frame": None,
            "action": None,
            "count": source["count"],
            "display_count": display_count,
            "source_frame_offset": source_frame_offset,
            "minimum_frame": video_start_frame,
            "detail": detail,
        }
        with _pending_lock:
            _pending[token] = state

        PromptServer.instance.send_sync("interactive_frame_selector.open", detail)
        print(f"[Interactive Frame Selector] Waiting for a frame choice ({source['count']} frames @ {source['fps']:.3f} fps).")
        try:
            with condition:
                while state["action"] is None:
                    condition.wait(timeout=1.0)
            if state["action"] == "discard":
                _last_selection.pop(cache_key, None)
                discard_revision = _record_discard(project_name)
                print(
                    "[Interactive Frame Selector] Generation discarded; downstream branch blocked "
                    f"and provisional latent state cleared (revision {discard_revision})."
                )
                blocker = ExecutionBlocker(None)
                return (blocker, blocker, blocker)
            if state["action"] == "cancel":
                raise RuntimeError("Frame selection cancelled by user.")
            selected = int(state["frame"])
            _last_selection[cache_key] = selected
            selected_image = _extract_frame(source, selected)
            saved = _save_cut(
                source, selected, project_name, video_start_frame, audio_start_frame, guidance_audio
            )
            previous_audio, next_audio = _record_accepted_cut(
                project_name, saved, selected, source["fps"], audio_start_frame, video_start_frame
            )
            selection_revision = int((_read_project_state(project_name) or {}).get("revision", 0))
            _save_selection_anchor(project_name, selected_image, selection_revision)
            _save_project_start_frame(project_name, selected_image, selection_revision)
            if latent_segment is not None:
                from .h3_latent_tools import commit_accepted_segment
                commit_accepted_segment(latent_segment, project_name)
            print(f"[Interactive Frame Selector] Selected frame {selected} ({selected / source['fps']:.3f}s).")
            print(f"[Interactive Frame Selector] Saved cut copy: {saved}")
            print(f"[Interactive Frame Selector] Audio position: {previous_audio:.3f}s -> {next_audio:.3f}s")
            accepted_seconds = (selected - video_start_frame + 1) / source["fps"]
            return (selected_image, accepted_seconds, selected / source["fps"])
        finally:
            with _pending_lock:
                _pending.pop(token, None)


def _projects_root():
    return Path(folder_paths.get_output_directory()) / "frame_selector_projects"


def _project_choices():
    root = _projects_root()
    if not root.is_dir():
        return ["(no projects found — refresh after creating one)"]
    names = sorted((p.name for p in root.iterdir() if p.is_dir()), key=str.casefold)
    return names or ["(no projects found — refresh after creating one)"]


def _project_videos(project_name):
    root = _projects_root().resolve()
    project = (root / str(project_name)).resolve()
    try:
        project.relative_to(root)
    except ValueError as exc:
        raise ValueError("Invalid project folder.") from exc
    if not project.is_dir():
        raise ValueError(f"Project folder does not exist: {project_name}")
    state_path = project / "project_state.json"
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            state = {}
        accepted = []
        for record in state.get("accepted_clips", []):
            candidate = (project / str(record.get("file", ""))).resolve()
            try:
                candidate.relative_to(project)
            except ValueError:
                continue
            if candidate.is_file():
                accepted.append(candidate)
        if accepted:
            return accepted
    extensions = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
    return sorted(
        (p for p in project.iterdir() if p.is_file() and p.suffix.lower() in extensions),
        key=lambda p: p.name.casefold(),
    )


def _project_video_frame_counts(project_name, files):
    """Return manifest-authoritative frame counts in the same order as files."""
    state = _read_project_state(project_name) or {}
    by_name = {
        Path(str(item.get("file", ""))).name: max(1, int(item.get("saved_frame_count", 1)))
        for item in state.get("accepted_clips", [])
    }
    counts = []
    for path in files:
        recorded = by_name.get(path.name)
        if recorded is None:
            cv2, cap = _open_video(str(path))
            try:
                recorded = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            finally:
                cap.release()
        counts.append(max(1, int(recorded)))
    return counts


def _cfr_video_inputs(frame_counts):
    parts = []
    for index, count in enumerate(frame_counts):
        parts.append(
            f"[{index}:v:0]fps={PROJECT_TIMELINE_FPS},"
            f"setpts=N/({PROJECT_TIMELINE_FPS}*TB),"
            f"tpad=stop_mode=clone:stop_duration={1 / PROJECT_TIMELINE_FPS:.9f},"
            f"trim=end_frame={int(count)},setpts=N/({PROJECT_TIMELINE_FPS}*TB)[v{index}];"
        )
    return "".join(parts)


def _sequence_fingerprint(files):
    digest = hashlib.sha1()
    for path in files:
        stat = path.stat()
        digest.update(f"{path.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
        sidecar = path.with_suffix(".guidance.wav")
        if sidecar.is_file():
            sidecar_stat = sidecar.stat()
            digest.update(
                f"{sidecar.name}\0{sidecar_stat.st_size}\0{sidecar_stat.st_mtime_ns}\n".encode()
            )
    return digest.hexdigest()


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _concat_project_preview(project_name, original_audio=None):
    files = _project_videos(project_name)
    if not files:
        raise ValueError(f"No video files found in project: {project_name}")
    frame_counts = _project_video_frame_counts(project_name, files)
    fingerprint = _sequence_fingerprint(files) + "_" + hashlib.sha1(
        repr(frame_counts).encode("utf-8")
    ).hexdigest()[:8]
    safe_project = re.sub(r"[^A-Za-z0-9._-]+", "_", str(project_name)).strip("._")[:50] or "project"
    subfolder = "interactive_frame_selector/sequences"
    target_dir = Path(folder_paths.get_temp_directory()) / subfolder
    target_dir.mkdir(parents=True, exist_ok=True)
    # v4 can rebuild the soundtrack directly from one original AUDIO value.
    original_signature = _audio_value_fingerprint(original_audio) if original_audio is not None else "sidecars"
    filename = f"sequence_v4_{safe_project}_{fingerprint[:12]}_{original_signature}.mp4"
    output = target_dir / filename
    if output.is_file() and output.stat().st_size > 0:
        return {"filename": filename, "subfolder": subfolder, "type": "temp"}, len(files)

    manifest = target_dir / f"concat_{uuid.uuid4().hex}.txt"
    # FFmpeg concat manifests quote apostrophes using: '\''
    lines = []
    for path in files:
        escaped = path.resolve().as_posix().replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    building = target_dir / f"building_{uuid.uuid4().hex}.mp4"
    guidance_sidecars = [path.with_suffix(".guidance.wav") for path in files]
    has_complete_guidance = all(path.is_file() for path in guidance_sidecars)
    preview_raw_audio = None
    if original_audio is not None:
        total_video_duration = sum(frame_counts) / float(PROJECT_TIMELINE_FPS)
        preview_raw_audio, sample_rate, channels = _guidance_audio_raw(original_audio, project_name)
        inputs = []
        for path in files:
            inputs.extend(("-i", str(path)))
        inputs.extend((
            "-f", "f32le", "-ar", str(sample_rate), "-ac", str(channels),
            "-i", str(preview_raw_audio),
        ))
        video_inputs = _cfr_video_inputs(frame_counts)
        video_concat = "".join(f"[v{index}]" for index in range(len(files)))
        state = _read_project_state(project_name) or {}
        records = {Path(str(item.get("file", ""))).name: item for item in state.get("accepted_clips", [])}
        first_record = records.get(files[0].name, {})
        original_start = max(0.0, float(first_record.get("guidance_source_start_seconds", 0.0)))
        audio_index = len(files)
        # The override is deliberately one extraction, not one extraction per
        # clip. There is therefore no audio boundary at any video join.
        audio_graph = (
            f"[{audio_index}:a:0]atrim=start={original_start:.9f}:"
            f"end={original_start + total_video_duration:.9f},"
            "asetpts=PTS-STARTPTS,apad,"
            f"atrim=end={total_video_duration:.9f}[aout]"
        )
        filter_graph = (
            video_inputs
            + f"{video_concat}concat=n={len(files)}:v=1:a=0[vout];"
            + audio_graph
        )
        command = [
            _find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y", *inputs,
            "-filter_complex", filter_graph,
            "-map", "[vout]", "-map", "[aout]", "-t", f"{total_video_duration:.9f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart", str(building),
        ]
    elif has_complete_guidance:
        total_video_duration = sum(frame_counts) / float(PROJECT_TIMELINE_FPS)
        inputs = []
        for path in files:
            inputs.extend(("-i", str(path)))
        for path in guidance_sidecars:
            inputs.extend(("-i", str(path)))
        video_inputs = _cfr_video_inputs(frame_counts)
        video_concat = "".join(f"[v{index}]" for index in range(len(files)))
        audio_offset = len(files)
        audio_inputs = "".join(
            f"[{audio_offset + index}:a:0]asetpts=PTS-STARTPTS[a{index}];"
            for index in range(len(files))
        )
        audio_concat = "".join(f"[a{index}]" for index in range(len(files)))
        filter_graph = (
            video_inputs
            + f"{video_concat}concat=n={len(files)}:v=1:a=0[vout];"
            + audio_inputs
            + f"{audio_concat}concat=n={len(files)}:v=0:a=1,"
              f"apad,atrim=end={total_video_duration:.9f}[aout]"
        )
        command = [
            _find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
            *inputs,
            "-filter_complex", filter_graph,
            "-map", "[vout]", "-map", "[aout]",
            "-t", f"{total_video_duration:.9f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
            str(building),
        ]
    elif len(files) > 1:
        total_video_duration = sum(frame_counts) / float(PROJECT_TIMELINE_FPS)
        inputs = []
        for path in files:
            inputs.extend(("-i", str(path)))
        video_inputs = _cfr_video_inputs(frame_counts)
        video_concat = "".join(f"[v{index}]" for index in range(len(files)))
        audio_inputs = "".join(f"[{index}:a:0]asetpts=PTS-STARTPTS[a{index}];" for index in range(len(files)))
        audio_chain = ""
        previous = "a0"
        for index in range(1, len(files)):
            output_label = f"ax{index}"
            audio_chain += (
                f"[{previous}][a{index}]"
                f"acrossfade=d=0.005:c1=tri:c2=tri[{output_label}];"
            )
            previous = output_label
        final_audio = "aout"
        filter_graph = (
            video_inputs
            + f"{video_concat}concat=n={len(files)}:v=1:a=0[vout];"
            + audio_inputs
            + audio_chain
            + f"[{previous}]apad,atrim=end={total_video_duration:.9f}[{final_audio}]"
        )
        command = [
            _find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
            *inputs,
            "-filter_complex", filter_graph,
            "-map", "[vout]", "-map", f"[{final_audio}]",
            "-t", f"{total_video_duration:.9f}",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
            str(building),
        ]
    else:
        total_video_duration = frame_counts[0] / float(PROJECT_TIMELINE_FPS)
        command = [
            _find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
            "-i", str(files[0]),
            "-filter_complex", _cfr_video_inputs(frame_counts).rstrip(";"),
            "-map", "[v0]", "-map", "0:a?",
            "-af", (
                f"apad,atrim=duration={total_video_duration:.9f},"
                "asetpts=PTS-STARTPTS"
            ),
            "-frames:v", str(frame_counts[0]), "-r", str(PROJECT_TIMELINE_FPS),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
            str(building),
        ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True)
        if (
            completed.returncode != 0
            and original_audio is None
            and len(files) > 1
            and not has_complete_guidance
        ):
            # Preserve compatibility with silent or heterogeneous third-party
            # videos; those cannot participate in an audio acrossfade.
            fallback = [
                _find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "concat", "-safe", "0", "-i", str(manifest),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-movflags", "+faststart",
                str(building),
            ]
            completed = subprocess.run(fallback, capture_output=True, text=True)
        if completed.returncode != 0:
            raise RuntimeError(f"FFmpeg could not build the project preview: {completed.stderr.strip()}")
        os.replace(building, output)
    finally:
        if preview_raw_audio is not None:
            try:
                preview_raw_audio.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            manifest.unlink(missing_ok=True)
            building.unlink(missing_ok=True)
        except OSError:
            pass
    return {"filename": filename, "subfolder": subfolder, "type": "temp"}, len(files)


class ProjectSequencePreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"project": (_project_choices(),)},
            "optional": {"original_audio": ("AUDIO",)},
        }

    # ANY keeps this usable across ComfyUI versions and third-party video packs,
    # which do not currently share one universal VIDEO wrapper type.
    RETURN_TYPES = (ANY, "STRING")
    RETURN_NAMES = ("video", "file_path")
    FUNCTION = "preview"
    CATEGORY = "video/interactive"
    DESCRIPTION = "Previews every accepted cut in a project folder as one seamless video."
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, project, **_kwargs):
        try:
            return _sequence_fingerprint(_project_videos(project))
        except Exception:
            return float("nan")

    def preview(self, project, original_audio=None):
        preview, count = _concat_project_preview(project, original_audio)
        path = _preview_path(preview)
        # Include the common direct-path, ComfyUI preview and VHS-style shapes.
        # Interactive Frame Selector accepts this bundle directly, while other
        # filename-oriented nodes can use the separate file_path output.
        item = (
            preview["filename"],
            preview.get("subfolder", ""),
            preview.get("type", "temp"),
            "video/mp4",
        )
        compatibility_video = {
            "path": path,
            "file": path,
            "filename": preview["filename"],
            "subfolder": preview.get("subfolder", ""),
            "type": preview.get("type", "temp"),
            "format": "video/mp4",
            "videos": [item],
            "files": [item],
        }
        video = _native_video_from_file(path) or compatibility_video
        print(f"[Project Sequence Preview] Built {project!r} preview from {count} video file(s).")
        return {
            "ui": {"sequence_preview": [preview], "sequence_count": [count]},
            "result": (video, path),
        }


def _video_duration(path):
    cv2, cap = _open_video(str(path))
    try:
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()
    if frames <= 0 or fps <= 0.001:
        raise RuntimeError(f"Could not read duration from project video: {path.name}")
    return frames / fps


class ProjectAudioStartTime:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_name": ("STRING", {"default": "My Project", "multiline": False}),
                "starting_value": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 86400.0, "step": 0.001}),
                "lead_in_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.001}),
                "control": (["continue", "reset_to_starting_value"],),
                "requested_output_seconds": ("FLOAT", {"default": 8.0, "min": 0.208333, "max": 3600.0, "step": 0.001}),
                "requested_context_frames": ("INT", {"default": 39, "min": 5, "max": 3599, "step": 1}),
            }
        }

    RETURN_TYPES = ("FLOAT", "FLOAT")
    RETURN_NAMES = ("audio_start_seconds", "internal_generation_duration")
    FUNCTION = "calculate"
    CATEGORY = "audio/continuation"
    DESCRIPTION = "Plans native H3 audio pre-roll and duration while advancing unchanged user times automatically."

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def calculate(
        self,
        project_name,
        starting_value=0.0,
        requested_output_seconds=8.0,
        requested_context_frames=39,
        lead_in_seconds=0.0,
        control="continue",
    ):
        has_saved_state = (
            _project_state_path(project_name).is_file()
            or _audio_position_path(project_name).is_file()
        )
        force_manual = control == "reset_to_starting_value"
        if force_manual or not has_saved_state:
            _write_audio_position(project_name, 0.0)

        current_value = float(starting_value)
        with _project_state_lock:
            state = _read_project_state(project_name)
            if state is None:
                # Migrate the old project-only counter without folding the
                # current user input into persistent state.
                legacy_frames = max(
                    0,
                    int(round(_legacy_audio_seconds(project_name, 0.0) * PROJECT_TIMELINE_FPS)),
                )
                state = {
                    "schema_version": 1,
                    "project": _clean_project_name(project_name),
                    "timeline_fps": PROJECT_TIMELINE_FPS,
                    "timeline_frames": legacy_frames,
                    "audio_base_frames": 0,
                    "revision": 0,
                    "accepted_clips": [],
                }
                project_frames = legacy_frames
            else:
                timeline_fps = int(state.get("timeline_fps", PROJECT_TIMELINE_FPS)) or PROJECT_TIMELINE_FPS
                timeline_frames = max(0, int(state.get("timeline_frames", 0)))
                if timeline_fps != PROJECT_TIMELINE_FPS:
                    timeline_frames = int(round(timeline_frames * PROJECT_TIMELINE_FPS / timeline_fps))

                if "audio_base_frames" in state:
                    old_base_frames = max(0, int(state["audio_base_frames"]))
                else:
                    clips = list(state.get("accepted_clips", []))
                    old_base_frames = (
                        max(0, int(clips[0].get("timeline_start_frame", 0)))
                        if clips else timeline_frames
                    )
                project_frames = max(0, timeline_frames - old_base_frames)

            has_tracked_value = "audio_user_value" in state
            previous_value = float(state.get("audio_user_value", current_value))
            unchanged = has_tracked_value and abs(current_value - previous_value) <= 0.0005
            # Every first-seen or changed value, including zero, is manual.
            # Automatic compensation begins only when that same value is seen
            # unchanged on a subsequent execution.  An established project
            # with accepted clips is the recovery exception: older state or a
            # crash may leave the newer tracking fields absent, but rerolling
            # must continue from accepted history rather than silently
            # restarting the song at zero.  The explicit reset control remains
            # the way to request a real restart.
            recovered_project = not has_tracked_value and bool(state.get("accepted_clips"))
            automatic = not force_manual and (unchanged or recovered_project)
            if automatic:
                checkpoint_frames = max(0, int(state.get("audio_user_checkpoint_frames", 0)))
                compensation_frames = max(0, project_frames - checkpoint_frames)
            else:
                checkpoint_frames = project_frames
                compensation_frames = 0

            state["audio_user_value"] = current_value
            state["audio_user_checkpoint_frames"] = checkpoint_frames
            state["updated_at"] = datetime.now().isoformat(timespec="seconds")
            _write_project_state(project_name, state)

        compensation_seconds = compensation_frames / float(PROJECT_TIMELINE_FPS)
        combined = current_value + compensation_seconds
        desired_start = max(0.0, combined - float(lead_in_seconds))

        requested_frames = max(1, int(round(float(requested_output_seconds) * PROJECT_TIMELINE_FPS)))
        native_fresh_frames = 5 + 17 * max(0, int(math.ceil((requested_frames - 5) / 17.0)))
        handover_frames = max(5, int(requested_context_frames))
        handover_frames = 5 + 17 * max(0, int(round((handover_frames - 5) / 17.0)))
        # Only the actual copied latent tail is repeated visual/audio context.
        # Earlier builds rounded this 39f handover up to a 51f cut prefix,
        # discarding the first 12 genuinely new transition frames and joining
        # directly onto a later composition. Keep those 12 frames at the end
        # as disposable H3-grid overhead instead.
        prefix_frames = handover_frames
        # Accepted clips are the durable continuation authority. The Builder
        # can reconstruct an exact 39f RGB/VAE handover if a crash, reroll, or
        # older pack version lost the optional latent cache. Do not silently
        # emit a first-run/direct plan merely because those .pt files vanished.
        has_context = bool(state.get("accepted_clips"))
        if has_context and desired_start >= prefix_frames / float(PROJECT_TIMELINE_FPS):
            # The current VDN Comfy acceleration path aborts at the short
            # continuation layouts T42/F42 (141f) and T47/F47 (158f), even
            # when F47 is tested without context. 243f/T72 is the established
            # stable continuation layout, so short extensions render harmless
            # tail overhead which the selector removes after decoding.
            required_frames = native_fresh_frames + prefix_frames
            grid_frames = 5 + 17 * max(
                0, int(math.ceil((required_frames - 5) / 17.0))
            )
            internal_frames = max(grid_frames, 243)
            output = desired_start - prefix_frames / float(PROJECT_TIMELINE_FPS)
            active_prefix = prefix_frames
            active_handover = handover_frames
        else:
            internal_frames = native_fresh_frames
            output = desired_start
            active_prefix = 0
            active_handover = 0

        internal_duration = internal_frames / float(PROJECT_TIMELINE_FPS)
        stability_padding_frames = max(
            0, internal_frames - active_prefix - native_fresh_frames
        )
        _write_generation_plan(project_name, {
            "schema_version": 1,
            "requested_output_frames_24fps": requested_frames,
            "native_fresh_frames_24fps": native_fresh_frames,
            "stability_padding_frames_24fps": stability_padding_frames,
            "internal_generation_frames_24fps": internal_frames,
            "cut_prefix_frames_24fps": active_prefix,
            "latent_handover_frames_24fps": active_handover,
            "desired_audio_start_seconds": desired_start,
            "internal_audio_start_seconds": output,
            "mode": "continuation" if active_prefix else "direct",
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        })
        print(
            f"[Project Audio Start Time] Mode={'automatic' if automatic else 'manual'}, "
            f"user input={current_value:.3f}s, compensation={compensation_seconds:.3f}s, "
            f"desired start={desired_start:.3f}s, internal start={output:.3f}s, "
            f"duration={requested_frames / PROJECT_TIMELINE_FPS:.3f}s "
            f"(native fresh={native_fresh_frames}f) -> {internal_duration:.3f}s, "
            f"pre-roll={active_prefix}f, stability tail={stability_padding_frames}f."
        )
        return (output, internal_duration)


class RunningTotal:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 0.0, "min": -1.0e12, "max": 1.0e12, "step": 0.001}),
                "starting_value": ("FLOAT", {"default": 0.0, "min": -1.0e12, "max": 1.0e12, "step": 0.001}),
                "control": (["continue", "reset_then_add"],),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = ("new_total",)
    FUNCTION = "add"
    CATEGORY = "utils/math"
    DESCRIPTION = "Adds each incoming value to a persistent running total for this node instance."

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # Stateful by design: it must execute once for every queued run.
        return float("nan")

    def add(self, value, starting_value=0.0, control="continue", unique_id=None):
        key = str(unique_id)
        start = float(starting_value)
        incoming = float(value)
        with _running_totals_lock:
            state = _running_totals.get(key)
            should_reset = control == "reset_then_add" or state is None or state["starting_value"] != start
            previous = start if should_reset else state["total"]
            total = previous + incoming
            _running_totals[key] = {"total": total, "starting_value": start}
        print(f"[Running Total] {previous:.6f} + {incoming:.6f} = {total:.6f}")
        return (total,)


def _start_frames_directory(project_name, create=False):
    project = _clean_project_name(project_name)
    folder = Path(folder_paths.get_output_directory()) / "frame_selector_projects" / project / "start_frames"
    if create:
        folder.mkdir(parents=True, exist_ok=True)
    return folder


def _start_frame_files(project_name):
    folder = _start_frames_directory(project_name)
    if not folder.is_dir():
        return []
    numbered = []
    for path in folder.glob("frame_*.png"):
        match = re.fullmatch(r"frame_(\d+)\.png", path.name, flags=re.IGNORECASE)
        if match:
            numbered.append((int(match.group(1)), path))
    return sorted(numbered, key=lambda item: item[0])


def _save_project_start_frame(project_name, image, selection_revision):
    """Save exactly one queue frame for an accepted selector revision."""
    from PIL import Image

    state = _read_project_state(project_name) or {}
    latest = state.get("latest_selection", {})
    if int(latest.get("selection_revision", -1)) != int(selection_revision):
        return None
    number = max(1, len(state.get("accepted_clips", [])))
    folder = _start_frames_directory(project_name, create=True)
    filename = f"frame_{number:06d}.png"
    path = folder / filename
    with _start_frames_lock:
        temporary = folder / f"frame_{number:06d}_{uuid.uuid4().hex}.tmp.png"
        rgb = np.clip(image[0, ..., :3].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
        Image.fromarray(rgb, mode="RGB").save(temporary, format="PNG")
        os.replace(temporary, path)
    with _project_state_lock:
        current = _read_project_state(project_name)
        current_latest = (current or {}).get("latest_selection", {})
        if current and int(current_latest.get("selection_revision", -1)) == int(selection_revision):
            current_latest["start_frame_file"] = str(
                path.resolve().relative_to(_project_state_path(project_name, True).parent.resolve())
            )
            current_latest["color_decode_version"] = 2
            _write_project_state(project_name, current)
    print(f"[Interactive Frame Selector] Saved project start frame: {path}")
    return path


def _repair_latest_project_start_frame(project_name):
    """Recover or colour-correct the latest accepted project queue frame."""
    state = _read_project_state(project_name) or {}
    latest = state.get("latest_selection", {})
    if latest.get("decision") != "accepted":
        return None
    number = len(state.get("accepted_clips", []))
    if number < 1:
        return None
    folder = _start_frames_directory(project_name, create=True)
    target = folder / f"frame_{number:06d}.png"
    if target.is_file() and int(latest.get("color_decode_version", 0)) >= 2:
        return target
    # Migrate pre-v63 anchors through FFmpeg's metadata-aware decoder. Those
    # PNGs were previously produced by OpenCV, which used the wrong YUV matrix
    # for BT.709 files and made the next generation visibly change colour.
    clips = list(state.get("accepted_clips", []))
    if clips:
        record = clips[-1]
        relative_cut = str(record.get("file") or "")
        project_root = _project_state_path(project_name, True).parent.resolve()
        cut_path = (project_root / relative_cut.replace("\\", os.sep)).resolve()
        try:
            cut_path.relative_to(project_root)
        except ValueError:
            cut_path = None
        frame_count = max(1, int(record.get("saved_frame_count", 1)))
        if cut_path is not None and cut_path.is_file():
            corrected = _extract_frame(
                {"kind": "file", "path": str(cut_path), "count": frame_count},
                frame_count - 1,
            )
            revision = int(latest.get("selection_revision", -1))
            _save_selection_anchor(project_name, corrected, revision)
            repaired = _save_project_start_frame(project_name, corrected, revision)
            if repaired is not None:
                print(
                    "[Load Project Start Frame] Migrated accepted endpoint to "
                    f"metadata-correct RGB: {repaired}"
                )
                return repaired
    relative = str(latest.get("anchor_image_file") or "")
    if not relative:
        return None
    project_root = _project_state_path(project_name, True).parent.resolve()
    source = (project_root / relative.replace("\\", os.sep)).resolve()
    try:
        source.relative_to(project_root)
    except ValueError:
        return None
    if not source.is_file():
        return None
    with _start_frames_lock:
        if not target.is_file():
            temporary = folder / f"frame_{number:06d}_{uuid.uuid4().hex}.tmp.png"
            shutil.copyfile(source, temporary)
            os.replace(temporary, target)
    with _project_state_lock:
        current = _read_project_state(project_name)
        if current and int(current.get("latest_selection", {}).get("selection_revision", -1)) == int(
            latest.get("selection_revision", -2)
        ):
            current["latest_selection"]["start_frame_file"] = str(
                target.resolve().relative_to(project_root)
            )
            _write_project_state(project_name, current)
    print(f"[Load Project Start Frame] Repaired missing accepted queue frame: {target}")
    return target


def _pil_to_image_and_mask(path):
    from PIL import Image, ImageOps

    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        if image.mode == "I":
            image = image.point(lambda value: value * (1 / 255))
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        if "A" in image.getbands():
            alpha = np.asarray(image.getchannel("A"), dtype=np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(alpha)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32)
    return torch.from_numpy(rgb).unsqueeze(0), mask.unsqueeze(0)


def _input_image_choices():
    """Build the upload choices directly for compatibility across ComfyUI versions."""
    root = Path(folder_paths.get_input_directory())
    if not root.is_dir():
        return [""]
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
    choices = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(choices, key=str.casefold) or [""]


class SaveProjectStartFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "project_name": ("STRING", {"default": "My Project", "multiline": False}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    CATEGORY = "image/project"
    DESCRIPTION = "Saves one start frame for the next run in the project's start_frames queue."
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def save(self, image, project_name="My Project"):
        from PIL import Image

        state = _read_project_state(project_name) or {}
        recorded = str(state.get("latest_selection", {}).get("start_frame_file") or "")
        if recorded:
            project_root = _project_state_path(project_name, True).parent.resolve()
            existing_path = (project_root / recorded.replace("\\", os.sep)).resolve()
            try:
                existing_path.relative_to(project_root)
            except ValueError:
                existing_path = None
            if existing_path is not None and existing_path.is_file():
                project = _clean_project_name(project_name)
                subfolder = f"frame_selector_projects/{project}/start_frames"
                print(
                    "[Save Project Start Frame] Selector already saved this accepted revision; "
                    f"using: {existing_path}"
                )
                return {"ui": {"images": [{"filename": existing_path.name, "subfolder": subfolder, "type": "output"}]}, "result": ()}

        with _start_frames_lock:
            existing = _start_frame_files(project_name)
            number = (existing[-1][0] + 1) if existing else 1
            folder = _start_frames_directory(project_name, create=True)
            filename = f"frame_{number:06d}.png"
            path = folder / filename
            temporary = folder / f"frame_{number:06d}_{uuid.uuid4().hex}.tmp.png"
            rgb = np.clip(image[0, ..., :3].detach().cpu().numpy() * 255.0, 0, 255).astype(np.uint8)
            Image.fromarray(rgb, mode="RGB").save(temporary, format="PNG")
            os.replace(temporary, path)
        project = _clean_project_name(project_name)
        subfolder = f"frame_selector_projects/{project}/start_frames"
        print(f"[Save Project Start Frame] Saved: {path}")
        return {"ui": {"images": [{"filename": filename, "subfolder": subfolder, "type": "output"}]}, "result": ()}


class LoadProjectStartFrame:
    @classmethod
    def INPUT_TYPES(cls):
        choices = _input_image_choices()
        return {
            "required": {
                "source_mode": (["standard_upload", "project_queue"],),
                "initial_image": (choices, {"image_upload": True}),
                "project_name": ("STRING", {"default": "My Project", "multiline": False}),
                "frame_index": ("INT", {"default": -1, "min": -1000000, "max": 1000000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load"
    CATEGORY = "image/project"
    DESCRIPTION = "Switches between a normal uploaded image and the next frame in a project's saved queue."

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        # The project folder may receive a new frame between workflow runs.
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, initial_image, **_kwargs):
        if not folder_paths.exists_annotated_filepath(initial_image):
            return f"Initial image does not exist: {initial_image}"
        return True

    def load(self, source_mode, initial_image, project_name="My Project", frame_index=-1):
        if source_mode == "project_queue":
            _repair_latest_project_start_frame(project_name)
            with _start_frames_lock:
                files = _start_frame_files(project_name)
                if files:
                    requested = int(frame_index)
                    position = requested if requested >= 0 else len(files) + requested
                    position = max(0, min(position, len(files) - 1))
                    number, path = files[position]
                    print(
                        f"[Load Project Start Frame] Indexed frame {requested} resolved to "
                        f"saved frame {number}: {path}"
                    )
                    return _pil_to_image_and_mask(path)
        path = folder_paths.get_annotated_filepath(initial_image)
        reason = "standard upload mode" if source_mode == "standard_upload" else "project folder is empty"
        print(f"[Load Project Start Frame] Using uploaded image ({reason}): {path}")
        return _pil_to_image_and_mask(path)


NODE_CLASS_MAPPINGS = {
    "InteractiveFrameSelector": InteractiveFrameSelector,
    "ProjectSequencePreview": ProjectSequencePreview,
    "ProjectAudioStartTime": ProjectAudioStartTime,
    "RunningTotal": RunningTotal,
    "SaveProjectStartFrame": SaveProjectStartFrame,
    "LoadProjectStartFrame": LoadProjectStartFrame,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InteractiveFrameSelector": "Interactive Frame Selector",
    "ProjectSequencePreview": "Project Sequence Preview",
    "ProjectAudioStartTime": "Project Audio Start Time",
    "RunningTotal": "Running Total (Accumulator)",
    "SaveProjectStartFrame": "Save Project Start Frame",
    "LoadProjectStartFrame": "Load Project Start Frame",
}
