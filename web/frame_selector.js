import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const css = document.createElement("style");
css.textContent = `
.ifs-shade { position:fixed; inset:0; z-index:100000; display:grid; place-items:center; background:#000b; font:14px system-ui; }
.ifs-panel { width:min(900px,92vw); color:#eee; background:#202124; border:1px solid #555; border-radius:12px; padding:16px; box-shadow:0 20px 70px #000; }
.ifs-panel header { display:flex; justify-content:space-between; align-items:center; font-size:18px; margin-bottom:12px; }
.ifs-panel header small { color:#aaa; font-size:13px; }
.ifs-panel video { display:block; width:100%; max-height:62vh; background:#000; border-radius:7px; }
.ifs-range { width:100%; margin:15px 0; }
.ifs-fields { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.ifs-fields label { display:grid; grid-template-columns:auto 1fr; align-items:center; gap:9px; }
.ifs-fields input { color:#fff; background:#111; border:1px solid #555; border-radius:5px; padding:8px; }
.ifs-actions { display:flex; justify-content:flex-end; gap:10px; margin-top:16px; }
.ifs-actions button { border:0; border-radius:6px; padding:9px 16px; color:#eee; background:#555; cursor:pointer; }
.ifs-actions .ifs-discard { margin-right:auto; background:#991b1b; }
.ifs-actions .ifs-select { color:white; background:#2563eb; font-weight:650; }
.ifs-actions button:disabled { opacity:.5; cursor:wait; }
.ifs-status { min-height:18px; margin-top:10px; color:#fca5a5; font-size:13px; }
`;
document.head.appendChild(css);

function clamp(value, low, high) {
    return Math.max(low, Math.min(high, value));
}

function parseTime(value) {
    const text = String(value).trim();
    if (!text) return NaN;
    const parts = text.split(":").map(Number);
    if (parts.some(Number.isNaN) || parts.length > 3) return NaN;
    if (parts.length === 1) return parts[0];
    if (parts.length === 2) return parts[0] * 60 + parts[1];
    return parts[0] * 3600 + parts[1] * 60 + parts[2];
}

function formatTime(seconds) {
    const ms = Math.max(0, seconds);
    const hours = Math.floor(ms / 3600);
    const minutes = Math.floor((ms % 3600) / 60);
    const secs = (ms % 60).toFixed(3).padStart(6, "0");
    return hours ? `${hours}:${String(minutes).padStart(2, "0")}:${secs}` : `${minutes}:${secs}`;
}

// Keep ComfyUI's other video widgets (including widgets in sibling tabs) from
// talking over the interactive selector. BroadcastChannel is scoped to the
// same ComfyUI origin, so separate portable installs/servers are unaffected.
const audioChannel = typeof BroadcastChannel !== "undefined"
    ? new BroadcastChannel("interactive-frame-selector-audio-v1")
    : null;
const audioLocks = new Set();
const resolvedSelectorTokens = new Set();
let lockedVideos = new Map();
let videoObserver = null;

function silenceVideo(video, selectorVideo = null) {
    if (!(video instanceof HTMLVideoElement) || video === selectorVideo) return;
    if (!lockedVideos.has(video)) lockedVideos.set(video, {muted: video.muted});
    video.pause();
    video.muted = true;
}

function applyAudioLock(lockId, selectorVideo = null) {
    audioLocks.add(lockId);
    document.querySelectorAll("video").forEach(video => silenceVideo(video, selectorVideo));
    if (!videoObserver) {
        videoObserver = new MutationObserver(() => {
            if (!audioLocks.size) return;
            const activeSelector = document.querySelector(".ifs-shade video");
            document.querySelectorAll("video").forEach(video => silenceVideo(video, activeSelector));
        });
        videoObserver.observe(document.documentElement, {childList: true, subtree: true});
    }
}

function releaseAudioLock(lockId) {
    audioLocks.delete(lockId);
    if (audioLocks.size) return;
    videoObserver?.disconnect();
    videoObserver = null;
    for (const [video, state] of lockedVideos) {
        if (video.isConnected) video.muted = state.muted;
    }
    lockedVideos = new Map();
}

audioChannel?.addEventListener("message", event => {
    const {action, lockId, token} = event.data || {};
    if (action === "resolved" && token) {
        resolvedSelectorTokens.add(String(token));
        return;
    }
    if (!lockId) return;
    if (action === "lock") applyAudioLock(lockId);
    if (action === "release") releaseAudioLock(lockId);
});

function openSelector(detail) {
    const count = Math.max(1, Number(detail.frame_count));
    const fps = Math.max(0.001, Number(detail.fps));
    const minimumFrame = clamp(Math.round(Number(detail.minimum_frame) || 0), 0, count - 1);
    let frame = 0;

    const shade = document.createElement("div");
    shade.className = "ifs-shade";
    shade.dataset.token = String(detail.token);
    shade.innerHTML = `
      <section class="ifs-panel" role="dialog" aria-modal="true">
        <header>Interactive Frame Selector <small>${count} frames · ${fps.toFixed(3)} fps</small></header>
        <video controls preload="auto"></video>
        <input class="ifs-range" type="range" min="0" max="${count - 1}" value="0" step="1">
        <div class="ifs-fields">
          <label>Frame <input class="ifs-frame" type="number" min="0" max="${count - 1}" value="0"></label>
          <label>Time <input class="ifs-time" type="text" value="${formatTime(0)}"></label>
        </div>
        <div class="ifs-actions">
          <button class="ifs-discard">Discard Generation</button>
          <button class="ifs-cancel">Cancel</button>
          <button class="ifs-select">Select This Frame</button>
        </div>
        <div class="ifs-status" role="alert"></div>
      </section>`;
    document.body.appendChild(shade);

    const video = shade.querySelector("video");
    const audioLockId = `${detail.token}-${Date.now()}`;
    applyAudioLock(audioLockId, video);
    audioChannel?.postMessage({action: "lock", lockId: audioLockId});
    video.muted = false;
    const range = shade.querySelector(".ifs-range");
    const frameInput = shade.querySelector(".ifs-frame");
    const timeInput = shade.querySelector(".ifs-time");
    const status = shade.querySelector(".ifs-status");
    const selectButton = shade.querySelector(".ifs-select");
    const preview = detail.preview;
    video.src = api.apiURL(`/view?filename=${encodeURIComponent(preview.filename)}&subfolder=${encodeURIComponent(preview.subfolder || "")}&type=${encodeURIComponent(preview.type || "temp")}`);

    const update = (next, seek = true) => {
        frame = clamp(Math.round(Number(next) || 0), 0, count - 1);
        range.value = frame;
        frameInput.value = frame;
        timeInput.value = formatTime(frame / fps);
        const protectedFrame = frame < minimumFrame;
        selectButton.disabled = protectedFrame;
        status.textContent = protectedFrame
            ? `Preview-only pre-roll. Select frame ${minimumFrame} or later.`
            : "";
        // Seek to the middle of the requested frame's display interval.  A
        // seek to the final frame's opening timestamp can be treated as EOF
        // by browsers when an MP4 lacks a terminal packet duration.
        if (seek && Number.isFinite(video.duration)) {
            video.currentTime = Math.min((frame + 0.5) / fps, Math.max(0, video.duration - 0.001));
        }
    };
    range.addEventListener("input", () => update(range.value));
    frameInput.addEventListener("change", () => update(frameInput.value));
    frameInput.addEventListener("keydown", e => { if (e.key === "Enter") update(frameInput.value); });
    timeInput.addEventListener("change", () => {
        const seconds = parseTime(timeInput.value);
        if (Number.isFinite(seconds)) update(seconds * fps);
        else timeInput.value = formatTime(frame / fps);
    });
    timeInput.addEventListener("keydown", e => { if (e.key === "Enter") timeInput.dispatchEvent(new Event("change")); });
    video.addEventListener("timeupdate", () => {
        if (!video.seeking && !range.matches(":active")) update(video.currentTime * fps, false);
    });
    shade.addEventListener("keydown", e => {
        if (e.key === "ArrowLeft") { e.preventDefault(); update(frame - 1); }
        if (e.key === "ArrowRight") { e.preventDefault(); update(frame + 1); }
    });

    const finish = async action => {
        shade.querySelectorAll("button").forEach(b => b.disabled = true);
        status.textContent = "Sending choice…";
        try {
            const controller = new AbortController();
            const timeout = setTimeout(() => controller.abort(), 15000);
            let response;
            try {
                response = await api.fetchApi("/interactive_frame_selector/select", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({token: detail.token, frame, action}),
                    signal: controller.signal,
                });
            } finally {
                clearTimeout(timeout);
            }
            if (!response.ok) {
                let message = `ComfyUI rejected the choice (${response.status}).`;
                try {
                    const data = await response.json();
                    if (data?.error) message = data.error;
                } catch (_) {}
                throw new Error(message);
            }
            video.pause();
            shade.remove();
            releaseAudioLock(audioLockId);
            audioChannel?.postMessage({action: "release", lockId: audioLockId});
            resolvedSelectorTokens.add(String(detail.token));
            audioChannel?.postMessage({action: "resolved", token: String(detail.token)});
        } catch (error) {
            status.textContent = error?.name === "AbortError"
                ? "ComfyUI did not answer within 15 seconds. Check the server console, then retry."
                : `Could not submit choice: ${error?.message || error}`;
            shade.querySelectorAll("button").forEach(b => b.disabled = false);
        }
    };
    selectButton.onclick = () => finish("select");
    shade.querySelector(".ifs-discard").onclick = () => finish("discard");
    shade.querySelector(".ifs-cancel").onclick = () => finish("cancel");
    update(0);
    frameInput.focus();
}

app.registerExtension({
    name: "OpenAI.InteractiveFrameSelector",
    setup() {
        const waiting = new Map();
        const showWhenVisible = detail => {
            if (!detail?.token || resolvedSelectorTokens.has(String(detail.token))) return;
            if (document.querySelector(`.ifs-shade[data-token="${CSS.escape(String(detail.token))}"]`)) return;
            if (document.visibilityState !== "visible") {
                waiting.set(String(detail.token), detail);
                return;
            }
            waiting.delete(String(detail.token));
            openSelector(detail);
        };
        api.addEventListener("interactive_frame_selector.open", event => showWhenVisible(event.detail));
        const recoverPending = async () => {
            try {
                const response = await api.fetchApi("/interactive_frame_selector/pending");
                if (!response.ok) return;
                const data = await response.json();
                for (const detail of data?.pending || []) showWhenVisible(detail);
            } catch (error) {
                console.warn("Interactive Frame Selector: pending review recovery failed", error);
            }
        };
        recoverPending();
        document.addEventListener("visibilitychange", () => {
            if (document.visibilityState !== "visible") return;
            for (const detail of waiting.values()) showWhenVisible(detail);
            recoverPending();
        });
    },
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "ProjectSequencePreview") return;
        const originalExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function(message) {
            originalExecuted?.apply(this, arguments);
            const info = message?.sequence_preview?.[0];
            if (!info) return;
            if (!this.sequenceVideo) {
                const container = document.createElement("div");
                container.style.width = "100%";
                container.style.background = "#000";
                container.style.borderRadius = "7px";
                container.style.overflow = "hidden";
                const video = document.createElement("video");
                video.controls = true;
                video.preload = "metadata";
                video.style.width = "100%";
                video.style.display = "block";
                container.appendChild(video);
                this.addDOMWidget("sequence_preview", "preview", container, {
                    serialize: false,
                    getMinHeight: () => 260,
                });
                this.sequenceVideo = video;
                this.setSize([Math.max(this.size[0], 520), Math.max(this.size[1], 390)]);
            }
            this.sequenceVideo.src = api.apiURL(`/view?filename=${encodeURIComponent(info.filename)}&subfolder=${encodeURIComponent(info.subfolder || "")}&type=${encodeURIComponent(info.type || "temp")}&t=${Date.now()}`);
            this.sequenceVideo.load();
            this.setDirtyCanvas(true, true);
        };
    },
});
