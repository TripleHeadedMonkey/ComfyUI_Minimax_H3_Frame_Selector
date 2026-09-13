# ComfyUI Interactive Frame Selector

## v59 maintenance correction

- Off-grid accepted endpoints with more than one residual frame now get an exact cached continuation tail automatically. The proven one-frame-tolerance path remains unchanged. For larger residuals, the next Builder run loads the final 39 frames of that accepted MP4, resizes them to the connected latent canvas (standard and custom 2× decode sizes are both handled), re-encodes them with the already-connected video VAE, and uses that exact tail instead of a sampler tail ending several frames early. The cache is keyed to the accepted selection revision. On-grid and one-frame-residual selections keep the original fast latent-only path.

## v58 maintenance correction

- Continuation pre-roll now equals the actual accepted latent handover (`39` frames by default), rather than being incorrectly rounded up to `51`. The extra 12 frames are retained as disposable H3-grid overhead at the end of the render. Previously, the selector removed those first 12 genuinely new transition frames and joined directly onto a later composition, producing an immediate positional jump at every saved-video boundary. Audio now begins 39 frames early and is cut at the same point, preserving the working H3 guidance alignment and exact requested output duration.

## v57 maintenance correction

- Restored the pre-v51 two-anchor conditioning layout: the user's existing frame-0 guide is preserved and the hidden-pre-roll endpoint anchor is appended. Relocating the original guide changed MiniMax H3's joint audiovisual solution and could make a correctly loaded continuation slice generate unrelated audio. The unsafe latent freeze remains removed. Frame-exact cutting, project manifests, discard safety, and v54 crash recovery remain active.

## v56 maintenance correction

- Removed the experimental `freeze_audio_latent` control. The combined H3 node uses guide audio as conditioning; its sampler target audio latent is not a directly decodable copy of that song. Freezing the target therefore produced a continuous beep rather than preserved guidance. Continuation audio behavior is restored to the pre-v53 path, while the v54 crash-recovery and v51 visual-anchor corrections remain.

## v55 maintenance correction

- **MiniMax H3 Continuation Builder** now exposes `freeze_audio_latent`. It defaults to `true`: the supplied music latent remains exact but still drives visual timing. Set it to `false` to restore the legacy behavior where H3 can regenerate, alter, or add to the audio. The toggle affects continuation runs only; the first project generation still passes through normally because no continuation context exists.

## v54 maintenance correction

- Crash/restart recovery now treats any project with accepted clips as an automatic audio continuation even if its newer `audio_user_value` tracking fields are missing. Previously, the first rerun of such a recovered/older project could interpret the unchanged `0` as a new manual start and replay the song from the beginning. Use `reset_to_starting_value` when a genuine restart is intended.

## v53 maintenance correction

- On continuation runs the Builder now freezes the complete fresh guide-audio latent (`audio noise_mask = 0`) while leaving it present for multimodal visual timing. The planner and audio loader were supplying the correct offset, but the previous Builder allowed H3/VDN to regenerate that audio during joint sampling; a visual-conditioning change could therefore make the embedded soundtrack restart. First runs retain their existing behavior because no continuation Builder context is active yet.

## v52 maintenance correction

- The interactive selector preview now uses its connected `guidance_audio` as the authoritative soundtrack and removes the same hidden continuation prefix from both audio and video. Previously, the selector UI always played the rendered video's model-audio stream even when the accepted save path was configured to use guidance audio, making a correctly planned continuation sound restarted or altered during selection.

## v51 maintenance correction

- Continuations no longer condition the saved project endpoint twice. When the same image is already present through `first_frame` at frame 0, the Builder relocates that guide to the final hidden pre-roll frame instead of appending another copy. Unrelated start/end/reference guides are preserved. This prevents the accepted endpoint from overwriting the beginning of the latent tail and then visibly correcting near the cut.

## v50 maintenance correction

- The exact selected endpoint anchor is now placed on the final hidden pre-roll frame. It is no longer delayed by `residual_decoded_frames`, which caused the visible correction jump by exactly that residual count.
- Saved cuts and selector previews use explicit constant-frame-rate timestamps and preserve the complete duration of their final frame.
- Project Sequence Preview follows `project_state.json` accepted-clip order, restores each recorded `saved_frame_count`, and no longer includes unrelated MP4 files found in the project directory.

A blocking frame picker for rendered video. When execution reaches the node, a modal video timeline opens in the ComfyUI browser. Scrub the video or enter a frame number/time, then press **Select This Frame**. The node saves a cut copy through the selected frame and resumes with exactly one `IMAGE` frame. **Discard Generation** finishes this branch without sending an image downstream and does not save a cut. When the sampler output is connected to optional `latent_segment`, the selector itself commits the accepted latent chain in the same decision. No provisional generation latent is stored, so discard or a crash leaves the last accepted tail untouched.

For latent continuations, Project Audio Start Time plans the longer sequence before the joint H3 latent is created. Connect `audio_start_seconds` to the audio loader's start-time input and `internal_generation_duration` to its duration input. On a continuation, an 8-second request becomes a native 10.125-second/243-frame audio selection beginning 1.625 seconds earlier. Builder protects the first 39 visual frames as copied context. Selector removes those 39 frames and saves the requested 192 frames/8 seconds; the remaining 12-frame H3-grid overhead is discarded from the end. The picker hides both regions and displays only the exact selectable 192-frame range as frame 0 onward; choices are translated back to source-frame coordinates internally. The Builder reads visual context from `latest_segment.pt`, avoiding cumulative handover drift.

Requested output duration is frame-exact at 24 fps, while H3 generation always rounds upward to its required `17k+5` frame grid. Short continuations are additionally padded to the proven stable 243-frame/`T72` VDN layout because the tested VDN Comfy build aborts at the intermediate 141-frame/`T42` and 158-frame/`T47` layouts. Thus 4 seconds requests 96 saved frames but renders 243 frames internally: 39 context frames, 96 accepted frames and 108 disposable tail frames. An 8-second request is 39 + 192 + 12 frames. A 16-second request renders 447 frames and saves 384. The selector hides and discards both the opening context and trailing grid/stability overhead, so it shows and saves exactly the user-requested duration.

Connect the exact same sliced `AUDIO` output from the audio loader both to the MiniMax guidance path and to Interactive Frame Selector's optional `guidance_audio` input. When connected, the selector discards the model-rendered audio and sample-accurately trims/muxes that guidance audio onto the accepted video range. It also stores that accepted guidance range as lossless float PCM beside the cut. Project Sequence Preview concatenates those lossless ranges without fades or overlaps and performs one AAC encode only after the complete soundtrack is assembled. Thus the saved soundtrack is the exact source section that drove each generation, without a lossy codec boundary at every join. If `guidance_audio` is disconnected, the selector retains backward-compatible model-audio saving and prints a warning.

Project Sequence Preview also accepts an optional `original_audio` input. Connect the full, unsliced source track to override every generated or embedded clip soundtrack. Preview reads the project's first recorded source-audio position and makes one continuous extraction for the entire joined-video duration, followed by one final AAC encode. It never splits or rejoins the override audio at video boundaries, so later joins cannot skip a beat. Older projects without a recorded opening position begin at zero.

While the selector is open, other ComfyUI video previews are paused and muted across browser tabs connected to the same server. This prevents overlapping audio; background previews stay paused after the selector closes.

## Install

1. Extract/copy `comfyui-interactive-frame-selector` into `ComfyUI/custom_nodes/`.
2. With ComfyUI's Python environment active, run `pip install -r requirements.txt` from this folder.
3. Restart ComfyUI and hard-refresh the browser (`Ctrl+F5`).

Find the node under **video → interactive → Interactive Frame Selector**.

The same category also contains **Project Sequence Preview**.

## Inputs and output

- `video`: accepts an `IMAGE` frame batch, a direct local video path, and common VHS/video-combine filename bundles.
- `fallback_fps`: used for frame batches or files without readable FPS metadata.
- `project_name`: names the folder receiving accepted video cuts.
- `selection_behavior`:
  - `always_pause` opens the picker on every execution.
  - `reuse_last_selection` reuses the last choice while the same source and node remain unchanged.
- Outputs:
  - `selected_frame` — one RGB `IMAGE`, shape `[1, H, W, 3]`.
  - `selected_time_seconds` — accepted new duration as a `FLOAT`; excludes any automatically removed visual-context prefix and is the value to use for audio advancement or `commit_signal`.
  - `source_selection_time_seconds` — the selected frame's raw timestamp in the rendered source, including its visual-context prefix.

Accepted cuts are written to:

`ComfyUI/output/frame_selector_projects/<project_name>/`

On the first generation the cut contains frame `0` through the selected frame, inclusive. When the visual-latent Builder supplied continuation context, the selector reads its exact length from the project automatically and excludes that repeated opening prefix from the video. The newly generated audio starts at its own time zero and is trimmed to the accepted video duration. It is re-encoded with FFmpeg so both video cuts occur on exact frames rather than existing keyframes.

Put **Project Audio Start Time** before the generator and give it the same `project_name` as the selector. It does not inspect saved videos. Every first-seen or changed `starting_value`, including `0`, is manual: the node uses the entered time directly (minus any lead-in) and records a new project checkpoint. If the value remains unchanged on the next run, the node returns to automatic mode and adds only the accepted project duration accumulated since that checkpoint. This allows an upstream value to switch or cycle between song sections without an old project offset being imposed on the first run of the new section. Discarded generations add no duration.

`lead_in_seconds` subtracts overlap from the output without changing the stored total. `reset_to_starting_value` resets the project timeline; switch it back to `continue` afterward. The selector advances the stored position only by the new portion after any automatically removed continuation prefix. The authoritative position is stored as an integer 24 fps frame count in `project_state.json`, so repeated extensions do not accumulate floating-point timing drift. That file also records each accepted clip, its timeline range, revision, and SHA-256 hash. `audio_position.json` remains as a compatibility mirror for existing workflows. Both survive ComfyUI restarts. Discarding does not advance either file.

## Running Total (Accumulator)

This stateful math node is under **utils → math**. On its first execution it returns `starting_value + value`. Each later execution adds the new `value` to its existing total and outputs the result. It is deliberately forced to execute on every queued run rather than being cached.

- `starting_value` defaults to `0`. Changing it reinitializes the total before adding the current input.
- `control: continue` adds normally.
- `control: reset_then_add` resets to `starting_value` and then adds the current input. Switch it back to `continue` afterward; leaving it selected resets on every run.
- State belongs to the individual node and lasts until ComfyUI restarts. Deleting/recreating the node also starts a fresh total.

For a recycled audio workflow, connect `selected_time_seconds` into `value`; `new_total` is then the accumulated time position. A discarded generation blocks the selector's time output, so it is not added.

## Project start-frame queue

Two nodes under **image → project** provide the image handoff between completed runs:

- **Load Project Start Frame** has the same upload/dropdown behavior as a normal image loader plus a two-way `source_mode` toggle. `standard_upload` uses the manually selected image. `project_queue` uses the separate `frame_index` value to select a chronological PNG from `project_name`: index `0` is the oldest, `1` is the second, and `-1` is the newest. Out-of-range indices clamp to the nearest end. If the project folder is empty, it falls back to the uploaded image. It returns `IMAGE` plus `MASK`.
- **Save Project Start Frame** remains available as a compatibility/manual output node. Interactive Frame Selector now saves the accepted start frame transactionally itself as `frame_000001.png`, `frame_000002.png`, and so on inside `ComfyUI/output/frame_selector_projects/<project>/start_frames/`. If the Save node is still connected, it detects that revision and does not create a duplicate.

The default `frame_index` is `-1`, so normal repeated runs always use the newest saved frame. After a discard, the newest frame is unchanged and is therefore reused automatically. Changing the index lets you scroll through earlier scenes without any third mode or cursor state. Give the related nodes exactly the same `project_name`. On loading an older project whose queue entry was missed by a downstream Save node, Load Project Start Frame repairs it from the selector's committed selection image.

Frame numbering is zero-based: the first frame is frame `0`. Time input accepts seconds, `MM:SS.mmm`, or `HH:MM:SS.mmm`. The chosen frame number is authoritative.

## Notes

- If the browser is refreshed while the server is still waiting, the selector is restored automatically. A ComfyUI/server restart still cancels the active execution.
- The modal does not freeze the web server; it only waits inside the current workflow execution.
- If a private node pack uses an unrecognized video wrapper, connect that renderer's filename/VHS output or its decoded `IMAGE` batch.
- FFmpeg is tested by launching it before use; a stale or non-executable path now produces a direct installation error instead of failing later during the cut.

## Project Sequence Preview

This output node lists the folders under `ComfyUI/output/frame_selector_projects/` in a dropdown. Run it to concatenate every video in the selected folder by filename order and display the complete sequence as one video inside the node. The combined preview is cached in ComfyUI's temporary directory and is rebuilt automatically when a source video is added, replaced, or changed.

It also exposes two workflow outputs:

- `video` — a native file-backed ComfyUI `VIDEO` on current builds. Older/portable builds automatically receive a permissive direct-path/VHS compatibility bundle instead. Either form connects directly to **Interactive Frame Selector → video**.
- `file_path` — the absolute path to the combined MP4 for nodes that accept a filename/string instead of a video bundle.

Use **Refresh Node Definitions** (the same refresh used after adding models/nodes) to repopulate the project dropdown after making a new project folder.

## MiniMax H3 Scoped Offset Anchor

This node is under **model → conditioning → minimax**. It places one continuation image at an exact non-zero frame without changing the normal behavior of unrelated H3 workflows.

Connect `positive`, `latent`, and `vae` from **MiniMaxH3ReferenceToVideo**, connect the continuation `image`, and choose a zero-based `frame_index` (default `12`). Send its `positive` and pass-through `latent` outputs through any normal **MiniMaxH3AddGuide** nodes and finally into **BasicGuider**. The index is clamped to the generated frame count. `offset_frames` reports the resolved index and `offset_seconds` converts it at H3's fixed 24 fps; connect `offset_seconds` to **Project Audio Start Time → lead_in_seconds** so audio begins early by exactly the regenerated overlap.

On current ComfyUI builds the node detects native **MiniMaxH3AddGuide** support and does not patch core at all. The compatibility wrapper exists only for older builds and activates only when conditioning contains this node's private marker. Native X2/X4 AddGuide workflows take the original ComfyUI core path unchanged. Do not install the separate global Keyframe Offset extension alongside this package, because its global patch would run before or around this scoped wrapper.

## MiniMax H3 latent continuation

Three optional nodes under **MiniMax H3 → continuation** preserve H3's joint video/audio state instead of reducing a continuation to one RGB frame.

### MiniMax H3 Continuation Tail

Connect the sampler-output H3 `LATENT`, the video VAE, and the audio VAE. The node resolves `requested_context_frames` to the nearest boundary that is valid for both H3's `17k+5` video grid and its exact 40 Hz audio-latent clock. The usable shared boundaries are `39`, `90`, `141`, and so on; `39` is the default.

Outputs are the joint `context_latent`, decoded `context_frames`, matching `context_audio`, the final decoded frame, and the resolved duration. Decoded spatial size is controlled entirely by the connected VAE: the standard H3 VAE returns standard-size images, while a compatible custom 2× VAE returns 2× images. Latent timing and slicing are unchanged.

### MiniMax H3 Continuation Builder

This is the bridge between an accepted previous latent and the next sampler. Connect the fresh next-generation `positive` conditioning and latent produced by **MiniMaxH3ReferenceToVideo** (or the equivalent H3 setup node) to the Builder, plus the video VAE. Connect Builder `positive` to the guider and `sampler_latent` to the sampler's latent input.

For the reusable one-workflow loop, use `source_mode: project_chain` and the same `project_name` as **Latent Chain** and **Interactive Frame Selector**. On the first run no project chain exists, so the fresh conditioning and target pass through unchanged. After the first accepted result is committed, later runs automatically load its video-latent tail, place it at the beginning of the fresh target, and preserve only its video mask. The fresh target audio samples and audio mask pass through unchanged, leaving song continuity entirely under the existing Project Audio Start Time/audio-input workflow. The selector's exact chosen image is also loaded from the project and inserted as a conditioning anchor at the handoff. Exact-grid selections (and the harmless initial one-frame residual) anchor the last hidden context frame. When cumulative off-grid rounding is larger, the accepted RGB tail is re-encoded and the selected endpoint anchors the first saveable frame, preventing the next extension from opening on a different view before correcting. `anchor_frame_index` reports where it was placed. `requested_context_frames` must match `handover_frames` on Latent Chain; `39` is the recommended starting value.

`connected_context` is available for a manually expanded workflow: connect `context_latent` from **Continuation Tail** directly. It is not needed for the reusable project loop.

### MiniMax H3 Latent Chain

Connect each completed sampler-output continuation latent to `segment` and use the same `project_name` on successive runs. The first run creates:

`ComfyUI/output/frame_selector_projects/<project>/latent_chain/`

After selection, the node reads `project_state.json` automatically and truncates the sampled latent to the last sliceable H3 boundary at or before the exact selected frame. Later runs load the assembled chain, remove the repeated joint AV prefix specified by `handover_frames`, append only the new latent region, and update `assembled.pt` plus `chain.json`. Audio length is recomputed against the absolute 24 fps project timeline so independent rounding at each 40 Hz seam cannot accumulate drift. Each selector decision has a revision, preventing the same accepted segment from being appended twice.

- `append_or_start` starts automatically when no chain exists and appends thereafter.
- `reset_and_start` replaces the stored chain with the incoming segment; switch it back after starting a new scene.
- Connect Interactive Frame Selector's `selected_frame` to the required `selector_commit_signal`. This forces the latent to be committed only after **Select This Frame**; **Discard Generation** blocks the commit. The required connection also prevents ComfyUI from scheduling Latent Chain ahead of the selector and accidentally leaving an older segment as the next continuation source.
- Use the same handover length that created the H3 continuation. Independent clips are not safe to concatenate in latent space.
- `current_segment` is the untouched sampler output for the normal per-run VAE decode, Save Video and Interactive Frame Selector path. It remains the original generated duration.
- `full_visual_project_latent` grows as accepted sections are accumulated and is only for an intentional full-project **video** decode. Do not connect it to the ordinary per-run preview or audio decode. Its audio member merely keeps the joint H3 structure valid; the original song remains authoritative. The stored tensors can be large because they preserve the full working visual latent rather than an MP4.
- The Builder never reuses the chain's generated audio. For music workflows, continue using the original song and the existing project audio-offset path as the authoritative audio source.

For the normal interactive workflow, the standalone Latent Chain node is no longer required. Connect the sampler output directly to Interactive Frame Selector's optional `latent_segment`. Pressing **Select This Frame** commits the latent synchronously; **Discard Generation**, Cancel, or a crash commits nothing. The standalone node remains available for non-interactive/manual chain workflows.

Reusable interactive loop wiring:

1. Fresh H3 setup conditioning and latent → **Continuation Builder `positive` and `target_latent`**; connect the video VAE too.
2. Builder `positive` → guider and `sampler_latent` → sampler latent input.
3. Sampler output latent → the normal VAE/video rendering path **and** → **Interactive Frame Selector `latent_segment`**.
4. Rendered video → **Interactive Frame Selector `video`**.
5. Use the same project name on Builder and Selector. The selector reads the planned handover automatically when it commits.
