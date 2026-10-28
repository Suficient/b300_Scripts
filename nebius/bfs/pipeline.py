"""The BFS LTX-2.5 v1.1 head-swap workflow as a ComfyUI API prompt.

Node ids, settings and wiring follow workflows/workflow_ltx2.5_v1.1_head_swap_drag_and_drop.json
(Alissonerdx/BFS-Best-Face-Swap-Video @ d0adb53). Changes from the drag-and-drop file:
  - bf16 transformer and text encoder instead of int8 (the B300 has little int8 throughput)
  - prompt comes from the pair's caption.txt instead of the Qwen3-VL auto prompt
  - guide video and face image load from paths; output size, fps and length follow the clip
  - only the second-pass result is saved (no first-pass video, no comparison grid)
The pixel-spatial upscaler LoRA (bypassed in the drag-and-drop file) is left out.
"""
import json
import math
import subprocess
from fractions import Fraction

UNET = "ltx-2.5-22b-distilled-transformer-bf16.safetensors"
TEXT_ENCODER = "gemma4-12b-with-proj-ltx-2.5-bf16.safetensors"
VIDEO_VAE = "ltx-2.5-video-vae-bf16.safetensors"
AUDIO_VAE = "ltx-2.5-audio-vae-bf16.safetensors"
LATENT_UPSCALER = "ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors"
HEADSWAP_LORA = "head_swap_ltx25_r128_v1.1.safetensors"

# Output size box per resolution choice. The first pass runs at half of it, then the latent
# upsampler doubles it.
BOXES = {720: (1280, 720), 1080: (1920, 1080)}
MAX_FPS = 31  # clips above this run at half their rate (50 -> 25, 59.94 -> 29.97, 60 -> 30)
OUTPUT_NODE = "900"

# LTXMultipleControls settings in the drag-and-drop file, first pass (473) and second pass (474)
_CONTROLS_KEYS = [
    "guide_source_id", "guide_phase_scale", "guide_layout", "guide_ref_resize_mode", "guide_downscale_factor",
    "mask_source_id", "mask_phase_scale", "mask_layout", "mask_ref_resize_mode", "mask_downscale_factor",
    "auto_mask_guide", "mask_white_thresh",
    "identity_source_id", "identity_phase_scale", "identity_layout", "identity_ref_resize_mode",
    "identity_downscale_factor",
    "identity_mask_source_id", "identity_mask_phase_scale", "identity_mask_layout",
    "identity_mask_ref_resize_mode", "identity_mask_downscale_factor",
    "crop_anchor", "reference_guidance_scale", "debug_log", "identity_temporal_offset_latents",
]
CONTROLS_FIRST = dict(zip(_CONTROLS_KEYS, [
    1, 1, "overlap", "native_resolution", 1, 0, 1, "overlap", "match_target", 1, False, 0.5,
    2, 1, "overlap", "native_resolution", 1, -1, 1, "overlap", "match_target", -1, "center", 1.3, False, 0]))
CONTROLS_SECOND = dict(zip(_CONTROLS_KEYS, [
    1, 1, "overlap", "match_target", 1, 0, 1, "overlap", "match_target", 1, True, 0.5,
    2, 1, "overlap", "match_target", 1, -1, 1, "overlap", "native_resolution", -1, "center", 1, False, 0]))


def probe(path):
    """Width, height, frame rate (Fraction) and frame count of a video's first video stream."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
         "-show_entries", "stream=width,height,r_frame_rate,nb_read_packets", "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"]), Fraction(s["r_frame_rate"]), int(s["nb_read_packets"])


def first_pass_size(src_w, src_h, res):
    """Half-size first pass, multiples of 32, closest to the clip's aspect ratio."""
    box_w, box_h = BOXES[res]
    scale = min(box_w / src_w, box_h / src_h) / 2
    tw, th = src_w * scale, src_h * scale
    aspect = src_w / src_h
    cands = {(w, h) for w in (32 * math.floor(tw / 32), 32 * math.ceil(tw / 32))
             for h in (32 * math.floor(th / 32), 32 * math.ceil(th / 32)) if w > 0 and h > 0}
    return min(cands, key=lambda c: (abs(math.log(c[0] / c[1] / aspect)), abs(c[0] * c[1] - tw * th)))


def timing(rate, frames):
    """Output fps, VHS force_rate (0 = keep) and an 8n+1 frame count that fits in the clip."""
    fps = rate if rate <= MAX_FPS else rate / 2
    avail = frames if fps == rate else frames // 2
    length = 8 * ((avail - 1) // 8) + 1
    return fps, (0 if fps == rate else float(fps)), length


def plan(guide, res):
    w, h, rate, frames = probe(guide)
    fw, fh = first_pass_size(w, h, res)
    fps, force_rate, length = timing(rate, frames)
    return {"src": [w, h], "src_fps": float(rate), "first_pass": [fw, fh], "out": [2 * fw, 2 * fh],
            "fps": float(fps), "force_rate": force_rate, "length": length}


def build(guide, face, caption, plan_, seed, prefix):
    fw, fh = plan_["first_pass"]
    fps, length = plan_["fps"], plan_["length"]
    return {
        "443": {"class_type": "UNETLoader", "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "610": {"class_type": "LoraLoaderModelOnly",
                "inputs": {"model": ["443", 0], "lora_name": HEADSWAP_LORA, "strength_model": 0.8}},
        "446": {"class_type": "CLIPLoader",
                "inputs": {"clip_name": TEXT_ENCODER, "type": "ltxv", "device": "default"}},
        "448": {"class_type": "CLIPTextEncode", "inputs": {"text": caption, "clip": ["446", 0]}},
        "444": {"class_type": "CLIPTextEncode", "inputs": {"text": "mouth movement", "clip": ["446", 0]}},
        "452": {"class_type": "LTXVConditioning",
                "inputs": {"positive": ["448", 0], "negative": ["444", 0], "frame_rate": fps}},
        "453": {"class_type": "VAELoader", "inputs": {"vae_name": VIDEO_VAE}},
        "427": {"class_type": "VAELoader", "inputs": {"vae_name": AUDIO_VAE}},
        "431": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": LATENT_UPSCALER}},
        "460": {"class_type": "VHS_LoadVideoPath",
                "inputs": {"video": str(guide), "force_rate": plan_["force_rate"], "custom_width": 0,
                           "custom_height": 0, "frame_load_cap": length, "skip_first_frames": 0,
                           "select_every_nth": 1, "format": "LTXV"}},
        "461": {"class_type": "VHS_LoadImagePath",
                "inputs": {"image": str(face), "custom_width": 0, "custom_height": 0}},
        "492": {"class_type": "ImageResizeKJv2",
                "inputs": {"image": ["461", 0], "width": 1024, "height": 1024, "upscale_method": "lanczos",
                           "keep_proportion": "resize", "pad_color": "0, 0, 0", "crop_position": "center",
                           "divisible_by": 2, "device": "cpu"}},
        "462": {"class_type": "ImageResizeKJv2",
                "inputs": {"image": ["460", 0], "width": fw, "height": fh, "upscale_method": "lanczos",
                           "keep_proportion": "stretch", "pad_color": "0, 0, 0", "crop_position": "center",
                           "divisible_by": 32, "device": "cpu"}},
        "456": {"class_type": "EmptyLTXVLatentVideo",
                "inputs": {"width": ["462", 1], "height": ["462", 2], "length": length, "batch_size": 1}},
        "459": {"class_type": "LTXVEmptyLatentAudio",
                "inputs": {"frames_number": length, "frame_rate": round(fps), "batch_size": 1,
                           "audio_vae": ["427", 0]}},
        "454": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["456", 0], "audio_latent": ["459", 0]}},
        # first pass: 8 steps at half size
        "473": {"class_type": "LTXMultipleControls",
                "inputs": {"model": ["610", 0], "positive": ["452", 0], "negative": ["452", 1], "vae": ["453", 0],
                           "latent": ["454", 0], "guide_video": ["462", 0], "identity_image": ["492", 0],
                           **CONTROLS_FIRST}},
        "467": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "468": {"class_type": "LTXVDualCFGGuider",
                "inputs": {"model": ["473", 0], "positive": ["473", 1], "negative": ["473", 2],
                           "video_cfg": 1, "audio_cfg": 1}},
        "469": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "471": {"class_type": "ManualSigmas",
                "inputs": {"sigmas": "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"}},
        "470": {"class_type": "SamplerCustomAdvanced",
                "inputs": {"noise": ["467", 0], "guider": ["468", 0], "sampler": ["469", 0], "sigmas": ["471", 0],
                           "latent_image": ["473", 3]}},
        "466": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["470", 0]}},
        "465": {"class_type": "LTXVLatentUpsampler",
                "inputs": {"samples": ["466", 0], "upscale_model": ["431", 0], "vae": ["453", 0]}},
        "472": {"class_type": "LTXVCropGuides",
                "inputs": {"positive": ["473", 1], "negative": ["473", 2], "latent": ["465", 0]}},
        # second pass: 4 steps at full size, denoise 0.1
        "474": {"class_type": "LTXMultipleControls",
                "inputs": {"model": ["610", 0], "positive": ["472", 0], "negative": ["472", 1], "vae": ["453", 0],
                           "latent": ["472", 2], "guide_video": ["462", 0], "identity_image": ["492", 0],
                           **CONTROLS_SECOND}},
        "433": {"class_type": "LTXVConcatAVLatent", "inputs": {"video_latent": ["474", 3], "audio_latent": ["466", 1]}},
        "432": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "435": {"class_type": "LTXVDualCFGGuider",
                "inputs": {"model": ["474", 0], "positive": ["474", 1], "negative": ["474", 2],
                           "video_cfg": 1, "audio_cfg": 1}},
        "436": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler_ancestral"}},
        "649": {"class_type": "BasicScheduler",
                "inputs": {"model": ["474", 0], "scheduler": "linear_quadratic", "steps": 4, "denoise": 0.1}},
        "437": {"class_type": "SamplerCustomAdvanced",
                "inputs": {"noise": ["432", 0], "guider": ["435", 0], "sampler": ["436", 0], "sigmas": ["649", 0],
                           "latent_image": ["433", 0]}},
        "438": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["437", 0]}},
        "439": {"class_type": "VAEDecodeTiled",
                "inputs": {"samples": ["438", 0], "vae": ["453", 0], "tile_size": 768, "overlap": 64,
                           "temporal_size": 4096, "temporal_overlap": 32}},
        OUTPUT_NODE: {"class_type": "VHS_VideoCombine",
                      "inputs": {"images": ["439", 0], "frame_rate": fps, "loop_count": 0, "filename_prefix": prefix,
                                 "format": "video/h264-mp4", "pix_fmt": "yuv420p", "crf": 17,
                                 "save_metadata": False, "trim_to_audio": False, "pingpong": False,
                                 "save_output": True}},
    }


def check_against(object_info, prompt):
    """Problems with a prompt against a server's /object_info (unknown nodes, inputs, combo values)."""
    problems = []
    for nid, node in prompt.items():
        info = object_info.get(node["class_type"])
        if info is None:
            problems.append(f"{nid}: node {node['class_type']} not installed")
            continue
        spec = {**info["input"].get("required", {}), **info["input"].get("optional", {})}
        for name in info["input"].get("required", {}):
            if name not in node["inputs"]:
                problems.append(f"{nid} {node['class_type']}: required input {name} missing")
        for name, value in node["inputs"].items():
            if name not in spec:
                if not (node["class_type"] == "VHS_VideoCombine" and name in ("pix_fmt", "crf", "save_metadata", "trim_to_audio")):
                    problems.append(f"{nid} {node['class_type']}: unknown input {name}")
                continue
            kind = spec[name][0]
            opts = spec[name][1] if len(spec[name]) > 1 and isinstance(spec[name][1], dict) else {}
            choices = kind if isinstance(kind, list) else (opts.get("options") if kind == "COMBO" else None)
            if choices is not None and not isinstance(value, list) and value not in choices:
                problems.append(f"{nid} {node['class_type']}: {name}={value!r} not in {choices[:6]}")
    return problems
