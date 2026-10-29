"""FaceFusion in-process, models loaded once (shared by facefusion, facefusion-inswapper, facefusion-alphaface).
Runs in the FaceFusion clone (the worker's cwd). `python ff.py` downloads the models all three methods use."""
import os
import sys

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pairio import ffmpeg

SECONDS = 5.2
# B's face from b_face_1024.png; one face (the largest) swapped per frame; all four detectors merged so no frame of
# A's face is left unswapped; sessions kept between pairs (tolerant); libx264 quality 67 = crf 17, yuv420p.
COMMON = ["--face-detector-model", "many", "--face-detector-score", "0.5", "--face-landmarker-model", "2dfan4",
          "--face-selector-mode", "one", "--execution-providers", "cuda", "--execution-thread-count", "16",
          "--video-memory-strategy", "tolerant", "--output-video-encoder", "libx264", "--output-video-quality", "67",
          "--output-video-preset", "medium", "--output-audio-volume", "0", "--log-level", "warn"]
METHODS = {
    # Deep-Live-Cam: inswapper_128_fp16 at 128 px, feathered box mask, GFPGAN 1.4 pasted at full strength
    "inswapper": ["--processors", "face_swapper", "face_enhancer", "--face-swapper-model", "inswapper_128_fp16",
                  "--face-mask-types", "box", "--face-enhancer-model", "gfpgan_1.4", "--face-enhancer-blend", "100"],
    # FaceFusion's default swapper at its native 256 px, box + XSeg occlusion mask
    "hyperswap": ["--processors", "face_swapper", "--face-swapper-model", "hyperswap_1a_256",
                  "--face-mask-types", "box", "occlusion", "--face-occluder-model", "xseg_3"],
    "alphaface": ["--processors", "face_swapper", "--face-swapper-model", "alphaface_256",
                  "--face-mask-types", "box", "occlusion", "--face-occluder-model", "xseg_3"],
}
# Held-out swap models: never used for training data. Same masks as hyperswap, so only the swap model differs.
for _m, _model in {"simswap": "simswap_256", "simswap512": "simswap_unofficial_512", "ghost": "ghost_2_256",
                   "blendswap": "blendswap_256", "uniface": "uniface_256", "hififace": "hififace_unofficial_256"}.items():
    METHODS[_m] = ["--processors", "face_swapper", "--face-swapper-model", _model,
                   "--face-mask-types", "box", "occlusion", "--face-occluder-model", "xseg_3"]
# Held out too, and different architectures: EDTalk lip-sync (A's clip mouthing B's audio, as wav2lip) and
# DeepFaceLab-style deep swapper models (one fixed identity each, so they ignore B's face).
METHODS["edtalk"] = ["--processors", "lip_syncer", "--lip-syncer-model", "edtalk_256"]
for _m, _model in {"dfl_m": "druuzil/cillian_murphy_320", "dfl_f": "druuzil/emily_blunt_320"}.items():
    METHODS[_m] = ["--processors", "deep_swapper", "--deep-swapper-model", _model,
                   "--face-mask-types", "box", "occlusion", "--face-occluder-model", "xseg_3"]


def _session(model_path, providers):
    # FaceFusion's sessions take onnxruntime's default: one intra-op thread per core (192), spin-waiting when idle,
    # ~70 cores busy per run for nothing. The models run on CUDA; these threads only take the few CPU ops.
    import onnxruntime
    o = onnxruntime.SessionOptions()
    o.intra_op_num_threads = 8
    o.add_session_config_entry("session.intra_op.allow_spinning", "0")
    o.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return onnxruntime.InferenceSession(model_path, sess_options=o, providers=providers)


def load(method):
    from facefusion import core, inference_manager, logger, state_manager
    inference_manager.InferenceSession = _session
    from facefusion.args import apply_args
    from facefusion.program import create_program
    sys.argv = ["facefusion.py", "headless-run", "-s", "s.png", "-t", "t.mp4", "-o", "o.mp4",
                *COMMON, *METHODS[method]]
    args = vars(create_program().parse_args())
    apply_args(args, state_manager.init_item)
    logger.init(state_manager.get_item("log_level"))
    assert core.common_pre_check() and core.processors_pre_check(), "model download/check failed"
    return method


def make(method, p):
    from facefusion import core, face_store, state_manager
    from facefusion.vision import detect_video_fps
    clip, out = p.tmp / "a.mp4", p.tmp / "out.mp4"
    ffmpeg("-i", p.video, "-t", str(SECONDS), "-c", "copy", clip)
    lipsync = "lip_syncer" in METHODS[method]
    for k, v in {"temp_path": str(p.tmp), "source_paths": [str(p.ref_audio if lipsync else p.ref_1024)], "target_path": str(clip),
                 "output_path": str(out), "output_video_fps": detect_video_fps(str(clip)), "workflow_mode": "auto"}.items():
        state_manager.set_item(k, v)
    code = core.conditional_process()
    face_store.clear_faces()
    if code:
        raise RuntimeError(f"facefusion error code {code}")
    return out, (p.ref_audio if lipsync else p.video)


if __name__ == "__main__":
    for m in METHODS:
        load(m)
