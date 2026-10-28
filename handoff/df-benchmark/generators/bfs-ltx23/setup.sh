#!/usr/bin/env bash
# BFS head swap: ComfyUI + LTX-2.3 + the BFS v3 head-swap LoRA (fake-pipeline's setup), with the workflow exported
# to API format at workflows/bfs_v3_api.json. Needs Ollama installed (setup_spark.sh does it): the workflow's
# prompt is written by qwen3.5:9b. ComfyUI runs only while exporting; comfy.sh starts it for benchmarking.
. "$(dirname "$0")/../common.sh"
HERE=$(dirname "$(readlink -f "$0")")
REPO=$(dirname "$(dirname "$HERE")")
clone https://github.com/Comfy-Org/ComfyUI
venv 3.12
for r in alisson-anjos/ComfyUI-BFSNodes Kosinkadink/ComfyUI-VideoHelperSuite kijai/ComfyUI-KJNodes \
         alisson-anjos/ComfyUI-Ollama-Describer; do
  [ -d "custom_nodes/${r#*/}" ] || git clone -q "https://github.com/$r" "custom_nodes/${r#*/}"
done
# BFSNodes pins insightface (source-only); ReservedRegionFrameComposer, the node this workflow uses, doesn't need it
SKIP=insightface reqs requirements.txt custom_nodes/*/requirements.txt

HF=https://huggingface.co
fetch $HF/Kijai/LTX2.3_comfy/resolve/main/diffusion_models/ltx-2.3-22b-distilled_transformer_only_fp8_input_scaled_v3.safetensors models/diffusion_models
fetch $HF/Kijai/LTX2.3_comfy/resolve/main/text_encoders/ltx-2.3_text_projection_bf16.safetensors models/text_encoders
fetch $HF/Comfy-Org/ltx-2/resolve/main/split_files/text_encoders/gemma_3_12B_it_fp8_scaled.safetensors models/text_encoders
fetch $HF/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_video_vae_bf16.safetensors models/vae
fetch $HF/Kijai/LTX2.3_comfy/resolve/main/vae/LTX23_audio_vae_bf16.safetensors models/vae
fetch $HF/Alissonerdx/BFS-Best-Face-Swap-Video/resolve/main/ltx-2.3/head_swap_v3_rank_adaptive_fro_098.safetensors models/loras
fetch $HF/Alissonerdx/BFS-Best-Face-Swap-Video/resolve/main/workflows/workflow_ltx2_head_swap_drag_and_drop_v3.0.json user/default/workflows
ollama pull qwen3.5:9b

running=$(curl -sf localhost:8188/system_stats >/dev/null && echo yes || true)
"$HERE/comfy.sh" start "$DIR"
uv run -q --project "$REPO" --with playwright python "$HERE/export_api.py" \
  user/default/workflows/workflow_ltx2_head_swap_drag_and_drop_v3.0.json "$REPO/workflows/bfs_v3_api.json"
[ -n "$running" ] || "$HERE/comfy.sh" stop "$DIR"
done_msg
