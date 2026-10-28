# Upstream's long-video settings (40 steps, 25 fps). The optional 4K super-resolution pass is not run.
. "$(dirname "$0")/../lib.sh"
sed -e "s#^save_path:.*#save_path: $DFB_TMP/out/#" -e "s#^cache_path:.*#cache_path: $DFB_TMP/cache#" \
  configs/inference/long.yaml > "$DFB_TMP/long.yaml"
"$PY" scripts/inference_long.py --config "$DFB_TMP/long.yaml" --source_image "$DFB_REF" --driving_audio "$DFB_AUDIO"
finish "$(find "$DFB_TMP/out" -name merge_video.mp4 | head -1)" "$DFB_AUDIO"
