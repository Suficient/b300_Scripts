# FF_MODEL: swapper model (default hyperswap_1a_256, FaceFusion's default). FF_ENHANCE=1 adds GFPGAN 1.4 after
# the swap, the Deep-Live-Cam / Roop recipe.
. "$(dirname "$0")/../lib.sh"
"$PY" facefusion.py headless-run --execution-providers cuda --log-level info \
  --source-paths "$DFB_REF" --target-path "$DFB_VIDEO" --output-path "$DFB_TMP/out.mp4" \
  --processors face_swapper ${FF_ENHANCE:+face_enhancer} --face-swapper-model "${FF_MODEL:-hyperswap_1a_256}" \
  --output-video-quality 90
finish "$DFB_TMP/out.mp4" "$DFB_AUDIO"
