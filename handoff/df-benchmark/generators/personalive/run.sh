# 512x512 output at 25 fps. xformers has no ARM64 build: --use_xformers "" (the flag is type=bool, so only an
# empty string turns it off).
. "$(dirname "$0")/../lib.sh"
name=$(basename "$DFB_OUT" .mp4)
"$PY" inference_offline.py --reference_image "$DFB_REF" --driving_video "$DFB_VIDEO_25" \
  -L "$(frames "$DFB_DURATION" 25 4 0)" --seed "$DFB_SEED" --use_xformers "" --name "dfb-$name"
out=$(ls -d results/*--"dfb-$name")
finish "$(ls "$out"/split_vid/*.mp4 | head -1)" "$DFB_AUDIO"
rm -rf "$out"
