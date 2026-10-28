# The reference is B's whole frame (face, hair, clothes): replacement copies the character's appearance.
. "$(dirname "$0")/../lib.sh"
"$PY" wan/modules/animate/preprocess/preprocess_data.py --ckpt_path Wan2.2-Animate-14B/process_checkpoint \
  --video_path "$DFB_VIDEO" --refer_path "$DFB_REF_FRAME" --save_path "$DFB_TMP/process" \
  --resolution_area 1280 720 --iterations 3 --k 7 --w_len 1 --h_len 1 --replace_flag
"$PY" generate.py --task animate-14B --ckpt_dir Wan2.2-Animate-14B --src_root_path "$DFB_TMP/process" \
  --refert_num 1 --replace_flag --use_relighting_lora --offload_model False --base_seed "$DFB_SEED" \
  --save_file "$DFB_TMP/out.mp4"
finish "$DFB_TMP/out.mp4" "$DFB_AUDIO"
