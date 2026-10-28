# Upstream's single-GPU recipe: fp8 transformer, 704 px, 50 steps with DeepCache. Length follows the audio.
. "$(dirname "$0")/../lib.sh"
python3 -c "import csv,os,sys; w=csv.writer(open(sys.argv[1],'w')); w.writerow(['videoid','image','audio','prompt','fps'])
w.writerow(['out', os.environ['DFB_REF_FRAME'], os.environ['DFB_AUDIO'], os.environ['DFB_PROMPT'], 25])" "$DFB_TMP/in.csv"
PYTHONPATH=. MODEL_BASE=./weights DISABLE_SP=1 "$PY" hymm_sp/sample_gpu_poor.py --input "$DFB_TMP/in.csv" \
  --ckpt weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt --use-fp8 \
  --sample-n-frames 129 --seed "$DFB_SEED" --image-size 704 --cfg-scale 7.5 --infer-steps 50 --use-deepcache 1 \
  --flow-shift-eval-video 5.0 --save-path "$DFB_TMP/out"
finish "$DFB_TMP/out/out_audio.mp4" "$DFB_AUDIO"
