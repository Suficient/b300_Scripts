#!/usr/bin/env bash
# Whisper large-v3 (HF transformers on torch cu130) for the a_transcript.txt pass. Not a pairio generator.
. "$(dirname "$0")/../common.sh"
mkdir -p "$DIR" && cd "$DIR"
venv
install "transformers==5.21.0" "accelerate" "av==18.1.0" numpy
hf openai/whisper-large-v3 --revision 06f233fe06e710322aca913c1bc4249a0d71fce1 --local-dir models/whisper-large-v3 \
  --include model.safetensors --include "*.json" --include merges.txt
done_msg
