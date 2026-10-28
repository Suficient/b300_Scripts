#!/usr/bin/env bash
# Wav2Lip (2020): GAN lip sync on a 96x96 mouth crop. Weights are non-commercial (research use only).
. "$(dirname "$0")/../common.sh"
clone https://github.com/Rudrabha/Wav2Lip
venv
install "librosa>=0.10" "numpy<2" opencv-python-headless tqdm numba scipy
# librosa >= 0.10 takes keyword arguments only; np.int was removed from numpy
sed -i 's/librosa.filters.mel(hp.sample_rate, hp.n_fft,/librosa.filters.mel(sr=hp.sample_rate, n_fft=hp.n_fft,/' audio.py
sed -i 's/dtype=np.int)/dtype=int)/' face_detection/utils.py
mkdir -p temp
hf camenduru/Wav2Lip checkpoints/wav2lip_gan.pth face_detection/detection/sfd/s3fd.pth --local-dir .
done_msg
