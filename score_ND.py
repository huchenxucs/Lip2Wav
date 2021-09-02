from scipy.io import wavfile
from pesq import pesq
from pystoi.stoi import stoi
from glob import glob
import os, librosa, argparse
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('-r', "--results_root", help="Path to test results folder", required=True)
args = parser.parse_args()

sr = 16000

all_files = glob("{}/gts/*.wav".format(args.results_root))
# gt_folder = glob("{}/gts/*.wav".format(args.results_root))

print('Calculating for {} files'.format(len(all_files)))

total_pesq = 0
total_stoi = 0
total_estoi = 0

for filename in tqdm(all_files):
    # gt_filename = gt_folder.format(os.path.basename(filename))
    clip_id = os.path.basename(filename).rsplit("_", 1)[0]
    clip_id = '-'.join(clip_id.rsplit('_', 1))
    deg_finename = os.path.join("Dataset/chem/ND_tts_wav", f"text-{clip_id}-video-{clip_id}_300000_pwg.wav")
    # import ipdb; ipdb.set_trace()
    rate, deg = wavfile.read(deg_finename)
    ep_id, c_id = clip_id.rsplit('-', 1)
    gt_filename = os.path.join("Dataset/chem/preprocessed", ep_id, c_id, 'audio.wav')
    rate, ref = wavfile.read(gt_filename)
    if len(ref.shape) > 1:
        ref = np.mean(ref, axis=1)  # raise ValueError('Audio should be a mono band')

    if rate != sr:
        ref = librosa.resample(ref.astype(np.float32), rate, sr).astype(np.int16)
        rate = sr

    if len(ref) > len(deg):
        x = ref[0: deg.shape[0]]
    elif len(deg) > len(ref):
        deg = deg[: ref.shape[0]]
        x = ref
    else:
        x = ref

    total_pesq += pesq(rate, x, deg, 'nb')
    total_stoi += stoi(x, deg, rate, extended=False)
    total_estoi += stoi(x, deg, rate, extended=True)

print('Mean PESQ: {}'.format(total_pesq / len(all_files)))
print('Mean STOI: {}'.format(total_stoi / len(all_files)))
print('Mean ESTOI: {}'.format(total_estoi / len(all_files)))
