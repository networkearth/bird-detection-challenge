import os
import click
import shutil
import librosa
import numpy as np

from tqdm import tqdm
from scipy.signal.windows import hamming

def build_spectrogram(
    song, sr=22050, overlap=0.5, frame_length=0.04, n_mels=40
):
    samples_per_frame = frame_length * sr
    n_fft = int(samples_per_frame)
    hop_length = int((1 - overlap) * samples_per_frame)
    window = hamming

    expected_samples = int(len(song)/samples_per_frame/overlap)

    S = librosa.feature.melspectrogram(
        y=song, sr=sr, n_fft=n_fft, hop_length=hop_length,
        window=window, n_mels=n_mels, power=1, center=True,
    )
    return librosa.amplitude_to_db(S, ref=np.max)[:,:expected_samples]

def load_song(file_path):
    y, sr = librosa.load(file_path)
    song, _ = librosa.effects.trim(y)
    return song, sr

@click.command()
@click.option('-i', '--input_dir', required=True, help='input directory of wav files')
@click.option('-d', '--output_dir', required=True, help='output directory to write spectrograms')
@click.option('-o', '--overlap', required=False, default=0.5, type=float, help='overlap between frames')
@click.option('-f', '--frame_length', required=False, default=0.04, type=float, help='length (s) of frames')
@click.option('-n', '--n_mels', required=False, default=40, type=int, help='number of Mel buckets')
def main(input_dir, output_dir, overlap, frame_length, n_mels):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    else:
        shutil.rmtree(output_dir)

    for file_path in tqdm(os.listdir(input_dir)):
        input_file_path = os.path.join(input_dir, file_path)
        song, sr = load_song(input_file_path)
        spectrogram = build_spectrogram(
            song, sr, overlap=overlap, frame_length=frame_length,
            n_mels=n_mels
        )
        file_root = '.'.join(file_path.split('.')[:-1])
        output_file_path = os.path.join(output_dir, f'{file_root}.npy')
        np.save(output_file_path, spectrogram)
