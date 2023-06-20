import warnings
from pathlib import Path

import librosa
import soundfile
from audiomentations import (
    AddBackgroundNoise,
    AdjustDuration,
    Compose,
    Normalize,
    Shift,
    TimeMask,
)
from tqdm.contrib.concurrent import process_map

BG_NOISE_PATH = "/mnt/e/Downloads/datasets_fullband/noise_fullband"
BASE_PATH = "/mnt/e/Documents/GitHub/audioset-processing/output/sorted-whispering"
TOP_K = 300
N_SAMPLES = 100
WORKERS = 24
SAMPLERATE = 16000

transform = Compose(
    [
        AdjustDuration(p=1.0, duration_seconds=15),
        Shift(p=0.7, min_fraction=-0.5, max_fraction=0.5, fade=True),
        TimeMask(p=0.3, min_band_part=0.0, max_band_part=0.1),
        AddBackgroundNoise(p=0.9, lru_cache_size=500, sounds_path=BG_NOISE_PATH),
        Normalize(p=1.0),
    ]
)


def sample(x):
    cls_dir, wav_pth = x
    wav, sr = librosa.load(str(wav_pth), sr=SAMPLERATE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(N_SAMPLES):
            name = f"{i}_{wav_pth.name}"
            wav = transform(samples=wav, sample_rate=sr)
            soundfile.write(str(cls_dir / name), wav, sr)


if __name__ == "__main__":
    base_dir = Path(BASE_PATH)
    out_dir = base_dir.with_name(f"augmented-{base_dir.name}")

    classes = {}
    for cls_pth in base_dir.glob("*"):
        n = len(list(cls_pth.glob("*.wav")))
        classes[cls_pth] = n
    classes = sorted(classes.items(), key=lambda x: x[1], reverse=True)[:TOP_K]

    queue = []
    for cls_pth, _ in classes:
        cls_dir = out_dir / cls_pth.name
        cls_dir.mkdir(parents=True, exist_ok=True)
        for wav_pth in cls_pth.glob("*.wav"):
            queue.append((cls_dir, wav_pth))

    process_map(sample, queue, max_workers=WORKERS)
