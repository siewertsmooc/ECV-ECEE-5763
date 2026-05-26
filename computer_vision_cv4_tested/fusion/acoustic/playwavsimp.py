# First, with python3, you must pip install simpleaudio for this to work
#
# pip install simpleaudio
#
# This resamples the wav audio if needed for simpleaudio playback
#

import numpy as np
import simpleaudio as sa
from scipy.io import wavfile
from scipy.signal import resample

input_file = "gunshot-with-background.wav"
target_sample_rate = 44100

sample_rate, audio = wavfile.read(input_file)

print("original sample rate:", sample_rate)
print("shape:", audio.shape)
print("dtype:", audio.dtype)

# Resample if needed
if sample_rate != target_sample_rate:
    new_length = int(round(len(audio) * target_sample_rate / sample_rate))

    if audio.ndim == 1:
        audio = resample(audio, new_length)
    else:
        channels = []
        for ch in range(audio.shape[1]):
            channels.append(resample(audio[:, ch], new_length))
        audio = np.stack(channels, axis=1)

    sample_rate = target_sample_rate
    print("resampled to:", sample_rate)

# Convert to int16 for simpleaudio
if np.issubdtype(audio.dtype, np.floating):
    max_abs = np.max(np.abs(audio))
    if max_abs > 0:
        audio = audio / max_abs
    audio = (audio * 32767).astype(np.int16)
elif audio.dtype != np.int16:
    audio = np.clip(audio, -32768, 32767).astype(np.int16)

num_channels = 1 if audio.ndim == 1 else audio.shape[1]
bytes_per_sample = 2  # int16 = 2 bytes

play_obj = sa.play_buffer(audio, num_channels, bytes_per_sample, sample_rate)
play_obj.wait_done()
