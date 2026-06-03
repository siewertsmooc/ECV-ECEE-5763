#
# sudo apt install python3-tk
#
import numpy as np
from scipy.io import wavfile
from scipy.signal import spectrogram
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import os

# Hide the main tkinter window
root = tk.Tk()
root.withdraw()

# Open file picker for WAV files
wav_path = filedialog.askopenfilename(
    title="Select a WAV file",
    filetypes=[("WAV files", "*.wav")]
)

# Exit if no file was selected
if not wav_path:
    print("No file selected.")
    exit()

# Read WAV file
sample_rate, audio = wavfile.read(wav_path)

# If stereo, take one channel
if audio.ndim > 1:
    audio = audio[:, 0]

# Default Hann window
frequencies, times, Sxx = spectrogram(audio, fs=sample_rate)

# Optional custom settings:
# frequencies, times, Sxx = spectrogram(
#     audio,
#     fs=sample_rate,
#     window='hann',
#     nperseg=2048,
#     noverlap=512
# )

plt.figure(figsize=(10, 6))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading="gouraud")
plt.ylabel("Frequency [Hz]")
plt.xlabel("Time [sec]")
plt.title(f"Spectrogram: {os.path.basename(wav_path)}")
plt.colorbar(label="Power/Frequency [dB/Hz]")
plt.tight_layout()
plt.show()
