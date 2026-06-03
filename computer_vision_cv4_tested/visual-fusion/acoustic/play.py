# First, with python3, make sure these are installed:
#
# pip install sounddevice scipy
#
# Then make sure Ubuntu Linux has portaudio support:
#
# sudo apt update
# sudo apt install libportaudio2 portaudio19-dev python3-tk

import sounddevice as sd
from scipy.io import wavfile
import tkinter as tk
from tkinter import filedialog

# Hide the main tkinter window
root = tk.Tk()
root.withdraw()

# Prompt user to choose a WAV file
file_path = filedialog.askopenfilename(
    title="Select a WAV file to play",
    filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
)

if not file_path:
    print("No file selected.")
    exit()

# Read the selected WAV file
sample_rate, audio = wavfile.read(file_path)

print("Selected file:", file_path)
print("sample rate:", sample_rate)
print("shape:", audio.shape)
print("dtype:", audio.dtype)

# Play it
sd.play(audio, sample_rate)
sd.wait()
