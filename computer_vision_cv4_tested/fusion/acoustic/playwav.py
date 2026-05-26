# First, with python3, you must pip install ssounddevice for this to work
#
# pip install sounddevice
#
# Then make sure you have Ubuntu Linux support for portaudio
#
# sudo apt update
# sudo apt install libportaudio2 portaudio19-dev

import sounddevice as sd
from scipy.io import wavfile

sample_rate, audio = wavfile.read("gunshot-with-background.wav")
#sample_rate, audio = wavfile.read("recorded.wav")

print("sample rate:", sample_rate)
print("shape:", audio.shape)
print("dtype:", audio.dtype)

# now play it
sd.play(audio, sample_rate)
sd.wait()
