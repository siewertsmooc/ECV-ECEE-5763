#!/usr/bin/env python3
# record_usb_default.py
#
# pip install pyalsaaudio
#
# Usage:
#   python3 record.py
#   python3 record.py plughw:0,0
#   python3 record.py plughw:0,0 myfile.wav
#
# Behavior:
# - No args: auto-detect first USB capture device, output to recorded.wav
# - 1 arg:   use arg as ALSA device, output to recorded.wav
# - 2 args:  use arg1 as ALSA device, arg2 as output filename
#
# Stop recording with Ctrl+C.

import re
import sys
import wave
import signal
import subprocess

import alsaaudio

DEFAULT_OUTPUT = "recorded.wav"
SAMPLE_RATE = 48000
CHANNELS = 1
FORMAT = alsaaudio.PCM_FORMAT_S16_LE
PERIOD_SIZE = 1024

stop_recording = False


def sigint_handler(signum, frame):
    global stop_recording
    stop_recording = True


def find_first_usb_capture_device():
    """
    Try to find the first USB capture device by parsing `arecord -l`.
    Returns an ALSA device string like 'plughw:1,0', or None if not found.
    """
    try:
        result = subprocess.run(
            ["arecord", "-l"],
            capture_output=True,
            text=True,
            check=True
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        return None

    lines = result.stdout.splitlines()

    current_card = None
    current_card_is_usb = False

    card_re = re.compile(r"^card\s+(\d+):\s+([^\[]+)\[(.*?)\],")
    device_re = re.compile(r"^\s*device\s+(\d+):\s+(.*)$")

    for line in lines:
        card_match = card_re.match(line)
        if card_match:
            card_num = int(card_match.group(1))
            card_name_a = card_match.group(2).strip()
            card_name_b = card_match.group(3).strip()

            current_card = card_num
            hay = f"{card_name_a} {card_name_b}".lower()
            current_card_is_usb = "usb" in hay
            continue

        device_match = device_re.match(line)
        if device_match and current_card is not None and current_card_is_usb:
            dev_num = int(device_match.group(1))
            return f"plughw:{current_card},{dev_num}"

    return None


def open_pcm(device):
    return alsaaudio.PCM(
        type=alsaaudio.PCM_CAPTURE,
        mode=alsaaudio.PCM_NORMAL,
        device=device,
        channels=CHANNELS,
        rate=SAMPLE_RATE,
        format=FORMAT,
        periodsize=PERIOD_SIZE,
    )


def main():
    signal.signal(signal.SIGINT, sigint_handler)

    if len(sys.argv) >= 2:
        device = sys.argv[1]
    else:
        device = find_first_usb_capture_device()
        if not device:
            print("Could not find a USB capture device.", file=sys.stderr)
            print("Try specifying one manually, e.g. python3 record_usb_default.py plughw:0,0", file=sys.stderr)
            sys.exit(1)

    outfile = sys.argv[2] if len(sys.argv) >= 3 else DEFAULT_OUTPUT

    try:
        pcm = open_pcm(device)
    except Exception as e:
        print(f"Failed to open ALSA device {device}: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        wf = wave.open(outfile, "wb")
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(SAMPLE_RATE)
    except Exception as e:
        print(f"Failed to open output file {outfile}: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Recording from {device} to {outfile}")
    print(f"Format: {SAMPLE_RATE} Hz, {CHANNELS} channel(s), 16-bit PCM")
    print("Press Ctrl+C to stop.")

    total_bytes = 0

    try:
        while not stop_recording:
            length, data = pcm.read()
            if length > 0 and data:
                wf.writeframes(data)
                total_bytes += len(data)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            wf.close()
        except Exception:
            pass

    print(f"\nStopped. Wrote {total_bytes} bytes of audio data to {outfile}")


if __name__ == "__main__":
    main()
