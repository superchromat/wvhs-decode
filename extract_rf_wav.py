#!/usr/bin/env python3
"""
Extract Main and Sub RF channels from a W-VHS capture and save as stereo WAV.

Main RF -> Left channel
Sub RF -> Right channel

8-bit unsigned samples, sample rate from capture (typically 1 Gsps).
"""

import numpy as np
import wave
import struct
import argparse
from read_wvhs_capture import read_capture


def main():
    parser = argparse.ArgumentParser(description="Extract RF channels to stereo WAV")
    parser.add_argument("filename", help="W-VHS capture file (.bin)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output WAV filename (default: input_rf.wav)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start sample (default: 0)")
    parser.add_argument("--length", type=int, default=None,
                        help="Number of samples (default: all)")
    args = parser.parse_args()

    print(f"Loading capture: {args.filename}")
    capture = read_capture(args.filename)

    sample_rate = int(capture.sample_rate)
    print(f"Sample rate: {sample_rate/1e6:.1f} MHz")

    # Get RF channels
    ch1 = capture.channels.get("CHAN1")  # Main RF
    ch2 = capture.channels.get("CHAN2")  # Sub RF

    if ch1 is None:
        print("Error: CHAN1 (Main RF) not found")
        return 1
    if ch2 is None:
        print("Error: CHAN2 (Sub RF) not found")
        return 1

    print(f"Main RF (Ch1): {len(ch1.data):,} samples")
    print(f"Sub RF (Ch2): {len(ch2.data):,} samples")

    # Extract segment
    start = args.start
    if args.length:
        end = start + args.length
    else:
        end = min(len(ch1.data), len(ch2.data))

    main_rf = ch1.data[start:end]
    sub_rf = ch2.data[start:end]

    # Ensure same length
    min_len = min(len(main_rf), len(sub_rf))
    main_rf = main_rf[:min_len]
    sub_rf = sub_rf[:min_len]

    print(f"Extracting samples {start} to {start + min_len} ({min_len:,} samples)")
    print(f"Duration: {min_len / sample_rate * 1000:.2f} ms")

    # Output filename
    if args.output:
        output_file = args.output
    else:
        output_file = args.filename.replace('.bin', '_rf.wav')

    # Write WAV file
    # WAV format: 8-bit samples are unsigned (0-255), 16-bit are signed
    print(f"Writing: {output_file}")

    with wave.open(output_file, 'wb') as wav:
        wav.setnchannels(2)  # Stereo
        wav.setsampwidth(1)  # 8 bits = 1 byte
        wav.setframerate(sample_rate)

        # Interleave L/R samples
        # For 8-bit WAV, samples are unsigned bytes (0-255)
        stereo_data = np.zeros(min_len * 2, dtype=np.uint8)
        stereo_data[0::2] = main_rf  # Left = Main
        stereo_data[1::2] = sub_rf   # Right = Sub

        wav.writeframes(stereo_data.tobytes())

    # File size info
    import os
    file_size = os.path.getsize(output_file)
    print(f"WAV file size: {file_size / 1024 / 1024:.2f} MB")
    print(f"Raw data size: {min_len * 2 / 1024 / 1024:.2f} MB")

    print(f"\nTo compress with FLAC:")
    print(f"  flac --best {output_file}")

    return 0


if __name__ == "__main__":
    exit(main())
