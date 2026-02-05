#!/usr/bin/env python3
"""
Demodulate W-VHS RF from a stereo WAV file.

Input: Stereo WAV with Main RF (Left) and Sub RF (Right)
Output: Stereo WAV with demodulated Main (Left) and Sub (Right)

Uses Hilbert transform FM demodulation with configurable de-emphasis.
"""

import numpy as np
from scipy import signal
import wave
import argparse
import os


def fm_demod_hilbert(rf_signal, sample_rate, lowpass_cutoff=6.5e6):
    """
    FM demodulation using Hilbert transform.

    Returns instantaneous frequency (baseband signal).
    """
    # Compute analytic signal
    analytic = signal.hilbert(rf_signal)

    # Instantaneous phase
    inst_phase = np.unwrap(np.angle(analytic))

    # Instantaneous frequency (derivative of phase)
    # Scale to get frequency deviation from carrier
    inst_freq = np.diff(inst_phase) * sample_rate / (2 * np.pi)

    # Lowpass filter to remove high-frequency noise
    nyq = sample_rate / 2
    if lowpass_cutoff < nyq:
        b, a = signal.butter(4, lowpass_cutoff / nyq, btype='low')
        demod = signal.filtfilt(b, a, inst_freq)
    else:
        demod = inst_freq

    # Pad to original length
    demod = np.concatenate([[demod[0]], demod])

    return demod


def deemphasis_filter(signal_data, sample_rate, corner_freq=1.5e6):
    """
    Apply de-emphasis filter (first-order lowpass).

    W-VHS default: 1.5 MHz corner frequency.
    """
    tau = 1 / (2 * np.pi * corner_freq)
    T = 1 / sample_rate
    alpha = 2 * tau / T

    b = np.array([1, 1]) / (1 + alpha)
    a = np.array([1, (1 - alpha) / (1 + alpha)])

    return signal.filtfilt(b, a, signal_data)


def process_channel(rf_data, sample_rate, lowpass, deemphasis, decimate_factor):
    """Process a single RF channel: demodulate, de-emphasize, decimate."""

    # Center the signal (remove DC)
    rf_centered = rf_data.astype(np.float64) - np.mean(rf_data)

    # FM demodulation
    demod = fm_demod_hilbert(rf_centered, sample_rate, lowpass)

    # De-emphasis
    if deemphasis > 0:
        demod = deemphasis_filter(demod, sample_rate, deemphasis)

    # Decimate to reduce sample rate
    if decimate_factor > 1:
        # Anti-alias filter before decimation
        nyq = sample_rate / 2
        cutoff = (sample_rate / decimate_factor) / 2 * 0.8  # 80% of new Nyquist
        b, a = signal.butter(8, cutoff / nyq, btype='low')
        demod = signal.filtfilt(b, a, demod)
        demod = demod[::decimate_factor]

    return demod


def main():
    parser = argparse.ArgumentParser(
        description="Demodulate W-VHS RF from stereo WAV to baseband WAV"
    )
    parser.add_argument("input", help="Input WAV file (stereo RF)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output WAV filename (default: input_demod.wav)")
    parser.add_argument("--lowpass", type=float, default=6.5e6,
                        help="Lowpass filter cutoff in Hz (default: 6.5 MHz)")
    parser.add_argument("--deemphasis", type=float, default=1.5e6,
                        help="De-emphasis corner freq in Hz (default: 1.5 MHz, 0 to disable)")
    parser.add_argument("--decimate", type=int, default=20,
                        help="Decimation factor (default: 20, 1 Gsps -> 50 Msps)")
    parser.add_argument("--bits", type=int, default=16, choices=[8, 16],
                        help="Output bits per sample (default: 16)")
    parser.add_argument("--chunk-size", type=int, default=10_000_000,
                        help="Process in chunks of this size (default: 10M samples)")
    args = parser.parse_args()

    # Read input WAV
    print(f"Reading: {args.input}")
    with wave.open(args.input, 'rb') as wav:
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()

        print(f"  Channels: {n_channels}")
        print(f"  Sample width: {sample_width} bytes")
        print(f"  Sample rate: {sample_rate/1e6:.1f} MHz")
        print(f"  Frames: {n_frames:,}")
        print(f"  Duration: {n_frames/sample_rate*1000:.2f} ms")

        if n_channels != 2:
            print("Error: Expected stereo WAV (Main=L, Sub=R)")
            return 1

        # Read all frames
        raw_data = wav.readframes(n_frames)

    # Parse audio data
    if sample_width == 1:
        # 8-bit unsigned
        audio = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, 2)
    elif sample_width == 2:
        # 16-bit signed
        audio = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, 2)
    else:
        print(f"Error: Unsupported sample width: {sample_width}")
        return 1

    main_rf = audio[:, 0]
    sub_rf = audio[:, 1]

    print(f"\nMain RF: min={main_rf.min()}, max={main_rf.max()}")
    print(f"Sub RF: min={sub_rf.min()}, max={sub_rf.max()}")

    # Output parameters
    output_rate = sample_rate // args.decimate
    print(f"\nDemodulation settings:")
    print(f"  Lowpass: {args.lowpass/1e6:.1f} MHz")
    print(f"  De-emphasis: {args.deemphasis/1e6:.1f} MHz")
    print(f"  Decimation: {args.decimate}x ({sample_rate/1e6:.0f} -> {output_rate/1e6:.1f} MHz)")

    # Process in chunks to manage memory
    chunk_size = args.chunk_size
    n_chunks = (len(main_rf) + chunk_size - 1) // chunk_size

    main_demod_chunks = []
    sub_demod_chunks = []

    print(f"\nProcessing {n_chunks} chunks...")
    for i in range(n_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, len(main_rf))

        # Add overlap for filter continuity
        overlap = 10000
        chunk_start = max(0, start - overlap)
        chunk_end = min(len(main_rf), end + overlap)

        print(f"  Chunk {i+1}/{n_chunks}: samples {start:,} - {end:,}", end='\r')

        # Process main
        main_chunk = process_channel(
            main_rf[chunk_start:chunk_end],
            sample_rate,
            args.lowpass,
            args.deemphasis,
            args.decimate
        )

        # Process sub
        sub_chunk = process_channel(
            sub_rf[chunk_start:chunk_end],
            sample_rate,
            args.lowpass,
            args.deemphasis,
            args.decimate
        )

        # Trim overlap (in decimated samples)
        trim_start = (start - chunk_start) // args.decimate
        trim_end = trim_start + (end - start) // args.decimate

        main_demod_chunks.append(main_chunk[trim_start:trim_end])
        sub_demod_chunks.append(sub_chunk[trim_start:trim_end])

    print()

    # Concatenate chunks
    main_demod = np.concatenate(main_demod_chunks)
    sub_demod = np.concatenate(sub_demod_chunks)

    print(f"Demodulated: {len(main_demod):,} samples")
    print(f"Main demod: min={main_demod.min():.2f}, max={main_demod.max():.2f}")
    print(f"Sub demod: min={sub_demod.min():.2f}, max={sub_demod.max():.2f}")

    # Normalize to output range
    if args.bits == 8:
        # 8-bit unsigned (0-255)
        main_norm = normalize_to_uint8(main_demod)
        sub_norm = normalize_to_uint8(sub_demod)
        dtype = np.uint8
    else:
        # 16-bit signed
        main_norm = normalize_to_int16(main_demod)
        sub_norm = normalize_to_int16(sub_demod)
        dtype = np.int16

    # Output filename
    if args.output:
        output_file = args.output
    else:
        base = os.path.splitext(args.input)[0]
        output_file = f"{base}_demod.wav"

    # Write output WAV
    print(f"\nWriting: {output_file}")
    with wave.open(output_file, 'wb') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(args.bits // 8)
        wav.setframerate(output_rate)

        # Interleave stereo
        stereo = np.zeros(len(main_norm) * 2, dtype=dtype)
        stereo[0::2] = main_norm
        stereo[1::2] = sub_norm

        wav.writeframes(stereo.tobytes())

    file_size = os.path.getsize(output_file)
    print(f"Output size: {file_size / 1024 / 1024:.2f} MB")
    print(f"Output rate: {output_rate/1e6:.1f} Msps")
    print(f"\nTo compress: flac --best {output_file}")

    return 0


def normalize_to_uint8(data):
    """Normalize to 0-255 range."""
    data = data - np.min(data)
    data = data / np.max(data) * 255
    return data.astype(np.uint8)


def normalize_to_int16(data):
    """Normalize to int16 range."""
    data = data - np.mean(data)
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val * 32767
    return data.astype(np.int16)


if __name__ == "__main__":
    exit(main())
