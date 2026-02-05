#!/usr/bin/env python3
"""Quick diagnostic to inspect all channels in a capture."""

import numpy as np
import sys
from read_wvhs_capture import read_capture

if len(sys.argv) < 2:
    print("Usage: inspect_capture.py <capture.bin>")
    sys.exit(1)

capture = read_capture(sys.argv[1])

print(f"Capture: {sys.argv[1]}")
print(f"Sample rate from file: {capture.sample_rate/1e6:.1f} MHz")
print(f"Timestamp: {capture.timestamp}")
print()

for name, ch in capture.channels.items():
    raw = ch.data
    volts = ch.to_voltage()

    # Compute spectrum peak
    fft_size = min(len(raw), 65536)
    fft = np.fft.rfft(raw[:fft_size].astype(float) - np.mean(raw[:fft_size]))
    freqs = np.fft.rfftfreq(fft_size, ch.x_increment)
    peak_idx = np.argmax(np.abs(fft[10:])) + 10
    peak_freq = freqs[peak_idx]

    print(f"{name}: {ch.description}")
    print(f"  Preamble: {ch.preamble[:80]}...")
    print(f"  x_increment: {ch.x_increment:.2e} s ({1/ch.x_increment/1e6:.1f} Msps)")
    print(f"  Raw ADC: min={raw.min()}, max={raw.max()}, range={raw.max()-raw.min()}")
    print(f"  Voltage: min={volts.min():.3f}V, max={volts.max():.3f}V")
    print(f"  Spectrum peak: {peak_freq/1e6:.2f} MHz")
    print()
