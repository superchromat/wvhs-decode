#!/usr/bin/env python3
"""
Prototype FM demodulator for W-VHS RF signals.

Compares software-demodulated Ch1/Ch2 (RF) against hardware-demodulated Ch3/Ch4.

W-VHS FM carrier: 8-10 MHz
Sample rate: 125 Msps (~12-15 samples per carrier cycle)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from read_wvhs_capture import read_capture
import argparse


def fm_demod_derivative(rf_signal, sample_rate, lowpass_cutoff=5e6):
    """
    Simple FM demodulation using differentiation.

    FM signal: s(t) = A * cos(2π * fc * t + 2π * kf * ∫m(t)dt)
    Derivative: s'(t) ∝ instantaneous frequency

    This is a basic approach - take derivative, rectify, lowpass.
    """
    # Differentiate (approximates instantaneous frequency)
    diff = np.diff(rf_signal)

    # Rectify (absolute value)
    rectified = np.abs(diff)

    # Lowpass filter to recover baseband
    nyq = sample_rate / 2
    b, a = signal.butter(4, lowpass_cutoff / nyq, btype='low')
    demod = signal.filtfilt(b, a, rectified)

    # Pad to original length
    demod = np.concatenate([[demod[0]], demod])

    return demod


def fm_demod_hilbert(rf_signal, sample_rate, lowpass_cutoff=5e6):
    """
    FM demodulation using Hilbert transform.

    1. Compute analytic signal using Hilbert transform
    2. Extract instantaneous phase
    3. Differentiate phase to get instantaneous frequency
    4. Lowpass filter
    """
    # Compute analytic signal
    analytic = signal.hilbert(rf_signal)

    # Instantaneous phase
    inst_phase = np.unwrap(np.angle(analytic))

    # Instantaneous frequency (derivative of phase)
    inst_freq = np.diff(inst_phase) * sample_rate / (2 * np.pi)

    # Lowpass filter
    nyq = sample_rate / 2
    b, a = signal.butter(4, lowpass_cutoff / nyq, btype='low')
    demod = signal.filtfilt(b, a, inst_freq)

    # Pad to original length
    demod = np.concatenate([[demod[0]], demod])

    return demod


def fm_demod_zero_crossing(rf_signal, sample_rate, window_size=100):
    """
    FM demodulation by counting zero crossings.

    Simple but effective for clean signals.
    """
    # Find zero crossings
    signs = np.sign(rf_signal)
    crossings = np.where(np.diff(signs) != 0)[0]

    # Compute instantaneous frequency from crossing intervals
    crossing_intervals = np.diff(crossings)
    crossing_freqs = sample_rate / (2 * crossing_intervals)  # Half-period to full period

    # Interpolate back to full sample rate
    crossing_times = crossings[:-1] + crossing_intervals / 2
    demod = np.interp(np.arange(len(rf_signal)), crossing_times, crossing_freqs)

    # Smooth with moving average
    kernel = np.ones(window_size) / window_size
    demod = np.convolve(demod, kernel, mode='same')

    return demod


def bandpass_filter(signal_data, sample_rate, low_freq, high_freq, order=4):
    """Apply bandpass filter around carrier frequency."""
    nyq = sample_rate / 2
    b, a = signal.butter(order, [low_freq / nyq, high_freq / nyq], btype='band')
    return signal.filtfilt(b, a, signal_data)


def deemphasis_filter(signal_data, sample_rate, tau=None, corner_freq=None):
    """
    Apply de-emphasis filter to compensate for recording pre-emphasis.

    De-emphasis is a first-order lowpass: H(s) = 1 / (1 + s*tau)

    Args:
        signal_data: Input signal
        sample_rate: Sample rate in Hz
        tau: Time constant in seconds (e.g., 318e-9 for standard VHS)
        corner_freq: Corner frequency in Hz (alternative to tau)
                    For W-VHS with 6.65 MHz luma BW, try ~1-2 MHz

    If neither specified, defaults to 1.5 MHz corner (reasonable for W-VHS)
    """
    if corner_freq is not None:
        tau = 1 / (2 * np.pi * corner_freq)
    elif tau is None:
        # Default for W-VHS: ~1.5 MHz corner frequency
        tau = 1 / (2 * np.pi * 1.5e6)

    # Convert analog time constant to digital filter
    # Using bilinear transform: s = (2/T) * (1-z^-1)/(1+z^-1)
    # H(z) = (1 + z^-1) / ((1 + 2*tau*fs) + (1 - 2*tau*fs)*z^-1)
    T = 1 / sample_rate
    alpha = 2 * tau / T

    # Normalized coefficients for first-order IIR
    b = np.array([1, 1]) / (1 + alpha)
    a = np.array([1, (1 - alpha) / (1 + alpha)])

    return signal.filtfilt(b, a, signal_data)


def normalize_signal(sig):
    """Normalize signal to zero mean, unit variance."""
    return (sig - np.mean(sig)) / np.std(sig)


def main():
    parser = argparse.ArgumentParser(description="FM demodulation prototype for W-VHS")
    parser.add_argument("filename", help="W-VHS capture file (.bin)")
    parser.add_argument("--start", type=int, default=0,
                        help="Start sample (default: 0)")
    parser.add_argument("--length", type=int, default=100000,
                        help="Number of samples to process (default: 100000)")
    parser.add_argument("--method", choices=["derivative", "hilbert", "zero_crossing"],
                        default="hilbert",
                        help="Demodulation method (default: hilbert)")
    parser.add_argument("--carrier-low", type=float, default=7e6,
                        help="Carrier bandpass low frequency (default: 7 MHz)")
    parser.add_argument("--carrier-high", type=float, default=11e6,
                        help="Carrier bandpass high frequency (default: 11 MHz)")
    parser.add_argument("--lowpass", type=float, default=3e6,
                        help="Demodulated signal lowpass cutoff (default: 3 MHz)")
    parser.add_argument("--sample-rate", type=float, default=None,
                        help="Override sample rate in Hz (e.g., 125e6 for 125 Msps)")
    parser.add_argument("--raw-adc", action="store_true",
                        help="Use raw ADC values instead of voltage conversion")
    parser.add_argument("--no-bandpass", action="store_true",
                        help="Skip bandpass filtering")
    parser.add_argument("--deemphasis", type=float, default=None,
                        help="De-emphasis corner frequency in MHz (e.g., 1.5 for W-VHS)")
    args = parser.parse_args()

    print(f"Loading capture: {args.filename}")
    capture = read_capture(args.filename)

    sample_rate = args.sample_rate if args.sample_rate else capture.sample_rate
    if args.sample_rate:
        print(f"Sample rate: {sample_rate/1e6:.1f} MHz (overridden)")
    else:
        print(f"Sample rate: {sample_rate/1e6:.1f} MHz (from file)")

    # Get channels
    ch1 = capture.channels.get("CHAN1")  # Main RF
    ch2 = capture.channels.get("CHAN2")  # Sub RF
    ch3 = capture.channels.get("CHAN3")  # Main demodulated
    ch4 = capture.channels.get("CHAN4")  # Sub demodulated

    if ch1 is None or ch3 is None:
        print("Error: Need at least Ch1 (RF) and Ch3 (demodulated) for comparison")
        return

    # Extract segment
    start = args.start
    end = start + args.length

    print(f"Processing samples {start} to {end}")

    # Check raw ADC data first
    rf_main_raw = ch1.data[start:end].astype(np.float64)
    print(f"RF Main RAW ADC: min={rf_main_raw.min():.0f}, max={rf_main_raw.max():.0f}, range={rf_main_raw.max()-rf_main_raw.min():.0f}")
    print(f"Ch1 preamble: {ch1.preamble[:100]}...")

    # Use raw ADC or voltage conversion
    if args.raw_adc:
        print("Using RAW ADC values (centered)")
        rf_main = rf_main_raw - np.mean(rf_main_raw)  # Center around zero
        demod_main_hw = ch3.data[start:end].astype(np.float64) - 128  # Center around zero
    else:
        rf_main = ch1.to_voltage()[start:end]
        demod_main_hw = ch3.to_voltage()[start:end]

    if ch2 is not None and ch4 is not None:
        rf_sub = ch2.to_voltage()[start:end]
        demod_sub_hw = ch4.to_voltage()[start:end]
        has_sub = True
    else:
        has_sub = False

    print(f"RF Main: min={rf_main.min():.3f}V, max={rf_main.max():.3f}V")
    print(f"Demod Main (HW): min={demod_main_hw.min():.3f}V, max={demod_main_hw.max():.3f}V")

    # Compute spectrum to check carrier frequency
    print("Analyzing RF spectrum...")
    fft_size = min(len(rf_main), 65536)
    rf_fft = np.fft.rfft(rf_main[:fft_size])
    freqs = np.fft.rfftfreq(fft_size, 1/sample_rate)
    peak_idx = np.argmax(np.abs(rf_fft[10:len(rf_fft)//2])) + 10  # Skip DC
    peak_freq = freqs[peak_idx]
    print(f"Peak frequency in RF (FFT): {peak_freq/1e6:.2f} MHz")

    # Also measure from zero crossings in time domain
    rf_centered = rf_main - np.mean(rf_main)
    zero_crossings = np.where(np.diff(np.sign(rf_centered)))[0]
    if len(zero_crossings) > 10:
        crossing_intervals = np.diff(zero_crossings)
        median_half_period = np.median(crossing_intervals)
        freq_from_crossings = sample_rate / (2 * median_half_period)
        print(f"Carrier freq (zero-crossing): {freq_from_crossings/1e6:.2f} MHz")
        print(f"  (half-period: {median_half_period:.1f} samples = {median_half_period/sample_rate*1e9:.1f} ns)")

    # Bandpass filter RF around carrier (optional)
    if args.no_bandpass:
        print("Skipping bandpass filter")
        rf_main_filtered = rf_main - np.mean(rf_main)  # Just remove DC
    else:
        print(f"Bandpass filtering RF: {args.carrier_low/1e6:.1f} - {args.carrier_high/1e6:.1f} MHz")
        rf_main_filtered = bandpass_filter(rf_main, sample_rate, args.carrier_low, args.carrier_high)
    print(f"RF Filtered: min={rf_main_filtered.min():.3f}, max={rf_main_filtered.max():.3f}")

    # Demodulate
    print(f"Demodulating using {args.method} method...")
    if args.method == "derivative":
        demod_main_sw = fm_demod_derivative(rf_main_filtered, sample_rate, args.lowpass)
    elif args.method == "hilbert":
        demod_main_sw = fm_demod_hilbert(rf_main_filtered, sample_rate, args.lowpass)
    elif args.method == "zero_crossing":
        demod_main_sw = fm_demod_zero_crossing(rf_main_filtered, sample_rate)

    # Apply de-emphasis if requested
    if args.deemphasis:
        print(f"Applying de-emphasis: {args.deemphasis} MHz corner frequency")
        demod_main_sw = deemphasis_filter(demod_main_sw, sample_rate, corner_freq=args.deemphasis * 1e6)

    # Normalize for comparison
    demod_main_hw_norm = normalize_signal(demod_main_hw)
    demod_main_sw_norm = normalize_signal(demod_main_sw)

    # Find best time alignment using cross-correlation
    print("Finding best time alignment...")
    max_shift = min(10000, len(demod_main_hw_norm) // 4)
    cross_corr = np.correlate(demod_main_hw_norm[:max_shift*2], demod_main_sw_norm[:max_shift*2], mode='full')
    best_shift = np.argmax(np.abs(cross_corr)) - (max_shift*2 - 1)
    best_corr_value = cross_corr[np.argmax(np.abs(cross_corr))]
    print(f"Best shift: {best_shift} samples ({best_shift/sample_rate*1e6:.2f} µs), cross-corr: {best_corr_value:.4f}")

    # Apply shift for comparison
    if best_shift > 0:
        demod_main_sw_aligned = np.concatenate([np.zeros(best_shift), demod_main_sw_norm[:-best_shift]])
    elif best_shift < 0:
        demod_main_sw_aligned = np.concatenate([demod_main_sw_norm[-best_shift:], np.zeros(-best_shift)])
    else:
        demod_main_sw_aligned = demod_main_sw_norm

    # Compute correlation (both normal and inverted)
    correlation = np.corrcoef(demod_main_hw_norm, demod_main_sw_norm)[0, 1]
    correlation_aligned = np.corrcoef(demod_main_hw_norm, demod_main_sw_aligned)[0, 1]
    correlation_inverted = np.corrcoef(demod_main_hw_norm, -demod_main_sw_norm)[0, 1]
    print(f"Correlation (raw): {correlation:.4f}")
    print(f"Correlation (aligned): {correlation_aligned:.4f}")
    print(f"Correlation (inverted): {correlation_inverted:.4f}")

    # Time axis in microseconds
    time_us = np.arange(len(rf_main)) / sample_rate * 1e6

    # Plot
    fig, axes = plt.subplots(5, 1, figsize=(14, 12))

    # Plot 1: RF Spectrum
    fft_plot_size = min(len(rf_main), 65536)
    rf_fft_plot = np.fft.rfft(rf_main[:fft_plot_size])
    freqs_plot = np.fft.rfftfreq(fft_plot_size, 1/sample_rate) / 1e6  # MHz
    axes[0].semilogy(freqs_plot, np.abs(rf_fft_plot), 'b-', linewidth=0.5)
    axes[0].axvline(args.carrier_low/1e6, color='r', linestyle='--', alpha=0.5, label='Bandpass')
    axes[0].axvline(args.carrier_high/1e6, color='r', linestyle='--', alpha=0.5)
    axes[0].axvline(peak_freq/1e6, color='g', linestyle='-', alpha=0.7, label=f'Peak: {peak_freq/1e6:.1f} MHz')
    axes[0].set_ylabel('RF Spectrum')
    axes[0].set_xlabel('Frequency (MHz)')
    axes[0].set_xlim(0, sample_rate/2e6)
    axes[0].legend(loc='upper right')
    axes[0].set_title(f'W-VHS FM Demodulation Comparison (method: {args.method})')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Raw RF (time domain, small segment)
    plot_samples = min(1000, len(rf_main))
    axes[1].plot(time_us[:plot_samples], rf_main[:plot_samples], 'b-', linewidth=0.5, alpha=0.7)
    axes[1].set_ylabel('RF Main (V)')
    axes[1].set_xlabel('Time (µs)')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Filtered RF (small segment)
    axes[2].plot(time_us[:plot_samples], rf_main_filtered[:plot_samples], 'g-', linewidth=0.5, alpha=0.7)
    axes[2].set_ylabel('RF Filtered (V)')
    axes[2].set_xlabel('Time (µs)')
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Demodulated comparison (normalized)
    axes[3].plot(time_us, demod_main_hw_norm, 'b-', linewidth=0.8, label='Hardware (Ch3)', alpha=0.7)
    axes[3].plot(time_us, demod_main_sw_norm, 'r-', linewidth=0.8, label=f'Software ({args.method})', alpha=0.7)
    axes[3].set_ylabel('Demodulated (normalized)')
    axes[3].set_xlabel('Time (µs)')
    axes[3].legend(loc='upper right')
    axes[3].grid(True, alpha=0.3)
    axes[3].set_title(f'Corr: raw={correlation:.3f}, aligned={correlation_aligned:.3f}, inverted={correlation_inverted:.3f}')

    # Plot 5: Aligned comparison
    axes[4].plot(time_us, demod_main_hw_norm, 'b-', linewidth=0.8, label='Hardware (Ch3)', alpha=0.7)
    axes[4].plot(time_us, demod_main_sw_aligned, 'r-', linewidth=0.8, label=f'Software (aligned by {best_shift})', alpha=0.7)
    axes[4].set_ylabel('Aligned (normalized)')
    axes[4].set_xlabel('Time (µs)')
    axes[4].legend(loc='upper right')
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_png = args.filename.replace('.bin', f'_fm_demod_{args.method}.png')
    plt.savefig(output_png, dpi=150)
    print(f"Saved plot: {output_png}")

    plt.show()


if __name__ == "__main__":
    main()
