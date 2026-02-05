#!/usr/bin/env python3
"""
Render W-VHS color video from demodulated stereo WAV file.

Input: Stereo WAV with Main demodulated (Left) and Sub demodulated (Right)
Output: TIFF images per field

Scanline structure (at 2048 pixels per line):
  - Pixels 0-50: H sync
  - Pixels 50-580: Chroma (Pr from Main/L, Pb from Sub/R)
  - Pixels 581-2022: Luma (Y even from Main/L, Y odd from Sub/R)

The chroma is subsampled and needs to be stretched to match luma width.
"""

import numpy as np
import sys
import wave
import argparse
from PIL import Image
from scipy import ndimage


# Scanline regions (at 2048 pixels per line)
HSYNC_END = 50
CHROMA_START = 50
CHROMA_END = 580
LUMA_START = 581
LUMA_END = 2022

# Derived dimensions
CHROMA_WIDTH = CHROMA_END - CHROMA_START  # 530 pixels
LUMA_WIDTH = LUMA_END - LUMA_START        # 1441 pixels


def ypbpr_to_rgb(y, pb, pr):
    """
    Convert YPbPr to RGB using ITU-R BT.601 matrix.

    Y is in range [0, 1] (black to white)
    Pb, Pr are in range [-0.5, 0.5] (centered at 0)

    Returns RGB in range [0, 255]
    """
    # BT.601 conversion matrix
    r = y + 1.402 * pr
    g = y - 0.344136 * pb - 0.714136 * pr
    b = y + 1.772 * pb

    # Clip and scale to 8-bit
    r = np.clip(r, 0, 1) * 255
    g = np.clip(g, 0, 1) * 255
    b = np.clip(b, 0, 1) * 255

    return r.astype(np.uint8), g.astype(np.uint8), b.astype(np.uint8)


def upsample_chroma(chroma, target_width):
    """Upsample chroma array to match luma width using linear interpolation."""
    if chroma.ndim == 1:
        # Single line
        x_old = np.linspace(0, 1, len(chroma))
        x_new = np.linspace(0, 1, target_width)
        return np.interp(x_new, x_old, chroma)
    else:
        # 2D array (multiple lines)
        zoom_factor = target_width / chroma.shape[1]
        return ndimage.zoom(chroma, (1, zoom_factor), order=1)


def read_wav_stereo(filename):
    """Read stereo WAV file and return (main_data, sub_data, sample_rate)."""
    with wave.open(filename, 'rb') as wav:
        n_channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        n_frames = wav.getnframes()

        if n_channels != 2:
            raise ValueError(f"Expected stereo WAV, got {n_channels} channels")

        raw_data = wav.readframes(n_frames)

        if sample_width == 1:
            # 8-bit unsigned
            audio = np.frombuffer(raw_data, dtype=np.uint8).reshape(-1, 2)
            # Convert to float centered around 0
            audio = audio.astype(np.float64) - 128.0
        elif sample_width == 2:
            # 16-bit signed
            audio = np.frombuffer(raw_data, dtype=np.int16).reshape(-1, 2)
            audio = audio.astype(np.float64)
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")

        main_data = audio[:, 0]  # Left = Main
        sub_data = audio[:, 1]   # Right = Sub

        return main_data, sub_data, sample_rate, sample_width


def render_fields_from_wav(filename, pixels_per_line=2048, sync_threshold=None,
                           invert=False, grayscale=False, saturation=1.0,
                           line_offset=9, brightness=-0.25, gamma=1.0,
                           output_prefix=None):
    """Render video fields from demodulated stereo WAV.

    W-VHS encodes:
    - Main (Left channel): Even Y lines + Pr chroma
    - Sub (Right channel): Odd Y lines + Pb chroma

    Each channel's sync pulses are detected independently for proper alignment.
    Main vsync occurs ~11 lines before sub vsync, requiring line offset compensation.
    """

    print(f"Reading WAV: {filename}")
    main_signal, sub_signal, sample_rate, sample_width = read_wav_stereo(filename)

    x_increment = 1.0 / sample_rate

    print(f"  Sample rate: {sample_rate/1e6:.1f} MHz")
    print(f"  Sample width: {sample_width * 8} bits")
    print(f"  Total samples: {len(main_signal):,}")
    print(f"  Duration: {len(main_signal)/sample_rate*1000:.2f} ms")

    if invert:
        print("Inverting signal...")
        main_signal = -main_signal
        sub_signal = -sub_signal

    # Signal statistics (main channel for sync detection)
    v_min = np.min(main_signal)
    v_max = np.max(main_signal)
    v_mean = np.mean(main_signal)
    v_p1 = np.percentile(main_signal, 1)
    v_p5 = np.percentile(main_signal, 5)
    v_p50 = np.percentile(main_signal, 50)
    v_p95 = np.percentile(main_signal, 95)
    v_p99 = np.percentile(main_signal, 99)

    print(f"\nMain channel signal statistics:")
    print(f"  Min: {v_min:.1f}  Max: {v_max:.1f}  Mean: {v_mean:.1f}")
    print(f"  Percentiles: 1%={v_p1:.1f}  5%={v_p5:.1f}  50%={v_p50:.1f}  95%={v_p95:.1f}  99%={v_p99:.1f}")

    # Find sync threshold - for demodulated video, sync tips are the lowest values
    if sync_threshold is None:
        sync_threshold = v_p1 + (v_p5 - v_p1) * 0.5

    print(f"\nSync threshold: {sync_threshold:.1f}")

    # Detect all sync pulses (from main channel)
    # Use center of sync pulse for more stable line alignment
    print("Detecting sync pulses...")
    is_sync = (main_signal < sync_threshold).astype(np.uint8)
    diff = np.diff(is_sync.astype(np.int8))
    sync_starts = np.where(diff == 1)[0] + 1   # Falling edge (entering sync)
    sync_ends = np.where(diff == -1)[0] + 1    # Rising edge (leaving sync)

    if is_sync[0] == 1:
        sync_starts = np.concatenate([[0], sync_starts])
    if is_sync[-1] == 1:
        sync_ends = np.concatenate([sync_ends, [len(is_sync)]])

    min_len = min(len(sync_starts), len(sync_ends))
    sync_starts = sync_starts[:min_len]
    sync_ends = sync_ends[:min_len]

    widths_us = (sync_ends - sync_starts) * x_increment * 1e6
    valid = (widths_us >= 1.0) & (widths_us < 100.0)
    sync_starts = sync_starts[valid]
    sync_ends = sync_ends[valid]
    widths_us = widths_us[valid]

    # Use START of sync pulse as line reference point
    # The scanline regions assume: pixel 0 = start of sync, pixel 50 = end of sync
    sync_refs = sync_starts

    print(f"Found {len(sync_refs)} sync pulses")

    if len(sync_refs) == 0:
        print("\nNo sync pulses detected! Possible causes:")
        print("  - Sync threshold is wrong (try --threshold)")
        print("  - Signal is not a demodulated video signal")
        print("  - Signal level is inverted (try --invert)")
        print(f"\nTry running with --threshold {v_mean:.1f} or adjust based on signal stats above")
        sys.exit(1)

    # Classify pulses (hsync vs vsync based on width)
    sorted_widths = np.sort(widths_us)
    gaps = np.diff(sorted_widths)
    if len(gaps) > 0:
        max_gap_idx = np.argmax(gaps)
        width_threshold = (sorted_widths[max_gap_idx] + sorted_widths[max_gap_idx + 1]) / 2
        width_threshold = np.clip(width_threshold, 6.0, 15.0)
    else:
        width_threshold = 8.0

    is_hsync = widths_us < width_threshold
    all_line_starts = sync_refs

    vsync_starts = sync_refs[~is_hsync]
    print(f"Found {np.sum(is_hsync)} hsync pulses, {len(vsync_starts)} vsync pulses")

    if len(vsync_starts) == 0:
        print("\nNo vsync pulses detected - treating entire capture as one field")
        vsync_starts = np.array([sync_refs[0]])

    # Group vsync pulses into field boundaries
    print("Identifying fields...")
    vsync_groups = []
    current_group_start = vsync_starts[0]
    for i in range(1, len(vsync_starts)):
        gap = (vsync_starts[i] - vsync_starts[i-1]) * x_increment * 1e3
        if gap > 1.0:
            vsync_groups.append(current_group_start)
            current_group_start = vsync_starts[i]
    vsync_groups.append(current_group_start)
    vsync_groups.append(len(main_signal))

    print(f"Found {len(vsync_groups) - 1} fields")

    # Detect sync on sub channel separately (also using sync centers)
    print("Detecting sync pulses on sub channel...")
    v_sub_p1 = np.percentile(sub_signal, 1)
    v_sub_p5 = np.percentile(sub_signal, 5)
    sync_threshold_sub = v_sub_p1 + (v_sub_p5 - v_sub_p1) * 0.5

    is_sync_sub = (sub_signal < sync_threshold_sub).astype(np.uint8)
    diff_sub = np.diff(is_sync_sub.astype(np.int8))
    sync_starts_sub = np.where(diff_sub == 1)[0] + 1
    sync_ends_sub = np.where(diff_sub == -1)[0] + 1

    if is_sync_sub[0] == 1:
        sync_starts_sub = np.concatenate([[0], sync_starts_sub])
    if is_sync_sub[-1] == 1:
        sync_ends_sub = np.concatenate([sync_ends_sub, [len(is_sync_sub)]])

    min_len_sub = min(len(sync_starts_sub), len(sync_ends_sub))
    sync_starts_sub = sync_starts_sub[:min_len_sub]
    sync_ends_sub = sync_ends_sub[:min_len_sub]

    widths_us_sub = (sync_ends_sub - sync_starts_sub) * x_increment * 1e6
    valid_sub = (widths_us_sub >= 1.0) & (widths_us_sub < 100.0)
    sync_starts_sub = sync_starts_sub[valid_sub]
    sync_ends_sub = sync_ends_sub[valid_sub]
    widths_us_sub = widths_us_sub[valid_sub]

    # Use start of sync pulse (same as main channel)
    sync_refs_sub = sync_starts_sub

    is_hsync_sub = widths_us_sub < width_threshold
    all_line_starts_sub = sync_refs_sub

    print(f"Found {len(sync_refs_sub)} sync pulses on sub ({np.sum(is_hsync_sub)} hsync)")

    # Sample luma and chroma regions for level analysis
    print("\nAnalyzing signal levels...")
    sample_lines = min(100, len(sync_refs) - 1)
    y_main_samples = []
    y_sub_samples = []
    pb_samples = []
    pr_samples = []

    for i in range(sample_lines):
        line_start = sync_refs[i]
        line_end = sync_refs[i + 1] if i + 1 < len(sync_refs) else len(main_signal)
        line_main = main_signal[line_start:line_end]
        if len(line_main) > pixels_per_line:
            indices = np.linspace(0, len(line_main) - 1, pixels_per_line).astype(int)
            resampled = line_main[indices]
            y_main_samples.extend(resampled[LUMA_START:LUMA_END])
            pr_samples.extend(resampled[CHROMA_START:CHROMA_END])  # Pr from main

        line_sub = sub_signal[line_start:line_end]
        if len(line_sub) > pixels_per_line:
            indices = np.linspace(0, len(line_sub) - 1, pixels_per_line).astype(int)
            resampled = line_sub[indices]
            y_sub_samples.extend(resampled[LUMA_START:LUMA_END])
            pb_samples.extend(resampled[CHROMA_START:CHROMA_END])  # Pb from sub

    # Calculate normalization parameters using same approach as render_wvhs_to_tiff.py
    # Use calibrated voltage scaling: 400mV = 100% (black to white), sync tip + 100mV = black
    main_sync_tip = np.percentile(main_signal, 0.1)
    measured_luma_range = np.percentile(y_main_samples, 99) - np.percentile(y_main_samples, 1)

    # Scale factor maps our signal units to "equivalent volts"
    # measured_luma_range should correspond to ~400mV of real video
    v_scale_factor = measured_luma_range / 0.400 if measured_luma_range > 0 else 1.0

    # Black level = sync tip + 100mV equivalent
    main_black = main_sync_tip + (0.100 * v_scale_factor)
    main_scale = 0.400 * v_scale_factor  # 400mV in our signal units

    sub_sync_tip = np.percentile(sub_signal, 0.1)
    sub_measured_range = np.percentile(y_sub_samples, 99) - np.percentile(y_sub_samples, 1)
    sub_v_scale = sub_measured_range / 0.400 if sub_measured_range > 0 else 1.0
    sub_black = sub_sync_tip + (0.100 * sub_v_scale)
    sub_scale = 0.400 * sub_v_scale

    # Chroma centers and scales
    pr_center = np.median(pr_samples) if pr_samples else 0
    pb_center = np.median(pb_samples) if pb_samples else 0
    pr_scale = main_scale
    pb_scale = sub_scale

    print(f"\nNormalization (400mV=100%, scale_factor={v_scale_factor:.2f}):")
    print(f"  Main: sync_tip={main_sync_tip:.1f}, black={main_black:.1f}, scale={main_scale:.1f}")
    print(f"  Sub:  sync_tip={sub_sync_tip:.1f}, black={sub_black:.1f}, scale={sub_scale:.1f}")
    print(f"  Pr center: {pr_center:.1f} (from main)")
    print(f"  Pb center: {pb_center:.1f} (from sub)")
    print(f"  Line offset: {line_offset} (content alignment between main/sub)")

    # Output filename base
    if output_prefix:
        base_filename = output_prefix
    else:
        import os
        base_filename = os.path.splitext(filename)[0]

    # Process each field
    for field_idx in range(len(vsync_groups) - 1):
        field_start = vsync_groups[field_idx]
        field_end = vsync_groups[field_idx + 1]

        field_line_starts = all_line_starts[(all_line_starts >= field_start) & (all_line_starts < field_end)]

        if len(field_line_starts) == 0:
            continue

        print(f"\nRendering field {field_idx} ({len(field_line_starts)} lines)...")

        y_even_rows = []  # From main (L) - even lines
        y_odd_rows = []   # From sub (R) - odd lines
        pr_rows = []      # Pr from main (L)
        pb_rows = []      # Pb from sub (R)

        # Get sub channel line starts in this field region
        field_line_starts_sub = all_line_starts_sub[
            (all_line_starts_sub >= field_start - 100000) &
            (all_line_starts_sub < field_end + 100000)
        ]

        for line_idx in range(len(field_line_starts) - 1):
            line_start = field_line_starts[line_idx]
            line_end = field_line_starts[line_idx + 1]

            # Process main channel (Y_even + Pr)
            line_main = main_signal[line_start:line_end]
            if len(line_main) > 0:
                indices = np.linspace(0, len(line_main) - 1, pixels_per_line).astype(int)
                resampled_main = line_main[indices]

                y_even_line = resampled_main[LUMA_START:LUMA_END]
                pr_line = resampled_main[CHROMA_START:CHROMA_END]
            else:
                y_even_line = np.full(LUMA_WIDTH, main_black)
                pr_line = np.zeros(CHROMA_WIDTH)

            y_even_rows.append(y_even_line)
            pr_rows.append(pr_line)

            # Process sub channel (Y_odd + Pb) - find sub line with matching content
            # Time-based matching + line_offset to account for vsync difference
            # Main vsync leads sub vsync, so sub content is offset
            if len(field_line_starts_sub) > 0:
                # Find sub line that starts closest to this main line's start
                sub_idx = np.searchsorted(field_line_starts_sub, line_start)

                # Check neighbors to find closest time match
                best_idx = None
                best_dist = float('inf')
                for candidate in [sub_idx - 1, sub_idx, sub_idx + 1]:
                    if 0 <= candidate < len(field_line_starts_sub):
                        dist = abs(field_line_starts_sub[candidate] - line_start)
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = candidate

                # Apply line offset to get content-aligned sub line
                # Main vsync leads, so we need a later sub line for matching content
                if best_idx is not None:
                    best_idx += line_offset

                if best_idx is not None and 0 <= best_idx < len(field_line_starts_sub) - 1:
                    sub_line_start = field_line_starts_sub[best_idx]
                    sub_line_end = field_line_starts_sub[best_idx + 1]

                    line_sub = sub_signal[sub_line_start:sub_line_end]
                    if len(line_sub) > 0:
                        indices = np.linspace(0, len(line_sub) - 1, pixels_per_line).astype(int)
                        resampled_sub = line_sub[indices]
                        y_odd_line = resampled_sub[LUMA_START:LUMA_END]
                        pb_line = resampled_sub[CHROMA_START:CHROMA_END]
                    else:
                        y_odd_line = np.full(LUMA_WIDTH, sub_black)
                        pb_line = np.zeros(CHROMA_WIDTH)
                else:
                    y_odd_line = np.full(LUMA_WIDTH, sub_black)
                    pb_line = np.zeros(CHROMA_WIDTH)

                y_odd_rows.append(y_odd_line)
                pb_rows.append(pb_line)

        if len(y_even_rows) == 0:
            print(f"  Skipping field {field_idx} - no valid lines")
            continue

        # Convert to numpy arrays
        y_even_image = np.array(y_even_rows)
        y_odd_image = np.array(y_odd_rows) if y_odd_rows else None
        pb_image = np.array(pb_rows)
        pr_image = np.array(pr_rows) if pr_rows else None

        # Normalize Y channels
        y_even_norm = np.clip((y_even_image - main_black) / main_scale + brightness, 0, 1)

        # Interleave even and odd Y lines
        if y_odd_image is not None and len(y_odd_image) > 0:
            y_odd_norm = np.clip((y_odd_image - sub_black) / sub_scale + brightness, 0, 1)

            num_lines = min(len(y_even_norm), len(y_odd_norm))
            y_norm = np.zeros((num_lines * 2, LUMA_WIDTH), dtype=np.float64)
            y_norm[0::2] = y_even_norm[:num_lines]
            y_norm[1::2] = y_odd_norm[:num_lines]

            # Chroma needs to be duplicated for each pair of Y lines
            pb_interleaved = np.repeat(pb_image[:num_lines], 2, axis=0)
            pr_interleaved = np.repeat(pr_image[:num_lines], 2, axis=0)
            pb_image = pb_interleaved
            pr_image = pr_interleaved
        else:
            y_norm = y_even_norm

        if grayscale or pr_image is None:
            # Grayscale output
            if gamma != 1.0:
                image_grey = (np.clip(y_norm ** gamma, 0, 1) * 255).astype(np.uint8)
            else:
                image_grey = (y_norm * 255).astype(np.uint8)
            output_filename = f"{base_filename}_field{field_idx:02d}.tiff"
            img = Image.fromarray(image_grey, mode='L')
            img.save(output_filename)
            print(f"  Saved {output_filename} ({image_grey.shape[0]} x {image_grey.shape[1]})")
        else:
            # Color output - upsample chroma to match luma
            pb_upsampled = upsample_chroma(pb_image, LUMA_WIDTH)
            pr_upsampled = upsample_chroma(pr_image, LUMA_WIDTH)

            # Normalize Pb, Pr to [-0.5, 0.5]
            pb_norm = np.clip((pb_upsampled - pb_center) / pb_scale * saturation, -0.5, 0.5)
            pr_norm = np.clip((pr_upsampled - pr_center) / pr_scale * saturation, -0.5, 0.5)

            # Convert to RGB
            r, g, b = ypbpr_to_rgb(y_norm, pb_norm, pr_norm)

            # Apply gamma correction if specified
            if gamma != 1.0:
                r = (r / 255.0) ** gamma * 255
                g = (g / 255.0) ** gamma * 255
                b = (b / 255.0) ** gamma * 255
                r = np.clip(r, 0, 255).astype(np.uint8)
                g = np.clip(g, 0, 255).astype(np.uint8)
                b = np.clip(b, 0, 255).astype(np.uint8)

            # Stack into RGB image
            rgb_image = np.stack([r, g, b], axis=-1)

            output_filename = f"{base_filename}_field{field_idx:02d}.tiff"
            img = Image.fromarray(rgb_image, mode='RGB')
            img.save(output_filename)
            print(f"  Saved {output_filename} ({rgb_image.shape[0]} x {rgb_image.shape[1]} x 3)")

    print(f"\nDone! Created {len(vsync_groups) - 1} TIFF files")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render W-VHS color video from demodulated stereo WAV to TIFF files."
    )
    parser.add_argument("filename", help="Demodulated WAV file (stereo: Main=L, Sub=R)")
    parser.add_argument("-o", "--output", default=None,
                        help="Output filename prefix (default: input basename)")
    parser.add_argument("-p", "--pixels", type=int, default=2048,
                        help="Pixels per line (default: 2048)")
    parser.add_argument("-t", "--threshold", type=float, default=None,
                        help="Sync threshold (auto-detected if not specified)")
    parser.add_argument("--invert", action="store_true",
                        help="Invert signal (sync pulses are positive)")
    parser.add_argument("--grayscale", action="store_true",
                        help="Output grayscale Y channel only (no color)")
    parser.add_argument("--saturation", type=float, default=1.0,
                        help="Color saturation (default: 1.0, lower = less saturated)")
    parser.add_argument("--line-offset", type=int, default=9,
                        help="Line offset for main/sub content alignment (default: 7)")
    parser.add_argument("--brightness", type=float, default=-0.25,
                        help="Brightness adjustment (default: -0.25, negative = darker)")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Gamma correction (default: 1.0)")

    args = parser.parse_args()

    try:
        render_fields_from_wav(
            args.filename,
            pixels_per_line=args.pixels,
            sync_threshold=args.threshold,
            invert=args.invert,
            grayscale=args.grayscale,
            saturation=args.saturation,
            line_offset=args.line_offset,
            brightness=args.brightness,
            gamma=args.gamma,
            output_prefix=args.output
        )
    except FileNotFoundError:
        print(f"Error: File '{args.filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
