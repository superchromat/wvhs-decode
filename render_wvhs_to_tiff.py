#!/usr/bin/env python3
"""
Render W-VHS color video from Main (Ch3) and Sub (Ch4) demodulated channels.

Scanline structure (at 2048 pixels per line):
  - Pixels 0-50: H sync
  - Pixels 50-580: Chroma (Pb from Main/Ch3, Pr from Sub/Ch4)
  - Pixels 581-2022: Luma (Y from Main/Ch3)

The chroma is subsampled and needs to be stretched to match luma width.
"""

import numpy as np
import sys
from PIL import Image
from read_wvhs_capture import read_capture
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


def voltage_to_greyscale(voltages, vmin, vmax):
    """Convert voltages to 0-255 greyscale."""
    normalized = np.clip((voltages - vmin) / (vmax - vmin), 0, 1)
    return (normalized * 255).astype(np.uint8)


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


def render_fields(filename, pixels_per_line=2048, sync_threshold=None, invert=False, grayscale=False, saturation=1.0, line_offset=50, brightness=0.0, gamma=1.0):
    """Render video fields from Main (Ch3) and Sub (Ch4) demodulated channels.

    W-VHS encodes:
    - Main (Ch3): Even Y lines + Pr chroma
    - Sub (Ch4): Odd Y lines + Pb chroma

    Each channel's sync pulses are detected independently for proper alignment.
    Main vsync occurs ~7 lines before sub vsync, requiring line offset compensation.
    """

    print(f"Reading capture: {filename}")
    capture = read_capture(filename)

    # Get Ch3 - Main Demodulated (Y + Pb)
    ch3 = capture.channels.get("CHAN3")
    if ch3 is None:
        print("Error: CHAN3 (Main Demodulated) not found in capture")
        sys.exit(1)

    # Get Ch4 - Sub Demodulated (Pr)
    ch4 = capture.channels.get("CHAN4")
    if ch4 is None and not grayscale:
        print("Error: CHAN4 (Sub Demodulated) not found in capture")
        print("Use --grayscale to render Y channel only")
        sys.exit(1)

    print(f"Main channel: {ch3.description}")
    if ch4:
        print(f"Sub channel: {ch4.description}")
    print(f"Total samples: {len(ch3.data):,}")
    print(f"Sample rate: {capture.sample_rate/1e9:.3f} Gsps")

    # Convert to voltages
    print("Converting to voltages...")
    voltages_main = ch3.to_voltage()
    voltages_sub = ch4.to_voltage() if ch4 else None
    x_increment = ch3.x_increment

    if invert:
        print("Inverting signal...")
        voltages_main = -voltages_main
        if voltages_sub is not None:
            voltages_sub = -voltages_sub

    # Voltage statistics (main channel for sync detection)
    v_min = np.min(voltages_main)
    v_max = np.max(voltages_main)
    v_mean = np.mean(voltages_main)
    v_p1 = np.percentile(voltages_main, 1)
    v_p5 = np.percentile(voltages_main, 5)
    v_p50 = np.percentile(voltages_main, 50)
    v_p95 = np.percentile(voltages_main, 95)
    v_p99 = np.percentile(voltages_main, 99)

    print(f"\nMain channel voltage statistics:")
    print(f"  Min: {v_min:.3f}V  Max: {v_max:.3f}V  Mean: {v_mean:.3f}V")
    print(f"  Percentiles: 1%={v_p1:.3f}V  5%={v_p5:.3f}V  50%={v_p50:.3f}V  95%={v_p95:.3f}V  99%={v_p99:.3f}V")

    # Find sync threshold - for demodulated video, sync tips are the lowest voltages
    if sync_threshold is None:
        sync_threshold = v_p1 + (v_p5 - v_p1) * 0.5

    print(f"\nSync threshold: {sync_threshold:.3f}V")

    # Detect all sync pulses (from main channel)
    print("Detecting sync pulses...")
    is_sync = (voltages_main < sync_threshold).astype(np.uint8)
    diff = np.diff(is_sync.astype(np.int8))
    rising = np.where(diff == 1)[0] + 1
    falling = np.where(diff == -1)[0] + 1

    if is_sync[0] == 1:
        rising = np.concatenate([[0], rising])
    if is_sync[-1] == 1:
        falling = np.concatenate([falling, [len(is_sync)]])

    min_len = min(len(rising), len(falling))
    rising = rising[:min_len]
    falling = falling[:min_len]

    widths_us = (falling - rising) * x_increment * 1e6
    valid = (widths_us >= 1.0) & (widths_us < 100.0)
    rising = rising[valid]
    widths_us = widths_us[valid]

    print(f"Found {len(rising)} sync pulses")

    if len(rising) == 0:
        print("\nNo sync pulses detected! Possible causes:")
        print("  - Sync threshold is wrong (try --threshold)")
        print("  - Signal is not a demodulated video signal")
        print("  - Signal level is inverted (try --invert)")
        print(f"\nTry running with --threshold {v_mean:.3f} or adjust based on voltage stats above")
        sys.exit(1)

    # Classify pulses (hsync vs vsync based on width)
    sorted_widths = np.sort(widths_us)
    gaps = np.diff(sorted_widths)
    if len(gaps) > 0:
        max_gap_idx = np.argmax(gaps)
        threshold = (sorted_widths[max_gap_idx] + sorted_widths[max_gap_idx + 1]) / 2
        threshold = np.clip(threshold, 6.0, 15.0)
    else:
        threshold = 8.0

    is_hsync = widths_us < threshold
    all_line_starts = rising

    vsync_starts = rising[~is_hsync]
    print(f"Found {np.sum(is_hsync)} hsync pulses, {len(vsync_starts)} vsync pulses")

    if len(vsync_starts) == 0:
        print("\nNo vsync pulses detected - treating entire capture as one field")
        vsync_starts = np.array([rising[0]])

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
    vsync_groups.append(len(voltages_main))

    print(f"Found {len(vsync_groups) - 1} fields")

    # Determine voltage scaling for Y, Pb, Pr
    # Y (luma): blanking level to white
    # Pb, Pr (chroma): centered around blanking level
    # We'll use percentiles from the appropriate regions to estimate

    print("\nAnalyzing signal levels...")


    # Detect sync on sub channel separately
    print("Detecting sync pulses on sub channel...")
    if voltages_sub is not None:
        # Use same threshold logic for sub channel
        v_sub_p1 = np.percentile(voltages_sub, 1)
        v_sub_p5 = np.percentile(voltages_sub, 5)
        sync_threshold_sub = v_sub_p1 + (v_sub_p5 - v_sub_p1) * 0.5

        is_sync_sub = (voltages_sub < sync_threshold_sub).astype(np.uint8)
        diff_sub = np.diff(is_sync_sub.astype(np.int8))
        rising_sub = np.where(diff_sub == 1)[0] + 1
        falling_sub = np.where(diff_sub == -1)[0] + 1

        if is_sync_sub[0] == 1:
            rising_sub = np.concatenate([[0], rising_sub])
        if is_sync_sub[-1] == 1:
            falling_sub = np.concatenate([falling_sub, [len(is_sync_sub)]])

        min_len_sub = min(len(rising_sub), len(falling_sub))
        rising_sub = rising_sub[:min_len_sub]
        falling_sub = falling_sub[:min_len_sub]

        widths_us_sub = (falling_sub - rising_sub) * x_increment * 1e6
        valid_sub = (widths_us_sub >= 1.0) & (widths_us_sub < 100.0)
        rising_sub = rising_sub[valid_sub]
        widths_us_sub = widths_us_sub[valid_sub]

        # Classify hsync vs vsync for sub
        is_hsync_sub = widths_us_sub < threshold
        all_line_starts_sub = rising_sub

        print(f"Found {len(rising_sub)} sync pulses on sub ({np.sum(is_hsync_sub)} hsync)")
    else:
        all_line_starts_sub = None

    # Sample luma and chroma regions for level analysis
    sample_lines = min(100, len(rising) - 1)
    y_main_samples = []
    y_sub_samples = []
    pb_samples = []
    pr_samples = []

    for i in range(sample_lines):
        line_start = rising[i]
        line_end = rising[i + 1] if i + 1 < len(rising) else len(voltages_main)
        line_main = voltages_main[line_start:line_end]
        if len(line_main) > pixels_per_line:
            indices = np.linspace(0, len(line_main) - 1, pixels_per_line).astype(int)
            resampled = line_main[indices]
            y_main_samples.extend(resampled[LUMA_START:LUMA_END])
            pr_samples.extend(resampled[CHROMA_START:CHROMA_END])  # Pr from main
        if voltages_sub is not None:
            line_sub = voltages_sub[line_start:line_end]
            if len(line_sub) > pixels_per_line:
                indices = np.linspace(0, len(line_sub) - 1, pixels_per_line).astype(int)
                resampled = line_sub[indices]
                y_sub_samples.extend(resampled[LUMA_START:LUMA_END])
                pb_samples.extend(resampled[CHROMA_START:CHROMA_END])  # Pb from sub

    # Print detailed voltage statistics
    print(f"\nLuma region voltage statistics:")
    print(f"  Main Y: min={np.min(y_main_samples):.3f}V, max={np.max(y_main_samples):.3f}V, "
          f"p1={np.percentile(y_main_samples, 1):.3f}V, p99={np.percentile(y_main_samples, 99):.3f}V")
    if y_sub_samples:
        print(f"  Sub Y:  min={np.min(y_sub_samples):.3f}V, max={np.max(y_sub_samples):.3f}V, "
              f"p1={np.percentile(y_sub_samples, 1):.3f}V, p99={np.percentile(y_sub_samples, 99):.3f}V")

    print(f"\nChroma region voltage statistics:")
    print(f"  Pb: min={np.min(pb_samples):.3f}V, max={np.max(pb_samples):.3f}V, median={np.median(pb_samples):.3f}V")
    if pr_samples:
        print(f"  Pr: min={np.min(pr_samples):.3f}V, max={np.max(pr_samples):.3f}V, median={np.median(pr_samples):.3f}V")

    # Calculate voltage scale factor from the data
    # User measured: sync tip=-1V, black=-0.9V, chroma center=-0.77V, 400mV=100%
    # The voltage conversion from scope preamble is wrong, so we need to find the scale factor

    # Find sync tip level (lowest voltage in the signal, from overall data not just luma)
    main_sync_tip = np.percentile(voltages_main, 0.1)

    # Black level should be ~100mV above sync tip
    # The ratio of measured range to actual range gives us the scale factor
    # Measured luma range (p1-p99) should correspond to roughly the video range
    measured_luma_range = np.percentile(y_main_samples, 99) - np.percentile(y_main_samples, 1)

    # Real video range is about 400mV (black to white)
    # Scale factor converts our measured voltages to real voltages
    v_scale_factor = measured_luma_range / 0.400

    # Black level = sync tip + 100mV (converted to our units)
    main_black = main_sync_tip + (0.100 * v_scale_factor)
    main_scale = 0.400 * v_scale_factor  # 400mV in our voltage units

    if y_sub_samples:
        sub_sync_tip = np.percentile(voltages_sub, 0.1)
        sub_measured_range = np.percentile(y_sub_samples, 99) - np.percentile(y_sub_samples, 1)
        sub_v_scale = sub_measured_range / 0.400
        sub_black = sub_sync_tip + (0.100 * sub_v_scale)
        sub_scale = 0.400 * sub_v_scale
    else:
        sub_black = main_black
        sub_scale = main_scale

    # Chroma centers (measured from data) and fixed 350mV scale
    pr_center = np.median(pr_samples) if pr_samples else main_black
    pb_center = np.median(pb_samples) if pb_samples else sub_black

    # Use same 350mV scale for chroma as luma
    pr_scale = main_scale
    pb_scale = sub_scale

    print(f"\nNormalization (400mV = 100%, scale factor={v_scale_factor:.2f}):")
    print(f"  Main: sync_tip={main_sync_tip:.3f}V, black={main_black:.3f}V, scale={main_scale:.3f}V")
    print(f"  Sub:  sync_tip={sub_sync_tip:.3f}V, black={sub_black:.3f}V, scale={sub_scale:.3f}V" if y_sub_samples else "")
    print(f"  Pr center: {pr_center:.3f}V (from main)")
    print(f"  Pb center: {pb_center:.3f}V (from sub)")
    print(f"  Line offset: {line_offset} (main vsync leads sub by this many lines)")

    # Process each field
    base_filename = filename.replace('.bin', '')

    for field_idx in range(len(vsync_groups) - 1):
        field_start = vsync_groups[field_idx]
        field_end = vsync_groups[field_idx + 1]

        field_line_starts = all_line_starts[(all_line_starts >= field_start) & (all_line_starts < field_end)]

        if len(field_line_starts) == 0:
            continue

        print(f"\nRendering field {field_idx} ({len(field_line_starts)} lines)...")

        y_even_rows = []  # From main (Ch3) - even lines
        y_odd_rows = []   # From sub (Ch4) - odd lines
        pr_rows = []  # Pr from main (Ch3)
        pb_rows = []  # Pb from sub (Ch4)

        # Get sub channel line starts in this field region
        if all_line_starts_sub is not None:
            field_line_starts_sub = all_line_starts_sub[
                (all_line_starts_sub >= field_start - 100000) &
                (all_line_starts_sub < field_end + 100000)
            ]
        else:
            field_line_starts_sub = None

        for line_idx in range(len(field_line_starts) - 1):
            line_start = field_line_starts[line_idx]
            line_end = field_line_starts[line_idx + 1]

            # Process main channel (Y_even + Pr)
            line_main = voltages_main[line_start:line_end]
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

            # Process sub channel (Y_odd + Pb) - find corresponding sync on sub
            # Main vsync occurs ~line_offset lines before sub vsync
            # So main line N corresponds to sub line (N - line_offset)
            if field_line_starts_sub is not None and len(field_line_starts_sub) > 0:
                # Find the sub line that corresponds to this main line
                # First find which main line number this is within the field
                main_line_num = line_idx

                # The corresponding sub line number (accounting for vsync offset)
                # Main vsync leads, so sub is behind - add offset to get corresponding sub line
                sub_line_num = main_line_num + line_offset

                if 0 <= sub_line_num < len(field_line_starts_sub) - 1:
                    sub_line_start = field_line_starts_sub[sub_line_num]
                    sub_line_end = field_line_starts_sub[sub_line_num + 1]

                    line_sub = voltages_sub[sub_line_start:sub_line_end]
                    if len(line_sub) > 0:
                        indices = np.linspace(0, len(line_sub) - 1, pixels_per_line).astype(int)
                        resampled_sub = line_sub[indices]
                        y_odd_line = resampled_sub[LUMA_START:LUMA_END]
                        pb_line = resampled_sub[CHROMA_START:CHROMA_END]
                    else:
                        y_odd_line = np.full(LUMA_WIDTH, sub_black)
                        pb_line = np.zeros(CHROMA_WIDTH)
                else:
                    # Line offset puts us outside valid sub range - use blank
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

        # Normalize each Y channel using measured range
        # Apply brightness adjustment (negative = darker, shifts black level up)
        y_even_norm = np.clip((y_even_image - main_black) / main_scale + brightness, 0, 1)

        # Interleave even and odd Y lines
        # Even lines (from main) go to rows 0, 2, 4, ...
        # Odd lines (from sub) go to rows 1, 3, 5, ...
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

            # Normalize Pb, Pr to [-0.5, 0.5] using fixed 350mV scale
            # Apply saturation control (lower = less saturated)
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
    import argparse

    parser = argparse.ArgumentParser(
        description="Render W-VHS color video from Main (Ch3) and Sub (Ch4) channels to TIFF files."
    )
    parser.add_argument("filename", help="W-VHS capture file (.bin)")
    parser.add_argument("-p", "--pixels", type=int, default=2048,
                        help="Pixels per line (default: 2048)")
    parser.add_argument("-t", "--threshold", type=float, default=None,
                        help="Sync threshold voltage (auto-detected if not specified)")
    parser.add_argument("--invert", action="store_true",
                        help="Invert signal (sync pulses are positive)")
    parser.add_argument("--grayscale", action="store_true",
                        help="Output grayscale Y channel only (no color)")
    parser.add_argument("--saturation", type=float, default=1.0,
                        help="Color saturation (default: 1.0, lower = less saturated)")
    parser.add_argument("--line-offset", type=int, default=11,
                        help="Line offset between main and sub vsync (default: 11, main leads sub)")
    parser.add_argument("--brightness", type=float, default=-0.25,
                        help="Brightness adjustment (default: -0.25, negative = darker)")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Gamma correction to apply (default: 1.0, try 0.9 or 1.1)")

    args = parser.parse_args()

    try:
        render_fields(args.filename, args.pixels, args.threshold, args.invert, args.grayscale, args.saturation, args.line_offset, args.brightness, args.gamma)
    except FileNotFoundError:
        print(f"Error: File '{args.filename}' not found")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
