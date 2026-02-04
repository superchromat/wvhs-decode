#!/usr/bin/env python3
"""
Capture W-VHS oscilloscope traces from 4 channels.

Ch1: Main FM/RF
Ch2: Sub FM/RF
Ch3: Main - demodulated
Ch4: Sub - demodulated

Saves to a binary file with header containing metadata for later analysis.
"""

import pyvisa
import struct
import sys
import time
import argparse
from datetime import datetime

# Configuration
SCOPE_ADDRESS = "TCPIP::192.168.100.99::INSTR"
POINTS_PER_CHANNEL = 125_000_000  # 125Mpts
CHUNK_SIZE = 1_000_000
DEFAULT_OUTPUT = "wvhs_capture"

CHANNELS = {
    "CHAN1": "Main FM/RF",
    "CHAN2": "Sub FM/RF",
    "CHAN3": "Main Demodulated",
    "CHAN4": "Sub Demodulated",
}


def capture_channel(scope, channel, points_to_capture, chunk_size):
    """Capture data from a single channel, returns (data, preamble)."""

    print(f"\n--- Capturing {channel} ({CHANNELS[channel]}) ---")

    # Configure waveform parameters for this channel
    scope.write(f":WAV:SOUR {channel}")
    scope.write(":WAV:MODE RAW")
    scope.write(":WAV:FORM BYTE")

    # Get preamble information
    preamble = scope.query(":WAV:PRE?").strip()
    print(f"Preamble: {preamble}")

    # Capture data in chunks
    num_chunks = (points_to_capture + chunk_size - 1) // chunk_size
    print(f"Capturing {points_to_capture} points in {num_chunks} chunks...")

    start_time = time.time()
    channel_data = bytearray()

    for chunk_idx in range(num_chunks):
        start_point = chunk_idx * chunk_size + 1
        end_point = min(start_point + chunk_size - 1, points_to_capture)

        # Set start and stop points
        scope.write(f":WAV:STAR {start_point}")
        scope.write(f":WAV:STOP {end_point}")

        # Request data
        scope.write(":WAV:DATA?")

        # Read the IEEE 488.2 definite length block header
        header_char = scope.read_bytes(1)
        if header_char != b'#':
            raise ValueError(f"Expected '#' but got {header_char}")

        num_digits_char = scope.read_bytes(1)
        num_digits = int(num_digits_char)

        length_str = scope.read_bytes(num_digits)
        length = int(length_str)

        # Read the actual data
        data = scope.read_bytes(length)

        # Read termination character
        scope.read_bytes(1)

        channel_data.extend(data)

        # Progress reporting
        elapsed = time.time() - start_time
        progress = (chunk_idx + 1) / num_chunks * 100
        rate = len(channel_data) / elapsed / 1024 / 1024 if elapsed > 0 else 0

        print(f"  Chunk {chunk_idx + 1}/{num_chunks} ({progress:.1f}%) - {rate:.2f} MB/s", end='\r')

    print()  # New line
    elapsed_total = time.time() - start_time
    print(f"  Channel complete: {len(channel_data)} bytes in {elapsed_total:.1f}s")

    return bytes(channel_data), preamble


def write_capture_file(filename, channels_data, sample_rate, timestamp, notes=""):
    """
    Write capture data to binary file.

    File format:
    - Magic: 8 bytes "WVHSCAP\0"
    - Version: 4 bytes (uint32, little-endian) = 1
    - Timestamp: 8 bytes (uint64, unix timestamp in microseconds)
    - Sample rate: 8 bytes (double, samples per second)
    - Num channels: 4 bytes (uint32)
    - Notes length: 4 bytes (uint32)
    - Notes: variable length UTF-8 string

    For each channel:
    - Channel name length: 4 bytes (uint32)
    - Channel name: variable length UTF-8 string
    - Channel description length: 4 bytes (uint32)
    - Channel description: variable length UTF-8 string
    - Preamble length: 4 bytes (uint32)
    - Preamble: variable length UTF-8 string
    - Data length: 8 bytes (uint64)
    - Data: raw bytes (unsigned 8-bit samples)
    """

    with open(filename, 'wb') as f:
        # Magic and version
        f.write(b'WVHSCAP\0')
        f.write(struct.pack('<I', 1))  # Version 1

        # Timestamp (microseconds since epoch)
        ts_us = int(timestamp.timestamp() * 1_000_000)
        f.write(struct.pack('<Q', ts_us))

        # Sample rate
        f.write(struct.pack('<d', sample_rate))

        # Number of channels
        f.write(struct.pack('<I', len(channels_data)))

        # Notes
        notes_bytes = notes.encode('utf-8')
        f.write(struct.pack('<I', len(notes_bytes)))
        f.write(notes_bytes)

        # Channel data
        for channel_name, (data, preamble, description) in channels_data.items():
            # Channel name
            name_bytes = channel_name.encode('utf-8')
            f.write(struct.pack('<I', len(name_bytes)))
            f.write(name_bytes)

            # Channel description
            desc_bytes = description.encode('utf-8')
            f.write(struct.pack('<I', len(desc_bytes)))
            f.write(desc_bytes)

            # Preamble
            preamble_bytes = preamble.encode('utf-8')
            f.write(struct.pack('<I', len(preamble_bytes)))
            f.write(preamble_bytes)

            # Data
            f.write(struct.pack('<Q', len(data)))
            f.write(data)


def main():
    parser = argparse.ArgumentParser(description="Capture W-VHS oscilloscope traces")
    parser.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                        help="Output filename (without extension)")
    parser.add_argument("-n", "--notes", default="",
                        help="Notes to include in capture file")
    parser.add_argument("--no-trigger", action="store_true",
                        help="Skip triggering, use existing acquisition")
    parser.add_argument("--address", default=SCOPE_ADDRESS,
                        help="Oscilloscope VISA address")
    parser.add_argument("--points", type=int, default=POINTS_PER_CHANNEL,
                        help="Points per channel to capture")
    args = parser.parse_args()

    timestamp = datetime.now()
    output_file = f"{args.output}_{timestamp.strftime('%Y%m%d_%H%M%S')}.bin"

    print(f"W-VHS Oscilloscope Capture")
    print(f"==========================")
    print(f"Output file: {output_file}")
    print(f"Points per channel: {args.points:,}")
    print(f"Channels: {', '.join(CHANNELS.values())}")
    print()

    # Connect to oscilloscope
    print(f"Connecting to oscilloscope at {args.address}...")
    rm = pyvisa.ResourceManager()

    try:
        scope = rm.open_resource(args.address)
        scope.timeout = 120000  # 120 second timeout
        scope.chunk_size = 100 * 1024 * 1024  # 100MB chunks for GigE
        scope.read_termination = '\n'
        scope.write_termination = '\n'

        idn = scope.query('*IDN?').strip()
        print(f"Connected to: {idn}")
    except Exception as e:
        print(f"Error connecting to oscilloscope: {e}")
        sys.exit(1)

    # Check/set memory depth
    print("\nChecking acquisition settings...")
    try:
        current_mdep = scope.query(":ACQ:MDEP?").strip()
        print(f"Current memory depth: {current_mdep}")

        # Get sample rate
        sample_rate_str = scope.query(":ACQ:SRAT?").strip()
        sample_rate = float(sample_rate_str)
        print(f"Sample rate: {sample_rate/1e9:.3f} Gsps")

        # Get timebase
        timebase = scope.query(":TIM:SCAL?").strip()
        print(f"Timebase: {timebase} s/div")

    except Exception as e:
        print(f"Warning: Could not query settings: {e}")
        sample_rate = 1e9  # Default to 1 Gsps

    # Trigger acquisition if requested
    if not args.no_trigger:
        print(f"\nTriggering single acquisition...")
        scope.write(":SINGLE")

        # Wait for acquisition to complete
        print("Waiting for trigger and acquisition to complete...")
        time.sleep(1)

        acquisition_complete = False
        start_wait = time.time()
        last_status = None

        while not acquisition_complete and (time.time() - start_wait) < 300:
            try:
                trig_stat = scope.query(":TRIG:STAT?").strip()

                if trig_stat != last_status:
                    print(f"Trigger status: {trig_stat}")
                    last_status = trig_stat

                if trig_stat == "STOP":
                    acquisition_complete = True
                    break

                time.sleep(0.5)
            except Exception as e:
                print(f"Status check error: {e}")
                time.sleep(1)

        if not acquisition_complete:
            print("Warning: Acquisition timeout - proceeding anyway")
        else:
            print("Acquisition complete!")
    else:
        print("\nSkipping trigger, using existing acquisition data")

    time.sleep(0.5)

    # Get actual memory depth
    try:
        mdep_str = scope.query(":ACQ:MDEP?").strip()
        available_points = int(float(mdep_str))
        print(f"\nMemory depth: {available_points:,} points")
    except:
        available_points = args.points

    points_to_capture = min(args.points, available_points)

    # Capture all channels
    channels_data = {}
    total_start = time.time()

    for channel, description in CHANNELS.items():
        try:
            data, preamble = capture_channel(scope, channel, points_to_capture, CHUNK_SIZE)
            channels_data[channel] = (data, preamble, description)
        except Exception as e:
            print(f"Error capturing {channel}: {e}")
            continue

    total_elapsed = time.time() - total_start

    # Write output file
    print(f"\nWriting output file: {output_file}")
    write_capture_file(output_file, channels_data, sample_rate, timestamp, args.notes)

    # Summary
    total_bytes = sum(len(d[0]) for d in channels_data.values())
    print(f"\n{'='*50}")
    print(f"Capture complete!")
    print(f"Total time: {total_elapsed:.1f} seconds")
    print(f"Total data: {total_bytes / 1024 / 1024:.2f} MB")
    print(f"Average rate: {total_bytes / total_elapsed / 1024 / 1024:.2f} MB/s")
    print(f"Output file: {output_file}")
    print(f"{'='*50}")

    # Close connection
    scope.close()
    rm.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCapture interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
