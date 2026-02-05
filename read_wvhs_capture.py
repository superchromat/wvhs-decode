#!/usr/bin/env python3
"""
Read W-VHS capture files created by capture_wvhs.py

Provides functions to load capture data and access individual channels.
"""

import struct
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ChannelData:
    """Data for a single captured channel."""
    name: str
    description: str
    preamble: str
    data: np.ndarray  # uint8 raw samples

    # Parsed from preamble (RIGOL format)
    x_increment: float = 0.0  # Time between samples
    x_origin: float = 0.0     # Time of first sample
    y_increment: float = 0.0  # Voltage per ADC step
    y_origin: float = 0.0     # Voltage offset
    y_reference: int = 0      # ADC reference level

    def __post_init__(self):
        self._parse_preamble()

    def _parse_preamble(self):
        """Parse RIGOL preamble string to extract scaling factors."""
        # RIGOL preamble format (comma-separated):
        # format, type, points, count, xincrement, xorigin, xreference, yincrement, yorigin, yreference
        try:
            parts = self.preamble.split(',')
            if len(parts) >= 10:
                self.x_increment = float(parts[4])
                self.x_origin = float(parts[5])
                self.y_increment = float(parts[7])
                self.y_origin = float(parts[8])
                self.y_reference = int(float(parts[9]))
        except (ValueError, IndexError):
            pass  # Keep defaults if parsing fails

    def to_voltage(self) -> np.ndarray:
        """Convert raw ADC values to voltage."""
        return (self.data.astype(np.float64) - self.y_reference) * self.y_increment + self.y_origin

    def get_time_axis(self) -> np.ndarray:
        """Get time axis for the samples."""
        return np.arange(len(self.data)) * self.x_increment + self.x_origin


@dataclass
class WVHSCapture:
    """Complete W-VHS capture with all channels."""
    timestamp: datetime
    sample_rate: float
    notes: str
    channels: Dict[str, ChannelData]

    @property
    def duration(self) -> float:
        """Duration of capture in seconds."""
        if not self.channels:
            return 0.0
        first_channel = next(iter(self.channels.values()))
        return len(first_channel.data) / self.sample_rate

    @property
    def num_samples(self) -> int:
        """Number of samples per channel."""
        if not self.channels:
            return 0
        return len(next(iter(self.channels.values())).data)

    def get_channel(self, name: str) -> Optional[ChannelData]:
        """Get channel by name (e.g., 'CHAN1' or 'Main FM/RF')."""
        if name in self.channels:
            return self.channels[name]
        # Search by description
        for ch_name, ch_data in self.channels.items():
            if name.lower() in ch_data.description.lower():
                return ch_data
        return None


def read_capture(filename: str) -> WVHSCapture:
    """Read a W-VHS capture file and return parsed data."""

    with open(filename, 'rb') as f:
        # Magic and version
        magic = f.read(8)
        if magic != b'WVHSCAP\0':
            raise ValueError(f"Invalid file magic: {magic}")

        version = struct.unpack('<I', f.read(4))[0]
        if version != 1:
            raise ValueError(f"Unsupported version: {version}")

        # Timestamp
        ts_us = struct.unpack('<Q', f.read(8))[0]
        timestamp = datetime.fromtimestamp(ts_us / 1_000_000)

        # Sample rate
        sample_rate = struct.unpack('<d', f.read(8))[0]

        # Number of channels
        num_channels = struct.unpack('<I', f.read(4))[0]

        # Notes
        notes_len = struct.unpack('<I', f.read(4))[0]
        notes = f.read(notes_len).decode('utf-8')

        # Read channels
        channels = {}
        for _ in range(num_channels):
            # Channel name
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8')

            # Description
            desc_len = struct.unpack('<I', f.read(4))[0]
            description = f.read(desc_len).decode('utf-8')

            # Preamble
            preamble_len = struct.unpack('<I', f.read(4))[0]
            preamble = f.read(preamble_len).decode('utf-8')

            # Data
            data_len = struct.unpack('<Q', f.read(8))[0]
            data = np.frombuffer(f.read(data_len), dtype=np.uint8)

            channels[name] = ChannelData(
                name=name,
                description=description,
                preamble=preamble,
                data=data
            )

    # Try to get actual sample rate from channel preamble (more reliable than header)
    # x_increment is time between samples, so sample_rate = 1/x_increment
    actual_sample_rate = sample_rate
    for ch in channels.values():
        if ch.x_increment > 0:
            actual_sample_rate = 1.0 / ch.x_increment
            break

    return WVHSCapture(
        timestamp=timestamp,
        sample_rate=actual_sample_rate,
        notes=notes,
        channels=channels
    )


def print_capture_info(capture: WVHSCapture):
    """Print summary information about a capture."""
    print(f"W-VHS Capture Info")
    print(f"==================")
    print(f"Timestamp: {capture.timestamp}")
    print(f"Sample rate: {capture.sample_rate/1e6:.3f} Msps")
    print(f"Duration: {capture.duration*1000:.3f} ms")
    print(f"Samples per channel: {capture.num_samples:,}")
    if capture.notes:
        print(f"Notes: {capture.notes}")
    print()
    print(f"Channels:")
    for name, ch in capture.channels.items():
        print(f"  {name}: {ch.description}")
        print(f"    - {len(ch.data):,} samples ({len(ch.data)/1024/1024:.1f} MB)")
        print(f"    - X increment: {ch.x_increment*1e9:.3f} ns")
        print(f"    - Y increment: {ch.y_increment*1000:.3f} mV/step")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: read_wvhs_capture.py <capture_file.bin>")
        sys.exit(1)

    capture = read_capture(sys.argv[1])
    print_capture_info(capture)
