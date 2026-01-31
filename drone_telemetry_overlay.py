#!/usr/bin/env python3
"""
DJI Drone Telemetry Overlay Script
Parses DJI SRT files and overlays telemetry data onto drone footage.
"""

import re
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import cv2
    import numpy as np
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python", "numpy"])
    import cv2
    import numpy as np


@dataclass
class TelemetryFrame:
    """Telemetry data for a single frame."""
    frame_num: int
    start_time_ms: float
    end_time_ms: float
    timestamp: str
    iso: int
    shutter: str
    fnum: float
    ev: float
    ct: int  # color temperature
    latitude: float
    longitude: float
    rel_alt: float  # relative altitude
    abs_alt: float  # absolute altitude
    # Calculated fields
    h_speed: float = 0.0  # horizontal speed (m/s)
    v_speed: float = 0.0  # vertical speed (m/s)


def parse_time_to_ms(time_str: str) -> float:
    """Convert SRT time format (HH:MM:SS,mmm) to milliseconds."""
    match = re.match(r"(\d+):(\d+):(\d+),(\d+)", time_str)
    if match:
        h, m, s, ms = map(int, match.groups())
        return (h * 3600 + m * 60 + s) * 1000 + ms
    return 0.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two GPS coordinates in meters."""
    R = 6371000  # Earth's radius in meters

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def parse_srt_file(srt_path: str) -> list[TelemetryFrame]:
    """Parse DJI SRT file and extract telemetry data."""
    frames = []

    with open(srt_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into subtitle blocks
    blocks = re.split(r'\n\n+', content.strip())

    for block in blocks:
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue

        # Parse frame number
        try:
            frame_num = int(lines[0])
        except ValueError:
            continue

        # Parse time range
        time_match = re.match(r"(\d+:\d+:\d+,\d+)\s*-->\s*(\d+:\d+:\d+,\d+)", lines[1])
        if not time_match:
            continue

        start_time = parse_time_to_ms(time_match.group(1))
        end_time = parse_time_to_ms(time_match.group(2))

        # Join remaining lines for metadata
        metadata_text = ' '.join(lines[2:])

        # Remove font tags
        metadata_text = re.sub(r'<[^>]+>', '', metadata_text)

        # Extract timestamp
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)', metadata_text)
        timestamp = timestamp_match.group(1) if timestamp_match else ""

        # Extract values using regex
        def extract_value(pattern: str, default=0):
            match = re.search(pattern, metadata_text)
            if match:
                try:
                    return float(match.group(1))
                except ValueError:
                    return default
            return default

        def extract_str(pattern: str, default=""):
            match = re.search(pattern, metadata_text)
            return match.group(1) if match else default

        frame = TelemetryFrame(
            frame_num=frame_num,
            start_time_ms=start_time,
            end_time_ms=end_time,
            timestamp=timestamp,
            iso=int(extract_value(r'\[iso:\s*(\d+)\]')),
            shutter=extract_str(r'\[shutter:\s*([^\]]+)\]'),
            fnum=extract_value(r'\[fnum:\s*([\d.]+)\]'),
            ev=extract_value(r'\[ev:\s*([+-]?[\d.]+)\]'),
            ct=int(extract_value(r'\[ct:\s*(\d+)\]')),
            latitude=extract_value(r'\[latitude:\s*([+-]?[\d.]+)\]'),
            longitude=extract_value(r'\[longitude:\s*([+-]?[\d.]+)\]'),
            rel_alt=extract_value(r'\[rel_alt:\s*([\d.]+)'),
            abs_alt=extract_value(r'\[abs_alt:\s*([\d.]+)'),
        )

        frames.append(frame)

    # Sort by frame number
    frames.sort(key=lambda f: f.frame_num)

    # Calculate speeds
    for i in range(1, len(frames)):
        prev = frames[i - 1]
        curr = frames[i]

        # Time delta in seconds
        dt = (curr.start_time_ms - prev.start_time_ms) / 1000.0
        if dt <= 0:
            dt = 0.033  # ~30fps fallback

        # Horizontal speed from GPS coordinates
        h_dist = haversine_distance(prev.latitude, prev.longitude,
                                     curr.latitude, curr.longitude)
        curr.h_speed = h_dist / dt

        # Vertical speed from altitude change
        v_dist = curr.rel_alt - prev.rel_alt
        curr.v_speed = v_dist / dt

    # Smooth speeds using moving average to reduce GPS noise
    window_size = 15
    for i in range(len(frames)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(frames), i + window_size // 2 + 1)

        h_speeds = [frames[j].h_speed for j in range(start_idx, end_idx)]
        v_speeds = [frames[j].v_speed for j in range(start_idx, end_idx)]

        frames[i].h_speed = sum(h_speeds) / len(h_speeds)
        frames[i].v_speed = sum(v_speeds) / len(v_speeds)

    return frames


def draw_telemetry_overlay(frame: np.ndarray, telemetry: TelemetryFrame,
                           video_width: int, video_height: int) -> np.ndarray:
    """Draw telemetry overlay on a video frame."""
    # Create overlay with transparency support
    overlay = frame.copy()

    # Colors (BGR format)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    SHADOW = (30, 30, 30)

    # Font settings - scale based on video resolution
    scale_factor = video_height / 1080.0
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale_large = 0.7 * scale_factor
    font_scale_small = 0.55 * scale_factor
    thickness = max(1, int(2 * scale_factor))

    # Padding and margins
    padding = int(20 * scale_factor)
    line_height = int(30 * scale_factor)

    def draw_text_with_shadow(img, text, pos, font_scale, color=WHITE):
        """Draw text with shadow for better visibility."""
        x, y = pos
        shadow_offset = max(1, int(2 * scale_factor))
        cv2.putText(img, text, (x + shadow_offset, y + shadow_offset),
                    font, font_scale, SHADOW, thickness + 1, cv2.LINE_AA)
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    # === TOP LEFT: Flight Data ===
    y_pos = padding + line_height

    # Altitude
    alt_text = f"ALT: {telemetry.rel_alt:.1f}m"
    draw_text_with_shadow(overlay, alt_text, (padding, y_pos), font_scale_large)
    y_pos += line_height

    # Horizontal Speed (convert m/s to km/h)
    h_speed_kmh = telemetry.h_speed * 3.6
    speed_text = f"H.SPD: {h_speed_kmh:.1f} km/h"
    draw_text_with_shadow(overlay, speed_text, (padding, y_pos), font_scale_large)
    y_pos += line_height

    # Vertical Speed
    v_speed_text = f"V.SPD: {telemetry.v_speed:+.1f} m/s"
    draw_text_with_shadow(overlay, v_speed_text, (padding, y_pos), font_scale_large)

    # === TOP RIGHT: Camera Settings ===
    y_pos = padding + line_height
    right_margin = video_width - padding

    # ISO
    iso_text = f"ISO {telemetry.iso}"
    text_size = cv2.getTextSize(iso_text, font, font_scale_small, thickness)[0]
    draw_text_with_shadow(overlay, iso_text, (right_margin - text_size[0], y_pos), font_scale_small)
    y_pos += line_height

    # Shutter
    shutter_text = f"{telemetry.shutter}s"
    text_size = cv2.getTextSize(shutter_text, font, font_scale_small, thickness)[0]
    draw_text_with_shadow(overlay, shutter_text, (right_margin - text_size[0], y_pos), font_scale_small)
    y_pos += line_height

    # Aperture
    fnum_text = f"f/{telemetry.fnum}"
    text_size = cv2.getTextSize(fnum_text, font, font_scale_small, thickness)[0]
    draw_text_with_shadow(overlay, fnum_text, (right_margin - text_size[0], y_pos), font_scale_small)
    y_pos += line_height

    # EV
    ev_text = f"EV {telemetry.ev:+.1f}"
    text_size = cv2.getTextSize(ev_text, font, font_scale_small, thickness)[0]
    draw_text_with_shadow(overlay, ev_text, (right_margin - text_size[0], y_pos), font_scale_small)

    # === BOTTOM LEFT: GPS Coordinates ===
    y_pos = video_height - padding - line_height

    # Coordinates
    lat_dir = "S" if telemetry.latitude < 0 else "N"
    lon_dir = "W" if telemetry.longitude < 0 else "E"
    coords_text = f"{abs(telemetry.latitude):.6f}{lat_dir}  {abs(telemetry.longitude):.6f}{lon_dir}"
    draw_text_with_shadow(overlay, coords_text, (padding, y_pos), font_scale_small)

    # === BOTTOM RIGHT: Timestamp ===
    if telemetry.timestamp:
        time_only = telemetry.timestamp.split(' ')[-1].split('.')[0]  # Get HH:MM:SS
        text_size = cv2.getTextSize(time_only, font, font_scale_small, thickness)[0]
        draw_text_with_shadow(overlay, time_only,
                             (right_margin - text_size[0], video_height - padding - line_height),
                             font_scale_small)

    # === Draw Speed Gauge (bottom center) ===
    gauge_center_x = video_width // 2
    gauge_center_y = video_height - int(80 * scale_factor)
    gauge_radius = int(50 * scale_factor)

    # Draw gauge background arc
    cv2.ellipse(overlay, (gauge_center_x, gauge_center_y), (gauge_radius, gauge_radius),
                0, 180, 360, SHADOW, max(2, int(4 * scale_factor)))
    cv2.ellipse(overlay, (gauge_center_x, gauge_center_y), (gauge_radius, gauge_radius),
                0, 180, 360, WHITE, max(1, int(2 * scale_factor)))

    # Speed indicator (0-50 km/h range, mapped to 180-360 degrees)
    max_speed = 50.0
    speed_ratio = min(h_speed_kmh / max_speed, 1.0)
    angle_deg = 180 + speed_ratio * 180
    angle_rad = math.radians(angle_deg)

    needle_length = gauge_radius - int(10 * scale_factor)
    needle_x = int(gauge_center_x + needle_length * math.cos(angle_rad))
    needle_y = int(gauge_center_y + needle_length * math.sin(angle_rad))

    cv2.line(overlay, (gauge_center_x, gauge_center_y), (needle_x, needle_y),
             (0, 200, 255), max(2, int(3 * scale_factor)), cv2.LINE_AA)

    # Speed value in center of gauge
    speed_val_text = f"{h_speed_kmh:.0f}"
    text_size = cv2.getTextSize(speed_val_text, font, font_scale_large, thickness)[0]
    draw_text_with_shadow(overlay, speed_val_text,
                         (gauge_center_x - text_size[0] // 2, gauge_center_y - int(10 * scale_factor)),
                         font_scale_large)

    # km/h label
    unit_text = "km/h"
    text_size = cv2.getTextSize(unit_text, font, font_scale_small * 0.8, thickness)[0]
    draw_text_with_shadow(overlay, unit_text,
                         (gauge_center_x - text_size[0] // 2, gauge_center_y + int(15 * scale_factor)),
                         font_scale_small * 0.8)

    return overlay


def get_telemetry_for_time(frames: list[TelemetryFrame], time_ms: float) -> Optional[TelemetryFrame]:
    """Find the telemetry frame for a given video time."""
    for frame in frames:
        if frame.start_time_ms <= time_ms < frame.end_time_ms:
            return frame

    # Return closest frame if exact match not found
    if frames:
        if time_ms < frames[0].start_time_ms:
            return frames[0]
        if time_ms >= frames[-1].end_time_ms:
            return frames[-1]

        # Binary search for closest
        for i, frame in enumerate(frames):
            if frame.start_time_ms > time_ms:
                return frames[i - 1] if i > 0 else frame

    return None


def process_video(video_path: str, srt_path: str, output_path: str):
    """Process video and add telemetry overlay."""
    print(f"Parsing telemetry from: {srt_path}")
    frames = parse_srt_file(srt_path)
    print(f"Loaded {len(frames)} telemetry frames")

    if not frames:
        print("Error: No telemetry data found!")
        return

    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file!")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print("Error: Could not create output video!")
        cap.release()
        return

    print(f"Writing to: {output_path}")
    print("Processing frames...")

    frame_num = 0
    while True:
        ret, video_frame = cap.read()
        if not ret:
            break

        # Calculate current time in ms
        current_time_ms = (frame_num / fps) * 1000

        # Get telemetry for current time
        telemetry = get_telemetry_for_time(frames, current_time_ms)

        if telemetry:
            # Draw overlay
            video_frame = draw_telemetry_overlay(video_frame, telemetry, width, height)

        # Write frame
        out.write(video_frame)

        frame_num += 1

        # Progress update
        if frame_num % 100 == 0 or frame_num == total_frames:
            progress = (frame_num / total_frames) * 100
            print(f"\rProgress: {frame_num}/{total_frames} ({progress:.1f}%)", end="", flush=True)

    print("\n")

    # Release resources
    cap.release()
    out.release()

    print(f"Video processing complete!")
    print(f"Output saved to: {output_path}")

    # Note about audio
    print("\nNote: The output video does not include audio.")
    print("To copy audio from the original video, run:")
    print(f'  ffmpeg -i "{output_path}" -i "{video_path}" -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 "{output_path.replace(".mp4", "_with_audio.mp4")}"')


def main():
    # Default paths (relative to script location)
    script_dir = Path(__file__).parent

    video_file = script_dir / "DJI_20260130095821_0076_D.MP4"
    srt_file = script_dir / "DJI_20260130095821_0076_D.SRT"
    output_file = script_dir / "DJI_20260130095821_0076_D_telemetry.mp4"

    # Allow command line arguments
    if len(sys.argv) >= 3:
        video_file = Path(sys.argv[1])
        srt_file = Path(sys.argv[2])
        output_file = Path(sys.argv[3]) if len(sys.argv) >= 4 else video_file.with_name(
            video_file.stem + "_telemetry.mp4"
        )

    if not video_file.exists():
        print(f"Error: Video file not found: {video_file}")
        sys.exit(1)

    if not srt_file.exists():
        print(f"Error: SRT file not found: {srt_file}")
        sys.exit(1)

    process_video(str(video_file), str(srt_file), str(output_file))


if __name__ == "__main__":
    main()
