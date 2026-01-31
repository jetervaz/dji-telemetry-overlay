#!/usr/bin/env python3
"""
DJI Drone Telemetry Overlay Script

A simple script to overlay telemetry data from DJI SRT files onto drone footage.
Uses the dji-telemetry library: https://github.com/jetervaz/dji-telemetry

Usage:
    python drone_telemetry_overlay.py                    # Process default files in current directory
    python drone_telemetry_overlay.py video.MP4          # Process specific video (auto-detect SRT)
    python drone_telemetry_overlay.py video.MP4 video.SRT output.mp4
"""

import subprocess
import sys
from pathlib import Path

# Auto-install dependencies if needed
def ensure_dependencies():
    try:
        import dji_telemetry
    except ImportError:
        print("Installing dji-telemetry library...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "git+https://github.com/jetervaz/dji-telemetry.git"
        ])

ensure_dependencies()

from dji_telemetry import parse_srt, process_video, add_audio


def progress_callback(current: int, total: int):
    """Display progress."""
    percent = (current / total) * 100
    if current % 100 == 0 or current == total:
        print(f"\rProgress: {current}/{total} ({percent:.1f}%)", end="", flush=True)
    if current == total:
        print()


def main():
    # Determine file paths
    script_dir = Path(__file__).parent

    if len(sys.argv) >= 2:
        video_file = Path(sys.argv[1])
        srt_file = Path(sys.argv[2]) if len(sys.argv) >= 3 else video_file.with_suffix('.SRT')
        output_file = Path(sys.argv[3]) if len(sys.argv) >= 4 else video_file.with_name(
            video_file.stem + '_telemetry.mp4'
        )
    else:
        # Find first MP4 file in current directory
        mp4_files = list(script_dir.glob('DJI_*.MP4'))
        if not mp4_files:
            mp4_files = list(script_dir.glob('*.MP4')) + list(script_dir.glob('*.mp4'))

        if not mp4_files:
            print("Usage: python drone_telemetry_overlay.py [video.MP4] [video.SRT] [output.mp4]")
            print("\nNo video files found in current directory.")
            sys.exit(1)

        video_file = mp4_files[0]
        srt_file = video_file.with_suffix('.SRT')
        output_file = video_file.with_name(video_file.stem + '_telemetry.mp4')

    # Validate files exist
    if not video_file.exists():
        print(f"Error: Video file not found: {video_file}")
        sys.exit(1)

    if not srt_file.exists():
        print(f"Error: SRT file not found: {srt_file}")
        sys.exit(1)

    # Parse telemetry
    print(f"Parsing telemetry from: {srt_file}")
    telemetry = parse_srt(srt_file)
    print(f"  Loaded {len(telemetry.frames)} frames")
    print(f"  Duration: {telemetry.duration_seconds:.1f}s")
    print(f"  Max altitude: {telemetry.max_altitude:.1f}m")
    print(f"  Max speed: {telemetry.max_speed * 3.6:.1f} km/h")

    # Process video
    print(f"\nProcessing video: {video_file}")
    temp_output = output_file.with_name(output_file.stem + '_temp.mp4')

    process_video(
        video_file,
        telemetry,
        temp_output,
        progress_callback=progress_callback
    )

    # Add audio
    print(f"Adding audio...")
    add_audio(temp_output, video_file, output_file)
    temp_output.unlink(missing_ok=True)

    print(f"\nOutput saved to: {output_file}")


if __name__ == "__main__":
    main()
