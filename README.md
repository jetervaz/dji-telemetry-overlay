# DJI Telemetry Overlay

Overlay telemetry data from DJI drone SRT files onto video footage.

Tested with **DJI Neo 2** but should work with other DJI drones that generate SRT telemetry files.

## Features

- Parses DJI SRT telemetry format (per-frame metadata)
- Calculates horizontal speed from GPS coordinates (Haversine formula)
- Calculates vertical speed from altitude changes
- Applies smoothing to reduce GPS noise
- Creates a clean overlay with:
  - **Top left**: Altitude, horizontal speed, vertical speed
  - **Top right**: Camera settings (ISO, shutter, f-stop, EV)
  - **Bottom left**: GPS coordinates
  - **Bottom right**: Timestamp
  - **Bottom center**: Speed gauge

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- NumPy

## Installation

```bash
# Clone the repository
git clone https://github.com/jetervaz/dji-telemetry-overlay.git
cd dji-telemetry-overlay

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install opencv-python numpy
```

## Usage

```bash
# Process a single video (uses matching .SRT file automatically)
python drone_telemetry_overlay.py

# Or specify files explicitly
python drone_telemetry_overlay.py video.MP4 video.SRT output.mp4
```

The script looks for files in its directory by default, or you can pass paths as arguments.

## Adding Audio

The output video does not include audio. To copy audio from the original:

```bash
ffmpeg -i output_telemetry.mp4 -i original.MP4 -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 final_with_audio.mp4
```

## DJI SRT Format

DJI drones generate SRT files alongside video files containing per-frame telemetry:

```
1
00:00:00,000 --> 00:00:00,033
<font size="28">FrameCnt: 1, DiffTime: 33ms
2026-01-30 09:58:21.637
[iso: 100] [shutter: 1/1250.0] [fnum: 2.2] [ev: -1.3] [ct: 6700] [latitude: -29.685883] [longitude: -53.777843] [rel_alt: 57.200 abs_alt: 204.644] ...</font>
```

### Extracted Data

| Field | Description |
|-------|-------------|
| `iso` | ISO sensitivity |
| `shutter` | Shutter speed |
| `fnum` | Aperture (f-number) |
| `ev` | Exposure compensation |
| `ct` | Color temperature (Kelvin) |
| `latitude` | GPS latitude |
| `longitude` | GPS longitude |
| `rel_alt` | Relative altitude (from takeoff point) |
| `abs_alt` | Absolute altitude (sea level) |

## License

MIT License
