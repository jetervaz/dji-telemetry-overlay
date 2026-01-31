# DJI Telemetry Overlay

A simple script to overlay telemetry data from DJI drone SRT files onto video footage.

Uses the [dji-telemetry](https://github.com/jetervaz/dji-telemetry) library.

## Features

- Parses DJI SRT telemetry format
- Calculates horizontal/vertical speeds from GPS data
- Creates overlay with:
  - **Top left**: Altitude, horizontal speed, vertical speed
  - **Top right**: Camera settings (ISO, shutter, f-stop, EV)
  - **Bottom left**: GPS coordinates
  - **Bottom right**: Timestamp
  - **Bottom center**: Speed gauge
- Copies audio from original video

Tested with **DJI Neo 2**.

## Usage

```bash
# Process video (auto-detects matching .SRT file)
python drone_telemetry_overlay.py video.MP4

# Or specify all files explicitly
python drone_telemetry_overlay.py video.MP4 video.SRT output.mp4

# Run in current directory (processes first DJI_*.MP4 found)
python drone_telemetry_overlay.py
```

The script will automatically install the required `dji-telemetry` library on first run.

## Requirements

- Python 3.8+
- ffmpeg (for audio copying)

## Advanced Usage

For more control, use the [dji-telemetry](https://github.com/jetervaz/dji-telemetry) library directly:

```bash
pip install git+https://github.com/jetervaz/dji-telemetry.git

# CLI tool
dji-telemetry overlay video.MP4 --audio
dji-telemetry export video.SRT -o telemetry.gpx
```

## DJI SRT Format Example

```
1
00:00:00,000 --> 00:00:00,033
<font size="28">FrameCnt: 1, DiffTime: 33ms
2026-01-30 09:58:21.637
[iso: 100] [shutter: 1/1250.0] [fnum: 2.2] [ev: -1.3] [ct: 6700]
[latitude: -29.685883] [longitude: -53.777843]
[rel_alt: 57.200 abs_alt: 204.644] ...</font>
```

## License

This project is licensed under the [MIT License](LICENSE).
