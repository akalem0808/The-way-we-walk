# Live Movement Color Dashboard

A Flask web application that streams live webcam video to your browser and overlays a color bar representing detected movement. The color bar at the top of the video changes according to the amount of movement, mapped to a color spectrum.

## Features
- Real-time webcam video streaming in your browser (MJPEG)
- Optical flow-based movement detection
- Color bar overlay that visualizes movement intensity
- Clean, modern dashboard UI

## Requirements
- Python 3.7+
- OpenCV (`opencv-python`)
- Flask
- matplotlib
- numpy

## Installation
1. Clone this repository:
   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```
2. Install dependencies:
   ```bash
   pip install opencv-python flask matplotlib numpy
   ```

## Usage
1. Run the Flask app:
   ```bash
   python live_dashboard.py
   ```
2. Open your browser and go to:
   ```
   http://localhost:8000
   ```
3. You will see your webcam feed with a large color bar at the top. The color represents the detected movement.
4. Press `Ctrl+C` in the terminal to stop the server.

## Troubleshooting
- **Permission denied / Port in use:**
  - Make sure port 8000 is free, or change the port in `live_dashboard.py`.
- **No webcam detected:**
  - Ensure your webcam is connected and accessible.
- **MJPEG stream not showing:**
  - Try a different browser (Chrome/Firefox recommended).
- **Linter errors about cv2:**
  - These are false positives if your code runs fine. Make sure you have `opencv-python` installed.

## Customization
- To change the color bar size, edit the `cv2.rectangle` line in `live_dashboard.py`.
- To adjust movement sensitivity, change the normalization range in the same file.

## License
MIT License 