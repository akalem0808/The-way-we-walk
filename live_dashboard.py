"""
Live Movement Color Dashboard
----------------------------
Flask app that streams live webcam video with a movement-based color bar using MJPEG streaming.
The color bar at the top of the video represents the detected movement, mapped to a color spectrum.
"""

from flask import Flask, render_template, Response
import cv2
import numpy as np
import matplotlib.pyplot as plt

app = Flask(__name__)

def movement_to_color(norm_value):
    """Map a normalized value [0,1] to an RGB color using the rainbow colormap."""
    cmap = plt.get_cmap('rainbow')
    rgba = cmap(norm_value)
    rgb = tuple(int(255 * x) for x in rgba[:3])
    return rgb

def gen_frames():
    """Generator that captures webcam frames, overlays a color bar, and yields MJPEG frames."""
    cap = cv2.VideoCapture(0)
    previous_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        display_frame = frame.copy()
        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_frame = cv2.resize(current_frame, (128, 64), interpolation=cv2.INTER_AREA)
        if previous_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(
                previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            average_movement = np.mean(magnitude)
            # Normalize movement (adjust min/max as needed)
            norm = np.clip((average_movement - 0) / (10 - 0), 0, 1)
            color = movement_to_color(norm)
            # Draw a large color bar at the top
            cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 150), color, -1)
            cv2.putText(
                display_frame,
                f"Movement: {average_movement:.2f}",
                (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 0),
                4,
            )
        previous_frame = current_frame
        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame = buffer.tobytes()
        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

@app.route('/')
def index():
    """Render the dashboard HTML page."""
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=8000) 