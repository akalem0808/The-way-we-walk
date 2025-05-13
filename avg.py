import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from flask import Flask, render_template, request, redirect, url_for, flash
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Required for flash messages

# Directory containing videos - using relative path
video_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'videos')

# Create videos directory if it doesn't exist
if not os.path.exists(video_dir):
    os.makedirs(video_dir)
    print(f"Created video directory at: {video_dir}")

print("Video directory set to:", video_dir)

# Allowed video extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    previous_frame = None
    movement_over_time = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        current_frame = cv2.resize(current_frame, (128, 64), interpolation=cv2.INTER_AREA)

        if previous_frame is not None:
            flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            average_movement = np.mean(magnitude)
            movement_over_time.append(average_movement)

        previous_frame = current_frame

    cap.release()
    return movement_over_time

def create_plot(movement_data, video_name):
    plt.figure(figsize=(10, 6))
    plt.plot(movement_data)
    plt.title(f'Average Movement Over Time - {video_name}')
    plt.xlabel('Frame')
    plt.ylabel('Average Movement')
    
    # Convert plot to base64 string
    canvas = FigureCanvas(plt.gcf())
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

def movement_to_color(norm_value):
    cmap = plt.cm.rainbow  # You can use any matplotlib colormap
    rgba = cmap(norm_value)
    rgb = tuple(int(255*x) for x in rgba[:3])
    return f'rgb{rgb}'

@app.route('/')
def index():
    try:
        video_files = [f for f in os.listdir(video_dir) if f.endswith(tuple(ALLOWED_EXTENSIONS))]
        
        if not video_files:
            return render_template('index.html', 
                                 videos=[], 
                                 message="No video files found. Please upload a video file.")
        
        video_data = []
        avg_movements = []
        for video_file in video_files:
            video_path = os.path.join(video_dir, video_file)
            movement_data = process_video(video_path)
            plot_data = create_plot(movement_data, video_file)
            
            # Calculate statistics
            avg_movement = np.mean(movement_data)
            max_movement = np.max(movement_data)
            avg_movements.append(avg_movement)
            
            video_data.append({
                'name': video_file,
                'plot': plot_data,
                'frames': len(movement_data),
                'avg_movement': avg_movement,
                'max_movement': max_movement
            })
        # Normalize average movements
        min_val = min(avg_movements)
        max_val = max(avg_movements)
        for i, v in enumerate(video_data):
            if max_val > min_val:
                norm = (v['avg_movement'] - min_val) / (max_val - min_val)
            else:
                norm = 0.5  # If all values are the same
            v['color'] = movement_to_color(norm)
        return render_template('index.html', videos=video_data)
    
    except Exception as e:
        return render_template('index.html', 
                             videos=[], 
                             message=f"Error processing videos: {str(e)}")

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        flash('No file selected')
        return redirect(url_for('index'))
    
    file = request.files['video']
    if file.filename == '':
        flash('No file selected')
        return redirect(url_for('index'))
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(video_dir, filename)
        file.save(file_path)
        flash('Video uploaded successfully')
    else:
        flash('Invalid file type. Please upload MP4, AVI, or MOV files only.')
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
