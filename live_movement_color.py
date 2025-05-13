import cv2
import numpy as np
import matplotlib.pyplot as plt

def movement_to_color(norm_value):
    cmap = plt.get_cmap('rainbow')
    rgba = cmap(norm_value)
    rgb = tuple(int(255*x) for x in rgba[:3])
    return rgb

cap = cv2.VideoCapture(0)  # 0 is the default webcam

previous_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display_frame = frame.copy()
    current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    current_frame = cv2.resize(current_frame, (128, 64), interpolation=cv2.INTER_AREA)

    if previous_frame is not None:
        flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        average_movement = np.mean(magnitude)
        # Normalize movement (tune min/max as needed)
        norm = np.clip((average_movement - 0) / (10 - 0), 0, 1)
        color = movement_to_color(norm)
        # Draw a rectangle with the color
        cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 50), color, -1)
        cv2.putText(display_frame, f"Movement: {average_movement:.2f}", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

    previous_frame = current_frame

    cv2.imshow('Live Movement Color', display_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows() 