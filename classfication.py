import cv2
import numpy as np

# Replace 'video_path' with the path to your video file
video_path = '/Users/amankaleem/Desktop/Aman/ MDE Fall -23/Quant-gsd/code/code/Video walking 2/OAW03-bottom.mp4'

# Initialize the video capture
cap = cv2.VideoCapture(video_path)

# Initialize a body detector with a confidence threshold
body_detector = cv2.HOGDescriptor()
body_detector.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Set a confidence threshold (adjust this value as needed)
confidence_threshold = 0.5

previous_area = None
circle_x = None

far_threshold = 5000  # Area in pixels for 'far'
middle_threshold = 15000  # Area in pixels for 'middle'
close_threshold = 30000  # Area in pixels for 'close'

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect bodies in the frame with the confidence threshold
    bodies, _ = body_detector.detectMultiScale(frame, winStride=(8, 8), padding=(32, 32), scale=1.05, hitThreshold=confidence_threshold)

    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Estimate the area of the detected body
        area = w * h

         # Classify the distance based on area
        if area < far_threshold:
            distance = "Far"
        elif area < middle_threshold:
            distance = "Middle"
        else:
            distance = "Close"

        # Display the distance text
        cv2.putText(frame, f"Distance: {distance}", (x, y - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Determine if the body is moving closer or farther
        if previous_area:
            if area > previous_area:
                direction = "Coming closer to the camera"
                if circle_x is not None:
                    circle_x += 10  # Move the blue circle to the right
                    if circle_x > frame.shape[1]:
                        circle_x = frame.shape[1]  # Limit circle's position within the frame
            elif area < previous_area:
                direction = "Moving away from the camera"
            else:
                direction = "Staying still"

            # Calculate the darkness of the red square based on movement (darker for moving away)
            darkness = int(255 * (previous_area - area) / previous_area)
            if darkness < 0:
                darkness = 0

            # Draw a red square that gets darker as the person moves away
            red_color = (0, 0, 255 - darkness)
            cv2.rectangle(frame, (x, y), (x + w, y + h), red_color, -1)

            # Display the direction text
            cv2.putText(frame, direction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        previous_area = area

    # Draw a blue circle that moves horizontally as the person moves closer
    if circle_x is not None:
        cv2.circle(frame, (circle_x, frame.shape[0] // 2), 30, (255, 0, 0), -1)

    # Show the frame
    cv2.imshow("Video Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
