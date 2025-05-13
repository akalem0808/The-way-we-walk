import cv2
import torch
from class import OpticalFlowDataset, OpticalFlowCNN 

# Load the model
model = OpticalFlowCNN()
model.load_state_dict(torch.load('/Users/amankaleem/Desktop/MDE/Quant-gsd/code/my_model2.pth'))
model.to(device)
model.eval()

# Initialize variables
prev_frame = None
predictions = []

# Open the video file
cap = cv2.VideoCapture('/Users/amankaleem/Desktop/Aman/ MDE Fall -23/Quant-gsd/code/code/Video walking 2/OAW04-top.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray_frame, (img_width, img_height))

    # Calculate optical flow
    if prev_frame is not None:
        flow = cv2.calcOpticalFlowFarneback(prev_frame, resized_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        flow_tensor = torch.from_numpy(flow).permute(2, 0, 1).unsqueeze(0).to(torch.float32).to(device)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(flow_tensor)
            predictions.append(prediction.cpu().item())
    
    prev_frame = resized_frame

cap.release()

# Process the predictions as needed (e.g., thresholding)
processed_predictions = [1 if p > 0.5 else 0 for p in predictions]

# Print the predictions
print(processed_predictions)
