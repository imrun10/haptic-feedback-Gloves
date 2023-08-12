import mediapipe as mp
import cv2
import uuid
import os


mp_drawing = mp.solutions.drawing_utils # Drawing helpers
mp_hands = mp.solutions.hands # Hands model


cap = cv2.VideoCapture(0) # Webcam object (0 for built-in webcam, 1 for external webcam) using openCV

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5,) as hands: # Hand model object

    while cap.isOpened():
        # Read webcam
        ret, frame = cap.read() # Read frame from webcam    
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        # detections (1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # Convert the BGR image to RGB before processing. because the hands model takes in RGB images only but openCV reads images in BGR format
        image.flags.writeable = False                  # Set flag which is not writable meaning that the image is not writable anymore otherwise it will throw an error
        results = hands.process(image)                 # Process the image to get results which is a dictionary
        image.flags.writeable = True                   # Set flag which is writable meaning that the image is writable again
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert the RGB image back to BGR so we can display it using openCV
        
        print(results.multi_hand_landmarks) # Print landmarks of all the hands in the frame as a dictionary or coordinates

        # show video

        #displat landmarks on video
        if results.multi_hand_landmarks: # If there are hands in the frame
            for num, hand in enumerate(results.multi_hand_landmarks):
                mp.solutions.drawing_utils.draw_landmarks(image, hand, mp_hands.HAND_CONNECTIONS) # Display landmarks on webcam frame
        cv2.imshow('Hand tracking', image) # Display webcam frame


       
        # how to quit
        if cv2.waitKey(10) & 0xFF == ord('q'): # Press q to exit
            break

#the with statement is like a try and catch except it automatically will close thanks to the __exit__ method in the class
#also the with statement will automatically call the __enter__ method in the class



cap.release() # Release the webcam
cv2.destroyAllWindows() # Close the webcam window

# Path: hand detection/handpose.py

""" 
(1)
The image is made not writable to avoid 
any potential errors that could occur during
the processing of the image. Making the image not writable 
ensures that the image is not accidentally modified during processing. 
Once processing is complete, the image is made writable again to allow 
for any modifications that may be required, such as converting the RGB 
image back toBGR for display using OpenCV.
"""