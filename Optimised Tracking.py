#Heads up: This code is quite old not sure if it will be that optimised and good havent really checked on it.
#PS, has been like 3 months since I last coded ngl
#Good luck understanding


import cv2
import mediapipe as mp
from cvzone.HandTrackingModule import HandDetector

# video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640) 
cap.set(4, 480)  

#hand detector
hand_detector = HandDetector(maxHands=2)

# face detector using Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

while True:
    # frame-by-frame (Probably what causes so much lag on my horrid laptop)
    ret, img = cap.read()
    if not ret:
        break


    # Convert image to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Hand Detection
    hands, img = hand_detector.findHands(img)

    # Pose Detection
    results = pose.process(img_rgb)
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Face Detection
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Display the resulting frame
    cv2.imshow("Tracking", img)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
