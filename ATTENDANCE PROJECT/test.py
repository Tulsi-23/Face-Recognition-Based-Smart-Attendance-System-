from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

Knn = KNeighborsClassifier(n_neighbors=5)
Knn.fit(FACES, LABELS)
imgBackground = cv2.imread("background.jpeg")
COL_NAMES = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    frame = cv2.resize(frame, (640, 480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = Knn.predict(resized_img)
        ts = time.time()
        dt = datetime.fromtimestamp(ts)

        date = dt.strftime("%d-%m-%y")
        timestamp = dt.strftime("%H:%M:%S")

        attendance_file = "Attendance/Attendance_" + date + ".csv"
        exit = os.path.isfile(attendance_file)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        attendance = [str(output), str(timestamp)]

    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", imgBackground)
    k = cv2.waitKey(1)
    if k == ord('o'):
        if exit:
            with open(attendance_file, "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open(attendance_file, "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)

    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
