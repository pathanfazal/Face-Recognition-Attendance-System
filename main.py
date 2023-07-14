import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime
import pickle

dataset_path = 'faces'

images = []
person_names = []
file_list = os.listdir(dataset_path)
for file_name in file_list:
    cur_img = cv2.imread(f'{dataset_path}/{file_name}')
    images.append(cur_img)
    person_names.append(os.path.splitext(file_name)[0])

def encode_faces(images):
    encoded_faces_list = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encoded_face = face_recognition.face_encodings(image)[0]
        encoded_faces_list.append(encoded_face)
    return encoded_faces_list

encoded_faces_train = encode_faces(images)

def mark_attendance(name):
    with open('Attendance.csv', 'r+') as f:
        my_data_list = f.readlines()
        name_list = []
        for line in my_data_list:
            entry = line.split(',\n')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            time = now.strftime('%I:%M:%S:%p')
            date = now.strftime('%d-%B-%Y')
            f.writelines('\n'f'{name}, {time}, {date}')

# Capture pictures from webcam
cap  = cv2.VideoCapture(0)
while True:
    x = 1
    success, img = cap.read()
    img_scaled = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_scaled = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2RGB)
    faces_in_frame = face_recognition.face_locations(img_scaled)
    encoded_faces_frame = face_recognition.face_encodings(img_scaled, faces_in_frame)
    for encoded_face, face_location in zip(encoded_faces_frame, faces_in_frame):
        matches = face_recognition.compare_faces(encoded_faces_train, encoded_face)
        face_distances = face_recognition.face_distance(encoded_faces_train, encoded_face)
        match_index = np.argmin(face_distances)
        if matches[match_index]:
            name = person_names[match_index].upper().lower()
            y1, x2, y2, x1 = face_location
            # Since we scaled down by 4 times
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1+6, y2-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            mark_attendance(name)
            print(name)
            x = int(input("NEXT: "))

    cv2.imshow('webcam', img)
    if x == 0:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
