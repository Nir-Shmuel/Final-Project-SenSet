import os
import cv2
import face_recognition
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
import numpy as np
import time

# Connect the path with 'dotenv' file name
load_dotenv(dotenv_path='dotenv')

emotions = [
    'Happy',
    'Neutral',
    'Sad'
]

prediction = [0.0] * len(emotions)
root_path = os.getenv('ROOT_PATH')
model_save_path = os.path.join(root_path, os.getenv('MODEL_SAVE_DIR'))
model_save_name = os.getenv('MODEL_SAVE_NAME')
model_path = os.path.join(model_save_path, model_save_name)
model = load_model(model_path, compile=False)
model.summary()

video_capture = cv2.VideoCapture(0)
shape = 96, 96
frames = 5
face_locations = []
i = 0
np_vid = np.empty(shape=(1, frames, shape[0], shape[1], 3), dtype=np.dtype('uint8'))
np_vid[0] = 1
label = 'checking'
start = time.time()
while True:

    # for each frame
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    if not ret:
        continue
    if str(type(frame)) == "<class 'NoneType'>":
        break

    # Convert the image from BGR to RGB
    rgb_frame = frame[:, :, ::-1]
    for j, emotion in enumerate(emotions):
        cv2.putText(frame, "%s: %.3f" % (emotion, prediction[j]), (10, 80 + 25 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 155,
                    0)

    face_locations = face_recognition.face_locations(rgb_frame)

    if not face_locations:
        i = 0

    for top, right, bottom, left in face_locations:
        i += 1
        crop_img = frame[top:bottom, left:right]
        resized = cv2.resize(crop_img, shape)

        np_vid[0][i % frames - 1] = resized

        # draw result
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if i % frames == 0:
            y = model.predict([np_vid] * 2)
            index = np.argmax(y, axis=1)
            label = emotions[index[0]]
            prediction = y[0]
            print('checking emotion %d seconds' % (time.time() - start))
            start = time.time()

    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
