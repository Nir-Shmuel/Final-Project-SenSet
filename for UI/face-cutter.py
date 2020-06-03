# import libraries
import cv2
import face_recognition
import glob

# video_path = glob.glob(r'C:\Users\nirsh\Desktop\CAER\train\Happy/*.avi')
video_path = glob.glob(r'C:\temp\videos/*.mp4')
print(len(video_path))


for i, video in enumerate(video_path):
    print(str(i) + '/' + str(len(video_path)))
    video_capture = cv2.VideoCapture(video)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    shape = (96, 96)
    output = cv2.VideoWriter(r'C:\Colman\פרוייקט גמר\videos\CAER\train\Surprise\0001.avi', fourcc, 20, shape)

    face_locations = []

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        if str(type(frame)) == "<class 'NoneType'>":
            break;
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces in the current frame of video
        face_locations = face_recognition.face_locations(rgb_frame)

        # Display the results
        for top, right, bottom, left in face_locations:
            # Draw a box around the face
            crop_img = frame[top:bottom, left:right]
            resized = cv2.resize(crop_img, shape)
            output.write(resized)

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release handle to the webcam
video_capture.release()
output.release()
cv2.destroyAllWindows()
