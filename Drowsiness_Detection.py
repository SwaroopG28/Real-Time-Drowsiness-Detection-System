from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import pyttsx3
import threading
import time

# Create a lock object
lock = threading.Lock()

# Initialize the text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 200)  # Speed of speech


def say_alert(message):
	with lock:
		engine.say(message)
		engine.runAndWait()


def repeat_alert(message, interval, stop_event):
	while not stop_event.is_set():
		say_alert(message)
		time.sleep(interval)


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear


thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")  # Dat file is the crux of the code

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
flag = 0
alert_triggered = False  # Track if alert has been triggered to avoid multiple threads
stop_event = threading.Event()

while True:
	ret, frame = cap.read()
	frame = imutils.resize(frame, width=800)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	subjects = detect(gray, 0)

	for subject in subjects:
		shape = predict(gray, subject)
		shape = face_utils.shape_to_np(shape)  # Converting to NumPy Array
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)
		ear = (leftEAR + rightEAR) / 2.0
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if ear < thresh:
			flag += 1
			if flag >= frame_check:
				if not alert_triggered:
					alert_triggered = True
					stop_event.clear()
					threading.Thread(target=repeat_alert, args=("Alert! DROWSINESS ALERT!", 0.1, stop_event)).start()
				(text_width, text_height), baseline = cv2.getTextSize("****************ALERT!****************", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
				x_center = (frame.shape[1] - text_width) // 2

				cv2.putText(frame, "****************ALERT!****************", (x_center, 50),
							cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
				cv2.putText(frame, "****************ALERT!****************", (x_center, frame.shape[0] - 50),
							cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
		else:
			flag = 0
			if alert_triggered:
				alert_triggered = False
				stop_event.set()  # Stop the repeating alert

	if alert_triggered:
		(text_width, text_height), baseline = cv2.getTextSize("****************ALERT!****************", cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
		x_center = (frame.shape[1] - text_width) // 2
		cv2.putText(frame, "****************ALERT!****************", (x_center, 50),
					cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
		cv2.putText(frame, "****************ALERT!****************", (x_center, frame.shape[0] - 50),
					cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("q"):
		break

cv2.destroyAllWindows()
cap.release()
