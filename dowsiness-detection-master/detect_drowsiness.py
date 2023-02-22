# USAGE
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat
# python detect_drowsiness.py --shape-predictor shape_predictor_68_face_landmarks.dat --alarm alarm.wav

# import the necessary packages
from scipy.spatial import distance as dist
from imutils import face_utils
import imutils
import dlib
import cv2
import os

class DrowsinessDetect:

	def __init__(self):
		self.shape_predictor = 'shape_predictor_68_face_landmarks.dat'
		# define two constants, one for the eye aspect ratio to indicate
		# blink and then a second constant for the number of consecutive
		# frames the eye must be below the threshold for to set off the
		# alarm
		self.EYE_AR_THRESH = 0.3
		self.EYE_AR_CONSEC_FRAMES = 1

		# initialize the frame counter as well as a boolean used to
		# indicate if the alarm is going off
		self.COUNTER = 0

		# initialize dlib's face detector (HOG-based) and then create
		# the facial landmark predictor
		self.detector = dlib.get_frontal_face_detector()
		self.predictor = dlib.shape_predictor(self.shape_predictor)

		# grab the indexes of the facial landmarks for the left and
		# right eye, respectively
		(self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
		(self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	def eye_aspect_ratio(self, eye):
		# compute the euclidean distances between the two sets of
		# vertical eye landmarks (x, y)-coordinates
		A = dist.euclidean(eye[1], eye[5])
		B = dist.euclidean(eye[2], eye[4])

		# compute the euclidean distance between the horizontal
		# eye landmark (x, y)-coordinates
		C = dist.euclidean(eye[0], eye[3])

		# compute the eye aspect ratio
		ear = (A + B) / (2.0 * C)

		# return the eye aspect ratio
		return ear

	def detect(self, image_path):
		frame = cv2.imread(image_path)
		frame = imutils.resize(frame, width=450)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detect faces in the grayscale frame
		rects = self.detector(gray, 0)

		if len(rects) == 0:
			return 'noface'

		# loop over the face detections
		for rect in rects:
			# determine the facial landmarks for the face region, then
			# convert the facial landmark (x, y)-coordinates to a NumPy
			# array
			shape = self.predictor(gray, rect)
			shape = face_utils.shape_to_np(shape)

			# extract the left and right eye coordinates, then use the
			# coordinates to compute the eye aspect ratio for both eyes
			leftEye = shape[self.lStart:self.lEnd]
			rightEye = shape[self.rStart:self.rEnd]
			leftEAR = self.eye_aspect_ratio(leftEye)
			rightEAR = self.eye_aspect_ratio(rightEye)

			# average the eye aspect ratio together for both eyes
			ear = (leftEAR + rightEAR) / 2.0

			# check to see if the eye aspect ratio is below the blink
			# threshold, and if so, increment the blink frame counter
			if ear < self.EYE_AR_THRESH:
				COUNTER += 1

				if COUNTER >= EYE_AR_CONSEC_FRAMES:
					os.remove(image_path)
					print('returning drowsy')
					return "drowsy"
				else:
					os.remove(image_path)
					print('returning none')
					return "none"

			# otherwise, the eye aspect ratio is not below the blink
			# threshold, so reset the counter
			else:
				COUNTER = 0
