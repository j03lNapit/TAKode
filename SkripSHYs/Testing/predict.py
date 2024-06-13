import cv2
import mediapipe as mp
import numpy as np
import csv
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    min_detection_confidence=0.5
)

def predict_kantuk(model_path, scaler_model_path,  mar, opennes):

    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    
    with open(scaler_model_path, 'rb') as file:
        scaler_model = pickle.load(file)
    
    # Prepare the data for prediction
    data = np.array([[mar, opennes]])
    data_scaled = scaler_model.transform(data)
    print(data_scaled)
    # Predict using the SVM model
    prediction = model.predict(data)
    
    return prediction


def eye_openness(eye_region):
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    _, binary_eye = cv2.threshold(gray_eye, 50, 255, cv2.THRESH_BINARY_INV)
    white_pixels = cv2.countNonZero(binary_eye)

   #Normalisasi dengan membagi total pixel dibagian mata
    openness = white_pixels / (binary_eye.shape[0] * binary_eye.shape[1])
    return openness, binary_eye

# ekstrak fitur mata 
def get_eye_region(image, landmarks, eye_indices):
    #mengambil titik koordinat landmark
    eye_points = np.array([(int(landmarks.landmark[i].x * image.shape[1]), int(landmarks.landmark[i].y * image.shape[0])) for i in eye_indices])
    eye_points = eye_points.reshape(-1, 1, 2)
    # menghitung bonding box
    x, y, w, h = cv2.boundingRect(eye_points)
    eye_region = image[y:y+h, x:x+w]
    return eye_region

def mouth_aspect_ratio(landmarks):
    # Indices for landmarks of the outer corners of the mouth
    left_mouth_corner = np.array([landmarks[61].x, landmarks[61].y])  # Left corner
    right_mouth_corner = np.array([landmarks[291].x, landmarks[291].y])  # Right corner

    # Indices for landmarks of the upper and lower inner lip
    upper_inner_lip = np.array([landmarks[13].x, landmarks[13].y])  # Upper inner lip
    lower_inner_lip = np.array([landmarks[14].x, landmarks[14].y])  # Lower inner lip

    # Calculate the distances
    horiz_dist = np.linalg.norm(left_mouth_corner - right_mouth_corner)
    vert_dist = np.linalg.norm(upper_inner_lip - lower_inner_lip)

    # Calculate MAR
    mar = vert_dist / horiz_dist * 1.5
    return mar


#masih percobaan
def draw_facial_features(image, landmarks, eye_indices, upper_lip_indices, lower_lip_indices):
    # Draw the eyes
    left_eye = np.array([(landmarks[i].x * image.shape[1], landmarks[i].y * image.shape[0]) for i in eye_indices[0]], dtype=np.int32)
    right_eye = np.array([(landmarks[i].x * image.shape[1], landmarks[i].y * image.shape[0]) for i in eye_indices[1]], dtype=np.int32)
    cv2.polylines(image, [left_eye], True, (0, 255, 0), 2)
    cv2.polylines(image, [right_eye], True, (0, 255, 0), 2)

    # Draw the lips
    upper_lip = np.array([(landmarks[i].x * image.shape[1], landmarks[i].y * image.shape[0]) for i in upper_lip_indices], dtype=np.int32)
    lower_lip = np.array([(landmarks[i].x * image.shape[1], landmarks[i].y * image.shape[0]) for i in lower_lip_indices], dtype=np.int32)
    cv2.polylines(image, [upper_lip], True, (0, 255, 0), 2)
    cv2.polylines(image, [lower_lip], True, (0, 255, 0), 2)

# #


# indeks mata di mp 468 titik
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246]
LipsUpperOuter = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
LipsLowerOuter = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]


# video_path = r"F:\SKRIPSI\TA\DROZY\videos_i8\6-1.mp4"  # Use the correct path or 0 for webcam
video_path = 0
cap = cv2.VideoCapture(video_path)

# # Get the base name of the video file without the extension
# video_basename = os.path.basename(video_path)
# video_title, _ = os.path.splitext(video_basename)

# # Define the CSV file name based on the video title
# csv_filename = f"{video_title}_data_analysis.csv"

csv_filename = os.path.join('data_real_time', 'data.csv')

fps = cap.get(cv2.CAP_PROP_FPS)
delay_between_frames = int(1000 / fps)
data_list = []

model = r'F:\SKRIPSI\TA\Fix_code\SkripSHYs\models\model_linear.pkl'
scaler_model = r'F:\SKRIPSI\TA\Fix_code\SkripSHYs\models\scaler.pkl'


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Reached the end of the video or the video cannot be read.")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            left_eye_region = get_eye_region(frame, face_landmarks, LEFT_EYE_INDICES)
            right_eye_region = get_eye_region(frame, face_landmarks, RIGHT_EYE_INDICES)

            left_openness, left_binary_eye = eye_openness(left_eye_region)
            right_openness, right_binary_eye = eye_openness(right_eye_region)
            openness = (left_openness + right_openness) / 2

            mar = mouth_aspect_ratio(landmarks)

            data_list.append([cap.get(cv2.CAP_PROP_POS_FRAMES), openness, mar])  

            draw_facial_features(frame, landmarks, (LEFT_EYE_INDICES, RIGHT_EYE_INDICES), LipsUpperOuter, LipsLowerOuter)

            cv2.imshow('Left Eye Binary', left_binary_eye)
            cv2.imshow('Right Eye Binary', right_binary_eye)
            cv2.putText(frame, f'Eye Openness: {openness:.2f}', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
            cv2.putText(frame, f'MAR: {mar:.2f}', (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)
    else:
        # No landmarks detected, so set values to None or np.nan
        left_openness, right_openness, mar = None, None, None

    cv2.imshow('MediaPipe Face Mesh', frame)
    if cv2.waitKey(delay_between_frames) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

with open(csv_filename, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # Write the header
    csvwriter.writerow(['Frame', 'Openness', 'MAR'])
    # Write the data
    csvwriter.writerows(data_list)
    print(f"Data saved to {csv_filename}")

# print(f"Processed {frame_number} frames from {video_title}")