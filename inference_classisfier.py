import pickle
import cv2
import mediapipe as mp
import numpy as np
from keras.preprocessing.sequence import pad_sequences


# Load the trained model and label encoder
with open('./model.p', 'rb') as f:
   model_dict = pickle.load(f)
model = model_dict['model']
label_encoder = model_dict['label_encoder']


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
   print("Error: Could not open webcam.")
   exit()


# Mapping from folder counter to labels
# folder_to_labels = {0: 'A', 1: 'B', 2: 'C'}  # Add more mappings as needed
# folder_to_labels = {0: 'A', 1: 'B', 2: 'L'}
folder_to_labels = {'A':'A','B':'B','C':'C','D':'D','del':'del','E':'E','F':'F','G':'G','H':'H','I':'I','J':'J','K':'K','L':'L','M':'M','N':'N','nothing':'nothing','O':'O','P':'P','Q':'Q','R':'R','S':'S','space':'space','T':'T','U':'U','V':'V','W':'W','X':'X','Y':'Y','Z':'Z'}


# The number of features the model expects (ensure this matches the training stage)
# Change this to 84 if error arise
EXPECTED_NUM_FEATURES = 84


while True:
   data_aux = []
   x_ = []
   y_ = []


   ret, frame = cap.read()
   if not ret:
       print("Error: Failed to capture image.")
       break


   H, W, _ = frame.shape
   frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


   results = hands.process(frame_rgb)
   if results.multi_hand_landmarks:
       for hand_landmarks in results.multi_hand_landmarks:
           mp_drawing.draw_landmarks(
               frame,  # image to draw
               hand_landmarks,  # model output
               mp_hands.HAND_CONNECTIONS,  # hand connections
               mp_drawing_styles.get_default_hand_landmarks_style(),
               mp_drawing_styles.get_default_hand_connections_style()
           )


       for hand_landmarks in results.multi_hand_landmarks:
           for lm in hand_landmarks.landmark:
               x_.append(lm.x)
               y_.append(lm.y)


           min_x, min_y = min(x_), min(y_)


           for lm in hand_landmarks.landmark:
               data_aux.extend([lm.x - min_x, lm.y - min_y])


       # Pad the data_aux to ensure consistency
       data_aux_padded = pad_sequences([data_aux], maxlen=EXPECTED_NUM_FEATURES, padding='post', dtype='float64')[0]


       x1 = int(min(x_) * W) - 10
       y1 = int(min(y_) * H) - 10
       x2 = int(max(x_) * W) + 10
       y2 = int(max(y_) * H) + 10


       # Make prediction
       prediction = model.predict([data_aux_padded])
       folder_counter = int(prediction[0])
       predicted_character = folder_to_labels.get(folder_counter, str(folder_counter))  # Use folder_to_labels mapping
       print(predicted_character)
       # Draw rectangle and put text
       cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
       cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)


   cv2.imshow('frame', frame)
   if cv2.waitKey(1) & 0xFF == ord('q'):
       break


cap.release()
cv2.destroyAllWindows()
