import os
import pickle
import cv2
import mediapipe as mp


# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)


DATA_DIR = './data'


# Ensure the data directory exists
if not os.path.exists(DATA_DIR):
   print(f"Error: Directory {DATA_DIR} does not exist.")
   exit()


data = []
labels = []




def process_image(img_path, label):
   img = cv2.imread(img_path)
   if img is None:
       print(f"Error: Unable to read image {img_path}")
       return None, None


   img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
   results = hands.process(img_rgb)


   if not results.multi_hand_landmarks:
       return None, None


   data_aux = []
   x_ = []
   y_ = []


   for hand_landmarks in results.multi_hand_landmarks:
       for lm in hand_landmarks.landmark:
           x_.append(lm.x)
           y_.append(lm.y)


       min_x, min_y = min(x_), min(y_)


       for lm in hand_landmarks.landmark:
           data_aux.extend([lm.x - min_x, lm.y - min_y])


   return data_aux, label




print("Processing images...")
for dir_ in os.listdir(DATA_DIR):
   class_dir = os.path.join(DATA_DIR, dir_)
   if not os.path.isdir(class_dir):
       continue
   for img_path in os.listdir(class_dir):
       img_full_path = os.path.join(class_dir, img_path)
       data_aux, label = process_image(img_full_path, dir_)
       if data_aux is not None and label is not None:
           data.append(data_aux)
           labels.append(label)
       print(f"Processed {img_full_path}")


# Save data to a pickle file
with open('data.pickle', 'wb') as f:
   pickle.dump({'data': data, 'labels': labels}, f)


print("Data processing complete. Data saved to data.pickle.")