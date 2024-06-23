import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences


# Constants
DATA_PATH = './data.pickle'
MODEL_PATH = 'model.p'
TEST_SIZE = 0.2
RANDOM_STATE = 42  # Ensures reproducibility


# Load the dataset
with open(DATA_PATH, 'rb') as f:
   data_dict = pickle.load(f)


data = data_dict['data']
labels = data_dict['labels']


# Pad sequences to ensure they are all the same length
max_length = max(len(seq) for seq in data)
data_padded = pad_sequences(data, maxlen=max_length, padding='post', dtype='float64')


# Encode labels as integers
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
   data_padded, labels_encoded, test_size=TEST_SIZE, shuffle=True, stratify=labels_encoded, random_state=RANDOM_STATE
)


# Initialize the model
model = RandomForestClassifier(random_state=RANDOM_STATE)


# Train the model
model.fit(x_train, y_train)


# Make predictions on the test set
y_predict = model.predict(x_test)


# Calculate accuracy
score = accuracy_score(y_predict, y_test)
print(f'{score * 100:.2f}% of samples were classified correctly!')


# Save the trained model and the label encoder
with open(MODEL_PATH, 'wb') as f:
   pickle.dump({'model': model, 'label_encoder': label_encoder}, f)


print(f'Model saved to {MODEL_PATH}')