import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
import os
import cv2
# Load the text classification model
text_model = load_model('lstm_model.h5')
tokenizer = Tokenizer()
train_data = pd.read_csv('train.csv')
tokenizer.fit_on_texts(train_data['text'])
max_sequence_length = 50

# Load the image classification model
image_model = load_model('CNN.model')
data_dir = "train"
class_names = os.listdir(data_dir)

# Tkinter setup
root = tk.Tk()
root.title("Multi-Model Prediction App")

# Text Classification Section
text_label = ttk.Label(root, text="Enter text:")
text_label.pack(pady=10)
text_entry = ttk.Entry(root, width=50)
text_entry.pack(pady=10)

result_text_label = ttk.Label(root, text="")
result_text_label.pack(pady=10)

def text_classification():
    user_input = text_entry.get()
    input_sequence = tokenizer.texts_to_sequences([user_input])
    input_padded = pad_sequences(input_sequence, maxlen=max_sequence_length)
    prediction_prob = text_model.predict(input_padded)[0][0]
    prediction = "Disaster" if prediction_prob > 0.5 else "Non-Disaster"

    # Retrieve location based on the provided text
    location_result = get_location_from_text(user_input)

    result_text_label.config(text=f"Text Prediction: {prediction} (Probability: {prediction_prob:.4f})\nLocation: {location_result}")

text_predict_button = ttk.Button(root, text="Get Text Prediction", command=text_classification)
text_predict_button.pack(pady=10)

# Function to retrieve location from text in 'train.csv'
def get_location_from_text(text):
    # You need to implement this function based on your data and matching criteria
    # For example, you can use pandas operations to filter the train_data DataFrame
    # based on the provided text.
    # Below is a sample, please adapt it according to your data and requirements.
    match_row = train_data[train_data['text'].str.contains(text, case=False, na=False)]
    if not match_row.empty:
        return match_row['location'].iloc[0]
    else:
        return "World Wide!!"

# Image Classification Section
def image_classification():
    file_path = filedialog.askopenfilename()
    if file_path:
        img = cv2.imread(file_path)
        img = cv2.medianBlur(img, 1)
        img = cv2.resize(img, (50, 50))
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        img = np.asarray(img)

        predictions = image_model.predict(img)
        class_index = np.argmax(predictions)
        class_label = class_names[class_index]
        
        result_image_label.config(text=f'Image Result: {class_label}')

image_select_button = tk.Button(root, text="Select Image for Classification", command=image_classification)
image_select_button.pack(pady=10)

result_image_label = tk.Label(root, text="")
result_image_label.pack(pady=10)

# Quit Button
quit_button = tk.Button(root, text="Quit", command=root.destroy)
quit_button.pack()

# Run the Tkinter main loop
root.mainloop()
