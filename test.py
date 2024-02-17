import tkinter as tk
from tkinter import ttk
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd

# Load the trained LSTM model
model = load_model('lstm_model.h5')

# Tokenizer for text preprocessing
tokenizer = Tokenizer()
tokenizer.fit_on_texts(pd.read_csv('train.csv')['text'])  # Use the entire dataset for fitting

# Define max_sequence_length
max_sequence_length = 50  # Adjust based on your model and data

# Function to get predictions based on user input
def get_prediction():
    user_input = entry.get()
    # Tokenize and pad the input sequence
    input_sequence = tokenizer.texts_to_sequences([user_input])
    input_padded = pad_sequences(input_sequence, maxlen=max_sequence_length)
    # Predict the result
    prediction_prob = model.predict(input_padded)[0][0]
    prediction = "Disaster" if prediction_prob > 0.5 else "Non-Disaster"
    result_label.config(text=f"Prediction: {prediction} (Probability: {prediction_prob:.4f})")

# Tkinter setup
root = tk.Tk()
root.title("Disaster Prediction App")

# Label and Entry for user input
input_label = ttk.Label(root, text="Enter text:")
input_label.pack(pady=10)
entry = ttk.Entry(root, width=50)
entry.pack(pady=10)

# Button to get prediction
predict_button = ttk.Button(root, text="Get Prediction", command=get_prediction)
predict_button.pack(pady=10)

# Label to display the result
result_label = ttk.Label(root, text="")
result_label.pack(pady=10)

# Run the Tkinter main loop
root.mainloop()
