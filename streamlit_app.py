import os
from sklearn.metrics import classification_report, confusion_matrix
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
import joblib
import cv2 
import tensorflow as tf
import streamlit as st




# Set opsi konfigurasi untuk menghilangkan warning PyplotGlobalUseWarning
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the trained RandomForestClassifier model
model = joblib.load('model.joblib')

# Load the trained CNN model 
cnn_model = tf.keras.models.load_model('cnn1_model.h5')

def classname():
    # Replace with your logic to get class names
    return ["Bicycling", "Eating", "Fitball Exercise", "Fishing", "Walking"]

def preprocess_image(uploaded_file):
    # Convert the uploaded image to a NumPy array
    image = Image.open(uploaded_file)
    image_array = np.array(image)

    # Resize and preprocess the image
    processed_image = cv2.resize(image_array, (256, 256))
    processed_image = processed_image / 255.0
    processed_image = processed_image.flatten()

    return processed_image

def load_classification_report(file_path):
    with open(file_path, 'r') as file:
        class_report = file.read()
    return class_report

def preprocess_image_cnn(uploaded_file):
    # Convert the uploaded image to a NumPy array
    image = Image.open(uploaded_file)
    image_array = np.array(image)

    # Resize and preprocess the image for CNN model
    processed_image = cv2.resize(image_array, (256, 256))
    processed_image = processed_image / 255.0  # Normalisasi pixel values
    processed_image = np.expand_dims(processed_image, axis=0)  # Tambahkan dimensi batch

    return processed_image


def plot_intensity_line_chart(data):
    # Plot grafik garis untuk rata-rata intensitas pixel
    plt.figure(figsize=(10, 6))

    for col in ['R_mean', 'G_mean', 'B_mean']:
        plt.plot(data['String'], data[col], label=col)

    plt.title('Rata-rata Intensitas Pixel untuk Setiap Kelas')
    plt.xlabel('Kelas')
    plt.ylabel('Rata-rata Intensitas Pixel')
    plt.legend()
    st.pyplot()
    
# Aplikasi Streamlit
def main():
    st.title("KLASIFIKASI ALGORITMA CNN DAN RANDOM FOREST PADA KLASIFIKASI POSE MANUSIA")
 
    data = pd.DataFrame({
        'Numerik': [0, 1, 2, 3, 4],
        'String': ['Bicycling', 'Eating', 'Fitball Exercise',
                   'Fishing', 'Walking'],
        'Jumlah' : ['112','88','114', '116', '104'],
        'R_mean': [149.5689239501953, 112.4098129272461, 133.79859924316406, 111.5028305053711, 116.2470703125],
        'G_mean': [152.3475341796875, 99.72533416748047, 135.12881469726562, 106.8306655883789, 111.39865112304688],
        'B_mean': [144.34524536132812, 91.16686248779297, 125.7186508178711, 113.65735626220703, 102.78475189208984]
    })

    # Ubah struktur data
    data['Jumlah'] = pd.to_numeric(data['Jumlah'])

    # Pilih opsi aplikasi (Klasifikasi Gambar atau EDA)
    app_option = st.radio("Pilih Opsi", ("Klasifikasi CNN", "Klasifikasi Random Forest", "EDA"))

    if app_option == "Klasifikasi Random Forest":
        st.subheader("Klasifikasi Random Forest")

        # Upload Gambar 
        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])
 
        if uploaded_file is not None:
            # Preprocess the uploaded image
            processed_image = preprocess_image(uploaded_file)

            # Reshape the processed image to 2D
            processed_image_2d = processed_image.reshape(1, -1)

            # Make prediction using the model
            prediction = model.predict(processed_image_2d)

            # Display the prediction
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            st.write("Prediction(Random Forest):", prediction) 
            
            # Get probability estimates for each class
            probabilities = model.predict_proba(processed_image_2d)
            
            # Membuat dataframe untuk menampilkan laporan klasifikasi dalam tabel
            class_labels = ['Bicycling', 'Eating', 'Fitball Exercise', 'Fishing', 'Walking']
            report_data = {'Kelas': class_labels,
                        'Presisi': [f"{probabilities[0][i]:.2f}" for i in range(len(class_labels))],
                        'Recall': [f"{probabilities[0][i]:.2f}" for i in range(len(class_labels))]}

            report_df = pd.DataFrame(report_data)

            # Menampilkan laporan klasifikasi dalam tabel
            st.write("Laporan Klasifikasi:")
            st.table(report_df)

            # Menampilkan akurasi keseluruhan
            overall_accuracy = max(probabilities[0])
            st.write(f"Akurasi Keseluruhan: {overall_accuracy:.2f}")

                   
    elif app_option == "Klasifikasi CNN":
        st.subheader("Klasifikasi CNN")

        # Upload Gambar
        uploaded_file = st.file_uploader("Choose a file", type=["jpg", "jpeg", "png"])

        # Inisialisasi predicted_class dengan nilai default -1
        predicted_class = -1
        
        # Load classification report
        class_report = load_classification_report('classification_report.txt')  # Ganti dengan metode sesuai format penyimpanan Anda

        # Display classification report
        st.subheader("Laporan Klasifikasi")
        st.text(class_report)

        # Load evaluation results
        evaluation_results = np.load('evaluation_results.npy', allow_pickle=True).item()
        
        
        plt.title('Confusion Matrix')
        # Menampilkan gambar statis dari file PNG
        image_path = 'coba.jpg'
        st.image(image_path, caption="Confusion Matrix", use_column_width=True)
 

        if uploaded_file is not None:
            # Preprocess the uploaded image for CNN
            processed_image_cnn = preprocess_image_cnn(uploaded_file)

            # Make prediction using the CNN model
            prediction_cnn = cnn_model.predict(processed_image_cnn)
            predicted_class = np.argmax(prediction_cnn)

            # Display the prediction
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            st.write("Predicted Class (CNN):", predicted_class)
            
  

    elif app_option == "EDA":
        st.subheader("Exploratory Data Analysis (EDA)")

        # Menampilkan data
        st.subheader('Data Preview')
        st.write(data.head())

        # Menampilkan kolom dataset
        st.subheader('Distribusi Label ')

        # Plot grafik batang untuk distribusi gambar
        plt.figure(figsize=(10, 6))
        plt.bar(data['String'], data['Jumlah'], color='pink')  
       
        plt.title('Distribusi Jumlah Gambar untuk Setiap Kelas')
        plt.xlabel('Kelas')
        plt.ylabel('Jumlah Gambar')
        st.pyplot()
        
        # Menampilkan kolom dataset
        st.subheader('Grafik Rata Rata Pixel Untuk Setiap Kelas')
        # Plot grafik garis untuk rata-rata intensitas pixel
        plot_intensity_line_chart(data)

if __name__ == "__main__":
    main()
