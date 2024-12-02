import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import pickle
from sklearn.tree import plot_tree
from sklearn.preprocessing import LabelEncoder

# Load image
img = plt.imread('jantungg.jpg')

# Load dataset
df = pd.read_csv('heart.csv')

# Load pre-trained model
model = pickle.load(open('model_prediksi_gagal_jantung.sav', 'rb'))

# Streamlit interface
st.title("Data Penyakit Jantung Explorer")

# Left sidebar
menu = st.sidebar.selectbox("Pilih Konten", ['Beranda', 'Dataset', 'Grafik', 'Prediksi'])

if menu == 'Beranda':
    st.image(img, caption='Gambar Jantung', use_container_width=True)
    st.write("Selamat datang di Data Penyakit Jantung Explorer!")
    st.write("Aplikasi ini memungkinkan Anda untuk menjelajahi data terkait penyakit jantung.")
    st.write("Silakan pilih menu di sebelah kiri untuk melihat konten yang tersedia.")

elif menu == 'Dataset':
    st.subheader("Dataset Penyakit Jantung")
    st.write(df)

    st.subheader("Analisis Deskriptif")
    st.write(df.describe())
    st.write(df['ChestPainType'].value_counts())

elif menu == 'Grafik':
    st.subheader("Visualisasi Data")
    
    # Histogram distribusi usia
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.hist(df['Age'], bins=20)
    ax.set_xlabel('Usia')
    ax.set_ylabel('Jumlah')
    ax.set_title('Distribusi Usia')
    st.pyplot(fig)

    # Scatter plot usia vs kolesterol
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(df['Age'], df['Cholesterol'])
    ax.set_xlabel('Usia')
    ax.set_ylabel('Kolesterol')
    ax.set_title('Hubungan Usia dan Kolesterol')
    st.pyplot(fig)


elif menu == 'Prediksi':
    st.subheader("Prediksi Penyakit Jantung")

    # Preprocessing
    kode_encoder = LabelEncoder()
    df['Sex'] = kode_encoder.fit_transform(df['Sex'])
    df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

    # Memisahkan fitur dan target
    X = df.drop('HeartDisease', axis=1)
    y = df['HeartDisease']

    # Streamlit interface
    st.title("Prediksi Penyakit Jantung dengan Decision Tree")
    st.header("Input Data Baru")

    # Form input menggunakan Streamlit
    age = st.number_input("Age", min_value=1, max_value=120, value=40)
    sex = st.selectbox("Sex", options=['M', 'F'])
    chest_pain_type = st.selectbox("Chest Pain Type", options=['ATA', 'NAP', 'ASY', 'TA'])
    resting_bp = st.number_input("RestingBP", min_value=50, max_value=200, value=120)
    cholesterol = st.number_input("Cholesterol", min_value=100, max_value=600, value=200)
    fasting_bs = st.selectbox("FastingBS", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
    resting_ecg = st.selectbox("RestingECG", options=['Normal', 'ST', 'LVH'])
    max_hr = st.number_input("MaxHR", min_value=50, max_value=250, value=150)
    exercise_angina = st.selectbox("Exercise Angina", options=['Y', 'N'])
    oldpeak = st.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)
    st_slope = st.selectbox("ST_Slope", options=['Up', 'Flat', 'Down'])

    # Data baru diubah ke dalam bentuk dictionary dan DataFrame
    data_baru = {
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain_type],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    }

    df_baru = pd.DataFrame(data_baru)

    # Preprocess data baru
    df_baru['Sex'] = kode_encoder.transform(df_baru['Sex'])
    df_baru = pd.get_dummies(df_baru, columns=['ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])
    df_baru = df_baru.reindex(columns=X.columns, fill_value=0)

    # Prediksi data baru
    if st.button("Prediksi"):
        prediksi = model.predict(df_baru)
        hasil = "Mengalami Risiko Penyakit Jantung" if prediksi[0] == 1 else "Tidak Mengalami Risiko Penyakit Jantung"
        st.subheader(f"Hasil Prediksi: {hasil}")

    # Visualisasi pohon keputusan
    st.subheader("Visualisasi Pohon Keputusan")
    fig, ax = plt.subplots(figsize=(12, 8))
    plot_tree(model, filled=True, feature_names=X.columns, class_names=['No Heart Disease', 'Heart Disease'], rounded=True, ax=ax)
    st.pyplot(fig)
