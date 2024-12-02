import pandas as pd
import pickle
import streamlit as st
import pyreadstat
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv('heart.csv')

# Preprocessing
kode_encoder = LabelEncoder()
df['Sex'] = kode_encoder.fit_transform(df['Sex'])
df = pd.get_dummies(df, columns=['ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope'])

# Memisahkan fitur dan target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Membagi data menjadi training dan testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=50)

# Membuat dan melatih model
model = DecisionTreeClassifier(random_state=50)
model.fit(X_train, y_train)

# Menyimpan model ke file .sav
with open('model_prediksi_gagal_jantung.sav', 'wb') as file:
    pickle.dump(model, file)



# Evaluasi model
y_pred = model.predict(X_test)
akurasi = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Streamlit interface
st.title("Prediksi Penyakit Jantung dengan Decision Tree")
st.sidebar.header("Input Data Baru")

# Form input menggunakan Streamlit
age = st.sidebar.number_input("Age", min_value=1, max_value=120, value=40)
sex = st.sidebar.selectbox("Sex", options=['M', 'F'])
chest_pain_type = st.sidebar.selectbox("Chest Pain Type", options=['ATA', 'NAP', 'ASY', 'TA'])
resting_bp = st.sidebar.number_input("RestingBP", min_value=50, max_value=200, value=120)
cholesterol = st.sidebar.number_input("Cholesterol", min_value=100, max_value=600, value=200)
fasting_bs = st.sidebar.selectbox("FastingBS", options=[0, 1], format_func=lambda x: 'Yes' if x == 1 else 'No')
resting_ecg = st.sidebar.selectbox("RestingECG", options=['Normal', 'ST', 'LVH'])
max_hr = st.sidebar.number_input("MaxHR", min_value=50, max_value=250, value=150)
exercise_angina = st.sidebar.selectbox("Exercise Angina", options=['Y', 'N'])
oldpeak = st.sidebar.number_input("Oldpeak", min_value=0.0, max_value=10.0, value=1.0)
st_slope = st.sidebar.selectbox("ST_Slope", options=['Up', 'Flat', 'Down'])

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
if st.sidebar.button("Prediksi"):
    prediksi = model.predict(df_baru)
    hasil = "Mengalami Risiko Penyakit Jantung" if prediksi[0] == 1 else "Tidak Mengalami Risiko Penyakit Jantung"
    st.subheader(f"Hasil Prediksi: {hasil}")
    st.write(f"Akurasi Model: {akurasi:.2f}")
    st.write("Confusion Matrix:")
    st.write(conf_matrix)
    st.write("Classification Report:")
    st.text(class_report)

# Visualisasi pohon keputusan
st.subheader("Visualisasi Pohon Keputusan")
fig, ax = plt.subplots(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['No Heart Disease', 'Heart Disease'], rounded=True, ax=ax)
st.pyplot(fig)
