import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
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

print("Model berhasil disimpan sebagai model_prediksi_gagal_jantung.sav")
