import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer

# 1. Load dataset
data = pd.read_csv("Air Quality.csv", sep=';')

# 2. Ganti -200 jadi NaN
data.replace(-200, np.nan, inplace=True)

# 3. Pilih target (CO sebagai indikator)
target_col = "CO(GT)"

# 4. Hapus baris yang targetnya kosong
data = data.dropna(subset=[target_col])

# 5. Buat label klasifikasi
def kategori(x):
    if x < 2:
        return "Baik"
    elif x < 5:
        return "Sedang"
    else:
        return "Buruk"

data["label"] = data[target_col].apply(kategori)

# 6. Pisahkan fitur dan label
X = data.drop(columns=[target_col, "label"])
y = data["label"]

# 7. Ambil hanya data numerik
X = X.select_dtypes(include=[np.number])

# 8. Imputasi (isi NaN dengan rata-rata)
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# 9. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 10. Model Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# 11. Prediksi
y_pred = model.predict(X_test)

# 12. Evaluasi
print("Akurasi:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))