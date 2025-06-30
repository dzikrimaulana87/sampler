
# Credit Card Default Prediction

## 1. Latar Belakang

Risiko gaggal bayar (galbar) adalah tantangan utama dalam industri keuangan, terutama pada produk kartu kredit. Prediksi lebih dini terhadap nasabah yang berpotensi gagal bayar sangat penting agar perusahaan dapat melakukan mitigasi risiko, seperti memperketat kredit atau menawarkan restrukturisasi.

Dengan memanfaatkan machine learning, kita dapat membangun model prediktif berbasis data historis untuk membantu proses pengambilan keputusan tersebut. Pada proyek ini digunakan dataset open-source dari UCI: [Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients).

> **Referensi:**  
> - [UCI ML Repo](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

---

## 2. Business Understanding

### Problem Statement
Bagaimana memprediksi apakah seorang nasabah kartu kredit akan gagal bayar tagihan bulan depan berdasarkan data profil dan transaksi mreka?

### Goals
- Membangun model klasifikasi yang akurat untuk memprediksi potensi gagal bayar.
- Mengidentifikasi fitur-fitur penting yang mempengaruhi risiko gagal bayar.
- Model harus cukup baik untuk digunakan sebagai dasar pengambilan keputusan di perusahaan keuangan.

### Solution Statement
- Menggunakan dua algoritma: **Logistic Regression** dan **Random Forest**.
- Mengukur performa dengan **F1-Score** (karena data imbalanced) dan **ROC-AUC** (untuk mengukur kemampuan model dalam memisahkan dua kelas).
- Tidak menghapus outlier pada fitur finansial, melainkan mengobservasi dan, jika perlu, melakukan capping (batas atas/bawah) agar tidak kehilangan data penting yang valid secara bisnis.

---



## 3. Data Understanding

### Pengantar
Pada bagian ini, kita akan memahami struktur dan kondisi data yang digunakan dalam proyek. Ini mencakup sumber data, jumlah baris dan kolom, kondisi data (missing, duplikat, outlier), dan penjelasan fitur-fitur.

### Sumber Data
Dataset diperoleh dari UCI Machine Learning Repository:  
[Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)

### Jumlah Data
- Jumlah baris (data observasi): **30.000**
- Jumlah kolom (fitur + target): **24**

### Kondisi Data
- **Missing Value**: Tidak terdapat missing value di dataset.
- **Data Duplikat**: Tidak ditemukan duplikasi baris.
- **Outlier**: Terdapat nilai ekstrim terutama pada fitur `LIMIT_BAL`, `BILL_AMT`, dan `PAY_AMT`. Namun setelah pemeriksaan, nilai-nilai tersebut masih valid secara domain bisnis dan tidak dihapus.
- **Fitur Non-Prediktif**: Kolom `ID` merupakan identifier dan tidak relevan untuk prediksi, sehingga akan dihapus pada tahap preparation.

### Uraian Fitur
| Fitur              | Deskripsi |
|--------------------|-----------|
| ID                 | Nomor identitas pelanggan |
| LIMIT_BAL          | Limit kartu kredit (dalam NT dollar) |
| SEX                | Jenis kelamin (1=pria, 2=wanita) |
| EDUCATION          | Tingkat pendidikan |
| MARRIAGE           | Status perkawinan |
| AGE                | Umur (tahun) |
| PAY_0 to PAY_6     | Status pembayaran dalam 6 bulan terakhir |
| BILL_AMT1 to 6     | Tagihan bulanan 6 bulan terakhir |
| PAY_AMT1 to 6      | Pembayaran bulanan 6 bulan terakhir |
| default.payment.next.month | Target: 1=gagal bayar, 0=tidak |

---
## 4. Exploratory Data Analysis (EDA)

### Correlation Heatmap
![image](https://github.com/user-attachments/assets/55ceae2c-c2a7-40b2-baa8-134482c76ec8)


- Heatmap memperlihatkan korelasi antar seluruh fitur dalam dataset.
- Diagonal utama berwarna merah menunjukkan korelasi sempurna setiap fitur dengan dirinya sendiri (nilai = 1).
- Area di luar diagonal memperlihatkan korelasi antar fitur yang berbeda: merah = korelasi positif, biru = korelasi negatif.
- Terlihat adanya korelasi positif cukup kuat di antara fitur-fitur urutan waktu, seperti antara `PAY_0` hingga `PAY_6`, serta antar `BILL_AMT1` sampai `BILL_AMT6`.  
  **Artinya:** Riwayat keterlambatan/pembayaran di bulan-bulan sebelumnya cenderung saling berhubungan.
- Korelasi antara fitur utama (`LIMIT_BAL`, `AGE`, dsb) dengan target (`default`) umumnya lemah, terlihat dari warna yang tidak mencolok di kolom/row `default`.
- **Implikasi:** Tidak ada satu fitur tunggal yang sangat dominan mempengaruhi label `default`, sehingga prediksi gagal bayar perlu mempertimbangkan kombinasi banyak fitur.

### Distribusi Usia (AGE)
![image](https://github.com/user-attachments/assets/8cf3e4b3-c661-4b0a-8db0-e8105b73257e)


- Distribusi usia nasabah cenderung menurun setelah usia 30 tahun, dengan puncak pada usia 28–30 tahun.
- Mayoritas nasabah berusia antara 25–40 tahun.
- Ada juga nasabah berusia hingga di atas 60 tahun, namun jumlahnya sedikit (ekor kanan distribusi).
- **Implikasi:** Bank harus memperhatikan strategi pemasaran dan mitigasi risiko pada kelompok usia muda, karena mereka mendominasi jumlah nasabah.
- Tidak tampak outlier ekstrem yang tidak wajar, meskipun ada beberapa nasabah di usia sangat muda atau sangat tua, ini masih wajar secara domain bisnis.

### Distribusi LIMIT_BAL (Limit Kartu Kredit)
![image](https://github.com/user-attachments/assets/7f8b1680-f234-4757-9ac5-29617dd80f09)



- Sebagian besar nasabah memiliki limit kartu kredit relatif kecil (sekitar di bawah 200.000 NT Dollar).
- Ada beberapa kelompok puncak pada kelipatan tertentu, kemungkinan akibat kebijakan bank dalam menentukan limit (misal: 50.000, 100.000, 200.000).
- Distribusi right-skewed: Terdapat ekor kanan yang panjang, menandakan ada sedikit nasabah dengan limit sangat besar (hingga mendekati 1.000.000 NT Dollar).
- Nilai limit yang sangat besar ini bukan error/outlier tidak valid, karena dalam industri keuangan, nasabah prioritas/super premium memang mendapatkan limit sangat tinggi.
- **Implikasi:** Model perlu mampu menangani variasi besar pada limit, dan sebaiknya tidak menghapus outlier pada fitur ini, karena outlier di sini mewakili segmen nasabah yang nyata dan strategis bagi bank.

### Rangkuman EDA

- Fitur-fitur utama saling berkorelasi erat dalam kelompoknya, terutama pada fitur berurutan waktu (BILL_AMT, PAY_n, dsb).
- Distribusi usia dan limit kredit sesuai ekspektasi bisnis, tidak ada indikasi data error, meski ada data ekstrim (tetap valid).
- Tidak ada alasan menghapus outlier pada LIMIT_BAL atau AGE, karena semua masuk akal di dunia nyata perbankan.


---
---



## 5. Data Preparation

### Penghapusan Kolom Non-Prediktif
Kolom `ID` dihapus dari dataset karena hanya merupakan penanda unik untuk masing-masing nasabah dan tidak memiliki kontribusi prediktif terhadap target (`default.payment.next.month`).

```python
df = df.drop(columns=["ID"])
```

### Pemisahan Fitur dan Target
Untuk melakukan pelatihan model, dataset perlu dipisahkan menjadi dua bagian:

- **Fitur (X)**: Semua kolom kecuali kolom target `default.payment.next.month`.
- **Target (y)**: Hanya kolom `default.payment.next.month`.

```python
X = df.drop(columns=["default.payment.next.month"])
y = df["default.payment.next.month"]
```

### Pembagian Data Training dan Testing
Dataset dibagi menjadi dua subset:
- **Training set (80%)** digunakan untuk melatih model.
- **Testing set (20%)** digunakan untuk mengevaluasi performa model terhadap data yang belum pernah dilihat.

Pembagian dilakukan secara acak dengan parameter `random_state=42` agar hasil dapat direproduksi.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Standardisasi Fitur
Karena Logistic Regression sensitif terhadap skala fitur, maka semua fitur numerik di-standarisasi agar memiliki distribusi dengan rata-rata 0 dan standar deviasi 1.

Random Forest sebenarnya tidak memerlukan scaling, namun standarisasi tetap dilakukan agar seluruh proses dapat berjalan seragam, terutama untuk baseline model.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

> Catatan: Scaling hanya dilakukan terhadap fitur X (bukan target), dan `fit()` hanya diterapkan pada training set untuk menghindari kebocoran data (data leakage).





## 6. Modeling

Pada tahap ini, dilakukan proses pelatihan model machine learning untuk memprediksi apakah nasabah kartu kredit akan gagal bayar bulan berikutnya.

Dua model dipilih:
- **Logistic Regression** (sebagai baseline model)
- **Random Forest Classifier** (untuk menangkap hubungan non-linier)

---

### Model 1: Logistic Regression

#### Cara Kerja
Logistic Regression adalah algoritma klasifikasi linier yang memodelkan hubungan antara fitur input dan probabilitas kelas menggunakan fungsi logit (sigmoid). Model ini menghasilkan output dalam bentuk probabilitas, kemudian dikonversi menjadi kelas (0 atau 1) berdasarkan threshold tertentu (default = 0.5).

#### Parameter dan Nilainya
```python
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
```

Karena tidak ada parameter eksplisit yang ditentukan, maka digunakan parameter default, yaitu:

| Parameter       | Nilai Default | Penjelasan |
|----------------|---------------|------------|
| penalty        | `'l2'`        | Regularisasi L2 (Ridge), digunakan untuk menghindari overfitting dengan menghukum koefisien besar |
| C              | `1.0`         | Kebalikan dari kekuatan regularisasi (semakin kecil nilai C → regularisasi lebih kuat) |
| solver         | `'lbfgs'`     | Algoritma optimisasi numerik yang cocok untuk dataset ukuran menengah dan mendukung L2 |
| max_iter       | `100`         | Batas maksimum iterasi untuk konvergensi algoritma |
| random_state   | `None`        | Tidak diset, sehingga hasilnya bisa berubah jika dijalankan ulang tanpa seed eksplisit |

#### Alasan Penggunaan
Logistic Regression dipilih karena:
- Mudah dijelaskan ke stakeholder non-teknis,
- Cepat dilatih,
- Cocok sebagai baseline model.

---

### Model 2: Random Forest Classifier

#### Cara Kerja
Random Forest adalah algoritma ensemble berbasis decision tree. Ia membuat banyak decision tree dari subset data dan fitur secara acak, lalu menghasilkan prediksi akhir berdasarkan agregasi (voting mayoritas untuk klasifikasi).

Keunggulan utama dari Random Forest adalah kemampuannya menangani:
- Hubungan non-linear antar fitur,
- Data dengan distribusi tidak normal,
- Outlier dan missing value (walau di dataset ini tidak ada missing).

#### Parameter dan Nilainya
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
```

Hanya `random_state` yang diset eksplisit, lainnya menggunakan nilai default:

| Parameter       | Nilai         | Penjelasan |
|----------------|---------------|------------|
| n_estimators   | `100`         | Jumlah pohon dalam hutan. Lebih banyak pohon bisa meningkatkan akurasi tapi memperlambat training |
| max_depth      | `None`        | Tidak dibatasi → setiap pohon tumbuh hingga semua daun homogen (atau kondisi minimum lainnya terpenuhi) |
| criterion      | `'gini'`      | Ukuran impurity node: Gini Impurity (biasa digunakan untuk klasifikasi) |
| min_samples_split | `2`       | Jumlah minimum sampel untuk membagi node internal |
| min_samples_leaf  | `1`       | Minimum sampel di tiap daun |
| bootstrap      | `True`        | Data bootstrap digunakan (sampling dengan pengembalian untuk tiap pohon) |
| random_state   | `42`          | Disetel untuk memastikan hasil dapat direproduksi |

#### Alasan Penggunaan
Random Forest dipilih karena:
- Mampu menangkap hubungan kompleks antar fitur,
- Tidak sensitif terhadap scaling atau distribusi data,
- Robust terhadap overfitting karena averaging banyak pohon.

---







## 7. Evaluation

### Metrik Evaluasi

Proyek ini menggunakan dua metrik utama untuk menilai kinerja model:
- **F1-Score**: Cocok untuk kasus data imbalance seperti ini. Menyeimbangkan antara precision dan recall.
- **ROC-AUC (Receiver Operating Characteristic - Area Under Curve)**: Mengukur kemampuan pemisahan antar kelas secara keseluruhan.

---

### Hasil Evaluasi per Model
![image](https://github.com/user-attachments/assets/58781308-bfd2-42b8-a41e-c6404686dc4f)

#### 📌 Logistic Regression

- **Confusion Matrix**:
  - True Negative (TN): 4528
  - False Positive (FP): 145
  - False Negative (FN): 1009
  - True Positive (TP): 318

- **Classification Report**:
  - Precision: 0.69 (kelas 1 / gagal bayar)
  - Recall: 0.24
  - F1-Score: 0.36
  - Accuracy: 81%
  - ROC-AUC: 0.7076

📈 **Analisis**:
- Model cenderung memprediksi kelas mayoritas (`tidak gagal`).
- Recall rendah (24%) artinya model gagal mendeteksi sebagian besar nasabah berisiko gagal bayar.
- Cocok untuk baseline interpretatif namun tidak cukup andal untuk deteksi risiko operasional.

---

#### 📌 Random Forest
![image](https://github.com/user-attachments/assets/5e287a0b-5318-4bc4-b70b-1e350339707a)


- **Confusion Matrix**:
  - True Negative (TN): 4397
  - False Positive (FP): 276
  - False Negative (FN): 852
  - True Positive (TP): 475

- **Classification Report**:
  - Precision: 0.63 (kelas 1 / gagal bayar)
  - Recall: 0.36
  - F1-Score: 0.46
  - Accuracy: 81%
  - ROC-AUC: 0.7506

📈 **Analisis**:
- Peningkatan recall dan F1-score signifikan dibanding Logistic Regression.
- ROC-AUC juga lebih tinggi → model lebih andal dalam memisahkan nasabah gagal dan tidak gagal.
- Lebih cocok untuk implementasi prediktif karena mendeteksi lebih banyak risiko.

---

### Komparasi dan Kesimpulan Model

| Metric           | Logistic Regression | Random Forest |
|------------------|---------------------|----------------|
| Precision (1)    | 0.69                | 0.63           |
| Recall (1)       | 0.24                | 0.36           |
| F1-score (1)     | 0.36                | 0.46           |
| ROC-AUC          | 0.7076              | 0.7506         |
| Accuracy         | 0.81                | 0.81           |

- Kedua model memiliki akurasi serupa, namun **Random Forest jauh lebih baik dalam mengenali nasabah berisiko** (kelas 1).
- Logistic Regression lebih baik untuk interpretasi dan justifikasi ke stakeholder.
- Random Forest direkomendasikan sebagai model utama untuk deteksi risiko gagal bayar.

---

### Keterkaitan dengan Business Understanding

- ✅ **Problem Statement** terjawab: model mampu memprediksi gagal bayar.
- ✅ **Goals tercapai**:
  - Membuat model akurat → Random Forest cocok.
  - Mengidentifikasi fitur penting → Logistic Regression & Feature Importance dari RF bisa digunakan.
- ✅ **Dampak Solusi**:
  - Model dapat digunakan sebagai sistem deteksi dini risiko kredit.
  - Memberi insight kepada tim kredit untuk menyusun strategi mitigasi, seperti penyesuaian limit kredit atau restrukturisasi.
  - Meningkatkan efisiensi dan akurasi proses screening nasabah berisiko.

---



## 8. Kesimpulan

- Model machine learning mampu meningkatkan efisiensi deteksi gagal bayar nasabah kartu kredit.
- Random Forest lebih unggul dari Logistic Regression dari segi F1 dan ROC/AUC, namun Logistic Regression tetap penting untuk explainability.
- Outlier tidak dihapus, karena secara bisnis merupakan kasus valid yang penting untuk prediksi model.
- Model siap digunakan sebagai decision support, namun bisa diimprove dengan data tambahan, feature engineering, atau balancing class (oversampling/undersampling).

---

## 9. Rekomendasi Pengembangan Lanjutan

- Lakukan feature selection untuk mengurangi dimensi jika model terlalu kompleks.
- Lakukan hyperparameter tuning lebih dalam.
- Eksplorasi teknik balancing untuk kelas target.
- Integrasi dengan data baru/real-time.

---

## 10. Referensi

- Card Default Dataset  
- Dokumentasi Scikit-learn  

---

## 11. Struktur Submission

- `credit_card_default_project.ipynb`: Notebook utama
- `credit_card_default_project.py`: Script python (versi py)
- `README.md`: Laporan lengkap proyek
