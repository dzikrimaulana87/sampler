
# Sistem Rekomendasi Film (MovieLens)  
_Prediksi Personal Film Pilihan Anda dengan Content-Based & Collaborative Filtering_

---

## 1. Latar Belakang

Di era digital dengan ribuan judul film, user membutuhkan sistem rekomendasi agar tidak kebingungan memilih konten. Sistem rekomendasi telah menjadi teknologi fundamental pada berbagai platform (Netflix, Amazon, dsb) untuk meningkatkan user engagement [1].  
Proyek ini membangun sistem rekomendasi film berbasis data MovieLens, menggabungkan pendekatan **content-based filtering** dan **collaborative filtering** [2][3].

---

## 2. Business Understanding

**Problem Statement:**  
Bagaimana membangun sistem yang dapat merekomendasikan film sesuai preferensi user, baik menggunakan kemiripan konten maupun pola interaksi user lain?

**Solution Statement:**  
- Menerapkan dua pendekatan:
  1. **Content-based filtering** (dengan TF-IDF dan cosine similarity pada metadata film) [3].
  2. **Collaborative filtering** (matrix factorization/SVD) [4].
- Model dievaluasi dengan metrik RMSE & MAE, serta insight bisnis.

---

## 3. Data Understanding

**Dataset:**  
- [MovieLens 100K (Small)](https://grouplens.org/datasets/movielens/)
- **Jumlah user:** 610  
- **Jumlah film:** 9724  
- **Jumlah rating:** 100.836  

### Uraian Fitur
| Fitur | Deskripsi | Tipe Data |
|---|---|---|
| `userId` | ID unik untuk setiap pengguna. | Numerik (Integer) |
| `movieId` | ID unik untuk setiap film. | Numerik (Integer) |
| `rating` | Skor rating yang diberikan pengguna (skala 0.5 - 5.0). | Numerik (Float) |
| `timestamp` | Waktu saat rating diberikan (format Unix). | Numerik (Integer) |
| `title` | Judul film beserta tahun rilis. | Teks (String) |
| `genres` | Genre film yang dipisahkan oleh |Teks (String) |

**Kondisi Data:**
- Tidak ada missing/null/duplikasi.
- Tidak ditemukan rating di luar rentang valid (0.5-5.0).
- Banyak film/user "rare" (sedikit interaksi) = tantangan cold start [4].

**Gambar:**  
- **gambar struktur dan preview data:**
> screenshot/visual tabel 5 baris awal dari `movies.csv` dan `ratings.csv`
![image](https://github.com/user-attachments/assets/ab698166-e0b8-4eaa-b951-c3ee4361fca4)
![image](https://github.com/user-attachments/assets/717b5d53-4321-475d-8ff9-1ddb192af581)

---

## 4. Exploratory Data Analysis (EDA)

EDA bertujuan memahami karakteristik data secara menyeluruh sebelum masuk tahap modeling.

### 4.1 Distribusi Rating  
Analisis distribusi rating user terhadap film, untuk memahami kecenderungan penilaian.

- **Langkah:** Visualisasikan histogram rating.
- **Insight:** Mayoritas rating >3.0, indikasi bias positif (positivity bias) pada data crowdsourcing.

**Gambar:**  
![image](https://github.com/user-attachments/assets/23c47911-7702-485b-8b56-8ea8744053da)


### 4.2 Film Paling Populer  
Analisis film dengan jumlah rating terbanyak, untuk memahami karakter film populer dan mendukung cold start user baru.

- **Langkah:** Ranking 10 film dengan rating terbanyak, tampilkan judul & genre.
- **Insight:** Film populer cenderung mudah direkomendasikan ke user baru.

**Gambar:**  
![image](https://github.com/user-attachments/assets/7aff29fe-a62a-4555-9282-31f79d426b93)

### 4.5 Jumlah Rating per User  
Memahami pola aktivitas user: siapa user aktif/pasif, potensi dampak user cold start.

- **Langkah:** Histogram banyaknya rating yang diberikan user.
- **Insight:** Sebagian besar user memberi sedikit rating (long tail), hanya sedikit user sangat aktif.

**Gambar:**  
![image](https://github.com/user-attachments/assets/b16dfec7-535b-4363-a6fd-031920dae108)


## 5. Data Preparation

Langkah-langkah menyiapkan data sebelum modeling:

- **Cleaning:** Data sudah bersih, tidak perlu imputasi/cleaning lebih lanjut.
- **Feature engineering:**  
  Gabungkan judul dan genre untuk content-based filtering (input ke TF-IDF vectorizer).
- **Transformasi:** Tidak perlu encoding tambahan, semua fitur sudah numerik/deskriptif.
- **Catatan cold start:** Rare user/film didokumentasikan untuk evaluasi.

---

## 6. Modeling

### 6.1 Content-Based Filtering

- **Prinsip:** Merekomendasikan film yang _mirip secara konten_ (judul/genre) dengan film yang sudah disukai user [3].
- **TF-IDF (Term Frequency-Inverse Document Frequency):**

  ![TF-IDF formula](https://github.com/user-attachments/assets/7fe4a952-9fe6-48f1-8398-57c505052da2)

- **Cosine Similarity:**  
  ![Cosine similarity formula](https://github.com/user-attachments/assets/9937cbe9-ef74-483b-9274-1d904a54fce7)

- **Langkah Modeling:**
  - Membuat representasi teks (judul + genre) setiap film sebagai vektor numerik (TF-IDF).
  - Hitung kemiripan antar film dengan cosine similarity.
  - Rekomendasikan film dengan skor kemiripan tertinggi.

- **Kelebihan:** Sangat baik untuk user dengan minat genre spesifik.
- **Kekurangan:** Sulit menemukan _hidden gem_ di luar genre user.

### 6.2 Collaborative Filtering (SVD)

- **Prinsip:**  
  Model mempelajari _representasi laten_ (matrix factorization) dari pola rating seluruh user [4].

- **Singular Value Decomposition (SVD):**  
  ![SVD formula](https://github.com/user-attachments/assets/f32fb734-9384-44bb-b205-dfd838c2aa9f)

- **Langkah Modeling:**
  - Membagi data menjadi train dan test set.
  - Melatih model SVD pada train set untuk memperoleh vektor laten user dan item.
  - Memvalidasi model dengan mengukur seberapa baik prediksi rating pada test set.

- **Kelebihan:** Menangkap pola "serupa" di seluruh data.
- **Kekurangan:** Lemah untuk user/film baru (cold start problem).

---

## 7. Evaluation

### Metrik Evaluasi

- **Root Mean Squared Error (RMSE):**
  ![RMSE formula](https://github.com/user-attachments/assets/2a669307-9870-4199-975e-703beb1ae57c)
- **Mean Absolute Error (MAE):**
  ![MAE formula](https://github.com/user-attachments/assets/5082c3e6-515f-4592-8af7-107e76255ea1)
- RMSE lebih sensitif terhadap outlier; MAE lebih stabil [5].

### Hasil Evaluasi
- **Collaborative filtering (SVD):**  

  - **RMSE rata-rata sebesar 0.8791** menunjukkan bahwa secara umum, prediksi rating dari model memiliki selisih sekitar +- 0.88 poin dari rating yang sebenarnya diberikan oleh pengguna. Dalam konteks skala rating 0.5-5.0, nilai ini tergolong sangat baik dan menandakan model memiliki akurasi prediksi yang tinggi.
  - **MAE rata-rata sebesar 0.6762** memberikan perspektif lain tentang rata-rata kesalahan, yang juga mengonfirmasi performa baik dari model.
- **Content-based:**  
  Dievaluasi secara kualitatif dengan memeriksa relevansi hasil rekomendasi ke user.

**Gambar:**  
![image](https://github.com/user-attachments/assets/09b18963-0b7c-4029-9c3c-79ae68e0b8a2)

![image](https://github.com/user-attachments/assets/04684e25-dcbd-4ce7-a104-d8cc09fb8494)



### Implikasi Bisnis

Sistem ini:
- Meningkatkan engagement user.
- Memudahkan user eksplorasi film baru & relevan.
- Menjadi basis sistem hybrid ke depan (kombinasi content & collaborative untuk atasi cold start) [1]

---

## 8. Kesimpulan

- Dua pendekatan rekomendasi dapat berjalan optimal di MovieLens.
- Tantangan utama: cold start pada film/user baru.
- Semua insight, evaluasi, dan proses mengacu pada literatur ilmiah [1][2][3][4][5].

---

## 9. Rekomendasi Pengembangan Lanjutan

- Hybrid system (menggabungkan kedua model)
- Context-aware (waktu, device, dsb)
- Interpretabilitas model (SHAP/LIME)
- Penerapan ke skala data yang lebih besar (MovieLens 1M/20M)

---

## 10. Referensi

[1] Ricci, F., Rokach, L., & Shapira, B. (2015). *Recommender Systems Handbook*.  
[2] Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*.  
[3] Pazzani, M. J., & Billsus, D. (2007). *Content-based recommendation systems*.  
[4] Koren, Y., Bell, R., & Volinsky, C. (2009). *Matrix Factorization Techniques for Recommender Systems*.  
[5] Willmott, C.J., & Matsuura, K. (2005). *Advantages of the mean absolute error (MAE) over the root mean square error (RMSE) in assessing average model performance*. *Climate Research*, 30, 79-82.

> Catatan: pembuat tidak secara menyeluruh membaca *keseluruhan* referensi di atas, melainkan dengan bantuan alat rangkum (SciSpace).

---

## 11. Struktur Submission

- `movie_recommendation_excellent.ipynb` : Notebook utama, lengkap dengan penjelasan dan sitasi.
- `README.md` : Laporan proyek.
- (Opsional) `movie_recommendation.py` : Script python.

---

**Catatan Teknis:**  
Jika menggunakan SVD (surprise), gunakan Numpy < 2.0.  
Jika dijalankan di Colab dan muncul warning dependency, abaikan selama hanya menjalankan sistem rekomendasi.

---
