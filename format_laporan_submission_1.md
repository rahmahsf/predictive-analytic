# Laporan Proyek Machine Learning - Rahmah Sary Fadiyah

## Domain Proyek : Kesehatan

Stroke merupakan salah satu penyebab utama kematian dan kecacatan jangka panjang di seluruh dunia. Menurut World Health Organization (WHO), sekitar 15 juta orang mengalami stroke setiap tahunnya, dan sekitar 5 juta di antaranya meninggal dunia sementara 5 juta lainnya mengalami kecacatan permanen [1].

Stroke terjadi ketika pasokan darah ke otak terganggu atau berkurang, sehingga jaringan otak tidak mendapatkan oksigen dan nutrisi yang cukup. Dalam banyak kasus, stroke bisa dicegah apabila faktor risikonya dapat dikenali sejak dini, seperti faktor utama penyebab stroke adalah hipertensi, selain itu juga faktor resiko laninya adalah merokok, diabetes melitus dan dispidemia [2].

Namun, banyak pasien tidak menyadari adanya risiko ini sampai mereka benar-benar mengalami stroke. Oleh karena itu, pemanfaatan teknologi prediktif berbasis machine learning dapat menjadi solusi untuk mendeteksi potensi stroke lebih awal dengan memanfaatkan data medis dan gaya hidup pasien.

Dengan pendekatan klasifikasi, proyek ini bertujuan membangun model prediksi yang dapat mengidentifikasi individu dengan risiko tinggi stroke, sehingga tindakan pencegahan dapat dilakukan secara proaktif.

## Business Understanding
### Problem Statements
Permasalahan dari statement ini adalah:
- Bagaimana cara mengidentifikasi individu dengan risiko tinggi terkena stroke menggunakan data kesehatan dasar dan gaya hidup?
- Apa saja fitur (variabel) yang paling berpengaruh terhadap risiko stroke pada individu?
- Seberapa akurat model klasifikasi yang dikembangkan dalam memprediksi kejadian stroke?

### Goals
Menjelaskan tujuan dari pernyataan masalah:
- Mengembangkan model klasifikasi berbasis machine learning untuk memprediksi apakah seseorang berisiko mengalami stroke atau tidak.
- Melakukan eksplorasi data dan analisis fitur untuk mengetahui faktor-faktor signifikan yang memengaruhi risiko stroke.
- Mengevaluasi performa model prediksi menggunakan metrik evaluasi seperti akurasi, precision, recall, dan F1-score 

### Solution statements
- Membangun baseline model menggunakan Logistic Regression sebagai pembanding awal untuk mengukur performa dasar dengan metrik evaluasi seperti akurasi, precision, recall, F1-score.
- Melakukan hyperparameter tuning menggunakan GridSearchCV atau RandomizedSearchCV untuk memperoleh kombinasi parameter terbaik pada model unggulan berdasarkan evaluasi ROC-AUC rata-rata.
- Menangani ketidakseimbangan data kelas dengan menerapkan teknik resampling seperti SMOTE, undersampling, atau penyesuaian class weight agar model lebih sensitif terhadap prediksi kelas minoritas.

## Data Understanding
[Dataset Stroke Prediction didapatkan dari Kaggle](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset), dataset ini digunakan untuk memprediksi apakah seorang pasien berisiko mengalami stroke berdasarkan beberapa parameter input seperti jenis kelamin, usia, riwayat penyakit (seperti hipertensi dan penyakit jantung), serta status merokok. Setiap baris dalam dataset ini memberikan informasi relevan mengenai masing-masing pasien, yang dapat membantu dalam membangun model prediktif untuk deteksi dini dan pencegahan stroke.
Informasi Metadata:
- Pemilik/Author: fedesoriano
- Kategori: Kesehatan, Klasifikasi Biner
- Lisensi: © Pemilik Asli (Original Authors)
- Frekuensi Pembaruan: Tidak pernah (Never)
- Tags: Health, Health Conditions, Healthcare, Public Health, Binary Classification
 

### Variabel-variabel pada Stroke Prediction dataset adalah sebagai berikut:
| No | Nama Atribut       | Tipe Data       | Deskripsi                                                                 |
|----|--------------------|-----------------|---------------------------------------------------------------------------|
| 1  | id                 | Integer         | ID unik pasien                                                            |
| 2  | gender             | Kategori        | Jenis kelamin pasien: "Male", "Female", atau "Other"                     |
| 3  | age                | Numerik (Float) | Usia pasien                                                               |
| 4  | hypertension       | Biner (0/1)     | 0 jika pasien tidak menderita hipertensi, 1 jika ya                      |
| 5  | heart_disease      | Biner (0/1)     | 0 jika pasien tidak memiliki penyakit jantung, 1 jika ya                 |
| 6  | ever_married       | Kategori        | Status pernikahan: "Yes" atau "No"                                       |
| 7  | work_type          | Kategori        | Jenis pekerjaan: "children", "Govt_job", "Never_worked", "Private", "Self-employed" |
| 8  | Residence_type     | Kategori        | Jenis tempat tinggal: "Urban" atau "Rural"                               |
| 9  | avg_glucose_level  | Numerik (Float) | Rata-rata kadar glukosa dalam darah                                      |
|10  | bmi                | Numerik (Float) | Indeks massa tubuh                                                       |
|11  | smoking_status     | Kategori        | Status merokok: "formerly smoked", "never smoked", "smokes", atau "Unknown" |
|12  | stroke             | Biner (0/1)     | 1 jika pasien pernah terkena stroke, 0 jika tidak                         |


**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

## Evaluation
Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
[1] [World Health Organization. (2021). Stroke: Key facts](https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death)
[2] [Utama, Y. A., & Nainggolan, S. S. (2022). Faktor resiko yang mempengaruhi kejadian stroke: sebuah tinjauan sistematis. *Jurnal Ilmiah Universitas Batanghari Jambi*, 22(1), 549-553.](https://ji.unbari.ac.id/index.php/ilmiah/article/view/1950)


