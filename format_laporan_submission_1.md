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
- **id**: Merupakan ID unik untuk setiap pasien, bertipe data integer.
- **gender** Menunjukkan jenis kelamin pasien dengan nilai "Male", "Female", atau "Other".
- **age**: Usia pasien dalam bentuk numerik (float).
- **hypertension**: Bernilai 0 jika pasien tidak menderita hipertensi, dan 1 jika menderita.
- **heart_disease**: Bernilai 0 jika pasien tidak memiliki penyakit jantung, dan 1 jika memiliki.
- **ever_married**: Status pernikahan pasien, dengan nilai "Yes" atau "No".
- **work_type**: Jenis pekerjaan pasien, bisa berupa "children", "Govt_job", "Never_worked", "Private", atau "Self-employed".
- **Residence_type**: Jenis tempat tinggal pasien, antara "Urban" dan "Rural".
- **avg_glucose_level**: Rata-rata kadar glukosa darah pasien dalam satuan numerik (float).
- **bmi**: Indeks massa tubuh pasien dalam bentuk numerik (float).
- **smoking_status**: Status merokok pasien, terdiri dari "formerly smoked", "never smoked", "smokes", atau "Unknown".
- **stroke**: Label target, bernilai 1 jika pasien pernah mengalami stroke, dan 0 jika tidak.

**Exploratory Data Analysis - Deskripsi Variabel**

| No | Kolom               | Non-Null Count | Tipe Data |
|----|---------------------|----------------|-----------|
| 0  | id                  | 5110 non-null  | int64     |
| 1  | gender              | 5110 non-null  | object    |
| 2  | age                 | 5110 non-null  | float64   |
| 3  | hypertension        | 5110 non-null  | int64     |
| 4  | heart_disease       | 5110 non-null  | int64     |
| 5  | ever_married        | 5110 non-null  | object    |
| 6  | work_type           | 5110 non-null  | object    |
| 7  | Residence_type      | 5110 non-null  | object    |
| 8  | avg_glucose_level   | 5110 non-null  | float64   |
| 9  | bmi                 | 4909 non-null  | float64   |
| 10 | smoking_status      | 5110 non-null  | object    |
| 11 | stroke              | 5110 non-null  | int64     |

Kolom `bmi` memiliki **201 nilai kosong (null)** dari total 5110 data. Nilai null ini perlu diatasi sebelum pelatihan model dengan cara **menghapus baris yang mengandung null** dan nanti jumlah data menjadi 4909.

Berikut adalah kolom yang dihapus atau tidak digunakan dalam proses pelatihan model karena tidak relevan terhadap prediksi atau berfungsi hanya sebagai pengidentifikasi:
- id karena kolom ini hanya berisi identitas unik pasien, tidak memiliki pengaruh terhadap risiko stroke.

Variabel kategorikal yang di-encode menggunakan LabelEncoder:
- gender: Jenis kelamin pasien, seperti "Male", "Female", dan "Other".
- ever_married: Status pernikahan pasien, dengan nilai "Yes" atau "No".
- work_type: Jenis pekerjaan pasien, contohnya "Private", "Self-employed", "Govt_job", "children", atau "Never_worked".
- Residence_type: Jenis tempat tinggal pasien, yaitu "Urban" atau "Rural".
- smoking_status: Status merokok pasien, seperti "never smoked", "formerly smoked", "smokes", dan "Unknown".
Kelima variabel tersebut diubah ke dalam format numerik agar bisa digunakan oleh model machine learning. Variabel lain seperti age, hypertension, heart_disease, avg_glucose_level, bmi, dan stroke tidak perlu di-encode karena sudah dalam bentuk numerik atau biner.

**Mengecek Duplikat data dan data yang outliner.**
Data duplikat dari data tersebut dan tidak ada duplikat data

| Statistik    | gender | age   | hypertension | heart_disease | ever_married | work_type | Residence_type | avg_glucose_level | bmi    | smoking_status |
|--------------|--------|-------|--------------|---------------|--------------|-----------|----------------|-------------------|--------|----------------|
| Count        | 4909   | 4909  | 4909         | 4909          | 4909         | 4909      | 4909           | 4909              | 4909   | 4909           |
| Mean         | 0.41   | 42.87 | 0.09         | 0.05          | 0.65         | 2.17      | 0.51           | 105.31            | 28.89  | 1.38           |
| Std Dev      | 0.49   | 22.56 | 0.29         | 0.22          | 0.48         | 1.09      | 0.50           | 44.42             | 7.85   | 1.07           |
| Min          | 0      | 0.08  | 0            | 0             | 0            | 0         | 0              | 55.12             | 10.30  | 0              |
| 25% (Q1)     | 0      | 25    | 0            | 0             | 0            | 2         | 0              | 77.07             | 23.50  | 0              |
| 50% (Median) | 0      | 44    | 0            | 0             | 1            | 2         | 1              | 91.68             | 28.10  | 2              |
| 75% (Q3)     | 1      | 60    | 0            | 0             | 1            | 3         | 1              | 113.57            | 33.10  | 2              |
| Max          | 2      | 82    | 1            | 1             | 1            | 4         | 1              | 271.74            | 97.60  | 3              |

Berikut penjelasan singkat dalam bentuk kalimat untuk setiap atribut pada dataset:
- Gender menunjukkan jenis kelamin responden yang sudah diubah ke bentuk numerik, mayoritas adalah perempuan (nilai 0).
- Age adalah usia responden. Terdapat nilai yang tidak logis yaitu 0.08 tahun (sekitar 29 hari), sehingga bisa dianggap sebagai outlier dan perlu dihapus.
- Hypertension adalah status hipertensi (0 = tidak, 1 = ya), dengan rata-rata 9% dari responden memiliki hipertensi.
- Heart Disease menunjukkan apakah responden memiliki penyakit jantung (0 = tidak, 1 = ya), dengan rata-rata hanya 5%.
- Ever Married adalah status pernikahan (0 = belum, 1 = sudah), dengan mayoritas responden sudah menikah.
- Work Type adalah jenis pekerjaan yang telah dikodekan, mayoritas bekerja di sektor swasta.
- Residence Type menunjukkan tempat tinggal (0 = pedesaan, 1 = perkotaan), distribusinya hampir seimbang.
- Average Glucose Level adalah kadar glukosa rata-rata dalam darah. Terdapat beberapa nilai tinggi (>200) yang masih masuk akal secara medis, namun perlu dicek sebagai outlier potensial.
- BMI (Body Mass Index) menunjukkan indeks massa tubuh. Terdapat nilai sangat rendah dan sangat tinggi yang kemungkinan adalah outlier.
- Smoking Status adalah status merokok yang telah diubah ke bentuk numerik. Beberapa data memiliki kategori "Unknown" yang jumlahnya cukup banyak.
Secara keseluruhan, data sudah cukup bersih namun terdapat beberapa nilai ekstrem yang perlu dianalisis lebih lanjut sebelum digunakan untuk pemodelan.

## Data Preparation
Data preparation adalah tahap penting sebelum membangun model machine learning. Tahapan ini bertujuan untuk membersihkan, mengubah, dan menyusun ulang data agar dapat digunakan secara optimal oleh algoritma. Berikut adalah tahapan yang dilakukan:

- **Menghapus missing value**
Dataset memiliki kolom bmi dengan sebagian nilai kosong (NaN). Untuk mengatasi hal ini, baris yang memiliki nilai kosong dihapus. Karena hanya sebagian kecil dari total baris yang memiliki missing value, dan kolom tersebut merupakan data numerik yang cukup sensitif, maka penghapusan lebih disarankan daripada imputasi. Imputasi bisa menambahkan bias jika tidak dilakukan dengan hati-hati.

- **Menghapus Nilai Tidak Logis (Outlier Ekstrem)**
  Pada kolom age, terdapat data dengan nilai minimum 0.08 tahun, yang jika dikonversi hanya sekitar 29 hari. Ini sangat tidak relevan untuk kasus stroke. Stroke hampir tidak pernah terjadi pada bayi baru lahir, sehingga nilai tersebut tidak logis dan dianggap sebagai outlier ekstrem. Data dengan age < 1 tahun dihapus.
  
- **Encoding fitur kategori**
  Kolom kategorikal seperti gender, ever_married, work_type, Residence_type, dan smoking_status diubah menjadi angka menggunakan LabelEncoder. Model machine learning umumnya hanya dapat memproses data numerik. Encoding mengubah string menjadi representasi numerik sehingga dapat digunakan dalam model.

- **Pembagian dataset dengan fungsi train_test_split dari library sklearn**
  Memisahkan data menjadi dua bagian: data pelatihan (80%) dan data pengujian (20%) dengan train_test_split dari sklearn. Model perlu diuji pada data yang belum pernah dilihat untuk mengetahui seberapa baik kemampuannya melakukan generalisasi.
  
- **Standarisasi.**
  Melakukan standardisasi pada fitur numerik menggunakan StandardScaler, agar semua fitur memiliki distribusi dengan mean 0 dan standar deviasi 1.Beberapa algoritma seperti K-Nearest Neighbors (KNN) atau algoritma berbasis jarak sangat dipengaruhi oleh skala data. Fitur dengan skala lebih besar dapat mendominasi hasil.
  
**KESIMPULAN perlu dilakukan data prepration**
- Proses data preparation sangat krusial karena data mentah seringkali mengandung berbagai masalah seperti nilai kosong, data tidak logis, skala tidak seragam, hingga tipe data yang tidak sesuai. Jika tidak diproses dengan benar, kualitas dan akurasi model yang dibangun bisa menurun drastis.
- Dengan membersihkan data dari missing value dan outlier, kita memastikan data yang digunakan representatif dan dapat dipercaya. Encoding diperlukan agar algoritma dapat memproses fitur kategorikal secara numerik. Pembagian dataset ke dalam data latih dan uji membantu menghindari overfitting dan mengevaluasi performa model secara objektif. Terakhir, standarisasi menjamin bahwa semua fitur numerik memiliki kontribusi yang seimbang dalam pelatihan model.
Tanpa tahapan ini, risiko kesalahan interpretasi model dan penurunan performa menjadi sangat tinggi. Oleh karena itu, data preparation adalah langkah fundamental sebelum memasuki tahap pemodelan machine learning.
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


