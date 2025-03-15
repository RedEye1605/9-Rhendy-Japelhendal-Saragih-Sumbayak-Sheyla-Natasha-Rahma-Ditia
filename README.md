# BCC Intern 2025 - Data Science Collaboration
 Repository ini dibuat untuk kolaborasi intern BCC 2025 dalam bidang Data Science. Repo ini digunakan untuk menyimpan dan berbagi notebook Jupyter, dataset, script Python, serta dokumentasi terkait proyek yang dikerjakan bersama.

## Struktur Repository
```
📂 BCC-Intern-2025-DataScience
├── 📁 datasets/
│   ├── 📁 raw/           # Data mentah/original
│   └── 📁 processed/     # Data yang sudah diproses
├── 📁 docs/                 # Dokumentasi proyek
│   └── codebook23_llcp-v2-508.HTML  # CDC BRFSS codebook
├── 📁 models/               # Model yang sudah dilatih
│   ├── best_xgb.pkl      # XGBoost model
│   ├── best_rf.pkl       # Random Forest model
│   └── best_lr.pkl       # Logistic Regression model
├── 📁 notebooks/         
│   ├── 📝 data_collection.ipynb      # Pengumpulan data
│   ├── 📝 data_preprocessing.ipynb   # Preprocessing data
│   ├── 📝 data_visualization.ipynb   # Visualisasi data
│   ├── 📝 modeling.ipynb            # Model machine learning
│   └── 📝 final_notebook.ipynb      # Notebook final
├── 📁 scripts/
│   ├── 📄 heart_attack_prediction_app.py   # Streamlit app
├── 📄 README.md            # Panduan repository
├── 📄 requirements.txt      # Dependensi Python
└── 📄 .gitignore           # File yang diabaikan git
```

## Cara Berkontribusi
1. Fork & Clone repository ini ke komputer lokal:
    ```bash
    git clone https://github.com/RedEye1605/9-Rhendy-Japelhendal-Saragih-Sumbayak-Sheyla-Natasha-Rahma-Ditia.git
    ```

2. Buat branch baru sesuai dengan task yang sedang dikerjakan:
    ```bash
    git checkout -b feature/nama-task
    ```

3. Tambahkan atau ubah file yang diperlukan:
    - Pastikan mengikuti struktur repository
    - Gunakan nama file yang deskriptif
    - Dokumentasikan perubahan dengan baik

4. Lakukan commit dengan pesan yang jelas:
    ```bash
    git add .
    git commit -m "feat: deskripsi perubahan yang dilakukan"
    ```

5. Push dan kirim Pull Request (PR):
    ```bash
    git push origin feature/nama-task
    ```
    - Buat PR melalui GitHub
    - Berikan deskripsi detail tentang perubahan
    - Tunggu review dari maintainer

6. Setelah PR di-merge:
    ```bash
    git checkout main
    git pull origin main
    ```

## Selalu Sinkronisasi dengan Main Branch
Sebelum memulai task baru, selalu update repository lokal:
```bash
git checkout main
git pull origin main
```

## Fitur Aplikasi
- Prediksi risiko serangan jantung berdasarkan informasi kesehatan
- Menggunakan ensemble 3 model machine learning:
    - XGBoost 
    - Random Forest
    - Logistic Regression
- Visualisasi penilaian risiko dengan gauge chart
- Identifikasi dan tampilan faktor risiko utama
- Rekomendasi personal berdasarkan tingkat risiko

## Model Machine Learning
Aplikasi menggunakan 3 model yang dilatih dengan dataset CDC BRFSS:
- **XGBoost**: Algoritma gradient boosting untuk data terstruktur
- **Random Forest**: Ensemble decision tree yang robust terhadap outlier 
- **Logistic Regression**: Model statistik untuk klasifikasi biner

## Sumber Data
Dataset berasal dari CDC's Behavioral Risk Factor Surveillance System (BRFSS) - survei kesehatan via telepon yang mengumpulkan data dari penduduk AS tentang:
- Perilaku berisiko terkait kesehatan
- Kondisi kesehatan kronis
- Penggunaan layanan preventif

## Deployment
Aplikasi telah dideploy menggunakan Streamlit Cloud dan dapat diakses melalui link berikut:
- [Heart Attack Risk Prediction App](https://9-rhendy-japelhendal-saragih-sumbayak-sheyla-natasha-rahma-dit.streamlit.app/)

### Menjalankan Aplikasi Secara Lokal
1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Jalankan aplikasi:
```bash
streamlit run scripts/heart_attack_prediction_app.py
```

3. Buka browser dan akses:
```
http://localhost:8501
```

## Kontributor
- Rhendy Japelhendal Saragih Sumbayak
- Sheyla Natasha Rahma Ditia

## Disclaimer
Aplikasi ini memberikan estimasi risiko serangan jantung berdasarkan model statistik. Tidak menggantikan saran, diagnosis, atau penanganan medis profesional. Selalu konsultasikan dengan tenaga kesehatan untuk masalah medis.
