<div align="center">

# 🇮🇩 Indonesia Automatic License Plate Recognition (ALPR)
**Intelligent Toll Road Monitoring System**

[![Python Version](https://img.shields.io/badge/Python-3.13.5-blue.svg)](https://www.python.org/downloads/release/python-3135/)
[![Framework](https://img.shields.io/badge/Framework-PyQt5-green.svg)](https://pypi.org/project/PyQt5/)
[![YOLO](https://img.shields.io/badge/Model-YOLOv8--OBB-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

</div>

---

## 📖 Deskripsi Singkat

**Indonesia ALPR** adalah sistem cerdas pengenalan pelat nomor kendaraan bermotor otomatis (Automatic License Plate Recognition) yang dioptimalkan untuk kondisi lalu lintas Indonesia. Proyek ini awalnya dirancang untuk pemantauan jalan tol dengan mengintegrasikan deteksi pelat yang sangat presisi (meskipun dalam posisi miring) dan pengenalan karakter optik yang cepat.

Aplikasi ini dibungkus dengan antarmuka pengguna (GUI) yang responsif dan dilengkapi mode gelap, membuatnya siap pakai untuk kebutuhan ruang kontrol *dashboard* monitoring jalan tol.

## ✨ Fitur Utama

- **Oriented Bounding Box (OBB) Detection**: Menggunakan YOLOv8-OBB untuk mendeteksi pelat nomor secara presisi meskipun gambar diambil dari sudut kamera yang miring atau tidak sejajar.
- **Robust Object Tracking**: Dilengkapi dengan algoritma pelacakan *Centroid Tracker* berbasis *Euclidean Distance* menggunakan algoritma Hungarian (*linear sum assignment*) untuk mencegah pengenalan ganda pada kendaraan yang sama.
- **Geometric Rectification**: Koreksi perspektif (Warp Perspective) otomatis untuk meluruskan potongan gambar pelat yang miring sebelum diteruskan ke OCR.
- **Deep Learning OCR**: Menggunakan arsitektur CRNN (Convolutional Recurrent Neural Network) dipadukan dengan CTC (Connectionist Temporal Classification) Loss untuk membaca pelat nomor.
- **Modern Dashboard UI**: Dibangun dengan antarmuka PyQt5 yang berjalan secara asynchronous (multi-threading) sehingga video *feed* tidak akan menyebabkan GUI membeku (*freeze*).
- **Auto Logging System**: Pencatatan pelat nomor hasil deteksi langsung ke UI dan _file_ log, lengkap dengan fitur penyimpanan _crop_ gambar pelat lokal.

## 🛠 Teknologi & Arsitektur

- **Bahasa Pemrograman**: Python 3.13.5
- **Computer Vision**: OpenCV (`opencv-python`)
- **Deteksi Objek**: Ultralytics YOLOv8
- **Model Pengenalan Teks (OCR)**: PyTorch & Torchvision (CRNN)
- **Komputasi & Algoritma**: NumPy, SciPy (Spatial & Optimize)
- **Antarmuka Pengguna (GUI)**: PyQt5
- **Pemrosesan Citra**: Pillow (PIL)

---

## ⚙️ Prasyarat (Prerequisites)

Sebelum menginstal dan menjalankan aplikasi ini, pastikan sistem Anda telah memiliki:

1. **Python 3.13.5** (atau yang lebih baru) yang sudah terdaftar di Environment Variables (`PATH`).
2. **Git** untuk meng-clone repositori (opsional).
3. **GPU (Opsional namun Sangat Disarankan)**: Sistem dapat berjalan menggunakan CPU (CPU only), namun untuk kinerja real-time dan kestabilan FPS (*Frames Per Second*) saat memproses video langsung, penggunaan **NVIDIA GPU dengan CUDA Toolkit** sangat disarankan.

---

## 🚀 Instalasi

Ikuti langkah-langkah berikut untuk mengatur _environment_ dan menjalankan proyek di lokal Anda:

**1. Clone Repositori**
```bash
git clone https://github.com/rzr06/Indonesia-ALPR.git
cd Indonesia-ALPR
```

**2. Buat Virtual Environment (Sangat Disarankan)**
```bash
python -m venv venv

# Aktivasi di Windows:
venv\Scripts\activate

# Aktivasi di Linux/Mac:
source venv/bin/activate
```

**3. Install Dependensi**
```bash
pip install -r requirements.txt
```

**4. Unduh Bobot Model (Weights)**
Dikarenakan ukuran model AI yang terlalu besar untuk repositori GitHub, silakan unduh kedua file model (`yolo_obb.pt` dan `crnn.pth`) secara manual dari tautan Google Drive di bawah ini:

🔗 **[Download Model ALPR dari Google Drive](https://drive.google.com/drive/folders/1v2W_7p3P4lKOHxG4FVC9yKvIvhtNPbcC)**

Setelah diunduh, letakkan kedua file tersebut tepat di dalam folder `models/` yang ada di root repositori.
```text
models/
├── crnn.pth
└── yolo_obb.pt
```

---

## 💻 Penggunaan (Usage)

Untuk menjalankan *dashboard* ALPR:

```bash
python main.py
```

Setelah antarmuka *dashboard* terbuka, Anda dapat berinteraksi dengan kontrol panel:
- **📸 Upload Image**: Untuk menguji sistem pada 1 file foto statis.
- **🎥 Upload Video**: Untuk memutar rekaman CCTV luring (misalnya `.mp4` atau `.avi`).
- **🔴 Start Live Camera**: Untuk menghubungkan sistem dengan webcam atau input kamera utama (Source Index `0`).
- **⏹ Stop Processing**: Menghentikan sesi pemrosesan video berjalan dan mereset antarmuka.

---

## 📂 Struktur Direktori

Proyek ini menggunakan pemisahan tanggung jawab (*Clean Architecture*) agar mudah untuk dipelihara dan dikembangkan lebih lanjut.

```text
.
├── main.py                  # Entry point aplikasi (Main Launcher)
├── requirements.txt         # Daftar dependensi library Python
├── README.md                # Dokumentasi Proyek
├── models/                  # File bobot Neural Network (Download manual)
│   ├── crnn.pth
│   └── yolo_obb.pt
├── core/                    # Engine utama sistem
│   ├── __init__.py
│   └── engine.py            # Logika sinkronisasi deteksi, tracking, dan OCR
├── inference/               # Wrapper dan arsitektur Model AI
│   ├── __init__.py
│   ├── crnn_network.py      # Definisi Neural Network CRNN
│   ├── detector.py          # Wrapper YOLOv8 OBB
│   └── recognizer.py        # Wrapper Inferensi CRNN
├── utils/                   # Fungsi utilitas & kalkulasi
│   ├── __init__.py
│   ├── geometry.py          # Logika perspective warp (transformasi gambar)
│   └── tracker.py           # Algoritma Centroid Tracker kustom
└── ui/                      # Komponen Graphic User Interface (GUI)
    ├── __init__.py
    ├── main_window.py       # Kelas utama Window PyQt5
    └── video_thread.py      # QThread untuk render frame secara asynchronous
```

---

## 🤝 Panduan Berkontribusi

Proyek ini sangat terbuka untuk kontribusi! Jika Anda menemukan kutu (*bug*), ingin mengoptimalkan performa (misalnya menambahkan model TensorRT), atau merapikan fitur UI:

1. Lakukan **Fork** pada repositori ini.
2. Buat *branch* fitur Anda: `git checkout -b feature/FiturBaruAnda`
3. Lakukan *commit* pada perubahan Anda: `git commit -m 'Menambahkan FiturBaruAnda'`
4. Lakukan *push* ke *branch* tersebut: `git push origin feature/FiturBaruAnda`
5. Buka **Pull Request** di repositori orisinal.

Kami akan melakukan tinjauan teknis atas kode yang Anda kirimkan secepatnya.

---

## 📜 Lisensi

Proyek ini dilisensikan di bawah lisensi **GPL-3.0 (GNU General Public License v3.0)**. 
Lisensi ini memastikan bahwa perangkat lunak yang ada akan selalu menjadi sumber terbuka (*open source*). Segala bentuk pengembangan, modifikasi, atau pendistribusian ulang dari kode ini **wajib** dilisensikan di bawah lisensi yang sama.

Silakan lihat file `LICENSE` untuk detail penuh.

---

## 📬 Kredit & Kontak

Dikembangkan dan dipelihara oleh **Haikal Purnama Aji**

- **GitHub**: [@rzr06](https://github.com/rzr06)
- **LinkedIn**: [Haikal Purnama Aji](https://www.linkedin.com/in/haikal-purnama-aji/)
- **Instagram**: [@_rzr06](https://www.instagram.com/_rzr06/)
- **Email**: [haikalpurnama004@gmail.com] *(Silakan hubungi saya melalui pesan langsung LinkedIn atau Instagram)*

<br>
<p align="center">
  <i>Dibuat untuk kemajuan Teknologi Cerdas di Indonesia.</i>
</p>
