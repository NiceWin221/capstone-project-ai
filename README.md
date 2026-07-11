# Aplikasi Navigasi Agen Cerdas & Mini-Game Strategi

Aplikasi berbasis web sederhana menggunakan **Streamlit** untuk mensimulasikan navigasi agen pintar menggunakan algoritma pencarian **A* Search** (Informed) dan **BFS** (Uninformed), visualisasi jalur dengan Bayesian Weather Cost, prediksi energi baterai dengan **Linear Regression**, serta mini-game **Tic-Tac-Toe** turn-based melawan Bot AI dengan algoritma **Minimax + Alpha-Beta Pruning**.

---

## 🛠️ Cara Menjalankan Aplikasi Secara Lokal

Ikuti langkah-langkah di bawah ini sesuai sistem operasi yang Anda gunakan.

### 1. Prasyarat (Prerequisites)
Pastikan Anda sudah menginstal Python di komputer Anda:
* **Windows/Mac/Linux**: Gunakan Python versi **3.10 s.d. 3.13** (Direkomendasikan **Python 3.12**).

---

### 2. Langkah-Langkah Jalankan Aplikasi

#### 💻 Pengguna MacOS / Linux
Buka Terminal Anda dan jalankan perintah berikut secara berurutan:

```bash
# 1. Masuk ke direktori project
cd capstone-project-ai

# 2. Buat virtual environment (.venv)
python3 -m venv .venv

# 3. Aktifkan virtual environment
source .venv/bin/activate

# 4. Install library pendukung
pip install -r requirements.txt

# 5. Jalankan aplikasi Streamlit
streamlit run app.py
```

#### 🪟 Pengguna Windows
Buka Command Prompt (cmd) atau PowerShell, lalu jalankan perintah berikut secara berurutan:

```bash
# 1. Masuk ke direktori/folder project
cd capstone-project-ai

# 2. Buat virtual environment (.venv)
python -m venv .venv

# 3. Aktifkan virtual environment
# Jika menggunakan Command Prompt (CMD):
.venv\Scripts\activate
# Jika menggunakan PowerShell:
.venv\Scripts\Activate.ps1

# 4. Install library pendukung
pip install -r requirements.txt

# 5. Jalankan aplikasi Streamlit
streamlit run app.py
```

---

## 📦 Isi Folder Project
* **`app.py`**: Berisi seluruh logika backend (A*, BFS, Minimax, Linear Regression) dan tampilan frontend web Streamlit.
* **`requirements.txt`**: Daftar library Python yang dibutuhkan aplikasi agar dapat berjalan dengan lancar.
* **`README.md`**: Panduan cara instalasi dan running aplikasi.
