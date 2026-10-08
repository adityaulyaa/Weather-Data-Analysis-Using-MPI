# Jawaban Analisis Proyek MPI dan CUDA

Dokumen ini menjawab pertanyaan berdasarkan isi dua folder proyek:

- `MPI_Project`: proyek **Weather Anomaly Detection Using MPI** dengan file utama `weather_anomaly.c`.
- `CUDA_Project`: proyek **Financial Risk Analysis / Monte Carlo Simulation Using CUDA** dengan file utama `Monte_Carlo_Program.ipynb`, output `simulation_summary.json`, `monte_carlo_simulation_results.csv`, dan dashboard HTML.

---

# 01. Klasifikasi Sistem Proyek pada Tingkat SIMD, SIMT, dan MIMD

## Ringkasan Klasifikasi

| Model | MPI_Project | CUDA_Project |
|---|---|---|
| SIMD | Tidak tertulis eksplisit, tetapi operasi loop numerik pada CPU berpotensi dioptimasi compiler sebagai SIMD | Operasi NumPy/CuPy seperti `dot`, `matmul`, dan `exp` dapat memakai vectorized operation, tetapi secara konsep CUDA lebih dominan SIMT |
| SIMT | Tidak digunakan | Digunakan pada eksekusi GPU CUDA melalui CuPy/cuBLAS/cuRAND |
| MIMD | Digunakan secara eksplisit melalui banyak proses MPI | Tidak eksplisit dalam notebook, kecuali jika dijalankan oleh banyak proses secara eksternal |

---

## Untuk MPI_Project

File utama: `MPI_Project/weather_anomaly.c`

Proyek MPI melakukan deteksi anomali suhu maksimum dari dataset cuaca. Program dijalankan dengan beberapa proses MPI. Setiap proses memiliki rank sendiri dan memproses bagian data masing-masing.

### 1. MIMD pada MPI

Klasifikasi utama proyek MPI adalah **MIMD (Multiple Instruction, Multiple Data)**.

Alasannya:

- Program dijalankan oleh beberapa proses MPI.
- Setiap proses memiliki ruang alamat sendiri.
- Setiap proses dapat menjalankan instruksi yang sama secara umum, tetapi pada cabang tertentu rank 0 menjalankan pekerjaan berbeda dari rank lain.
- Data yang diproses setiap rank berbeda.

Bagian kode yang menunjukkan MIMD:

```c
MPI_Comm_rank(MPI_COMM_WORLD, &rank);
MPI_Comm_size(MPI_COMM_WORLD, &size);
```

Setelah rank diketahui, terdapat pembagian peran:

```c
if (rank == 0) {
    full_dataset = (WeatherRecord *)malloc(MAX_RECORDS * sizeof(WeatherRecord));
    total_records = load_weather_csv("dataset_10000_rows.csv", full_dataset);
}
```

Artinya:

- Rank 0 bertindak sebagai master.
- Rank 0 membaca dataset dari file CSV.
- Rank lain tidak membaca file secara langsung.

Kemudian data dibagikan:

```c
MPI_Scatterv(all_temps, send_counts, displs, MPI_DOUBLE,
             local_temps, local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

Setiap rank memproses potongan data berbeda:

```c
for (int i = 0; i < local_count; i++) {
    if (fabs(local_temps[i] - global_mean) > (2.0 * global_std)) {
        local_anomalies++;
    }
}
```

Hasil lokal digabungkan:

```c
MPI_Reduce(&local_anomalies, &total_anomalies, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
```

Jadi, bagian deteksi anomali menggunakan pola:

```text
Rank 0      : baca data, hitung mean/std, kirim data, terima hasil akhir
Rank 1..N   : terima data lokal, hitung anomali lokal, kirim hasil
Semua rank  : menjalankan program yang sama tetapi pada data berbeda
```

Ini termasuk MIMD karena setiap proses MPI adalah unit eksekusi independen dengan memori sendiri.

### 2. SIMD pada MPI

SIMD tidak ditulis secara eksplisit menggunakan intrinsics seperti AVX/SSE. Namun, beberapa loop numerik dapat berpotensi dioptimasi oleh compiler menjadi SIMD.

Contoh loop:

```c
for (int i = 0; i < total_records; i++) {
    sum += full_dataset[i].max_temp;
}
```

```c
for (int i = 0; i < total_records; i++) {
    sq_diff_sum += pow(full_dataset[i].max_temp - global_mean, 2);
}
```

```c
for (int i = 0; i < local_count; i++) {
    if (fabs(local_temps[i] - global_mean) > (2.0 * global_std)) {
        local_anomalies++;
    }
}
```

Batas klasifikasi SIMD di proyek ini:

- Kode tidak secara eksplisit menggunakan instruksi SIMD.
- SIMD hanya mungkin terjadi jika compiler melakukan auto-vectorization.
- Fungsi seperti `pow()` dan cabang `if` pada loop anomali dapat membatasi vectorization.
- Jadi SIMD pada proyek MPI bersifat potensial, bukan desain utama.

### 3. SIMT pada MPI

Proyek MPI tidak menggunakan SIMT.

Alasannya:

- Tidak ada GPU kernel.
- Tidak ada CUDA thread/block/grid.
- Tidak ada eksekusi warp seperti pada GPU NVIDIA.

Jadi untuk MPI_Project:

```text
Klasifikasi utama : MIMD
SIMD              : mungkin pada level instruksi CPU jika compiler mengoptimasi loop
SIMT              : tidak digunakan
```

---

## Untuk CUDA_Project

File utama: `CUDA_Project/Monte_Carlo_Program.ipynb`

Proyek CUDA melakukan simulasi Monte Carlo untuk analisis risiko finansial portofolio saham AAPL, MSFT, GOOGL, AMZN, dan NVDA. Program membandingkan eksekusi CPU NumPy dan GPU CuPy.

### 1. SIMT pada CUDA

Klasifikasi utama proyek CUDA adalah **SIMT (Single Instruction, Multiple Threads)**.

Bagian notebook yang relevan:

```python
z_gpu = cp.random.normal(0, 1, size=(BIG_SIMULATIONS, len(tickers)))
correlated_returns_gpu = cp.matmul(z_gpu, chol_matrix_gpu.T)
portfolio_returns_gpu = cp.matmul(correlated_returns_gpu, weights_gpu)
simulated_portfolio_values_gpu = INITIAL_INVESTMENT * cp.exp(portfolio_returns_gpu * cp.sqrt(NUM_DAYS))
cp.cuda.Stream.null.synchronize()
```

Walaupun tidak ada kernel CUDA manual seperti `__global__`, CuPy menjalankan operasi tersebut menggunakan backend CUDA seperti:

- cuRAND untuk random number generation.
- cuBLAS untuk matrix multiplication.
- CUDA elementwise kernel untuk `exp`, perkalian, dan operasi vektor.

Pada GPU NVIDIA, operasi tersebut dijalankan oleh banyak thread dalam model SIMT. Satu instruksi kernel dieksekusi oleh banyak thread pada data berbeda.

Contoh pemetaan konseptual:

```text
Data simulasi Monte Carlo:
10.000.000 path x 5 aset

GPU:
Thread 0     -> memproses elemen/path tertentu
Thread 1     -> memproses elemen/path tertentu
Thread 2     -> memproses elemen/path tertentu
...
Thread N     -> memproses elemen/path tertentu
```

SIMT terlihat pada operasi berikut:

1. Generate random number:

```python
z_gpu = cp.random.normal(0, 1, size=(BIG_SIMULATIONS, len(tickers)))
```

2. Matrix multiplication:

```python
correlated_returns_gpu = cp.matmul(z_gpu, chol_matrix_gpu.T)
```

3. Portfolio weighted return:

```python
portfolio_returns_gpu = cp.matmul(correlated_returns_gpu, weights_gpu)
```

4. Transformasi eksponensial nilai portofolio:

```python
simulated_portfolio_values_gpu = INITIAL_INVESTMENT * cp.exp(...)
```

Batas klasifikasi SIMT:

- SIMT hanya berlaku pada bagian yang menggunakan `cp.*`, yaitu objek CuPy yang berada di GPU.
- Bagian `np.*` berjalan di CPU, bukan SIMT.
- Download data saham dengan `yfinance`, perhitungan awal `pandas`, dan sebagian analisis berbasis NumPy berjalan di CPU.
- CuPy menyembunyikan detail kernel, sehingga SIMT tidak tampak sebagai kode CUDA C eksplisit, tetapi tetap dieksekusi melalui CUDA runtime.

### 2. SIMD pada CUDA_Project

SIMD dapat muncul pada bagian CPU NumPy.

Bagian kode CPU:

```python
z_cpu = np.random.normal(0, 1, size=(BIG_SIMULATIONS, len(tickers)))
correlated_returns_cpu = np.dot(z_cpu, chol_matrix.T)
portfolio_returns_cpu = np.dot(correlated_returns_cpu, WEIGHTS)
simulated_portfolio_values_cpu = INITIAL_INVESTMENT * np.exp(portfolio_returns_cpu * np.sqrt(NUM_DAYS))
```

NumPy biasanya menggunakan pustaka numerik seperti BLAS yang dapat memanfaatkan instruksi SIMD CPU, misalnya SSE, AVX, atau AVX2, tergantung environment.

Batas SIMD pada CUDA_Project:

- SIMD hanya berlaku pada bagian CPU/NumPy.
- Tidak dikontrol langsung oleh kode notebook.
- Bergantung pada build NumPy, library BLAS, dan kemampuan CPU.

### 3. MIMD pada CUDA_Project

Proyek CUDA tidak secara eksplisit menggunakan MIMD.

Tidak ada:

- MPI rank.
- Multi-process GPU execution.
- Pembagian kerja antarnode.

Namun, jika notebook ini dijalankan di beberapa proses berbeda, misalnya melalui MPI eksternal atau scheduler cluster, maka bisa menjadi MIMD. Tetapi pada kode yang ada, CUDA_Project adalah single process yang menggunakan GPU melalui CuPy.

Jadi untuk CUDA_Project:

```text
Klasifikasi utama : SIMT pada eksekusi GPU CUDA
SIMD              : mungkin pada bagian CPU NumPy/BLAS
MIMD              : tidak eksplisit pada kode saat ini
```

---

# 03. Pengaruh Shared Memory, Distributed Memory, dan Host-Device Memory terhadap Kepemilikan Data

## Ringkasan Model Memori

| Model Memori | MPI_Project | CUDA_Project |
|---|---|---|
| Shared memory | Tidak digunakan antarrank; hanya memori lokal dalam satu proses | CPU memiliki memori host, GPU memiliki memori device; di dalam GPU terdapat memory hierarchy |
| Distributed memory | Digunakan eksplisit melalui MPI | Tidak digunakan eksplisit |
| Host-device memory | Tidak digunakan | Digunakan eksplisit melalui transfer NumPy ke CuPy dan CuPy ke NumPy |

---

## Untuk MPI_Project

### 1. Distributed Memory

MPI_Project menggunakan model **distributed memory**.

Setiap proses MPI memiliki ruang alamat sendiri. Variabel dengan nama sama pada rank berbeda bukan variabel yang sama secara fisik.

Contoh variabel:

```c
int rank, size;
WeatherRecord *full_dataset = NULL;
int total_records = 0;
double global_mean = 0.0, global_std = 0.0;
double *local_temps = NULL;
```

Meskipun semua rank memiliki variabel tersebut, isi dan alokasi memorinya berbeda pada setiap proses.

### 2. Kepemilikan Data

Kepemilikan data pada MPI_Project:

| Data | Pemilik Awal | Diakses Oleh | Mekanisme Distribusi |
|---|---|---|---|
| `full_dataset` | Rank 0 | Rank 0 saja | Tidak dibagikan langsung |
| `total_records` | Rank 0 | Semua rank | `MPI_Bcast` |
| `global_mean` | Rank 0 | Semua rank | `MPI_Bcast` |
| `global_std` | Rank 0 | Semua rank | `MPI_Bcast` |
| `all_temps` | Rank 0 | Rank 0 sebagai send buffer | `MPI_Scatterv` |
| `local_temps` | Masing-masing rank | Rank pemilik saja | Hasil `MPI_Scatterv` |
| `local_anomalies` | Masing-masing rank | Rank pemilik lalu dikirim ke rank 0 | `MPI_Reduce` |
| `total_anomalies` | Rank 0 | Rank 0 | `MPI_Reduce` |

### 3. Ruang Alamat MPI

Diagram ruang alamat:

```text
+-------------------------+        MPI_Bcast / MPI_Scatterv / MPI_Reduce        +-------------------------+
| Rank 0 Address Space    | <-------------------------------------------------> | Rank 1 Address Space    |
|                         |                                                     |                         |
| full_dataset            |                                                     | full_dataset = NULL     |
| all_temps               |                                                     | local_temps             |
| send_counts             |                                                     | local_anomalies         |
| displs                  |                                                     | total_records copy      |
| local_temps             |                                                     | global_mean copy        |
| total_anomalies         |                                                     | global_std copy         |
+-------------------------+                                                     +-------------------------+

+-------------------------+                                                     +-------------------------+
| Rank 2 Address Space    |                                                     | Rank N Address Space    |
| local_temps             |                                                     | local_temps             |
| local_anomalies         |                                                     | local_anomalies         |
| total_records copy      |                                                     | total_records copy      |
| global_mean copy        |                                                     | global_mean copy        |
| global_std copy         |                                                     | global_std copy         |
+-------------------------+                                                     +-------------------------+
```

### 4. Akses Legal

Akses legal:

- Rank 0 boleh mengakses `full_dataset`, `all_temps`, `send_counts`, dan `displs` karena hanya rank 0 yang mengalokasikannya.
- Setiap rank hanya boleh mengakses `local_temps` miliknya sendiri.
- Rank selain 0 tidak boleh mengakses `full_dataset` rank 0 secara langsung.
- Semua komunikasi data antarrank harus melalui MPI.

Contoh akses legal rank 0:

```c
if (rank == 0) {
    full_dataset = malloc(...);
    total_records = load_weather_csv(...);
}
```

Contoh akses legal semua rank setelah scatter:

```c
for (int i = 0; i < local_count; i++) {
    if (fabs(local_temps[i] - global_mean) > (2.0 * global_std)) {
        local_anomalies++;
    }
}
```

### 5. Lokasi Komunikasi Eksplisit

Komunikasi eksplisit terjadi pada:

1. Broadcast jumlah data:

```c
MPI_Bcast(&total_records, 1, MPI_INT, 0, MPI_COMM_WORLD);
```

2. Broadcast statistik global:

```c
MPI_Bcast(&global_mean, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
MPI_Bcast(&global_std, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

3. Distribusi suhu maksimum:

```c
MPI_Scatterv(all_temps, send_counts, displs, MPI_DOUBLE,
             local_temps, local_count, MPI_DOUBLE, 0, MPI_COMM_WORLD);
```

4. Penggabungan jumlah anomali:

```c
MPI_Reduce(&local_anomalies, &total_anomalies, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
```

### 6. Sinkronisasi

MPI collective operation berperan sebagai titik koordinasi logis. Contohnya:

- `MPI_Bcast` memastikan semua rank menerima nilai sebelum lanjut.
- `MPI_Scatterv` memastikan setiap rank menerima bagian data yang benar.
- `MPI_Reduce` memastikan hasil lokal digabung ke rank 0.

Tidak ada shared memory antarrank, sehingga tidak ada race condition pada variabel bersama. Risiko utama adalah mismatch jumlah data, tipe MPI, atau rank yang tidak ikut collective.

---

## Untuk CUDA_Project

### 1. Host Memory dan Device Memory

CUDA_Project menggunakan model **host-device memory**.

Host adalah CPU dan RAM utama. Device adalah GPU dan VRAM.

Pada notebook:

- Data awal diambil dengan `yfinance` dan diproses memakai `pandas`/`NumPy` di host.
- Data tertentu dipindahkan ke GPU memakai `cp.array()`.
- Hasil GPU dipindahkan kembali ke CPU memakai `cp.asnumpy()`.

### 2. Kepemilikan Data

| Data | Lokasi | Pemilik/Akses Legal |
|---|---|---|
| `data` dari yfinance | Host CPU | Python/pandas |
| `returns` | Host CPU | pandas/NumPy |
| `mean_returns` | Host CPU | NumPy |
| `cov_matrix` | Host CPU | NumPy |
| `chol_matrix` | Host CPU | NumPy |
| `WEIGHTS` | Host CPU | NumPy |
| `z_cpu` | Host CPU | NumPy |
| `correlated_returns_cpu` | Host CPU | NumPy |
| `chol_matrix_gpu` | Device GPU | CuPy/CUDA |
| `weights_gpu` | Device GPU | CuPy/CUDA |
| `z_gpu` | Device GPU | CuPy/CUDA |
| `correlated_returns_gpu` | Device GPU | CuPy/CUDA |
| `portfolio_returns_gpu` | Device GPU | CuPy/CUDA |
| `simulated_portfolio_values_gpu` | Device GPU | CuPy/CUDA |
| `portfolio_sim_results` | Host CPU | NumPy setelah `cp.asnumpy()` |

### 3. Ruang Alamat Host-Device

Diagram:

```text
+--------------------------------------------------+
| Host / CPU RAM                                   |
|                                                  |
| pandas DataFrame: data                           |
| NumPy arrays: returns, cov_matrix, chol_matrix   |
| NumPy arrays: z_cpu, correlated_returns_cpu      |
| NumPy array : portfolio_sim_results              |
+-------------------------+------------------------+
                          |
                          | cp.array() / transfer host -> device
                          v
+--------------------------------------------------+
| Device / GPU VRAM                                |
|                                                  |
| CuPy arrays: chol_matrix_gpu                     |
| CuPy arrays: weights_gpu                         |
| CuPy arrays: z_gpu                               |
| CuPy arrays: correlated_returns_gpu              |
| CuPy arrays: portfolio_returns_gpu               |
| CuPy arrays: simulated_portfolio_values_gpu      |
+-------------------------+------------------------+
                          |
                          | cp.asnumpy() / transfer device -> host
                          v
+--------------------------------------------------+
| Host / CPU RAM                                   |
| portfolio_losses, VaR, CVaR calculation          |
+--------------------------------------------------+
```

### 4. Akses Legal

Akses legal:

- NumPy hanya dapat memproses array host.
- CuPy memproses array device.
- CPU tidak boleh langsung membaca isi VRAM GPU tanpa transfer.
- GPU tidak langsung memproses `numpy.ndarray`; harus dikonversi ke `cupy.ndarray`.

Contoh transfer host ke device:

```python
chol_matrix_gpu = cp.array(chol_matrix)
weights_gpu = cp.array(WEIGHTS)
```

Contoh data dibuat langsung di device:

```python
z_gpu = cp.random.normal(0, 1, size=(BIG_SIMULATIONS, len(tickers)))
```

Contoh transfer device ke host:

```python
portfolio_sim_results = cp.asnumpy(simulated_portfolio_values_gpu)
```

### 5. Lokasi Komunikasi atau Sinkronisasi Eksplisit

Komunikasi eksplisit host-device:

1. Transfer matriks Cholesky dari CPU ke GPU:

```python
chol_matrix_gpu = cp.array(chol_matrix)
```

2. Transfer bobot portofolio dari CPU ke GPU:

```python
weights_gpu = cp.array(WEIGHTS)
```

3. Transfer hasil simulasi dari GPU ke CPU:

```python
portfolio_sim_results = cp.asnumpy(simulated_portfolio_values_gpu)
```

Sinkronisasi eksplisit:

```python
cp.cuda.Stream.null.synchronize()
```

Fungsi ini penting karena operasi GPU bersifat asynchronous. Tanpa sinkronisasi, pengukuran waktu GPU bisa salah karena CPU mungkin lanjut sebelum GPU selesai.

### 6. Shared Memory pada CUDA

Notebook tidak menggunakan shared memory CUDA secara eksplisit seperti kernel CUDA C:

```c
__shared__ float tile[...];
```

Namun, library seperti cuBLAS yang dipanggil melalui `cp.matmul()` kemungkinan menggunakan optimisasi internal termasuk shared memory GPU, register, cache, dan tiling. Detail tersebut disembunyikan oleh CuPy/cuBLAS.

Jadi batas klasifikasinya:

- Shared memory GPU tidak dikontrol langsung oleh kode proyek.
- Host-device memory digunakan jelas melalui CuPy.
- Distributed memory tidak digunakan dalam CUDA_Project.

---

# 09. Matriks Kompatibilitas Compiler Host, Compiler CUDA, Driver GPU, MPI, dan Arsitektur Target

## Kondisi Aktual Proyek

Pada folder yang dianalisis:

### MPI_Project

Tersedia:

- `weather_anomaly.c`
- `weather_anomaly.exe`
- `.vscode/c_cpp_properties.json`
- Dataset CSV

Namun tidak ditemukan:

- `Makefile`
- `CMakeLists.txt`
- Script otomatis untuk validasi compiler MPI
- Script otomatis untuk validasi runtime MPI

### CUDA_Project

Tersedia:

- `Monte_Carlo_Program.ipynb`
- `simulation_summary.json`
- `monte_carlo_simulation_results.csv`
- `MonteCarloDahboard.html`

Namun tidak ditemukan:

- Source CUDA C/C++ `.cu`
- `Makefile` atau `CMakeLists.txt`
- Script validasi driver GPU
- Script validasi versi CUDA Toolkit
- Script validasi versi CuPy terhadap driver CUDA

Jadi, matriks kompatibilitas belum dibangun secara formal di proyek. Namun, berikut adalah rancangan yang tepat untuk proyek ini.

---

## Matriks Kompatibilitas yang Disarankan

| Komponen | MPI_Project | CUDA_Project | Pemeriksaan |
|---|---|---|---|
| Host compiler | GCC / MSVC / MinGW | Compiler Python extension tidak langsung, NumPy/CuPy binary | `gcc --version`, `cl`, atau environment check |
| MPI implementation | MS-MPI / OpenMPI / MPICH | Tidak digunakan langsung | `mpiexec --version`, compile test `mpi.h` |
| CUDA compiler | Tidak digunakan | Tidak wajib untuk CuPy notebook, wajib jika ada `.cu` | `nvcc --version` jika pakai CUDA C/C++ |
| GPU driver | Tidak digunakan | Wajib untuk CuPy CUDA execution | `nvidia-smi` atau `cp.cuda.runtime.driverGetVersion()` |
| CUDA runtime | Tidak digunakan | Wajib | `cp.cuda.runtime.runtimeGetVersion()` |
| GPU architecture | Tidak digunakan | Harus kompatibel dengan CuPy/CUDA | `device.compute_capability` |
| Target architecture | CPU + MPI process | NVIDIA GPU CUDA-capable | runtime detection |

---

## Untuk MPI_Project

### Kompatibilitas yang Perlu Dicek

MPI_Project membutuhkan:

1. Compiler C yang dapat menemukan `mpi.h`.
2. Linker yang dapat menghubungkan library MPI.
3. Runtime `mpiexec` atau `mpirun`.
4. Konsistensi implementasi MPI antara compile dan run.

Contoh kombinasi valid:

| OS/Environment | Compiler | MPI | Status |
|---|---|---|---|
| Windows | MSVC | MS-MPI | Valid |
| Windows | MinGW GCC | MS-MPI dengan konfigurasi include/lib benar | Bisa valid, perlu hati-hati ABI |
| Linux | GCC | OpenMPI | Valid |
| Linux | GCC | MPICH | Valid |

Potensi masalah:

- Program dikompilasi dengan OpenMPI tetapi dijalankan dengan MPICH.
- Header `mpi.h` berasal dari satu implementasi, library link dari implementasi lain.
- `mpiexec` yang dipakai bukan milik MPI implementation yang sama.

### Pemeriksaan Otomatis MPI

Karena proyek saat ini belum memiliki script, pemeriksaan dapat dibuat sebagai pre-run check.

Contoh script shell konseptual:

```bash
#!/usr/bin/env bash
set -e

command -v mpicc >/dev/null 2>&1 || {
  echo "ERROR: mpicc tidak ditemukan. Install OpenMPI/MPICH atau konfigurasi PATH."
  exit 1
}

command -v mpiexec >/dev/null 2>&1 || command -v mpirun >/dev/null 2>&1 || {
  echo "ERROR: mpiexec/mpirun tidak ditemukan. Runtime MPI belum tersedia."
  exit 1
}

mpicc --showme:version 2>/dev/null || mpicc -v
mpicc weather_anomaly.c -o weather_anomaly -lm || {
  echo "ERROR: build MPI gagal. Periksa compiler, mpi.h, dan library MPI."
  exit 1
}

mpiexec -n 2 ./weather_anomaly || {
  echo "ERROR: eksekusi MPI gagal. Periksa runtime MPI dan dataset_10000_rows.csv."
  exit 1
}
```

Untuk Windows/MS-MPI, konsepnya dapat berupa:

```powershell
if (-not (Get-Command mpiexec -ErrorAction SilentlyContinue)) {
    Write-Error "ERROR: mpiexec tidak ditemukan. Install MS-MPI dan pastikan PATH benar."
    exit 1
}

if (-not (Test-Path "dataset_10000_rows.csv")) {
    Write-Error "ERROR: dataset_10000_rows.csv tidak ditemukan di folder eksekusi."
    exit 1
}

mpiexec -n 2 .\weather_anomaly.exe
if ($LASTEXITCODE -ne 0) {
    Write-Error "ERROR: eksekusi MPI gagal."
    exit $LASTEXITCODE
}
```

### Diagnosis yang Harus Diberikan

Jika tidak didukung, program/build harus berhenti dengan pesan seperti:

```text
ERROR: MPI runtime tidak ditemukan.
Solusi: install MS-MPI/OpenMPI/MPICH dan pastikan mpiexec tersedia di PATH.
```

```text
ERROR: dataset_10000_rows.csv tidak ditemukan.
Solusi: jalankan executable dari folder MPI_Project atau ubah path input dataset.
```

```text
ERROR: compile gagal karena mpi.h tidak ditemukan.
Solusi: gunakan compiler wrapper mpicc atau tambahkan include path MPI.
```

---

## Untuk CUDA_Project

### Kompatibilitas yang Perlu Dicek

CUDA_Project menggunakan Python, NumPy, pandas, yfinance, matplotlib, dan CuPy.

Bagian GPU membutuhkan:

- GPU NVIDIA CUDA-capable.
- Driver NVIDIA aktif.
- CuPy yang sesuai dengan CUDA runtime.
- CUDA runtime kompatibel dengan driver.

Notebook menggunakan:

```python
import cupy as cp
```

Operasi GPU:

```python
cp.random.normal(...)
cp.matmul(...)
cp.exp(...)
cp.cuda.Stream.null.synchronize()
```

Jika CuPy tidak kompatibel dengan driver CUDA, bagian ini gagal.

### Pemeriksaan Otomatis CUDA/CuPy

Pemeriksaan yang bagus untuk proyek ini sebaiknya dilakukan di awal notebook sebelum benchmark.

Contoh kode Python:

```python
import sys
import numpy as np

try:
    import cupy as cp
except ImportError:
    raise RuntimeError(
        "CuPy tidak terinstal. Install CuPy yang sesuai dengan versi CUDA/driver GPU."
    )

try:
    device_count = cp.cuda.runtime.getDeviceCount()
except cp.cuda.runtime.CUDARuntimeError as e:
    raise RuntimeError(
        f"CUDA runtime tidak dapat mengakses GPU. Detail: {e}"
    )

if device_count < 1:
    raise RuntimeError("Tidak ada GPU CUDA yang terlihat oleh proses ini.")

runtime_version = cp.cuda.runtime.runtimeGetVersion()
driver_version = cp.cuda.runtime.driverGetVersion()

if driver_version < runtime_version:
    raise RuntimeError(
        f"Driver CUDA terlalu lama. Driver={driver_version}, Runtime={runtime_version}."
    )

dev = cp.cuda.Device(0)
props = cp.cuda.runtime.getDeviceProperties(0)
major = props["major"]
minor = props["minor"]
name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]

if major < 6:
    raise RuntimeError(
        f"Compute capability GPU terlalu lama: {major}.{minor}. GPU={name}."
    )

print("CUDA environment valid")
print(f"GPU: {name}")
print(f"Compute capability: {major}.{minor}")
print(f"CUDA runtime version: {runtime_version}")
print(f"CUDA driver version: {driver_version}")
```

Catatan:

- Batas `major < 6` adalah contoh konservatif untuk workload modern.
- Jika hanya operasi CuPy dasar, GPU dengan compute capability lebih lama mungkin masih bisa, tergantung paket CuPy.
- Pemeriksaan aktual harus disesuaikan dengan versi CuPy yang dipakai.

### Jika Menggunakan CUDA C/C++ di Masa Depan

Jika proyek CUDA dikembangkan menjadi `.cu`, perlu dicek:

- `nvcc --version`
- host compiler yang didukung oleh NVCC
- target architecture seperti `sm_75`, `sm_80`, `sm_86`, `sm_89`

Contoh CMake check konseptual:

```cmake
enable_language(CUDA)

if (CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 11.0)
    message(FATAL_ERROR "CUDA Toolkit minimal 11.0 diperlukan")
endif()

set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
```

Namun pada proyek saat ini, karena menggunakan CuPy, `nvcc` tidak wajib untuk menjalankan notebook. Yang lebih penting adalah kompatibilitas CuPy, CUDA runtime, dan driver GPU.

---

# 14. Validasi Konfigurasi Block, Grid, Shared Memory, dan Fitur GPU

Sebelum kernel diluncurkan, program membaca properti GPU lalu memvalidasi konfigurasi:

1. Periksa `blockDim.x * blockDim.y * blockDim.z` terhadap `maxThreadsPerBlock`.
2. Periksa setiap dimensi `gridDim` terhadap `maxGridSize`.
3. Hitung kebutuhan shared memory dan bandingkan dengan `sharedMemPerBlock`.
4. Periksa compute capability dan fitur CUDA yang diperlukan kernel.
5. Jika semua valid, kernel diluncurkan; jika tidak, program menghentikan eksekusi.

Contoh pesan kesalahan:

```text
ERROR: jumlah thread per block (1024) melebihi batas GPU (512).
Parameter pelanggar: blockDim=(32,32,1).
```

```text
ERROR: ukuran grid pada dimensi X (200000) melebihi batas GPU (2147483647).
Parameter pelanggar: gridDim=(200000,1,1).
```

```text
ERROR: shared memory yang diminta (49152 byte) melebihi batas per block (32768 byte).
```

```text
ERROR: compute capability GPU 5.0 tidak memenuhi kebutuhan kernel 7.0.
```

Jika validasi gagal, kernel tidak diluncurkan, pesan dicetak oleh rank/proses terkait, lalu program keluar dengan kode error. Jika menggunakan MPI, semua rank menyebarkan status validasi melalui `MPI_Allreduce`; apabila satu rank gagal, seluruh rank berhenti menggunakan `MPI_Abort` agar tidak ada proses yang menunggu.

---

# 12. Verifikasi Perangkat GPU yang Terlihat oleh Setiap Proses saat Scheduler atau Container Membatasi Akses

## Konteks Pertanyaan

Dalam sistem cluster, container, atau scheduler seperti SLURM, PBS, Kubernetes, atau Docker, GPU yang terlihat oleh proses dapat dibatasi.

Contoh:

```bash
CUDA_VISIBLE_DEVICES=2,3 python main.py
```

Di dalam proses, GPU fisik 2 dan 3 bisa muncul sebagai ordinal lokal 0 dan 1. Karena itu, program tidak boleh mengasumsikan bahwa ordinal GPU bersifat global dan seragam.

---

## Kondisi Aktual Proyek

### MPI_Project

MPI_Project tidak menggunakan GPU, sehingga tidak ada verifikasi GPU.

### CUDA_Project

CUDA_Project menggunakan GPU melalui CuPy, tetapi notebook saat ini belum membuat inventaris eksplisit seperti:

- rank
- hostname
- ordinal lokal GPU
- nama GPU
- UUID GPU
- compute capability
- memory total

Notebook hanya memakai default device CuPy, biasanya device 0 yang terlihat oleh proses.

Karena itu, untuk menjawab kebutuhan pertanyaan, perlu ditambahkan rancangan pemeriksaan/inventaris GPU.

---

## Untuk CUDA_Project Single Process

Untuk notebook CUDA saat ini, verifikasi minimal yang bagus adalah:

```python
import socket
import cupy as cp

hostname = socket.gethostname()
count = cp.cuda.runtime.getDeviceCount()

print(f"Hostname: {hostname}")
print(f"Visible CUDA devices: {count}")

for local_ordinal in range(count):
    props = cp.cuda.runtime.getDeviceProperties(local_ordinal)
    name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    major = props["major"]
    minor = props["minor"]
    total_mem = props["totalGlobalMem"]
    pci_bus_id = cp.cuda.runtime.deviceGetPCIBusId(local_ordinal)

    print({
        "hostname": hostname,
        "local_ordinal": local_ordinal,
        "gpu_name": name,
        "compute_capability": f"{major}.{minor}",
        "total_memory_bytes": total_mem,
        "pci_bus_id": pci_bus_id,
    })
```

Tujuannya:

- Mengetahui GPU apa saja yang terlihat oleh proses Python.
- Tidak mengasumsikan device global.
- Menggunakan `local_ordinal`, yaitu ordinal setelah filtering oleh scheduler/container.

---

## Untuk MPI + CUDA / Hybrid

Jika CUDA_Project dikembangkan menjadi hybrid MPI+CUDA, setiap rank perlu membuat inventaris sendiri.

Informasi yang harus dicatat:

| Field | Fungsi |
|---|---|
| `rank` | Identitas proses MPI global |
| `local_rank` | Rank lokal pada node yang sama |
| `hostname` | Nama node |
| `visible_device_count` | Jumlah GPU yang terlihat oleh proses |
| `selected_local_ordinal` | GPU lokal yang dipakai rank |
| `gpu_name` | Nama GPU |
| `pci_bus_id` | Identitas lokasi PCI |
| `uuid` | Identitas unik GPU jika tersedia |
| `compute_capability` | Kemampuan arsitektur GPU |
| `total_memory` | Kapasitas VRAM |

### Mengapa Tidak Boleh Mengasumsikan Ordinal Global

Misal scheduler memberi:

```text
Node A punya GPU fisik: 0,1,2,3
Scheduler memberi rank akses ke GPU fisik 2
CUDA_VISIBLE_DEVICES=2
```

Di dalam proses:

```text
GPU fisik 2 terlihat sebagai cuda:0
```

Jadi ordinal lokal `0` bukan berarti GPU fisik global `0`.

Karena itu, inventaris harus menyimpan identitas lain seperti PCI bus ID atau UUID.

---

## Contoh Inventaris Rank, Hostname, dan GPU dengan mpi4py + CuPy

Untuk Python hybrid MPI+CUDA, contoh pendekatan:

```python
from mpi4py import MPI
import socket
import os
import cupy as cp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
hostname = socket.gethostname()

visible_count = cp.cuda.runtime.getDeviceCount()

if visible_count == 0:
    raise RuntimeError(f"Rank {rank} pada {hostname}: tidak ada GPU terlihat")

# Buat local rank berdasarkan hostname
all_hosts = comm.allgather(hostname)
local_rank = sum(1 for i in range(rank) if all_hosts[i] == hostname)

selected = local_rank % visible_count
cp.cuda.Device(selected).use()

props = cp.cuda.runtime.getDeviceProperties(selected)
name = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
major = props["major"]
minor = props["minor"]
total_mem = props["totalGlobalMem"]
pci_bus_id = cp.cuda.runtime.deviceGetPCIBusId(selected)

record = {
    "rank": rank,
    "size": size,
    "hostname": hostname,
    "local_rank": local_rank,
    "visible_device_count": visible_count,
    "selected_local_ordinal": selected,
    "gpu_name": name,
    "compute_capability": f"{major}.{minor}",
    "total_memory_bytes": total_mem,
    "pci_bus_id": pci_bus_id,
    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
}

records = comm.gather(record, root=0)

if rank == 0:
    for r in records:
        print(r)
```

Penjelasan:

- `rank` adalah rank global MPI.
- `hostname` dipakai untuk mengetahui proses mana berada pada node yang sama.
- `local_rank` dihitung dari jumlah rank sebelumnya yang memiliki hostname sama.
- `selected = local_rank % visible_count` memilih GPU berdasarkan ordinal lokal yang terlihat oleh proses.
- `CUDA_VISIBLE_DEVICES` dicatat untuk diagnosis.
- `pci_bus_id` dipakai sebagai identitas perangkat yang lebih stabil daripada ordinal lokal.

---

## Contoh Output Inventaris

Contoh output yang diharapkan:

```text
{
  'rank': 0,
  'size': 4,
  'hostname': 'node-a',
  'local_rank': 0,
  'visible_device_count': 2,
  'selected_local_ordinal': 0,
  'gpu_name': 'NVIDIA Tesla T4',
  'compute_capability': '7.5',
  'total_memory_bytes': 15835660288,
  'pci_bus_id': '0000:00:04.0',
  'CUDA_VISIBLE_DEVICES': '2,3'
}
{
  'rank': 1,
  'size': 4,
  'hostname': 'node-a',
  'local_rank': 1,
  'visible_device_count': 2,
  'selected_local_ordinal': 1,
  'gpu_name': 'NVIDIA Tesla T4',
  'compute_capability': '7.5',
  'total_memory_bytes': 15835660288,
  'pci_bus_id': '0000:00:05.0',
  'CUDA_VISIBLE_DEVICES': '2,3'
}
```

Dari output tersebut, terlihat bahwa ordinal lokal 0 dan 1 adalah GPU yang terlihat oleh proses, bukan selalu ordinal global fisik.

---

## Integrasi dengan CUDA_Project Saat Ini

Pada notebook saat ini, bagian berikut:

```python
chol_matrix_gpu = cp.array(chol_matrix)
weights_gpu = cp.array(WEIGHTS)
```

akan memakai device default CuPy. Agar lebih aman, sebelum alokasi GPU sebaiknya dipilih device eksplisit:

```python
device_id = 0
cp.cuda.Device(device_id).use()
```

Namun jika dijalankan di scheduler/container, `device_id = 0` berarti device ordinal lokal yang terlihat oleh proses, bukan GPU global 0. Maka inventaris perlu dicetak sebelum benchmark.

---

## Diagnosis Jika GPU Tidak Sesuai

Program sebaiknya berhenti jika:

1. Tidak ada GPU terlihat:

```text
ERROR: Rank X pada host Y tidak melihat GPU CUDA apa pun.
Periksa CUDA_VISIBLE_DEVICES, konfigurasi scheduler, atau konfigurasi container.
```

2. Jumlah rank lokal melebihi jumlah GPU dan oversubscription tidak diizinkan:

```text
ERROR: Host node-a memiliki 4 rank lokal tetapi hanya 2 GPU terlihat.
Solusi: kurangi rank per node atau izinkan sharing GPU secara eksplisit.
```

3. Compute capability tidak memenuhi syarat:

```text
ERROR: GPU pada rank X memiliki compute capability 5.0, minimal yang dibutuhkan 6.0.
```

4. Driver/runtime tidak cocok:

```text
ERROR: CUDA driver lebih lama dari runtime CuPy yang digunakan.
```

---

# Kesimpulan Umum

1. `MPI_Project` paling tepat diklasifikasikan sebagai **MIMD berbasis distributed memory**, karena menggunakan banyak proses MPI dengan ruang alamat terpisah dan komunikasi eksplisit melalui `MPI_Bcast`, `MPI_Scatterv`, dan `MPI_Reduce`.

2. `CUDA_Project` paling tepat diklasifikasikan sebagai **SIMT berbasis GPU CUDA**, karena operasi CuPy seperti `cp.random.normal`, `cp.matmul`, dan `cp.exp` dijalankan oleh banyak thread GPU melalui CUDA/cuBLAS/cuRAND.

3. SIMD tidak ditulis eksplisit pada kedua proyek, tetapi dapat muncul sebagai optimisasi internal pada CPU melalui compiler atau library numerik seperti NumPy/BLAS.

4. Proyek saat ini belum memiliki build system formal atau compatibility matrix otomatis. MPI_Project hanya memiliki source C dan executable, sedangkan CUDA_Project berbentuk notebook. Karena itu, pemeriksaan compiler/runtime sebaiknya ditambahkan sebagai pre-build atau pre-run validation.

5. Untuk GPU verification, CUDA_Project sebaiknya menambahkan inventaris device menggunakan CuPy agar tidak mengasumsikan ordinal GPU global, terutama jika dijalankan di scheduler, container, atau lingkungan multi-GPU.
