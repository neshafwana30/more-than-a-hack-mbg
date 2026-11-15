# Project Brief — Hackathon Barça: Context, Research Direction & Collaboration

## 1. Project Objective

Project ini merupakan solusi untuk challenge *More Than A Hack 2027 — FC Barcelona Possession Detection*.

Tujuan utamanya adalah menghasilkan satu label untuk setiap frame pertandingan Barcelona vs Athletic Club:

- `H`: Barcelona/home menguasai bola saat permainan berlangsung.
- `A`: Athletic Club/away menguasai bola saat permainan berlangsung.
- `D`: Bola tidak dalam permainan (*dead ball*).
- `X`: Abstention atau tidak memberikan prediksi.

Pertandingan terdiri dari **199.319 frame pada 25 FPS**, dengan **158.857 graded frames**.

Metrik utama kompetisi adalah **spellF1**, sedangkan macroF1 merupakan metrik sekunder.

Perbedaan keduanya sangat penting:

- **macroF1** mengevaluasi kesesuaian label pada setiap frame.
- **spellF1** mengevaluasi kualitas interval possession menggunakan pencocokan temporal IoU pada beberapa threshold.
- Prediksi dengan akurasi frame tinggi masih bisa menghasilkan spellF1 rendah apabila possession terlalu terfragmentasi, batasnya tidak tepat, atau stoppage tidak terdeteksi.

Dengan demikian, objective sistem bukan sekadar menghasilkan klasifikasi frame yang tampak masuk akal, tetapi membangun **temporal possession segmentation** yang akurat.

## 2. Competition Constraints

Baca dan pahami dokumentasi challenge sebelum mengusulkan perubahan metodologi:

- `CHALLENGE.md`
- `DATA.md`
- `EVALUATION.md`
- `RULES.md`
- `FAQ.md`
- `SUBMISSION.md`

Beberapa aturan penting:

1. Label harus dihasilkan oleh algoritma. Tidak boleh melakukan hand-labeling maupun hand-correction pada pertandingan yang disediakan.
2. Public ground truth lima menit hanya boleh digunakan untuk memeriksa kebenaran implementasi, alignment, dan format. **Tidak boleh digunakan untuk training, tuning, atau memilih model berdasarkan skor.**
3. Ground truth pertandingan penuh bersifat private.
4. Data pertandingan yang disediakan panitia tidak boleh didistribusikan atau diunggah sembarangan ke layanan pihak ketiga. Patuhi juga larangan Terms and Conditions terkait cloud-hosted LLM dan data turunannya.
5. Submission harus reproducible dari repository dan data yang diberikan.

Jangan mengasumsikan bahwa suatu segment salah hanya karena secara visual terlihat mencurigakan. Diagnosis dan debugging diperbolehkan; mengubah observasi manual menjadi label atau aturan khusus frame pertandingan tidak diperbolehkan.

## 3. Current System Architecture

Pipeline di `src/` merupakan sumber utama seluruh perhitungan.

Secara konseptual, pipeline terdiri dari:

**Data → Features → Quality & Evidence → Fusion → Temporal Decoding → Boundary Refinement → Submission**

Komponen yang sudah ada meliputi:

- Tracking dan feature extraction
- Observation quality
- Event candidates dan Smart Tagging
- Team control evidence
- Live/dead evidence
- Fusion utilities
- Temporal decoder
- Ambiguity detection
- Optional video-based recovery
- Boundary refinement
- Submission generation
- Artifact persistence dan diagnostics

Pahami implementasi yang benar-benar ada di repository. Jangan menganggap diagram konseptual tersebut sebagai spesifikasi lengkap.

**Penting: skor model adalah additive utilities, bukan probabilitas terkalibrasi.** Jangan menafsirkan skor 0,8 sebagai confidence 80%.

Konfigurasi dipisahkan menjadi:

- `config/v2.yaml`: konfigurasi model dan masalah.
- `config/artifacts.yaml`: konfigurasi penyimpanan artefak.

Setiap pipeline run menghasilkan folder unik:

`artifacts/<run_id>/`

Folder tersebut menyimpan manifest, snapshot konfigurasi, fitur, intermediate evidence, skor, hasil decoder, final labels, serta diagnostik.

Gunakan `manifest.final_core_pass` untuk mengetahui hasil core yang dipakai pada submission final. Jangan selalu berasumsi bahwa hasil akhir berasal dari `core/initial`.

Untuk membaca artefak, gunakan utilitas repository seperti `src.artifacts.read_table()` dan `read_json()` bila tersedia dan sesuai dengan implementasi aktual.

Jangan menggabungkan tabel berdasarkan posisi baris tanpa memvalidasi indeks `frame`.

## 4. Current Baseline and Known Findings

Pipeline telah menghasilkan `baseline_v2.csv`, dengan run yang telah selesai:

`20261115_152745_650609_05d95801`

Run ini menghasilkan submission valid dengan 199.319 baris.

Video recovery dinonaktifkan, sehingga prediksi berasal dari core initial dan boundary refinement.

Temuan berdasarkan eksplorasi sebelumnya:

### A. Missing-ball observations

- Ball coverage pada graded frames sekitar 59,31%.
- Sekitar 40,69% graded frames tidak memiliki observasi bola.
- Terdapat 150 missing-ball gaps.
- Median gap duration sekitar 10,72 detik.
- P95 gap duration sekitar 50,80 detik.
- Maximum gap duration sekitar 221,48 detik.

Missing ball tidak berarti dead ball.

Kita perlu memahami bagaimana kehilangan observasi memengaruhi team control, live/dead scoring, dan state transitions.

### B. Dead-ball detection

Pada 155.923 graded frames di luar interval forced D:

- Dead utility tidak pernah menang terhadap H/A.
- Dead utility tidak pernah seri dengan utility tertinggi H/A.
- Maximum dead advantage sekitar −0,002716.

Prediksi final hanya menghasilkan satu dead-ball segment sepanjang 2.934 frame, yaitu interval yang dipaksakan menjadi D pada awal babak kedua.

Ini adalah **indikasi kuat adanya masalah dalam natural dead-ball detection**, bukan bukti final mengenai penyebabnya.

Kemungkinan penyebab mencakup live/dead evidence, skala fusion, prior utility, atau komponen lainnya.

Jangan langsung memperbaikinya dengan menaikkan bias D sebelum melakukan score decomposition.

### C. Possession fragmentation

Prediksi final menghasilkan 1.070 possession spells:

- 535 H segments.
- 535 A segments.
- 29,0% predicted possession berlangsung kurang dari satu detik.
- 48,9% berlangsung kurang dari dua detik.
- Terdapat 1.069 state transitions, hampir seluruhnya H ↔ A.

Ini mengindikasikan potensi fragmentasi possession yang tinggi.

Namun, possession singkat tidak otomatis salah. Jangan menetapkan minimum possession duration hanya untuk membuat distribusinya terlihat realistis.

Kita harus membedakan apakah fragmentasi berasal dari:

1. Evidence geometris yang tidak stabil.
2. Team control score yang terlalu sensitif.
3. Kualitas tracking yang buruk.
4. Temporal decoder yang terlalu mudah berganti state.
5. Interaksi beberapa penyebab tersebut.

### D. Boundary refinement

Pada run tersebut:

- 1.069 boundaries diperiksa.
- 115 boundaries mengalami pergeseran.
- Total transitions tidak berubah setelah refinement.

Boundary refinement terutama menggeser posisi transisi, bukan mengurangi jumlah possession.

Jangan menganggap refinement sebagai solusi terhadap seluruh masalah fragmentasi.

**Seluruh temuan di atas merupakan diagnosis perilaku prediksi, bukan pengukuran error terhadap private ground truth.**

## 5. Previous Exploratory Work

Empat notebook sebelumnya sudah tersedia:

- `notebooks/01_data_audit.ipynb`
- `notebooks/02_feature_exploration.ipynb`
- `notebooks/03_baseline_possession.ipynb`
- `notebooks/04_error_analysis.ipynb`

Fungsinya:

**Notebook 01:** Audit struktur dataset, tracking coverage, missing-ball gaps, dan Smart Tagging.

**Notebook 02:** Eksplorasi fitur, observation quality, control evidence, live/dead scores, dan fusion utilities.

**Notebook 03:** Menjalankan pipeline, menghasilkan baseline, dan memeriksa validitas submission.

**Notebook 04:** Analisis output, possession segments, state transitions, boundary refinement, dan anomali prediksi.

Jangan mengulang eksplorasi ini tanpa alasan yang jelas.

Notebook baru telah disiapkan:

`notebooks/05_root_cause_diagnostics.ipynb`

Notebook 05 merupakan tempat utama untuk investigasi lanjutan.

Seluruh investigasi berikutnya sebaiknya dikerjakan di notebook ini dengan menambahkan cell, bukan terus membuat notebook baru.

Gunakan heading Markdown yang bisa di-collapse untuk memisahkan investigasi.

## 6. Current Research Direction

Kita sekarang berada pada tahap **Root-Cause Diagnostics**, bukan langsung Model Optimization.

Tujuan eksplorasi berikutnya adalah menjawab:

> Apa yang menyebabkan kegagalan atau ketidakstabilan prediksi, pada kondisi seperti apa masalah itu muncul, dan bagian pipeline mana yang bertanggung jawab?

Ada enam investigasi yang direncanakan.

### Investigasi 1 — Dead-Ball Score Diagnostics

Pertanyaan utama:

**Mengapa state D tidak pernah memiliki utility tertinggi di luar forced interval?**

Analisis:

- Distribusi dead advantage terhadap H/A.
- Live/dead raw evidence vs fusion utility.
- Pengaruh stoppage/restart anchors.
- Pengaruh ball motion dan player motion.
- Pengaruh observation quality.
- Perbedaan kondisi saat D mendekati H/A dan saat D sangat lemah.

Objective: Mengisolasi apakah kelemahan berada pada evidence generation atau fusion scoring, sebelum mengubah model.

### Investigasi 2 — H/A Transition Diagnostics

Pertanyaan utama:

**Mengapa model menghasilkan banyak possession pendek dan pergantian H/A?**

Analisis:

- Raw preference switches vs decoder switches.
- H/A score margins di sekitar transitions.
- Switch-back patterns seperti H → A → H.
- Durasi persistensi evidence sebelum perubahan state.
- Transitions ketika ball tracking missing.
- Distribusi durasi dan kualitas evidence pada short spells.

Objective: Membedakan instabilitas control evidence dari keputusan temporal decoder.

### Investigasi 3 — Observability and Missing-Ball Diagnostics

Pertanyaan utama:

**Bagaimana kualitas observasi memengaruhi stabilitas prediksi?**

Analisis:

- Ball observed vs missing.
- Reliable vs unreliable motion.
- Short vs long missing-ball gaps.
- Player tracking quality.
- Transition rate berdasarkan kondisi observability.
- State stability pada bagian awal, tengah, dan akhir gap.

Gunakan ukuran yang dinormalisasi, misalnya transitions per minute atau per 1.000 frames, bukan hanya raw count.

Objective: Memahami apakah dan bagaimana missing data berkontribusi terhadap instabilitas model.

### Investigasi 4 — Smart Tagging and Event Evidence

Pertanyaan utama:

**Sejauh mana Smart Tagging memberikan evidence yang berguna?**

Analisis:

- Konsistensi team tags dengan control evidence.
- Stoppage dan restart anchors.
- Pengaruh set-piece context terhadap D.
- Perbedaan event intervals dengan state boundaries.
- Coverage dan potensi konflik antar-evidence.

Smart Tagging adalah prediksi otomatis, bukan ground truth.

Objective: Mengukur apakah event evidence berkontribusi pada pengambilan keputusan atau hanya hadir sebagai fitur yang pengaruhnya lemah.

### Investigasi 5 — Decoder and Boundary Refinement

Pertanyaan utama:

**Apakah temporal inference menstabilkan prediksi atau justru mempertahankan fragmentasi?**

Analisis:

- Frame-wise utility argmax vs decoded labels.
- Transition penalties dan transition hints.
- Short-spell behavior.
- Perbandingan decoder output dengan final refined labels.
- Distribusi boundary shifts.
- Konsistensi perubahan state dengan evidence lokal.

Objective: Memisahkan kelemahan observation/fusion layer dari temporal decoding.

### Investigasi 6 — Controlled Experiments

Dilakukan setelah investigasi sebelumnya menghasilkan hipotesis yang cukup kuat.

Contoh:

- Fusion balance experiment.
- Dead-ball evidence refinement.
- Control stabilization.
- Transition-cost adjustment.
- Quality-aware inference.

Setiap eksperimen perlu memiliki:

1. Observasi awal.
2. Hipotesis.
3. Komponen atau parameter yang diubah.
4. Ekspektasi efek perubahan.
5. Hasil run.
6. Perbandingan terhadap baseline.
7. Interpretasi.
8. Keterbatasan kesimpulan.

Jangan mengubah banyak variabel sekaligus apabila tujuan eksperimennya adalah mengidentifikasi penyebab.

## 7. Engineering and Notebook Conventions

Prinsip utama:

**`src/` adalah sumber perhitungan; `notebooks/` adalah tempat eksplorasi dan interpretasi.**

Namun, perlu dibedakan antara dua jenis perhitungan:

**Reusable pipeline computations**

Perhitungan yang mengubah fitur, evidence, scoring, decoder, atau prediction harus berada di `src/`.

**Exploratory diagnostics**

Agregasi, visualisasi, korelasi, pivot, distribusi, dan analisis sementara boleh dilakukan di notebook.

Jika diagnostik baru akan dipakai terus-menerus, pertimbangkan memindahkannya ke modul diagnostik di `src/`, kemudian panggil dari notebook.

Hindari menduplikasi implementasi model di notebook.

Semua visualisasi, tabel, dan interpretasi harus bisa dilihat dalam output notebook, bukan hanya melalui hasil terminal.

Jangan menghasilkan analisis terpisah yang tidak terhubung ke notebook.

Gunakan pola:

**Question → Data → Analysis → Visualization → Interpretation → Finding → Next Question**

Setiap investigasi sebaiknya memiliki heading, tujuan, sumber data, kode analisis, hasil visual, dan Markdown interpretasi.

Jangan menyebut suatu temuan sebagai confirmed error jika belum dibandingkan dengan ground truth yang sah.

Gunakan istilah seperti *observed behavior*, *anomaly*, *hypothesis*, dan *potential failure mode*.

## 8. Collaboration Protocol

Cara kita bekerja harus transparan, incremental, dan bisa diikuti langsung.

Untuk setiap langkah:

1. **Inspect:** Baca implementasi dan artefak yang relevan terlebih dahulu.
2. **Plan:** Jelaskan pertanyaan yang akan dijawab, data yang dibutuhkan, dan metode analisis.
3. **Implement:** Tambahkan analisis di notebook yang sesuai. Jangan mengubah logic model jika tugasnya hanya diagnosis.
4. **Execute:** Jalankan cell atau notebook pada lingkungan lokal yang tersedia.
5. **Verify:** Pastikan hasil valid, tidak ada schema mismatch, join error, atau kekeliruan temporal indexing.
6. **Interpret:** Jelaskan hasil, penyebab potensial, dan ketidakpastian.
7. **Decide:** Tentukan apakah evidence cukup untuk melakukan eksperimen atau perlu investigasi tambahan.

Jangan melakukan refactor besar, menambah model baru, atau mengubah banyak komponen tanpa alasan yang berasal dari diagnosis.

Jangan menganggap setiap permintaan eksplorasi sebagai izin otomatis untuk mengubah pipeline.

Jika ada ketidaksesuaian antara dokumentasi dalam brief ini dan kode repository, **periksa implementasi aktual serta manifest run**, lalu jelaskan ketidaksesuaian tersebut.

Folder `tools/` tidak menjadi bagian dari pekerjaan ini kecuali diminta secara eksplisit.

## 9. Immediate Next Task

Langkah berikutnya adalah memeriksa isi dan kesiapan:

`notebooks/05_root_cause_diagnostics.ipynb`

Kemudian mulai dengan:

**Investigasi 1 — Dead-Ball Score Diagnostics.**

Sebelum melakukan perubahan model:

- Periksa implementasi live/dead scoring dan fusion di `src/`.
- Baca config efektif dari canonical run.
- Periksa artefak live/dead, event, quality, dan fusion.
- Identifikasi komponen yang membuat utility D kalah dari H/A.
- Bangun visualisasi dan diagnostic tables pada notebook 05.
- Jelaskan temuan beserta hipotesis mekanismenya.

Jangan langsung mengubah dead bias atau transition penalties.

Tujuan tahap ini adalah menghasilkan **pemahaman kausal yang cukup kuat untuk merancang eksperimen**, bukan sekadar membuat output terlihat lebih baik.

---

## 10. Guiding Principle

Project ini tidak lagi berada pada fase sekadar membangun algoritma yang dapat menghasilkan submission.

Kita sedang berusaha memahami dan memperbaiki sistem secara ilmiah.

Prinsip utama:

**Understand the failure mechanism before optimizing the model.**

Setiap perubahan sebaiknya mempunyai alasan yang dapat diuji, didukung evidence, dan dapat direproduksi.

Prioritaskan peningkatan kualitas penalaran dan validitas eksperimen, bukan hanya kompleksitas algoritma atau banyaknya visualisasi.