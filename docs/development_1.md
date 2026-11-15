# Technical Discussion for Codex — Baseline v3 Development Strategy

## Context

Kita telah menyelesaikan root-cause diagnostics untuk `baseline_v2` melalui `notebooks/05_root_cause_diagnostics.ipynb`.

Berdasarkan hasil diagnosis, ada beberapa hipotesis mengenai kelemahan arsitektur model saat ini. Aku ingin kamu mengevaluasi hipotesis tersebut secara kritis sebelum melakukan perubahan.

**Jangan langsung mengimplementasikan seluruh rekomendasi.** Periksa kode aktual, konfigurasi, data, dan artefak terlebih dahulu. Jika menemukan pendekatan yang lebih baik, jelaskan argumentasinya.

Tujuan diskusi ini adalah menentukan strategi pengembangan baseline v3 yang memiliki dasar teknis kuat, tetap sederhana jika memungkinkan, dan sesuai aturan kompetisi.

---

## 1. Diagnosis Utama

### A. Dead-ball evidence kalah dalam fusion

Hasil notebook 05 menunjukkan:

- Raw dead evidence mengalahkan raw live evidence pada 169 frame non-forced.
- Setelah fusion dengan team-control evidence, D tidak pernah mengalahkan H/A pada frame non-forced.
- Ada tambahan utility kontrol tim yang membuat H/A lebih kompetitif daripada D.
- Interval raw D yang unggul cenderung sangat pendek, dengan median sekitar 0,12 detik.

**Hipotesis:**

Fusion saat ini mungkin mencampurkan dua konsep yang seharusnya dibedakan:

1. Apakah permainan berlangsung (Live/Dead)?
2. Tim mana yang menguasai bola jika permainan berlangsung (H/A)?

Team-control evidence yang tinggi tidak selalu berarti permainan sedang berlangsung. Pada throw-in, corner, free kick, atau situasi stoppage, pemain tetap bisa berada dekat dengan bola.

Karena itu, kedekatan dan kontrol geometris tidak seharusnya secara otomatis menenggelamkan dead-ball evidence.

### B. Fragmentasi possession bukan semata masalah decoder

Diagnosis menunjukkan:

- Raw frame-wise argmax menghasilkan sekitar 1.708 transitions.
- Decoder menguranginya menjadi 1.069 transitions.
- Sekitar 48,9% predicted possession berlangsung kurang dari dua detik.
- Analisis local net gain pada short reversals tidak menemukan kandidat dengan keuntungan negatif berdasarkan objective yang diaudit.

**Hipotesis:**

Decoder sudah melakukan stabilisasi, tetapi emission utilities masih terlalu sering mendukung pergantian H/A.

Menaikkan transition cost atau melakukan smoothing agresif mungkin mengurangi fragmentasi, tetapi juga berisiko menghapus genuine short possessions.

Karena itu, prioritaskan pemeriksaan pembentukan control evidence sebelum hanya mengubah smoothing.

### C. Pergantian H/A banyak terjadi ketika bola terlihat

Sebagian besar transitions terjadi pada frame ketika bola terobservasi.

Hal ini menarik karena awalnya missing-ball diduga menjadi sumber utama flickering.

**Hipotesis:**

Model mungkin terlalu sensitif terhadap nearest-player geometry atau perubahan dukungan kontrol sesaat.

Tetapi ini belum membuktikan hubungan sebab-akibat. Perlu diperiksa apakah tingkat aktivitas permainan dan opportunity for turnover berbeda antara kondisi bola terlihat dan hilang.

---

## 2. Proposal A — Redesign Live/Dead and Possession Fusion

Ini adalah prioritas eksperimen pertamaku.

Saat ini, team-control evidence kemungkinan memberikan tambahan positif pada utility H/A, sehingga D harus mengalahkan bukan hanya live evidence, tetapi juga tambahan kontrol tim.

Aku mengusulkan pemisahan konseptual:

**Game-state evidence**

Menentukan apakah permainan Live atau Dead.

**Conditional possession evidence**

Menentukan apakah H atau A lebih mungkin menguasai bola ketika permainan Live.

Ada dua opsi arsitektur.

### Option A — Hierarchical Two-Stage Decoder

1. Decode Live/Dead.
2. Pada interval Live, decode H/A.

Keuntungan:

- D tidak perlu bersaing langsung dengan tambahan utility team control.
- Interpretabilitas tinggi.
- Masing-masing modul dapat didiagnosis secara terpisah.

Risiko:

- Kesalahan Live/Dead dapat menyebar ke seluruh hasil possession.
- Dua tahap decoding mungkin membutuhkan penanganan boundary yang cermat.

### Option B — Joint Three-State Decoder with Factorized Evidence

Pertahankan Viterbi tiga state (H/A/D), tetapi bangun fusion yang memisahkan:

- Live/Dead preference.
- Relative H/A possession preference.

Sebagai contoh konseptual:

\[
U_H = L + C_H^*
\]

\[
U_A = L + C_A^*
\]

\[
U_D = D
\]

Di sini \(C_H^*\) dan \(C_A^*\) merupakan relative control evidence yang telah dinormalisasi atau dipusatkan, bukan tambahan kontrol positif absolut yang otomatis menguntungkan Live.

Kita juga bisa menggunakan score decomposition atau log-odds representation lain, tergantung struktur aktual implementasi.

**Preferensiku: Option B lebih dahulu.**

Alasannya:

- Lebih sedikit perubahan arsitektur.
- Decoder yang ada dapat dipertahankan.
- Memungkinkan eksperimen terisolasi terhadap fusion.
- Mengurangi risiko error propagation dari hard two-stage decoding.

Namun, jangan menganggap rumus di atas sebagai implementasi final. Pertimbangkan normalisasi, skala utility, dan invariansi terhadap penambahan common offset pada H/A.

### Yang perlu kamu investigasi

1. Bagaimana `live_score`, `dead_score`, dan team-control scores digabungkan sekarang?
2. Apakah team-control evidence menambah utility Live secara sistematis?
3. Apakah representasi relative H/A dapat menghilangkan keunggulan artifisial tersebut?
4. Setelah perbaikan fusion, apakah D cukup kuat secara temporal untuk melewati transition costs?
5. Apakah diperlukan perubahan pada model live/dead selain fusion?

Lakukan eksperimen fusion sebagai satu perubahan terisolasi, bukan bersamaan dengan perubahan decoder atau control features.

---

## 3. Proposal B — Controlled Possession Acquisition Evidence

Aturan challenge membedakan proximity, touch, dan controlled possession.

Model seharusnya tidak berpindah possession hanya karena pemain lawan sesaat lebih dekat ke bola.

Aku mengusulkan agar kita mengevaluasi representasi evidence yang lebih eksplisit tentang **control acquisition**.

Konsepnya:

- Proximity menunjukkan peluang berinteraksi dengan bola.
- Persistence menunjukkan konsistensi evidence.
- Ball-motion consistency dapat memberikan petunjuk apakah bola sedang dikendalikan.
- Control acquisition menunjukkan bahwa tim penantang benar-benar memperoleh kontrol baru.

Kemungkinan fitur:

- Persistence of challenger-team proximity.
- Relative nearest-player distance.
- Reliable ball-motion evidence.
- Stable ball-player spatial relationship.
- Consistency of team evidence over a temporal window.
- Observation-quality-aware control margin.
- Evidence of recent ball reception or interception.

Penting: Track slots bukan identitas pemain permanen. Hindari asumsi bahwa satu slot berarti satu pemain sepanjang pertandingan.

### Transition-aware evidence

Daripada hanya menghitung siapa yang unggul pada frame sekarang, pertimbangkan evidence yang mendukung **pergantian dari pemegang possession saat ini ke tim penantang**.

Misalnya, ketika model sedang mempertahankan H, pergantian H → A harus membutuhkan cukup bukti bahwa A memperoleh controlled possession, bukan hanya satu atau beberapa frame dengan nearest-player A.

Bentuk matematisnya dapat berupa acquisition score, transition hint yang lebih informatif, atau evidence-aware transition cost.

### Pertanyaan untuk Codex

- Apakah fitur saat ini sudah merepresentasikan acquisition secara implisit?
- Bagian mana dari `control_margin`, `proxy_persistence`, dan `transition_hint` yang masih terlalu sensitif?
- Apakah fitur baru benar-benar memberikan informasi tambahan atau hanya menduplikasi evidence yang sudah ada?
- Bisakah kita membuat synthetic scenarios untuk menguji loose ball, contested ball, dan genuine turnover?

Jangan menambah fitur hanya karena terlihat intuitif. Setiap fitur harus memiliki fungsi yang jelas dan dapat diuji.

---

## 4. Proposal C — Evidence-Aware Temporal Inference

Aku tidak merekomendasikan mengganti Viterbi sebagai langkah pertama.

Decoder saat ini tampaknya sudah mengurangi transisi dibanding frame-wise argmax.

Namun, kita bisa mempertimbangkan pengembangan setelah emission scores lebih baik.

### Option 1 — Hysteresis

Gunakan kriteria mempertahankan possession yang berbeda dari kriteria berpindah possession.

Ketika evidence ambigu, state sebelumnya dipertahankan.

Ketika challenger-team acquisition evidence kuat, transition dapat terjadi lebih cepat.

### Option 2 — Adaptive Transition Cost

Biaya H ↔ A dapat dipengaruhi evidence akuisisi kontrol dan observability.

Contoh konseptual:

\[
C_{H\rightarrow A}(t)=
\operatorname{clip}
\left(C_0-\alpha S_{\text{acquire},A}(t), C_{\min}, C_{\max}\right)
\]

Periksa terlebih dahulu apakah decoder saat ini sudah mempunyai mekanisme serupa melalui `transition_hint`.

Jangan membuat fitur baru yang sebenarnya menduplikasi mekanisme yang sudah tersedia.

### Option 3 — Explicit Duration Modeling / HSMM

HSMM dapat memodelkan durasi possession dan stoppage secara eksplisit.

Tetapi aku tidak merekomendasikannya sebagai prioritas sekarang karena:

- Duration distribution yang benar tidak tersedia.
- Duration prior berisiko dipilih berdasarkan asumsi yang tidak tervalidasi.
- Kompleksitas implementasi meningkat.
- Deadline kompetisi sudah dekat.

Pertimbangkan HSMM hanya jika diagnosis menunjukkan keterbatasan mendasar pada decoder saat ini dan terdapat dasar yang sah untuk duration prior.

### Smoothing

Majority-vote smoothing atau minimum-duration filtering boleh diuji sebagai baseline pembanding.

Namun, jangan otomatis menggabungkan seluruh possession pendek.

Short possession tidak sama dengan false possession.

---

## 5. Proposal D — Strengthen Dead-Ball Temporal Evidence

Memperbaiki fusion saja mungkin belum cukup.

Diagnosis menunjukkan raw dead evidence yang unggul umumnya berlangsung sangat singkat.

Kita perlu mengevaluasi apakah modul live/dead menangkap stoppage sebagai **temporal state**, bukan hanya kondisi gerak lokal.

Pertimbangkan:

- Persistent stoppage evidence.
- Event-based restart context.
- Ball motion sebelum dan sesudah kandidat stoppage.
- Pergerakan kolektif pemain.
- Out-of-pitch evidence ketika reliable.
- Konteks set piece tanpa menganggap interval tag sebagai ground truth.

Hindari dua asumsi:

1. Missing ball berarti dead ball.
2. Low ball speed berarti dead ball.

Keduanya dapat menghasilkan kesalahan sistematis.

Jika memungkinkan, bedakan bukti bahwa permainan berhenti dari sekadar tidak adanya evidence bahwa permainan berlangsung.

---

## 6. Evaluation Without Training or Tuning on Public GT

Ini sangat penting.

Kompetisi melarang penggunaan public five-minute ground truth untuk training, tuning, atau model selection.

Karena itu, jangan menggunakan public spellF1 untuk memilih parameter, smoothing window, fusion weights, atau arsitektur.

Strategi evaluasi yang diperbolehkan:

### A. Synthetic Tests

Buat synthetic evidence sequences untuk menguji perilaku model pada:

- Stable possession.
- Temporary challenger proximity.
- Contested ball.
- Genuine control acquisition.
- Ball occlusion.
- Stoppage.
- Restart.
- Rapid but valid turnover.

Expected states pada synthetic tests boleh ditentukan karena datanya bukan label manual dari pertandingan kompetisi.

### B. Ablation Tests

Uji pengaruh perubahan komponen secara terisolasi:

- Existing fusion vs factorized fusion.
- Existing control vs additional acquisition evidence.
- Fixed transition cost vs evidence-aware transition cost.
- With/without Smart Tagging.
- With/without selected motion evidence.

### C. Behavioral Diagnostics

Bandingkan:

- Utility distributions.
- Dead advantage.
- State transition rates.
- Short reversal characteristics.
- Observation-quality sensitivity.
- Stability across halves.
- Model responsiveness to evidence.

Namun, behavioral metrics merupakan diagnostic indicators, bukan pengganti ground-truth accuracy.

Jangan memilih model hanya karena jumlah possession terlihat lebih realistis.

### D. External Validation

Jika tersedia data berlabel dari pertandingan lain yang memenuhi aturan kompetisi, gunakan sebagai validasi independen.

Untuk sekarang, jangan mengasumsikan data tersebut tersedia.

---

## 7. Proposed Experiment Sequence

Aku mengusulkan urutan berikut:

**Experiment 1 — Fusion Redesign**

Pertahankan seluruh modul lain, hanya ubah fusion.

Objective: Memastikan D dapat bersaing tanpa tambahan kontrol tim yang secara struktural menguntungkan H/A.

**Experiment 2 — Dead-Ball Evidence Persistence**

Pertahankan hasil Experiment 1 sebagai kandidat terpisah, lalu evaluasi evidence stoppage yang lebih berkelanjutan.

Objective: Meningkatkan kemampuan mengenali dead-ball intervals, bukan sekadar memperbesar skor D.

**Experiment 3 — Controlled Possession Acquisition**

Tambahkan atau revisi evidence turnover.

Objective: Mengurangi spurious H/A transitions tanpa menghilangkan genuine turnover.

**Experiment 4 — Temporal Transition Improvements**

Pertimbangkan hysteresis atau adaptive transition cost setelah evidence diperbaiki.

Objective: Meningkatkan temporal consistency dengan tetap menjaga responsiveness.

Setiap experiment harus menghasilkan run terpisah, konfigurasi efektif, artefak, dan hasil diagnostik yang dapat dibandingkan pada notebook 05.

Jangan menimpa baseline v2.

---

## 8. What I Need From You Before Implementation

Sebelum mengubah kode, aku ingin kamu memberikan evaluasi kritis terhadap proposal ini.

Tolong:

1. **Audit implementasi aktual** pada modul live/dead, control, fusion, decoder, dan refinement.
2. **Konfirmasi atau koreksi diagnosis** berdasarkan kode dan artefak lokal. Jika angka atau asumsi dalam proposal tidak sesuai, utamakan bukti repository.
3. **Evaluasi Proposal A–D**, termasuk kelebihan, kelemahan, potensi failure modes, dan kompleksitas implementasinya.
4. **Identifikasi perubahan terkecil dengan potensi manfaat terbesar**, mempertimbangkan deadline.
5. **Usulkan experiment plan** yang dapat membedakan hipotesis tanpa bergantung pada public ground truth.
6. **Jelaskan rekomendasi akhir**: apa yang sebaiknya dikerjakan sekarang, apa yang ditunda, dan mengapa.

Untuk tahap pertama, **jangan mengubah logic model atau konfigurasi produksi terlebih dahulu**.

Kita akan menggunakan hasil evaluasimu untuk menentukan perubahan baseline v3.

Aku ingin diskusi ini bersifat kolaboratif. Kamu tidak harus menyetujui seluruh usulan di atas. Jika ada asumsi yang keliru atau pendekatan yang lebih baik, sampaikan dengan bukti dan alasan teknis.

Prinsipnya:

**Understand the failure mechanism → Form a testable hypothesis → Run a controlled experiment → Interpret the evidence → Decide whether to adopt the change.**