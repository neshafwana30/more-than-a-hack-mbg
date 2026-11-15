# Hasil eksplorasi data dan diagnosis pipeline

Audit lokal dimulai pada **2026-11-14T13:09:29.900791+07:00**. Sumber fitur: `src/features.py` dan konfigurasi tetap `config/v2.yaml`. Tidak ada training, tuning GT, perubahan parameter inferensi, atau penulisan submission. Statistik dihitung ulang dari aset lokal. Pengujian proyek yang dijalankan pada sesi ini: **26 passed**.

## Temuan utama

1. **Dead-ball sudah kalah pada bukti sebelum decoder.** Dari 155,923 frame graded di luar interval kickoff wajib, skor D mengungguli kedua skor tim pada **0 frame** dan seri pada 0 frame. Sesudah decoding/refinement, D di luar interval wajib: **0 frame**. Distribusi selisih D terhadap tim terbaik: median -1.3471, maksimum -0.0027. Diagnosis pertama adalah skala/konstruksi bukti fusion, kemudian durasi stoppage; mengganti penalti decoder saja belum menyelesaikan kelemahan bukti.
2. **Fragmentasi tetap besar.** Output menghasilkan 1,070 possession, median 2.04 detik dan rata-rata 5.83 detik. Ini deskripsi prediksi, bukan estimasi jumlah possession benar. Soft short-spell refinement menggabungkan 0 run; boundary refinement menggeser 115 dari 1069 batas yang diperiksa.
3. **Missing ball bukan gap singkat yang dapat diasumsikan aman untuk interpolasi.** Terdapat 150 gap, median 10.72 detik, P95 50.80 detik, maksimum 221.48 detik. Sebanyak 71 gap menyentuh konteks set piece. Perbedaan proxy di tepi gap tidak membuktikan turnover.
4. **Metadata ATD broadcast kosong.** Raw tracking dan homography broadcast dapat diaudit terpisah, tetapi pemuatan tracking lengkap dengan metadata terblokir. Sinkronisasi broadcast-tactical tetap belum tersedia.
5. **Ada perbedaan jumlah frame video broadcast.** OpenCV melaporkan 164,487 frame video broadcast, sementara kontrak/raw tracking/homography menggunakan 164,488. Status decode frame terakhir yang diharapkan dicatat di video_metadata.csv. Ini isu integritas sumber/decoder yang perlu diselesaikan sebelum sinkronisasi lintas kamera.
6. **Pemilihan jendela video bukan pemulihan bola.** Konfigurasi video enabled=False; weights configured=False. Selektor memilih 40 jendela / 2840 frame, mencakup 2057 dari 64643 missing-ball frame graded. Anggaran ini berorientasi transisi/anomali, bukan menutup seluruh gap.

## Input, proses, output

Input utama berupa ATD tactical + metadata + Smart Tagging; homography dan video merupakan tambahan opsional. `build_features` menghasilkan tabel fitur kanonis 0-based, event ternormalisasi, slot-to-team, serta array posisi pemain di memori. Modul kualitas, event, kontrol tim dan live/dead membentuk bukti; fusion membuat utility H/A/D; Viterbi berjalan per babak; refinement menggeser batas yang sudah ada. Output submission adalah `frame,label` sepanjang 199.319 baris. `spellF1` menilai segmen possession dan `macroF1` menilai per frame. Audit ini tidak menghitung atau mengoptimasi skor GT.

## Inventaris dan integritas sumber

| source | asset | exists | bytes | status |
| --- | --- | --- | --- | --- |
| tactical | video | True | 5222307334 | ok |
| tactical | atd_meta | True | 59057 | ok |
| tactical | atd_raw | True | 195714285 | ok |
| tactical | homography | True | 23187043 | ok |
| tactical | smart_tagging | True | 261055 | ok |
| broadcast | video | True | 13962360320 | ok |
| broadcast | atd_meta | True | 0 | empty_or_missing |
| broadcast | atd_raw | True | 141966465 | ok |
| broadcast | homography | True | 15401635 | ok |

| source | records | first | last | discontinuities | expected | ok |
| --- | --- | --- | --- | --- | --- | --- |
| tactical | 199319 | 1 | 199319 | 0 | 199319 | True |
| broadcast | 164488 | 1 | 164488 | 0 | 164488 | True |

| source | opened | frames | fps | width | height | expected_frames | frame_count_matches_expected | expected_last_frame_decodes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tactical | True | 199319 | 25.0000 | 1920 | 1080 | 199319 | True | - |
| broadcast | True | 164487 | 25.0000 | 1920 | 1080 | 164488 | False | False |

Frame tracking yang diberikan adalah 1-based berurutan; indeks submission = frame_id - 1. Offset IdFrame di games.py adalah counter berbeda dan tidak diterapkan lagi ke ATD yang diberikan. Smart Tagging dibulatkan dari detik x 25 dengan interval analisis [start,end). Homography memakai frame 0-based.

## Skema dan kualitas observasi

ATD terurai: 199,319 baris x 193 kolom. Fitur aktual: 199,319 baris x 45 kolom. Slot per tim: {'H': 24, 'A': 22}. Slot bukan identitas pemain.

| field | non_null | rows |
| --- | --- | --- |
| ball_state | 0 | 199319 |
| ball_owning_team_id | 0 | 199319 |
| ball_z | 0 | 199319 |
| ball_speed | 0 | 199319 |

| region | frames | ball_seen_pct | mean_players | over_22_frames |
| --- | --- | --- | --- | --- |
| all | 199319 | 56.9379 | 18.9767 | 7102 |
| graded | 158857 | 59.3074 | 20.4874 | 3749 |
| half_1 | 79467 | 59.3492 | 20.1211 | 695 |
| half_2 | 79390 | 59.2657 | 20.8541 | 3054 |
| ungraded | 40462 | 47.6348 | 13.0456 | 3353 |

| group | count | median_seconds | p95_seconds | set_piece_overlap |
| --- | --- | --- | --- | --- |
| (1, 'different') | 29 | 10.7600 | 55.0320 | 15 |
| (1, 'same') | 37 | 11.9200 | 56.2640 | 16 |
| (1, 'unavailable') | 3 | 17.2000 | 35.1640 | 1 |
| (2, 'different') | 31 | 10.4000 | 48.0200 | 20 |
| (2, 'same') | 49 | 9.8000 | 30.0640 | 19 |
| (2, 'unavailable') | 1 | 4.8800 | 4.8800 | 0 |

Kolom optional yang seluruhnya null tidak memberikan label possession/live-dead tersembunyi. Fitur dengan nama centroid_speed sekarang merupakan alias median gerak slot yang telah disaring, bukan kecepatan centroid. Fitur frames_until/next memakai observasi masa depan dan sesuai inferensi offline. Kamus seluruh fitur tersedia di [DATA_DICTIONARY.md](DATA_DICTIONARY.md).

## Kontrak data turunan di src

Tabel di bawah dibuat di memori saat inferensi. Audit menyimpan statistik agregatnya, bukan mengekspor ulang tracking atau matriks skor per frame.

| Modul | Hasil turunan | Granularitas dan kegunaan |
| --- | --- | --- |
| features | frames, events, slots, player_positions | 199.319 baris fitur; interval event; slot-to-team; array x/y/valid/sides |
| observation_quality | QualityResult.frames, observation_weight | Per frame: visibilitas, jump, motion reliability, kelas gap, dan bobot kualitas |
| event_evidence | EventResult.frames, candidates | Per frame: stoppage/restart/out-of-pitch; daftar kandidat otomatis berbasis tag |
| team_control | ControlResult.frames, scores, transition_hint | Per frame; skor N x 2 untuk H/A, margin dan persistensi |
| live_dead | LiveDeadResult.frames, live_score, dead_score | Per frame; utility live/dead dan ketersediaan gerak pemain |
| evidence_fusion | EvidenceBundle | Utility N x 3 dalam urutan H/A/D; hint transisi, stop, restart; metadata semantik |
| temporal_decoder | DecodeResult | Label N; tabel transisi frame/from/to/half; jumlah run pendek yang digabung |
| boundary_refinement | BoundaryResult | Label N; laporan original/refined/shift/from/to/ambiguous per batas yang diperiksa |
| ambiguity_detector | AmbiguityResult | Flag per frame dan jendela start/end/center/reason/priority yang dipilih |
| visual_analyzer | VisualResult | Observasi frame/x/y/quality jika detektor aktif; belum ada pemulihan pada konfigurasi audit |
| pipeline | PipelineResult | Label, diagnostik agregat, informasi output dan transisi awal; seluruh tabel antara tidak dikembalikan |
| submission | segments_for_labels, aggregate_diagnostics | Segmen possession/dead-ball dan ringkasannya; CSV submission tetap tidak ditulis oleh audit |

Skema mentah 193 kolom, fitur aktual 45 kolom, kontribusi skor, dan ringkasan tahap decoder disimpan terpisah. Modul visual belum dijalankan sebagai detektor karena tidak ada bobot lokal. Intermediate schema dapat berubah jika kode sumber berubah; hash sumber di summary.json mengikat hasil audit ke implementasi ini.

## Geometri dan Smart Tagging

| feature | count | median | p95 | max |
| --- | --- | --- | --- | --- |
| nearest_h_m | 94030 | 3.5430 | 16.5757 | 59.2446 |
| nearest_a_m | 94214 | 4.6618 | 16.0370 | 86.9043 |
| abs_distance_margin_m | 94030 | 3.5294 | 15.6802 | 57.1378 |
| ball_speed2d_mps | 94065 | 5.0818 | 17.3515 | 109.5979 |
| player_motion_robust_h | 154297 | 1.3739 | 3.7444 | 8.1780 |
| player_motion_robust_a | 155309 | 1.3514 | 3.7438 | 7.9324 |

| instances | unknown_team | coverage_pct | both_team_active_frames | both_attacking_support_frames | invalid_or_outside_intervals |
| --- | --- | --- | --- | --- | --- |
| 638 | 36 | 82.2740 | 66993 | 14610 | 0 |

| code | instances | unknown_team | median_duration_s | graded_frames | missing_ball_pct |
| --- | --- | --- | --- | --- | --- |
| ATTACKING TRANSITION | 24 | 0 | 15.4400 | 9109 | 8.2007 |
| BALL IN FINAL THIRD | 93 | 0 | 27.0000 | 58690 | 32.0003 |
| BALL IN THE BOX | 72 | 0 | 17.4400 | 28965 | 30.0017 |
| BUILD UP | 22 | 0 | 20.5200 | 11011 | 7.2745 |
| CREATING CHANCES | 30 | 2 | 30.0000 | 22384 | 42.7448 |
| DEFENDING IN ATTACKING THIRD | 22 | 0 | 20.5200 | 11011 | 7.2745 |
| DEFENDING IN DEFENSIVE THIRD | 30 | 2 | 30.0000 | 22384 | 42.7448 |
| DEFENDING IN MIDDLE THIRD | 66 | 0 | 17.6400 | 31118 | 7.1952 |
| DEFENSIVE TRANSITION | 24 | 0 | 15.4400 | 9109 | 8.2007 |
| GOALS | 4 | 0 | 169.2600 | 16503 | 88.3355 |
| LONG BALL | 29 | 0 | 12.4000 | 9433 | 11.9368 |
| PLAYERS IN THE BOX | 97 | 0 | 27.6800 | 72639 | 42.4538 |
| PROGRESSION | 66 | 0 | 17.6400 | 31118 | 7.1952 |
| SET PIECES | 59 | 32 | 35.0000 | 43899 | 31.9347 |

Tag kedua tim bisa aktif bersamaan, termasuk karena sebuah tim bertahan saat tim lain menyerang. Overlap tag mentah bukan otomatis konflik. Tabel both_attacking_support_frames mengukur dukungan menyerang simultan secara terpisah. Distribusi spasial dihitung hanya saat bola terlihat; missing ball tidak dapat diberi zona tanpa inferensi tambahan. Tidak ada asumsi arah serang yang dipakai untuk zona koordinat.

## Diagnosis live/dead dan fusion

Fusion memakai H = home_control + live_score, A = away_control + live_score, D = dead_score pada bobot default. Bukti tim menambah utility kelas live tanpa tambahan setara pada kelas D. Saat seluruh bukti observasi/tag netral, prior menghasilkan H=A=0.59 dan D=0.25. Tambahan ini bukan probabilitas terkalibrasi. Analisis berikut mengecualikan interval kickoff wajib yang memakai skor +/-1e6.

| term | mean_unforced | nonzero_frames_unforced | max_unforced |
| --- | --- | --- | --- |
| live_prior | 0.5500 | 155923 | 0.5500 |
| ball_motion_live | 0.0963 | 91572 | 0.2200 |
| players_motion_live | 0.0607 | 153686 | 0.1200 |
| dead_prior | 0.2500 | 155923 | 0.2500 |
| outside_dead | 0.0049 | 2272 | 0.3500 |
| stoppage_dead | 0.0029 | 2793 | 0.3200 |
| restart_dead | 0.0013 | 2891 | 0.1400 |
| stationary_dead | 0.0002 | 313 | 0.1200 |

| component | count | median | p95 | max |
| --- | --- | --- | --- | --- |
| live | 155923 | 0.6743 | 0.8900 | 0.8900 |
| dead | 155923 | 0.2500 | 0.2500 | 0.9200 |
| home_control | 155923 | 0.1900 | 1.6857 | 2.0815 |
| away_control | 155923 | 0.0687 | 1.5468 | 2.0664 |
| dead_minus_best_team | 155923 | -1.3471 | -0.3544 | -0.0027 |

![Distribusi data dan skor](../evaluation/reports/exploration/data_and_evidence.png)

![Profil sekitar set piece](../evaluation/reports/exploration/set_piece_evidence.png)

Batas tag adalah konteks sekitar restart, belum tentu awal stoppage atau saat bola dimainkan. Implementasi saat ini membuat pulsa stoppage di awal tag dan restart di akhir tag; kualitas asumsi ini perlu dievaluasi secara algoritmis, bukan dijadikan label pasti. Seluruh jendela yang menyentuh interval kickoff wajib dikeluarkan dari profil agar skor paksa +/-1e6 tidak mencemari rerata. Profil agregat di atas memperlihatkan perubahan skor di sekitar batas tag, tidak memvalidasi waktu restart sebenarnya.

## Struktur temporal dan provenance prediksi

| stage | possessions | dead_segments | transitions | median_possession_s | spells_lt_1s | spells_lt_2s |
| --- | --- | --- | --- | --- | --- | --- |
| emission_argmax | 1709 | 1 | 1708 | 1.2800 | 758 | 1048 |
| decoded | 1070 | 1 | 1069 | 2.0400 | 311 | 526 |
| refined | 1070 | 1 | 1069 | 2.0400 | 310 | 523 |

| from | to | count | missing_ball_pct | weak_margin_pct | set_piece_context_pct |
| --- | --- | --- | --- | --- | --- |
| A | H | 534 | 6.5543 | 83.3333 | 30.5243 |
| D | H | 1 | 0.0000 | 0.0000 | 100.0000 |
| H | A | 534 | 4.4944 | 79.2135 | 30.3371 |

Run pendek (<2 detik) yang diapit state sama: {'count': 523, 'any_missing_ball': 5, 'set_piece_overlap': 149}. Ini kandidat diagnosis, bukan daftar kesalahan berlabel.

| path | valid | same_labels_as_current_run | possession_spells | dead_segments | possession_median_s |
| --- | --- | --- | --- | --- | --- |
| evaluation\results\baseline_raw.csv | True | False | 1949 | 150 | 0.9200 |
| evaluation\results\baseline_smooth125.csv | True | False | 398 | 79 | 8.1400 |
| evaluation\results\baseline_smooth25.csv | True | False | 746 | 133 | 3.0400 |
| evaluation\results\baseline_smooth50.csv | True | False | 594 | 133 | 4.6000 |
| evaluation\results\baseline_v1.csv | True | False | 1152 | 154 | 2.0400 |
| evaluation\results\baseline_v2.csv | True | True | 1070 | 1 | 2.0400 |

Perbandingan di atas adalah label per frame setelah validasi, sehingga tidak terpengaruh perbedaan line ending CSV. Hash byte masing-masing file ada di prediction_artifacts.csv. Artefak v3/outputs yang disebut pada identifikasi sebelumnya tidak tersedia dalam workspace saat audit ini. Notebook lama tetap dipertahankan sebagai snapshot historis.

![Diagnosis temporal](../evaluation/reports/exploration/temporal_diagnostics.png)

## Homography, video, dan validasi

| source | region | rows | valid_rows | invalid_finite_mismatch | near_singular_valid | error_median | error_p95 | error_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tactical | all | 199319 | 199319 | 0 | 0 | 0.2392 | 0.5357 | 5.4755 |
| tactical | half_1 | 79467 | 79467 | 0 | 0 | 0.2237 | 0.3516 | 5.4523 |
| tactical | half_2 | 79390 | 79390 | 0 | 0 | 0.2369 | 0.3631 | 5.4531 |
| tactical | ungraded | 40462 | 40462 | 0 | 0 | 0.3510 | 4.6021 | 5.4755 |
| broadcast | all | 164488 | 108248 | 0 | 0 | 0.2165 | 0.7427 | 303.8238 |

| deterministic_samples | decoded_samples | players_available_total | players_projected_inside_total | limitation |
| --- | --- | --- | --- | --- |
| 8 | 8 | 140 | 132 | Numeric projection/decode QA only; in-bounds projection does not prove visual alignment. |

Video QA menggunakan sampel deterministik di kedua babak, decode video dan proyeksi numerik koordinat ke ukuran gambar; tidak menyimpan cuplikan. Proyeksi di dalam gambar tidak membuktikan objek tepat menempel pada pemain. Pemeriksaan overlay visual rinci dan deteksi bola dari video belum dilakukan; bobot detektor lokal belum tersedia.

| tracking_counters | canonical_features | labels_valid | contributions_reconstructed | public_gt_format_valid | homography_canonical | video_samples_decoded | profiles_exclude_forced_kickoff |
| --- | --- | --- | --- | --- | --- | --- | --- |
| True | True | True | True | True | True | True | True |

GT publik diperiksa hanya kontrak format/alignment: {'rows': 199319, 'graded': 7500, 'first': 11185, 'last': 18684, 'labels': {'H': 3454, 'A': 2577, 'D': 1469}, 'use': 'Format/alignment contract only; no accuracy comparison or tuning.'}. Tidak ada model yang dilatih/dipilih berdasarkan jendela tersebut.

## Backlog perbaikan berdasarkan bukti

| Prioritas | Pekerjaan | Bukti/tujuan verifikasi |
| --- | --- | --- |
| P0 | Perbaiki desain kompetisi skor live/dead terhadap kontrol tim; kaji pemisahan keputusan live/dead dan H/A | D harus dapat memperoleh dukungan dari observasi stoppage yang masuk akal; uji sintetis memastikan bola hilang saja tidak memaksa D, dan gerak pemain saja tidak memaksa live |
| P0 | Modelkan konteks persiapan, stoppage, dan restart sebagai bukti temporal, bukan salin batas tag | Hindari asumsi awal/akhir tag adalah transisi pasti; uji sintetis set piece yang menyeberangi restart |
| P1 | Audit persistensi kontrol dan skala biaya transisi terhadap akumulasi emission | Tinjau run singkat dan pergantian balik berdasarkan kekuatan bukti; jangan memaksakan jumlah possession tertentu |
| P1 | Kelompokkan gap panjang, kualitas pemain, dan dukungan tag untuk strategi inferensi | Hindari interpolasi panjang atau membawa state tanpa evaluasi dukungan sebelum/sesudah gap |
| P1 | Selaraskan notebook baseline lama dengan implementasi src | Audit terbaru sekarang disematkan; implementasi baseline inline lama tetap historis |
| P2 | Pulihkan metadata dan periksa selisih jumlah frame video broadcast sebelum eksperimen lintas kamera | Jangan menebak team mapping atau sinkronisasi dari indeks frame |
| P2 | Evaluasi manfaat detektor lokal pada jendela berisiko | Ukur observasi bola yang diterima, validitas proyeksi, dan cakupan anggaran; detektor belum tersedia |

## Reproduksi dan keterbatasan

Dari root proyek:

```powershell
.\mbg\Scripts\python.exe tools/explore_data.py
.\mbg\Scripts\python.exe tools/render_exploration_report.py
.\mbg\Scripts\python.exe -m pytest -q
```

Hasil lengkap berada di `evaluation/reports/exploration/` (gitignored): inventaris, statistik skema mentah/fitur, kamus fitur, kualitas homography, metadata video, geometri/zona, gap, tag, kontribusi skor, transisi, segmen, profil set piece, provenance prediksi, tiga grafik, dan summary.json. Hash sumber/config/input tabular dicatat; video tidak di-hash ulang agar tidak membaca 19 GB tambahan. Tidak ada dataset tracking per frame atau cuplikan video yang diekspor.

Ringkasan agregat dan grafik bersifat deskriptif. Satu pertandingan tanpa GT penuh tidak memberi ukuran akurasi generalisasi. Uji sintetis/regresi memeriksa perilaku kode, bukan kebenaran prediksi pertandingan. Dokumen ini mencatat diagnosis dan backlog; parameter inferensi dan submission tetap belum diubah.
