# Kamus data fitur aktual

Dihasilkan pada 2026-11-14T13:09:29.900791+07:00 dari `src/features.py`; 45 kolom. Null % dan keunikan dihitung pada 158.857 frame graded.

| column | family | unit | meaning | dtype | null_pct | unique_non_null |
| --- | --- | --- | --- | --- | --- | --- |
| frame | timeline | frame / babak / boolean | Indeks video 0-based; half=0 di luar dua blok penilaian. | int32 | 0.0000 | 158857 |
| half | timeline | frame / babak / boolean | Indeks video 0-based; half=0 di luar dua blok penilaian. | int8 | 0.0000 | 2 |
| graded | timeline | frame / babak / boolean | Indeks video 0-based; half=0 di luar dua blok penilaian. | bool | 0.0000 | 1 |
| ball_x | posisi | koordinat 0-1 | Observasi posisi; dapat NaN, periksa outlier di luar lapangan. | float64 | 40.6926 | 58686 |
| ball_y | posisi | koordinat 0-1 | Observasi posisi; dapat NaN, periksa outlier di luar lapangan. | float64 | 40.6926 | 61542 |
| ball_ok | kualitas/konteks | boolean / kategori | Indikator observasi atau event; bukan label kebenaran. | bool | 0.0000 | 2 |
| ball_missing_run_frames | visibilitas/event | frame | Panjang gap/jarak ke observasi atau batas tag; bukan durasi possession. | int32 | 0.0000 | 122 |
| frames_since_ball_seen | visibilitas/event | frame | Panjang gap/jarak ke observasi atau batas tag; bukan durasi possession. | float64 | 0.0013 | 5538 |
| frames_until_ball_seen | visibilitas/event | frame | Panjang gap/jarak ke observasi atau batas tag; bukan durasi possession. | float64 | 0.3475 | 5538 |
| home_count | cakupan pemain | jumlah slot | Slot dengan x/y finite; bisa lebih dari 11 karena duplikasi tracking. | int64 | 0.0000 | 20 |
| away_count | cakupan pemain | jumlah slot | Slot dengan x/y finite; bisa lebih dari 11 karena duplikasi tracking. | int64 | 0.0000 | 15 |
| nearest_h_m | geometri | meter | Jarak/margin dari koordinat normalisasi dikali ukuran lapangan 105.3 x 68 m. | float64 | 40.8084 | 93994 |
| nearest_a_m | geometri | meter | Jarak/margin dari koordinat normalisasi dikali ukuran lapangan 105.3 x 68 m. | float64 | 40.6926 | 94190 |
| distance_margin_a_minus_h_m | geometri | meter | Jarak/margin dari koordinat normalisasi dikali ukuran lapangan 105.3 x 68 m. | float64 | 40.8084 | 94030 |
| abs_distance_margin_m | geometri | meter | Jarak/margin dari koordinat normalisasi dikali ukuran lapangan 105.3 x 68 m. | float64 | 40.8084 | 94030 |
| nearest_team_proxy | kontrol proxy | H/A/? | Tim terdekat secara geometri; bukan kepemilikan bola ground truth. | str | 0.0000 | 3 |
| ball_displacement_m | geometri | meter | Jarak/margin dari koordinat normalisasi dikali ukuran lapangan 105.3 x 68 m. | float64 | 40.7864 | 86316 |
| ball_speed2d_mps | gerak bola | m/s | Turunan posisi 2D pada frame berurutan dalam babak yang sama; bukan kecepatan 3D. | float64 | 40.7864 | 86311 |
| ball_speed_quality | kualitas/konteks | boolean / kategori | Indikator observasi atau event; bukan label kebenaran. | bool | 0.0000 | 2 |
| ball_visibility_age_s | visibilitas | detik | Jarak waktu ke observasi bola; fitur until/next memakai masa depan dalam inferensi offline. | float64 | 0.0013 | 5538 |
| ball_next_visibility_s | visibilitas | detik | Jarak waktu ke observasi bola; fitur until/next memakai masa depan dalam inferensi offline. | float64 | 0.3475 | 5538 |
| valid_h_fraction | cakupan pemain | 0-1 | Cakupan dibagi 11 dan dicap 1; duplikasi tidak terlihat setelah clipping. | float64 | 0.0000 | 12 |
| valid_a_fraction | cakupan pemain | 0-1 | Cakupan dibagi 11 dan dicap 1; duplikasi tidak terlihat setelah clipping. | float64 | 0.0000 | 11 |
| player_motion_robust_h | gerak pemain | m/s | Median perpindahan slot berurutan yang masuk akal; slot bukan identitas permanen. | float64 | 2.8705 | 113187 |
| player_motion_robust_a | gerak pemain | m/s | Median perpindahan slot berurutan yang masuk akal; slot bukan identitas permanen. | float64 | 2.2335 | 108256 |
| centroid_speed_H_mps | alias gerak | m/s | Alias player_motion_robust, bukan kecepatan centroid tim. | float64 | 2.8705 | 113187 |
| centroid_speed_A_mps | alias gerak | m/s | Alias player_motion_robust, bukan kecepatan centroid tim. | float64 | 2.2335 | 108256 |
| centroid_valid_h | kualitas gerak | boolean | Validitas gerak robust; nama centroid merupakan kompatibilitas historis. | bool | 0.0000 | 2 |
| centroid_valid_a | kualitas gerak | boolean | Validitas gerak robust; nama centroid merupakan kompatibilitas historis. | bool | 0.0000 | 2 |
| nearest_h_x | posisi | koordinat 0-1 | Observasi posisi; dapat NaN, periksa outlier di luar lapangan. | float64 | 40.8084 | 58012 |
| nearest_h_y | posisi | koordinat 0-1 | Observasi posisi; dapat NaN, periksa outlier di luar lapangan. | float64 | 40.8084 | 60229 |
| nearest_a_x | posisi | koordinat 0-1 | Observasi posisi; dapat NaN, periksa outlier di luar lapangan. | float64 | 40.6926 | 57634 |
| nearest_a_y | posisi | koordinat 0-1 | Observasi posisi; dapat NaN, periksa outlier di luar lapangan. | float64 | 40.6926 | 60011 |
| video_recovered_ball | kualitas/konteks | boolean / kategori | Indikator observasi atau event; bukan label kebenaran. | bool | 0.0000 | 1 |
| active_tag_count | tagging | jumlah interval | Tag aktif menggunakan interval [start,end); kedua tim bisa bertag bersamaan. | int64 | 0.0000 | 13 |
| active_h_tags | tagging | jumlah interval | Tag aktif menggunakan interval [start,end); kedua tim bisa bertag bersamaan. | int64 | 0.0000 | 8 |
| active_a_tags | tagging | jumlah interval | Tag aktif menggunakan interval [start,end); kedua tim bisa bertag bersamaan. | int64 | 0.0000 | 9 |
| set_piece_tag | kualitas/konteks | boolean / kategori | Indikator observasi atau event; bukan label kebenaran. | bool | 0.0000 | 2 |
| set_piece_start | kualitas/konteks | boolean / kategori | Indikator observasi atau event; bukan label kebenaran. | bool | 0.0000 | 2 |
| set_piece_end | kualitas/konteks | boolean / kategori | Indikator observasi atau event; bukan label kebenaran. | bool | 0.0000 | 2 |
| tag_set_piece_type | tagging | kategori / 0-1 | Konteks otomatis. Support mencap keluarga tag menyerang yang berkorelasi. | str | 0.0000 | 7 |
| tag_set_piece_team | tagging | kategori / 0-1 | Konteks otomatis. Support mencap keluarga tag menyerang yang berkorelasi. | str | 0.0000 | 3 |
| tag_boundary_distance_frames | visibilitas/event | frame | Panjang gap/jarak ke observasi atau batas tag; bukan durasi possession. | float64 | 0.0000 | 4351 |
| tag_h_support | tagging | kategori / 0-1 | Konteks otomatis. Support mencap keluarga tag menyerang yang berkorelasi. | float64 | 0.0000 | 3 |
| tag_a_support | tagging | kategori / 0-1 | Konteks otomatis. Support mencap keluarga tag menyerang yang berkorelasi. | float64 | 0.0000 | 3 |

Array posisi pemain dan tabel event/slot hanya berada di memori. Skor antarproses bukan probabilitas terkalibrasi. Detail skema ATD 193 kolom dan statistik all/graded tersedia di evaluation/reports/exploration/.
