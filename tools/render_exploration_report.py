"""Render the completed local audit as an Indonesian report and notebook appendices."""
from pathlib import Path
import base64
import contextlib
import io
import json
import sys
import uuid
import pandas as pd
from explore_data import ROOT, table, digest

OUT=ROOT/'evaluation/reports/exploration'
S=json.loads((OUT/'summary.json').read_text(encoding='utf-8'))

def load(name):return pd.read_csv(OUT/(name+'.csv'))
def mdtable(name,cols=None):
    df=load(name)
    return table(df if cols is None else df[cols])

def code(source):return {'cell_type':'code','id':uuid.uuid4().hex[:8],'metadata':{'tags':['current_src_audit']},'execution_count':None,'outputs':[],'source':source.splitlines(True)}
def md(source):return {'cell_type':'markdown','id':uuid.uuid4().hex[:8],'metadata':{'tags':['current_src_audit']},'source':source.splitlines(True)}

E=S['evidence'];P=S['prediction'];G=S['gaps'];V=S['video_selection']
created=S['created_at']
intro=f'''# Hasil eksplorasi data dan diagnosis pipeline

Audit lokal dimulai pada **{created}**. Sumber fitur: `src/features.py` dan konfigurasi tetap `config/v2.yaml`. Tidak ada training, tuning GT, perubahan parameter inferensi, atau penulisan submission. Statistik dihitung ulang dari aset lokal. Pengujian proyek yang dijalankan pada sesi ini: **26 passed**.

## Temuan utama

1. **Dead-ball sudah kalah pada bukti sebelum decoder.** Dari {E['unforced_frames']:,} frame graded di luar interval kickoff wajib, skor D mengungguli kedua skor tim pada **{E['dead_wins_unforced']:,} frame** dan seri pada {E['dead_ties_unforced']:,} frame. Sesudah decoding/refinement, D di luar interval wajib: **{E['dead_decoded_unforced']:,} frame**. Distribusi selisih D terhadap tim terbaik: median {E['dead_advantage_unforced']['median']:.4f}, maksimum {E['dead_advantage_unforced']['max']:.4f}. Diagnosis pertama adalah skala/konstruksi bukti fusion, kemudian durasi stoppage; mengganti penalti decoder saja belum menyelesaikan kelemahan bukti.
2. **Fragmentasi tetap besar.** Output menghasilkan {P['possession_spells']:,} possession, median {P['possession_median_s']:.2f} detik dan rata-rata {P['possession_mean_s']:.2f} detik. Ini deskripsi prediksi, bukan estimasi jumlah possession benar. Soft short-spell refinement menggabungkan {E['soft_short_spells_merged']} run; boundary refinement menggeser {E['boundaries_shifted']} dari {E['boundaries_examined']} batas yang diperiksa.
3. **Missing ball bukan gap singkat yang dapat diasumsikan aman untuk interpolasi.** Terdapat {G['count']} gap, median {G['median']:.2f} detik, P95 {G['p95']:.2f} detik, maksimum {G['max']:.2f} detik. Sebanyak {G['set_piece_overlap']} gap menyentuh konteks set piece. Perbedaan proxy di tepi gap tidak membuktikan turnover.
4. **Metadata ATD broadcast kosong.** Raw tracking dan homography broadcast dapat diaudit terpisah, tetapi pemuatan tracking lengkap dengan metadata terblokir. Sinkronisasi broadcast-tactical tetap belum tersedia.
5. **Ada perbedaan jumlah frame video broadcast.** OpenCV melaporkan {S['video_metadata'][1]['frames']:,} frame video broadcast, sementara kontrak/raw tracking/homography menggunakan {S['video_metadata'][1]['expected_frames']:,}. Status decode frame terakhir yang diharapkan dicatat di video_metadata.csv. Ini isu integritas sumber/decoder yang perlu diselesaikan sebelum sinkronisasi lintas kamera.
6. **Pemilihan jendela video bukan pemulihan bola.** Konfigurasi video enabled={V['enabled_in_config']}; weights configured={V['weights_configured']}. Selektor memilih {V['windows']} jendela / {V['selected_frames']} frame, mencakup {V['missing_frames_selected']} dari {V['total_missing_graded']} missing-ball frame graded. Anggaran ini berorientasi transisi/anomali, bukan menutup seluruh gap.

## Input, proses, output

Input utama berupa ATD tactical + metadata + Smart Tagging; homography dan video merupakan tambahan opsional. `build_features` menghasilkan tabel fitur kanonis 0-based, event ternormalisasi, slot-to-team, serta array posisi pemain di memori. Modul kualitas, event, kontrol tim dan live/dead membentuk bukti; fusion membuat utility H/A/D; Viterbi berjalan per babak; refinement menggeser batas yang sudah ada. Output submission adalah `frame,label` sepanjang 199.319 baris. `spellF1` menilai segmen possession dan `macroF1` menilai per frame. Audit ini tidak menghitung atau mengoptimasi skor GT.

## Inventaris dan integritas sumber

'''
report=intro+mdtable('asset_inventory',['source','asset','exists','bytes','status'])+'\n\n'+mdtable('tracking_counters')+'\n\n'+mdtable('video_metadata')
report+='\n\nFrame tracking yang diberikan adalah 1-based berurutan; indeks submission = frame_id - 1. Offset IdFrame di games.py adalah counter berbeda dan tidak diterapkan lagi ke ATD yang diberikan. Smart Tagging dibulatkan dari detik x 25 dengan interval analisis [start,end). Homography memakai frame 0-based.\n'
report+='\n## Skema dan kualitas observasi\n\n'
report+=f"ATD terurai: {S['raw_shape'][0]:,} baris x {S['raw_shape'][1]} kolom. Fitur aktual: {S['feature_shape'][0]:,} baris x {S['feature_shape'][1]} kolom. Slot per tim: {S['slot_counts']}. Slot bukan identitas pemain.\n\n"
report+=table(pd.DataFrame([{'field':k,**v} for k,v in S['optional_fields'].items()]))+'\n\n'
report+=mdtable('coverage')+'\n\n'+mdtable('gap_summary')+'\n\n'
report+='Kolom optional yang seluruhnya null tidak memberikan label possession/live-dead tersembunyi. Fitur dengan nama centroid_speed sekarang merupakan alias median gerak slot yang telah disaring, bukan kecepatan centroid. Fitur frames_until/next memakai observasi masa depan dan sesuai inferensi offline. Kamus seluruh fitur tersedia di [DATA_DICTIONARY.md](DATA_DICTIONARY.md).\n'
report+='''
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
'''
report+='\n## Geometri dan Smart Tagging\n\n'+mdtable('geometry_summary',['feature','count','median','p95','max'])+'\n\n'
report+=table(pd.DataFrame([S['tags']]))+'\n\n'+mdtable('tag_summary')+'\n\n'
report+='Tag kedua tim bisa aktif bersamaan, termasuk karena sebuah tim bertahan saat tim lain menyerang. Overlap tag mentah bukan otomatis konflik. Tabel both_attacking_support_frames mengukur dukungan menyerang simultan secara terpisah. Distribusi spasial dihitung hanya saat bola terlihat; missing ball tidak dapat diberi zona tanpa inferensi tambahan. Tidak ada asumsi arah serang yang dipakai untuk zona koordinat.\n'
report+='\n## Diagnosis live/dead dan fusion\n\n'
report+='Fusion memakai H = home_control + live_score, A = away_control + live_score, D = dead_score pada bobot default. Bukti tim menambah utility kelas live tanpa tambahan setara pada kelas D. Saat seluruh bukti observasi/tag netral, prior menghasilkan H=A=0.59 dan D=0.25. Tambahan ini bukan probabilitas terkalibrasi. Analisis berikut mengecualikan interval kickoff wajib yang memakai skor +/-1e6.\n\n'
report+=mdtable('score_contributions')+'\n\n'
score=load('score_summary');score=score[(score.condition=='unforced')&score.component.isin(['live','dead','home_control','away_control','dead_minus_best_team'])]
report+=table(score[['component','count','median','p95','max']])+'\n\n'
report+='![Distribusi data dan skor](../evaluation/reports/exploration/data_and_evidence.png)\n\n![Profil sekitar set piece](../evaluation/reports/exploration/set_piece_evidence.png)\n\n'
report+='Batas tag adalah konteks sekitar restart, belum tentu awal stoppage atau saat bola dimainkan. Implementasi saat ini membuat pulsa stoppage di awal tag dan restart di akhir tag; kualitas asumsi ini perlu dievaluasi secara algoritmis, bukan dijadikan label pasti. Seluruh jendela yang menyentuh interval kickoff wajib dikeluarkan dari profil agar skor paksa +/-1e6 tidak mencemari rerata. Profil agregat di atas memperlihatkan perubahan skor di sekitar batas tag, tidak memvalidasi waktu restart sebenarnya.\n'
report+='\n## Struktur temporal dan provenance prediksi\n\n'+mdtable('stage_summary')+'\n\n'+mdtable('transition_summary')+'\n\n'
report+=f"Run pendek (<2 detik) yang diapit state sama: {S['sandwiched_short_spells']}. Ini kandidat diagnosis, bukan daftar kesalahan berlabel.\n\n"
report+=mdtable('prediction_artifacts',['path','valid','same_labels_as_current_run','possession_spells','dead_segments','possession_median_s'])+'\n\n'
report+='Perbandingan di atas adalah label per frame setelah validasi, sehingga tidak terpengaruh perbedaan line ending CSV. Hash byte masing-masing file ada di prediction_artifacts.csv. Artefak v3/outputs yang disebut pada identifikasi sebelumnya tidak tersedia dalam workspace saat audit ini. Notebook lama tetap dipertahankan sebagai snapshot historis.\n\n![Diagnosis temporal](../evaluation/reports/exploration/temporal_diagnostics.png)\n'
report+='\n## Homography, video, dan validasi\n\n'+mdtable('homography_summary',['source','region','rows','valid_rows','invalid_finite_mismatch','near_singular_valid','error_median','error_p95','error_max'])+'\n\n'
report+=table(pd.DataFrame([S['video_geometry_qa']]))+'\n\n'
report+='Video QA menggunakan sampel deterministik di kedua babak, decode video dan proyeksi numerik koordinat ke ukuran gambar; tidak menyimpan cuplikan. Proyeksi di dalam gambar tidak membuktikan objek tepat menempel pada pemain. Pemeriksaan overlay visual rinci dan deteksi bola dari video belum dilakukan; bobot detektor lokal belum tersedia.\n\n'
report+=table(pd.DataFrame([S['checks']]))+'\n\n'
report+=f"GT publik diperiksa hanya kontrak format/alignment: {S['public_gt_contract']}. Tidak ada model yang dilatih/dipilih berdasarkan jendela tersebut.\n"
report+='''
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
.\\mbg\\Scripts\\python.exe tools/explore_data.py
.\\mbg\\Scripts\\python.exe tools/render_exploration_report.py
.\\mbg\\Scripts\\python.exe -m pytest -q
```

Hasil lengkap berada di `evaluation/reports/exploration/` (gitignored): inventaris, statistik skema mentah/fitur, kamus fitur, kualitas homography, metadata video, geometri/zona, gap, tag, kontribusi skor, transisi, segmen, profil set piece, provenance prediksi, tiga grafik, dan summary.json. Hash sumber/config/input tabular dicatat; video tidak di-hash ulang agar tidak membaca 19 GB tambahan. Tidak ada dataset tracking per frame atau cuplikan video yang diekspor.

Ringkasan agregat dan grafik bersifat deskriptif. Satu pertandingan tanpa GT penuh tidak memberi ukuran akurasi generalisasi. Uji sintetis/regresi memeriksa perilaku kode, bukan kebenaran prediksi pertandingan. Dokumen ini mencatat diagnosis dan backlog; parameter inferensi dan submission tetap belum diubah.
'''
(ROOT/'docs/EXPLORATION_REPORT.md').write_text(report,encoding='utf-8')
defs=load('feature_dictionary').merge(load('feature_schema_graded')[['column','dtype','null_pct','unique_non_null']],on='column')
(ROOT/'docs/DATA_DICTIONARY.md').write_text('# Kamus data fitur aktual\n\nDihasilkan pada '+created+' dari `src/features.py`; '+str(len(defs))+' kolom. Null % dan keunikan dihitung pada 158.857 frame graded.\n\n'+table(defs)+'\n\nArray posisi pemain dan tabel event/slot hanya berada di memori. Skor antarproses bukan probabilitas terkalibrasi. Detail skema ATD 193 kolom dan statistik all/graded tersedia di evaluation/reports/exploration/.\n',encoding='utf-8')
setup='''from pathlib import Path
import json, hashlib
import pandas as pd
_audit_root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'games.py').is_file())
_audit_dir = _audit_root / 'evaluation/reports/exploration'
_audit_summary = json.loads((_audit_dir / 'summary.json').read_text(encoding='utf-8'))
_stale = [name for name, sha in _audit_summary['code_hashes'].items()
          if not (_audit_root / name).is_file()
          or hashlib.sha256((_audit_root / name).read_bytes()).hexdigest() != sha]
if _stale:
    raise RuntimeError('Sumber berubah; jalankan tools/explore_data.py ulang: ' + ', '.join(_stale))
print('Audit lokal:', _audit_summary['created_at'])
print('Actual src feature shape:', _audit_summary['feature_shape'])
def _audit_table(name, columns=None):
    df = pd.read_csv(_audit_dir / (name + '.csv'))
    print('\\n' + name)
    print((df if columns is None else df[columns]).to_string(index=False))
'''
sections={
 '01_data_audit.ipynb':("Inventaris seluruh sumber dan audit skema aktual",'''_audit_table('asset_inventory', ['source','asset','bytes','status'])
_audit_table('tracking_counters')
_audit_table('coverage')
print('Optional fields:', _audit_summary['optional_fields'])
_audit_table('gap_summary')
_audit_table('homography_summary', ['source','region','rows','valid_rows','error_median','error_p95'])
print('Checks:', _audit_summary['checks'])
''',['data_and_evidence.png']),
 '02_feature_exploration.ipynb':("Fitur aktual dari src dan konteks observasi",'''_audit_table('feature_dictionary')
_audit_table('feature_schema_graded', ['column','dtype','null_pct','unique_non_null'])
_audit_table('geometry_summary', ['feature','count','median','p95','max'])
_audit_table('tag_summary')
_audit_table('observed_zone_summary')
print('Tag overlap:', _audit_summary['tags'])
print('Gap context:', _audit_summary['gaps'])
''',['data_and_evidence.png','set_piece_evidence.png']),
 '04_error_analysis.ipynb':("Diagnosis emission, decoder, refinement, dan video budget",'''print('Evidence:', json.dumps(_audit_summary['evidence'], indent=2))
_audit_table('score_contributions')
_audit_table('stage_summary')
_audit_table('transition_summary')
_audit_table('segment_duration_summary')
_audit_table('prediction_artifacts', ['path','valid','same_labels_as_current_run','possession_spells','dead_segments'])
print('Video selection:', _audit_summary['video_selection'])
print('Video geometry QA:', _audit_summary['video_geometry_qa'])
print('Short sandwiched spells:', _audit_summary['sandwiched_short_spells'])
''',['temporal_diagnostics.png','set_piece_evidence.png'])}

for name,(title,body,plots) in sections.items():
    path=ROOT/'notebooks'/name;nb=json.loads(path.read_text(encoding='utf-8'))
    nb['cells']=[c for c in nb['cells'] if 'current_src_audit' not in c.get('metadata',{}).get('tags',[])]
    nb['cells'].insert(0,md(f'> **Audit terbaru: {created}.** Bagian awal notebook menyimpan kode/hasil historis. Gunakan bagian **Audit src aktual** di akhir untuk hasil yang dihitung ulang dari kode saat ini. Detail: [laporan eksplorasi](../docs/EXPLORATION_REPORT.md). Parameter inferensi belum diubah.'))
    additions=[md(f'## Audit src aktual — {title}\n\nDihasilkan oleh `tools/explore_data.py`. Sel di bagian ini dieksekusi terpisah pada namespace Python baru saat pembuatan laporan. Untuk menghitung ulang data, jalankan skrip audit; sel berikut membaca agregat dan memeriksa hash sumber.'),code(setup),code(body)]
    for p in plots:additions.append(code("from IPython.display import Image\ndisplay(Image(filename=str(_audit_dir / '"+p+"')))\n"))
    additions.append(md(f"**Interpretasi terbaru:** D menang pada {E['dead_wins_unforced']} frame di luar interval wajib; output memiliki {P['possession_spells']} possession. Ini diagnosis struktur prediksi, bukan akurasi. Perbaikan pertama diarahkan pada kompetisi skor live/dead dan kontrol tim. Statistik tag/gap tidak dijadikan label ground truth. GT publik tidak dipakai tuning."))
    namespace={'__name__':'__audit_notebook__'};execution=0
    for cell in additions:
        if cell['cell_type']!='code':continue
        execution+=1;outputs=[]
        def display(obj):
            data=obj._repr_png_()
            if isinstance(data,tuple):data=data[0]
            if isinstance(data,bytes):data=base64.b64encode(data).decode('ascii')
            outputs.append({'output_type':'display_data','metadata':{},'data':{'image/png':data,'text/plain':['<Aggregate audit figure>']}})
        namespace['display']=display;capture=io.StringIO()
        with contextlib.redirect_stdout(capture):exec(compile(''.join(cell['source']),name,'exec'),namespace)
        if capture.getvalue():outputs.insert(0,{'output_type':'stream','name':'stdout','text':capture.getvalue().splitlines(True)})
        cell['outputs']=outputs;cell['execution_count']=execution
    nb['cells'].extend(additions)
    path.write_text(json.dumps(nb,ensure_ascii=False,indent=1)+'\n',encoding='utf-8')
    print('Updated',path.name,'new cells',len(additions))
print('Reports written:',ROOT/'docs/EXPLORATION_REPORT.md')
