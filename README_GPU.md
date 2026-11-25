# AzOCR-V2: NVIDIA GPU Təlim Paketi Bələdçisi (README_GPU.md)

Bu paket **AzOCR-V2** layihəsinin CRNN + CTC modellərini NVIDIA GPU dəstəkli başqa bir kompüterdə müstəqil şəkildə öyrətmək üçün hazırlanmışdır.

Paket 100% daşına biləndir (portable): bütün təmizlənmiş təlim dəstləri, model arxitekturası, CUDA skriptləri və konfiqurasiyalar daxil edilmişdir. Heç bir xarici verilənlər bazasına və ya font lisenziyasına ehtiyac yoxdur.

---

## 1. Sistem Tələbləri

1. **Əməliyyat Sistemi:** Windows 10 və ya Windows 11 (64-bit)  
2. **Qrafik Kartı (NVIDIA GPU):** NVIDIA GeForce RTX seriyası (RTX 3060, 3070, 3080, 4060, 4070, 4080, 4090 və s.) və ya GTX 1660 / Ti (Minimum 4 GB VRAM).
3. **NVIDIA Sürücüsü:** Yenilənmiş NVIDIA Driver (versiya $\ge 525$).
4. **Python:** Python 3.10 və ya Python 3.11 quraşdırılmış olmalı və quraşdırma zamanı *"Add Python to PATH"* seçilməlidir.

---

## 2. Addım-addım Quraşdırma və Təlim Bələdçisi

### Addım 1: Arxivin Açılması və Mühitin Qurulması
1. `AzOCR_GPU_Training_Package.zip` arxivini kompüterinizdə rahat bir yerə çıxarın (məsələn, `C:\AzOCR_GPU` və ya `D:\AzOCR_GPU`).
2. Qovluğun daxilindəki **`setup_env.bat`** faylını iki dəfə klikləyərək işə salın.
   - Bu skript avtomatik olaraq `.venv_gpu` virtual mühitini yaradacaq.
   - NVIDIA GPU dəstəkli PyTorch (CUDA 12.1) və zəruri kitabxanaları (`torch`, `torchvision`, `opencv`, `numpy`, `pyyaml`) quraşdıracaq.
   - Sonda GPU modelini ekrana çıxararaq əlaqəni yoxlayacaq.

---

### Addım 2: CUDA Diaqnostikası və 1-Epoxluq Smoke Test
Tam təlimlərə başlamazdan əvvəl GPU-nun, CUDA tensolarının və yaddaş qeydinin problemsiz işlədiyini yoxlamaq üçün:
- **`run_smoke_test.bat`** faylını iki dəfə klikləyərək işə salın.
- Bu skript:
  1. CUDA hardware diaqnostikasını icra edir (GPU modeli, VRAM həcmi, CUDA versiyası və $2048 \times 2048$ tensorda matmul sürət testi).
  2. 1 epoxluq kiçik bir alt-dəst üzərində sınaq təlimi keçirir (~15–25 saniyə).
- Ekranda `[+] TƏBRİKLƏR! Smoke test uğurla tamamlandı` yazısını gördükdə hər şeyin tam hazır olduğuna əmin ola bilərsiniz.

---

## 3. 1 Epox Ölçümündən Ümumi Təlim Müddətinin Hesablanması (ETA Düsturu)

Smoke test zamanı ekranda hər epoxun neçə saniyə çəkdiyi göstərilir:
`Epoch 1/1 (X.X s) | Train Loss: ...`

Buradan istifadə edərək tam təlimlərin nə qədər vaxt aparacağını asanlıqla hesablaya bilərsiniz:

$$\text{Ümumi Müddət (dəqiqə)} = \frac{T_{\text{epox (saniyə)}} \times N_{\text{epox}} \times N_{\text{model}}}{60}$$

Burada:
- $N_{\text{epox}}$: Erkən dayandırma (early stopping) ilə orta epox sayı $\approx 25$ epox.
- $N_{\text{model}}$: 12 model (əsas 2x2 matrisi) və ya 14 model (əlavə padding və variable-width ilə).

### Nümunəvi Təxminlər (GPU Kateqoriyalarına Görə):
| GPU Modeli | 1 Epox Müddəti ($T_{\text{epox}}$) | 1 Modelin Müddəti (~25 epox) | 12 Modelin Cəmi Müddəti | 14 Modelin Cəmi Müddəti |
| :--- | :---: | :---: | :---: | :---: |
| **NVIDIA RTX 4090 / 4080** | ~3 – 5 saniyə | ~1.5 – 2 dəqiqə | **~20 – 25 dəqiqə** | **~25 – 30 dəqiqə** |
| **NVIDIA RTX 4070 / 3080** | ~6 – 8 saniyə | ~2.5 – 3.5 dəqiqə | **~35 – 45 dəqiqə** | **~40 – 50 dəqiqə** |
| **NVIDIA RTX 3060 / 4060** | ~9 – 12 saniyə | ~4 – 5 dəqiqə | **~50 – 65 dəqiqə** | **~60 – 75 dəqiqə** |
| **GTX 1660 Ti / RTX 2060** | ~14 – 18 saniyə | ~6 – 7.5 dəqiqə | **~75 – 90 dəqiqə** | **~90 – 105 dəqiqə** |

---

## 4. Tam Təlimlərin İcrası

Smoke test uğurla bitdikdən sonra:
- **`run_full_training.bat`** faylını iki dəfə klikləyərək işə salın.
- Sizə 3 seçim təqdim olunacaq:
  - **`[1]` 12 Əsas Ablasiya Eksperimenti (Tövsiyə Edilən Əsas):**
    - Qrup A (V1 10k Clean, Basic Aug) x Seeds 42, 43, 44
    - Qrup B (V1 10k Clean, Photo Aug) x Seeds 42, 43, 44
    - Qrup C (V2 15.7k Clean, Basic Aug) x Seeds 42, 43, 44
    - Qrup D (V2 15.7k Clean, Photo Aug) x Seeds 42, 43, 44
  - **`[2]` Bütün 14 Eksperiment:**
    - 12 Əsas model + `Exp_RandomPadding` (0-12 px) + `Exp_VariableWidth` (dinamik kətan)
  - **`[3]` Yalnız Əlavə Eksperimentlər:**
    - `Exp_RandomPadding` və `Exp_VariableWidth`
- İstədiyiniz rəqəmi (`1` və ya `2`) daxil edib Enter basın.
- Təlim tam avtomatlaşdırılmışdır; hər epox və model bitdikdə nəticələr avtomatik qeyd olunur.

---

## 5. GPU Vəziyyətinin Monitorinqi
Təlim gedərkən GPU temperaturunu və VRAM istifadəsini izləmək üçün ayrıca bir PowerShell pəncərəsində bu əmri işə sala bilərsiniz:
```powershell
nvidia-smi -l 2
```

---

## 6. Nəticələrin Təhvil Verilməsi
Bütün təlimlər yekunlaşdıqda layihə daxilində `runs/` qovluğunda bütün nəticələr və `runs/experiments_summary.md` müqayisə cədvəli yaranacaq.

Hər model qovluğunda:
- `best.pt` (Ən yaxşı model çəkisi)
- `last.pt` (Son epox çəkisi)
- `train_log.jsonl` (Hər epoxun CER, loss, müddət qeydləri)
- `run_summary.json` (GPU məlumatı, VRAM, seed, ümumi vaxt)
- `config.yaml` (Konfiqurasiya)

Zəhmət olmasa:
1. `runs/` qovluğunu zip faylına sıxın (`AzOCR_Trained_Runs_Results.zip`).
2. Həmin zip arxivini bizə geri göndərin. Ətraflı məlumat üçün **`IMPORT_RESULTS.md`** faylına baxa bilərsiniz.

Dəstəyiniz və elmi əməkdaşlığınız üçün təşəkkür edirik!
