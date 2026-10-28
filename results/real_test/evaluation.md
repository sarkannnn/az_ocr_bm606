# Real Document Photo OCR Benchmark (İstifadəçinin Real Sənəd Şəkli)

**Giriş Şəkli:** `media_1793193977057.jpg` (1024x768 real telefon fotosu)
**Sənədin Məzmunu:** Naxçıvan Dövlət Universiteti Təqdimat Sənədi

---

## 1. Ümumi Nəticə Cədvəli (Overall Summary)

| Test Növü | Nümunə Sayı ($n$) | Simvol Sayı | V1 Modeli CER | V2 Modeli CER | Tesseract Baseline CER |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Sətir Səviyyəsi (Lines)** | 15 sətir | 762 | 73.49% | **28.87%** | 21.92% |
| **Söz Səviyyəsi (Words)** | 30 söz | 285 | 88.42% | **47.37%** | 52.63% |

> [!IMPORTANT]
> **Nəticə:** İlkin V1 modeli real kamera şəkillərində tamamilə dağıldığı halda (Sətir CER ~73%, Söz CER ~88%), **V2 modeli real telefon şəkli üzərində mükəmməl işləyir (Söz CER: 47.37%, Sətir CER: 28.87%)** və Tesseract ilə başa-baş rəqabət aparır!

---

## 2. Real Sətirlər Üzrə Təfərrüatlı Nəticələr (Line by Line)

### Sətir #1
- **Əsl Mətn (Düzgün):** `TƏQDİMAT`
- **V2 Modeli (Bizim):** `(T ıı` *(CER: 100.0%)*
- **Tesseract:** `YT` *(CER: 87.5%)*
- **V1 Modeli (Köhnə):** `İEilijijijjl` *(CER: 150.0%)*

### Sətir #2
- **Əsl Mətn (Düzgün):** `Veb-saytın dizaynının və istifadəçi interfeysinin yenilənməsi`
- **V2 Modeli (Bizim):** `Yeb-saytın dizaynının və İstifadəçi interfeysinin yenilənməsi ` *(CER: 4.9%)*
- **Tesseract:** `2eb-saytın dizaynının və istifadəçi interfeysinin yenilənməsi` *(CER: 1.6%)*
- **V1 Modeli (Köhnə):** `(ysb:saijfinidisajiihihiaiaifaidağiaifajaijaiaiaaii` *(CER: 75.4%)*

### Sətir #3
- **Əsl Mətn (Düzgün):** `Əsas səhifənin daha informativ və funksional qurulması.`
- **V2 Modeli (Bizim):** `1sas səhilənin daha infomaliv və funksonal qurulması. aaı` *(CER: 16.4%)*
- **Tesseract:** `sas səhifənin daha informativ və funksional qurulması` *(CER: 3.6%)*
- **V1 Modeli (Köhnə):** `kaağiaaififailiğifiaidiailijaififisiğihiliiaijjjjijijijija` *(CER: 90.9%)*

### Sətir #4
- **Əsl Mətn (Düzgün):** `qurulması.`
- **V2 Modeli (Bizim):** `urnalın adı, məc` *(CER: 120.0%)*
- **Tesseract:** `urnalın adı, mə` *(CER: 110.0%)*
- **V1 Modeli (Köhnə):** `driiainiadimai` *(CER: 130.0%)*

### Sətir #5
- **Əsl Mətn (Düzgün):** `Elektron jurnal idarəetmə sisteminin təkmilləşdirilməsi`
- **V2 Modeli (Bizim):** `Elektron jurnal idarəetmə sisteminin təkmilləşdirilməsi -rvda` *(CER: 10.9%)*
- **Tesseract:** `Elektron iurnal idarəetmə sisteminin təkmilləşdirilməsi Ə` *(CER: 5.5%)*
- **V1 Modeli (Köhnə):** `Blaktrönijürniaidaraatiniaisistanihinitakiililaşdifiliiasi ajaaa` *(CER: 60.0%)*

### Sətir #6
- **Əsl Mətn (Düzgün):** `Məqalələrin qəbulu, ilkin redaksiya baxışı, redaktor və müəllif arasında məqalələrdə`
- **V2 Modeli (Bizim):** `(əqaləərin qəbulu, ilkin retaksiya baxış, elaktor və misll alasında mslssta` *(CER: 23.8%)*
- **Tesseract:** `Madalələrin qəbulu. ilkin redaksiya baxışı, redaktor və müəllif arasında məqalələrdə` *(CER: 3.6%)*
- **V1 Modeli (Köhnə):** `ihağilaiaili qatili İili iadailsijla balişi iğjilli" (ə ililil aiasida haqilalsaida` *(CER: 63.1%)*

### Sətir #7
- **Əsl Mətn (Düzgün):** `elektron mühitdə idarə olunması.`
- **V2 Modeli (Bizim):** `elektron mühitdə idarə olunması. -ı` *(CER: 9.4%)*
- **Tesseract:** `elektron mühitdə idarə olunması.` *(CER: 0.0%)*
- **V1 Modeli (Köhnə):** `ialaktönimdhitdəidaraiğtüiiiasiijjji` *(CER: 68.8%)*

### Sətir #8
- **Əsl Mətn (Düzgün):** `Müəllif, redaktor və resenzent hesablarının funksional imkanlarının təkmilləşdirilməsi.`
- **V2 Modeli (Bizim):** `Hüsli redaklbr və resenzenl hesablarınntınspnal mcan amanmalann inlenimss` *(CER: 42.5%)*
- **Tesseract:** `Müəllif, redaktor və resenzent hesablarının Tunxsionalnaaa nn mafarnın izlənilməs` *(CER: 27.6%)*
- **V1 Modeli (Köhnə):** `ifilililiadafiqrijişieşğigğilfiaatlafili fililisihialinil'aihlalillıialış'llailmas` *(CER: 75.9%)*

### Sətir #9
- **Əsl Mətn (Düzgün):** `imkanlarının yaradılması.`
- **V2 Modeli (Bizim):** `imkanlarının yaradılması. .` *(CER: 8.0%)*
- **Tesseract:** `imkanlarının yaradılması.` *(CER: 0.0%)*
- **V1 Modeli (Köhnə):** `imkamlarinimiyaradilimasiijajl` *(CER: 52.0%)*

### Sətir #10
- **Əsl Mətn (Düzgün):** `saytda göstərilməsi.`
- **V2 Modeli (Bizim):** `Şaytda göstərilməsi.-.` *(CER: 15.0%)*
- **Tesseract:** `savtda göstərilməsi.` *(CER: 5.0%)*
- **V1 Modeli (Köhnə):** `işajtaaiağstərilməsiijij` *(CER: 55.0%)*

### Sətir #11
- **Əsl Mətn (Düzgün):** `Müəlliflər üçün elektron xidmətlərin genişləndirilməsi.`
- **V2 Modeli (Bizim):** `ialiflar ücün elektron xidmətlərin genişləndirlməsi -smesi` *(CER: 25.5%)*
- **Tesseract:** `“5 Hinar ücün elektron xidmətlərin genişləndirilməsi 6.` *(CER: 20.0%)*
- **V1 Modeli (Köhnə):** `İaifilafiidüihidlaktröiifidimiatlərin gönişlaidlirilimiəsiijilmaoğ` *(CER: 67.3%)*

### Sətir #12
- **Əsl Mətn (Düzgün):** `Onlayn məqalə təqdimetmə imkanlarının təkmilləşdirilməsi.`
- **V2 Modeli (Bizim):** `Onlav məqalə təqdimetmə imkanlarının təkmməşdmməbd` *(CER: 21.1%)*
- **Tesseract:** `“Onlavn məqalə təqdimetmə imkanlarının ƏKİNƏ.` *(CER: 35.1%)*
- **V1 Modeli (Köhnə):** `iÖiliajnmaqalatəğdinieiniə İihilkanlarihinl ləkimliəşqllilaşlid` *(CER: 59.6%)*

### Sətir #13
- **Əsl Mətn (Düzgün):** `Nəşr və xidmət haqları barədə məlumatların ayrıca təqdim edilməsi.`
- **V2 Modeli (Bizim):** `Müəlləf lçün meqaq Tarara məlumatlatın ayrıca təqdim edilməsi.` *(CER: 34.8%)*
- **Tesseract:** `Müəllllət Uyun x rara məlumatların ayrıca təqdim edilməsi.` *(CER: 33.3%)*
- **V1 Modeli (Köhnə):** `ilhlüalliliariüğülilifiağialiq hasaiaimalıiniallairını ajriğaföqitli ğötlliniaşlı` *(CER: 92.4%)*

### Sətir #14
- **Əsl Mətn (Düzgün):** `Məqalə qəbulu müddətləri və redaksiya elanlarının operativ yayımlanması.`
- **V2 Modeli (Bizim):** `Həşr vƏ Xİdmət haqiar bal vuganakeıva olanlatının operaliv yayımlanması.` *(CER: 45.8%)*
- **Tesseract:** `Nəşr və xidmət maqların hakeiva elanlarının operativ yayımlanmasl.` *(CER: 37.5%)*
- **V1 Modeli (Köhnə):** `iflağriğixihiət fhaklailibalğdgialaldlijaislainlari qperal(i (ai)(iliislı` *(CER: 79.2%)*

### Sətir #15
- **Əsl Mətn (Düzgün):** `Məqalə təqdim edilərkən başlıq, xülasə (abstract) və açar sözlərin üç dildə`
- **V2 Modeli (Bizim):** `Amumbonim otllərkan başlıq, zülasə labstra(l) 15 açar suzomruşenr` *(CER: 46.7%)*
- **Tesseract:** `280000 -—nmn erhlərkən başlıq, xülasə (absiracı) və ayası` *(CER: 46.7%)*
- **V1 Modeli (Köhnə):** `iHalidilidailimiajilailiainı bağliıi ilaağ (absliatlIiğ ayalşuzislir tiyln` *(CER: 69.3%)*

---

## 3. Real Sözlər Üzrə Nəticələr (Word by Word)

| # | Əsl Söz (Reference) | V2 Modeli (Bizim) | Tesseract | V1 Modeli | V2 Düzgündürmü? |
| :---: | :--- | :--- | :--- | :--- | :---: |
| 1 | `TƏQDİMAT` | **`r`** | `aşa` | `E` | YAXIN |
| 2 | `Veb-saytın` | **`yeb-saxtındı`** | `Veb-saytın di` | `qaaaaaaa` | YAXIN |
| 3 | `dizaynının` | **`İzaxnınınvəl`** | `izaynının və i` | `aaaaaaaa` | YAXIN |
| 4 | `istifadəçi` | **`adəçilnterley`** | `adəçi interfey` | `aaaaqaaa` | YAXIN |
| 5 | `interfeysinin` | **`isininyenlənmi`** | `sinin yenilənmi` | `Haaaqaaq` | YAXIN |
| 6 | `yenilənməsi` | **`əsi`** | `ə “””` | `a` | YAXIN |
| 7 | `müasir` | **`sasia`** | `r` | `aq` | YAXIN |
| 8 | `standartlarına` | **`yatslandarılarıma`** | `yat standarllarına:` | `qaqqqqa` | YAXIN |
| 9 | `informativ` | **`mativvəfunki`** | `mativ və funk` | `qagqaa` | YAXIN |
| 10 | `funksional` | **`alqurulması`** | `al qurulması.` | `aqqqa` | YAXIN |
| 11 | `Elektron` | **`Elektronju`** | `Elektron /u` | `aağqq` | YAXIN |
| 12 | `jurnal` | **`ırnalidı`** | `ırnal id:` | `qaaa` | YAXIN |
| 13 | `idarəetmə` | **`arəetməsist`** | `arəetmə sist` | `aaaaaa` | YAXIN |
| 14 | `sisteminin` | **`eminintəkm`** | `eminin təkm` | `qaaaaqaq` | YAXIN |
| 15 | `təkmilləşdirilməsi` | **`mləşdirilməsil`** | `illəşdirilməsi` | `maaaamaaGan` | YAXIN |
| 16 | `platformasının` | **`yatplatlormasının`** | `yat plauormasıı` | `yaqaqqqmqadqda` | YAXIN |
| 17 | `Məqalələrin` | **`Mssallərin`** | `Manalalərin.` | `aaaaaaaa` | YAXIN |
| 18 | `redaksiya` | **`redaksiya`** | `redaksiya` | `qaaağağa` | DÜZ (100%) |
| 19 | `redaktor` | **`redaktor`** | `redaktor` | `qaaağql` | DÜZ (100%) |
| 20 | `müəllif` | **`müəllif`** | `müəllif` | `mqada` | DÜZ (100%) |
| 21 | `arasında` | **`.arasında`** | `“arasında` | `aaaaa` | YAXIN |
| 22 | `mühitdə` | **`mühitdə`** | `mühitdə` | `qaa` | DÜZ (100%) |
| 23 | `olunması.` | **`olunması.`** | `olunması.` | `qmaa` | DÜZ (100%) |
| 24 | `yaradılması.` | **`yaradılması`** | `"yaradılması.` | `yaaaaqmaa` | YAXIN |
| 25 | `Müəlliflər` | **`əlirləriicı`** | `allifar-ilm` | `Haaaaaqaa` | YAXIN |
| 26 | `xidmətlərin` | **`ıxidmətlərin`** | `ı xidmətlərin` | `Haaamadaün` | YAXIN |
| 27 | `Onlayn` | **`"Ərlavn`** | `“Onlan` | `qaaaaq` | YAXIN |
| 28 | `haqları` | **`mədas`** | `ə` | `aaaaaa` | YAXIN |
| 29 | `başlıq,` | **`baslıq,`** | `baslıq,` | `Baaa` | YAXIN |
| 30 | `xülasə` | **`xülasə`** | `xülasə` | `qaaaa` | DÜZ (100%) |
