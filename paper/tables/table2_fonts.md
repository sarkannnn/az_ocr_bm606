# Table 2: Font Inventory, Training Provenance, and Taxonomic Classification

| Font Filename | Family Name | Style / Weight | In V1 Train? | In V2 Train? | Audit Classification | Reason / Status | Source File |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| `arial.ttf` | Arial | Regular | **YES** | **YES** | Group A: SEEN | V1 base training font | `fonts/arial.ttf` |
| `calibri.ttf` | Calibri | Regular | **YES** | **YES** | Group A: SEEN | V1 base training font | `fonts/calibri.ttf` |
| `candara.ttf` | Candara | Regular | **YES** | **YES** | Group A: SEEN | V1 base training font | `fonts/candara.ttf` |
| `corbel.ttf` | Corbel | Regular | **YES** | **YES** | Group A: SEEN | V1 base training font | `fonts/corbel.ttf` |
| `cour.ttf` | Courier New | Regular | **YES** | **YES** | Group A: SEEN | V1 base training font | `fonts/cour.ttf` |
| `segoeui.ttf` | Segoe UI | Regular | **YES** | **YES** | Group A: SEEN | V1 base training font | `fonts/segoeui.ttf` |
| `times.ttf` | Times New Roman | Regular | **YES** | **YES** | Group A: SEEN | V1 base training font | `fonts/times.ttf` |
| `verdana.ttf` | Verdana | Regular | **YES** | **YES** | Group A: SEEN | V1 base training font | `fonts/verdana.ttf` |
| `tahoma.ttf` | Tahoma | Regular | NO | NO | Group C: NEW FAMILY | Reserved for V1/V2 test set (20% held-out) | `fonts/tahoma.ttf` |
| `consola.ttf` | Consolas | Regular | NO | NO | Group B: FAMILY LEAKAGE | Test font, but bold sibling `consolab` in V2 | `fonts/consola.ttf` |
| `CascadiaCode.ttf` | Cascadia Code | Monospace | NO | **YES** | V2 Train Addition | Added in V2 (+15k words/+6k lines) | `fonts_new/CascadiaCode.ttf` |
| `bahnschrift.ttf` | Bahnschrift | Sans-serif | NO | **YES** | V2 Train Addition | Added in V2 | `fonts_new/bahnschrift.ttf` |
| `consolab.ttf` | Consolas | Bold | NO | **YES** | V2 Train Addition | Causes leakage into `consola.ttf` test set | `fonts_new/consolab.ttf` |
| `ebrima.ttf` | Ebrima | Sans-serif | NO | **YES** | V2 Train Addition | Added in V2 | `fonts_new/ebrima.ttf` |
| `l_10646.ttf` | Lucida Sans Unicode| Sans-serif | NO | **YES** | V2 Train Addition | Added in V2 | `fonts_new/l_10646.ttf` |
| `palab.ttf` | Palatino Linotype | Bold | NO | **YES** | V2 Train Addition | Causes leakage into `pala.ttf` ladder test | `fonts_new/palab.ttf` |
| `calibrib.ttf` | Calibri | Bold | NO | NO | Group B: NEAR-RELATIVE | Weight sibling of `calibri.ttf` | `C:/Windows/Fonts/calibrib.ttf` |
| `seguisb.ttf` | Segoe UI | Semi-Bold | NO | NO | Group B: NEAR-RELATIVE | Weight sibling of `segoeui.ttf` | `C:/Windows/Fonts/seguisb.ttf` |
| `verdanab.ttf` | Verdana | Bold | NO | NO | Group B: NEAR-RELATIVE | Weight sibling of `verdana.ttf` | `C:/Windows/Fonts/verdanab.ttf` |
| `CascadiaMono.ttf` | Cascadia Mono | Monospace | NO | NO | Group B: NEAR-RELATIVE | Letter glyph sibling of `CascadiaCode` | `C:/Windows/Fonts/CascadiaMono.ttf` |
| `pala.ttf` | Palatino Linotype | Regular | NO | NO | Group B: NEAR-RELATIVE | Weight sibling of `palab.ttf` | `C:/Windows/Fonts/pala.ttf` |
| `cambria.ttc` | Cambria | Regular | NO | NO | Group C: NEW FAMILY | Unseen serif family | `C:/Windows/Fonts/cambria.ttc` |
| `micross.ttf` | MS Sans Serif | Sans-serif | NO | NO | Group C: NEW FAMILY | Unseen sans-serif family | `C:/Windows/Fonts/micross.ttf` |
| `Moba_Fira.ttf` | Moba Fira | Monospace | NO | NO | Group C: NEW FAMILY | Unseen monospace family | `fonts/Moba_Fira.ttf` |
| `Moba_Font.ttf` | Moba Font | Monospace | NO | NO | Group C: NEW FAMILY | Unseen monospace family | `fonts/Moba_Font.ttf` |
| `Moba_Fanta.ttf` | Moba Fanta | Sans-serif | NO | NO | Group C: NEW FAMILY | Unseen decorative sans family | `fonts/Moba_Fanta.ttf` |
| `tahomabd.ttf` | Tahoma | Bold | NO | NO | Group C: NEW FAMILY | Bold variant of unseen Tahoma family | `C:/Windows/Fonts/tahomabd.ttf` |
| `SansSerifCollection.ttf`| Sans Serif Coll. | Sans-serif | NO | NO | Group C: NEW FAMILY | Unseen sans family (subject to bbox fix) | `C:/Windows/Fonts/SansSerifCollection.ttf` |
| `Inkfree.ttf` | Ink Free | Casual Script | NO | NO | CASUAL SCRIPT | Excluded from standard-print benchmarks | `C:/Windows/Fonts/Inkfree.ttf` |
| `georgia.ttf` | Georgia | Serif | NO | NO | **REJECTED** | Missing Azerbaijani diacritics: ə, Ə | `generate_data.py:font_ok` |
| `trebuc.ttf` | Trebuchet MS | Sans-serif | NO | NO | **REJECTED** | Missing Azerbaijani diacritics: ə, Ə | `generate_data.py:font_ok` |
| `constan.ttf` | Constantia | Serif | NO | NO | **REJECTED** | Missing Azerbaijani diacritics: ə, Ə | `generate_data.py:font_ok` |
| `framd.ttf` | Franklin Gothic | Sans-serif | NO | NO | **REJECTED** | Missing Azerbaijani diacritics: ə, Ə | `generate_data.py:font_ok` |
| `lucon.ttf` | Lucida Console | Monospace | NO | NO | **REJECTED** | Missing Azerbaijani diacritics: ə, Ə | `generate_data.py:font_ok` |

Source: generate_data.py:18-36, results/v2_audit/summary.md:8-25, results/v2_audit3/summary.md:40-62, results/v2_audit4/summary.md:53-66
