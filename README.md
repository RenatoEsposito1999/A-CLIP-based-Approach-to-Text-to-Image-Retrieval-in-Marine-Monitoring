# 🌊 A CLIP-based Approach to Text-to-Image Retrieval in Marine Monitoring 🐢

Tesi di Laurea Magistrale in **Applied Computer Science (Machine Learning and Big Data)**
[cite_start]presentata presso l'**Università degli Studi di Napoli "Parthenope"** [cite: 1, 3]
[cite_start]Dipartimento di Scienza e Tecnologia [cite: 2]

[cite_start]**Candidato:** Renato Esposito (Matricola: 0120000278) [cite: 7, 9]
[cite_start]**Relatore:** Prof. Antonino Staiano [cite: 11]
[cite_start]**Correlatore:** Dr. Alessio Pierluigi Placitelli [cite: 16]
[cite_start]**Esaminatore:** Prof. Francesco Camastra [cite: 13]
[cite_start]**Anno Accademico:** 2024-2025 [cite: 14]

---

## 💡 Introduzione e Obiettivi del Progetto

[cite_start]Questo progetto di tesi si concentra sull'applicazione di tecniche di Intelligenza Artificiale (AI) per supportare il monitoraggio della biodiversità marina, in linea con il **14° Obiettivo di Sviluppo Sostenibile (SDG) dell'Agenda 2030 dell'ONU**: "**Life below water**" ("Conservare e utilizzare in modo durevole gli oceani, i mari e le risorse marine per uno sviluppo sostenibile")[cite: 24, 28, 29].

[cite_start]L'obiettivo principale è stato la **progettazione e l'implementazione di un sistema di *Text-to-Image Retrieval*** specializzato nel dominio delle **tartarughe marine**[cite: 35, 41].

### 🎯 Obiettivi Specifici
1. [cite_start]**Allineare le rappresentazioni visuali e testuali** in uno spazio di *embedding* condiviso[cite: 42].
2. [cite_start]Consentire l'**interrogazione di dataset non strutturati tramite descrizioni testuali in linguaggio naturale** (es. *“tartaruga con una rete sul guscio”*) per recuperare le immagini più pertinenti[cite: 43].
3. [cite_start]Affrontare la sfida critica e complessa di **monitorare la presenza di tartarughe** e distinguerle da elementi potenzialmente dannosi (*distractor*), come detriti marini o altri animali[cite: 36, 37].

---

## 🛠 Metodo Proposto: Fine-Tuning di CLIP

[cite_start]Il sistema è basato su **CLIP (Contrastive Language–Image Pretraining)** di OpenAI (2021), un modello che apprende rappresentazioni congiunte visuali e testuali in uno spazio semantico comune[cite: 45, 46].

### 1. Architettura CLIP
CLIP utilizza un'architettura basata su:
* [cite_start]**Image Encoder (ViT-B/32):** L'immagine viene divisa in *patch* e un *token* speciale `[CLS]` ne rappresenta la *feature* globale[cite: 53, 54, 57].
* [cite_start]**Text Encoder (Standard Transformer Encoder):** Utilizza l'output del *token* finale (es. `[EOS]`) come rappresentazione globale del testo[cite: 59, 62].
* [cite_start]Le rappresentazioni di testo e immagine sono proiettate in uno **spazio di *embedding* comune** e normalizzate[cite: 64, 65].

### 2. Strategia di Loss Combinata
Per specializzare il modello, pur mantenendo le sue capacità di generalizzazione, è stata adottata una strategia di perdita combinata:
[cite_start]$$L_{Loss} = \frac{L_{BiC} + L_{CLIP}}{2}$$ [cite: 82]

* [cite_start]**$L_{CLIP}$ (InfoNCE-based Symmetric Cross-Entropy Loss):** Mira a massimizzare la similarità del coseno tra le coppie testo-immagine allineate e a minimizzarla per quelle non corrispondenti[cite: 66, 68]. [cite_start]È bidirezionale, consentendo il recupero di immagini da testo e viceversa[cite: 73].
* [cite_start]**$L_{BiC}$ (Bilinear Unified Contrastive Learning Loss):** Derivata da Unified Contrastive Learning, estende la perdita contrastiva al **caso *multi-positive***, permettendo più di un campione corrispondente per ancore (*anchor*)[cite: 78, 79, 80]. [cite_start]Questo aiuta a mantenere la separazione tra le categorie e a preservare la variabilità intra-classe delle tartarughe[cite: 83].

### 3. Fine-Tuning Efficace (LoRA)
[cite_start]Per un *fine-tuning* efficiente del modello pre-addestrato, è stata utilizzata la tecnica **LoRA (Low-Rank Adaptation)**[cite: 84]. LoRA aggiunge pochi pesi allenabili congelando il resto del modello, rendendo CLIP adattabile senza costi computazionali proibitivi. [cite_start]È stata applicata ai *layer* di attenzione degli encoder di testo e visuale[cite: 85, 86].

---

## 🧪 Esperimenti e Valutazione

### 1. Dataset
Sono stati utilizzati circa **35.100 immagini** provenienti da quattro dataset, suddivisi in:
* [cite_start]**Tartarughe Marine:** *Proprietary Dataset*, *Dataset Roboflow*, *SeaTurtleID2022*[cite: 89].
* [cite_start]**Distrattori:** *COCO Validation Set*[cite: 90].
[cite_start]La divisione è stata 70-20-10 (Train-Validation-Test), garantendo la presenza della classe "tartaruga" in ogni *batch* nonostante la sua sottorappresentazione rispetto ai distrattori[cite: 92, 93].

### 2. Pipeline di Captioning
[cite_start]Dato che la maggior parte delle immagini era priva di *caption*, è stata implementata una *pipeline* di *captioning* automatico con i seguenti passaggi[cite: 96, 97]:
* [cite_start]**Generazione Automatica:** Utilizzo di **BLIP** (Image → Caption)[cite: 97].
* [cite_start]**Correzione e Paraphrasing:** Post-elaborazione con **LLaMA** per rimuovere errori semantici e ripetitività, e aumentare la diversità lessicale[cite: 98, 100].
* [cite_start]**Validazione Manuale:** Per garantire accuratezza, unicità e consistenza semantica delle descrizioni[cite: 101, 102].

### 3. Setup Sperimentale
* [cite_start]**Hardware:** NVIDIA RTX 4090 (24 GB)[cite: 109].
* [cite_start]**Modelli base:** `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` e `openai/clip-vit-base-patch32`[cite: 111, 112].
* [cite_start]**Ottimizzazione:** AdamW, $lr=3e^{-4}$, 50 *epoch*[cite: 114].

### 4. Metriche di Valutazione
[cite_start]La valutazione è stata condotta a due livelli[cite: 122]:

* [cite_start]**Recupero a Livello di Categoria (*Category-level retrieval*):** Corretto se l'immagine recuperata appartiene alla stessa categoria della *query* (distinzione tra classi)[cite: 124, 125].
    * *Metriche:* Recall@K (R@1, R@5, R@10) e Mean Rank per la categoria generale (cat\_all).
* [cite_start]**Recupero Esatto a Livello di Istanza (*Exact instance-level retrieval*):** Corretto solo se l'immagine recuperata è esattamente quella associata alla *caption* (distinzione tra esempi molto simili)[cite: 127, 129].
    * *Metriche:* Recall@K (R@1, R@5, R@10) e Mean Rank per la classe "turtle".

---

## 📊 Risultati e Conclusioni

I risultati sperimentali mostrano che il **fine-tuning di CLIP** migliora significativamente le performance rispetto ai modelli base, in particolare per il recupero esatto delle istanze di tartarughe.

| Model Name | cat\_all R@1 | cat\_all R@5 | cat\_all R@10 | mean\_rank |
| :--- | :--- | :--- | :--- | :--- |
| CLIP\_OpenAI\_base | 0.3737 | 0.6085 | 0.7151 | 19 |
| **CLIP\_OpenAI\_tuned** | **0.4531** | **0.6970** | **0.7979** | **12** |
| CLIP\_LAION\_base | 0.4821 | 0.7140 | 0.8009 | 13 |
| CLIP\_LAION\_tuned | 0.4245 | 0.6758 | 0.7733 | 14 |

| Model Name | turtle\_R@1 | turtle\_R@5 | turtle\_R@10 | mean\_rank |
| :--- | :--- | :--- | :--- | :--- |
| CLIP\_OpenAI\_base | 0.0536 | 0.1429 | 0.1984 | 126 |
| **CLIP\_OpenAI\_tuned** | **0.1884** | **0.4841** | **0.5853** | **52** |
| CLIP\_LAION\_base | 0.0773 | 0.2222 | 0.2896 | 87 |
| CLIP\_LAION\_tuned | 0.1488 | 0.3829 | 0.5000 | 68 |

* [cite_start]Il modello **`CLIP_OpenAI_tuned`** ha mostrato i **migliori risultati** a livello di categoria e di istanza, riducendo drasticamente il *Mean Rank* per il recupero delle tartarughe (da 126 a **52**)[cite: 130].

[cite_start]**Conclusioni:** Il progetto ha dimostrato risultati promettenti nel riconoscimento delle tartarughe [cite: 135] grazie a:
* [cite_start]La costruzione di un **dataset specializzato e curato** (crop, *caption* automatiche + revisione manuale)[cite: 132].
* [cite_start]L'***efficient fine-tuning* di CLIP** con una combinazione di *CLIP Loss* e *UniCL*[cite: 133].
* [cite_start]La definizione di un **protocollo di valutazione** mirato[cite: 134].

---

## 🚀 Sviluppi Futuri

Le future direzioni di ricerca includono:
* [cite_start]**Miglioramento del Dataset:** Arricchimento e estensione[cite: 137].
* [cite_start]**Estensione a Nuove Classi:** Inclusione di altri elementi rilevanti per il monitoraggio marino (es. detriti, delfini)[cite: 138].
* [cite_start]**Generative Models:** Utilizzo di modelli generativi per arricchire i dati[cite: 139].
* [cite_start]**Architetture Multimodali Avanzate:** Esplorazione di nuove architetture e strategie di *loss* avanzate[cite: 141].
* [cite_start]**Integrazione Multimodale:** Andare oltre testo/immagini (es. integrazione della voce)[cite: 142].
