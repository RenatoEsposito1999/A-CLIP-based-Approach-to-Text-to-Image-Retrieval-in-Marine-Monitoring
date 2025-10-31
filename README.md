[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
# A CLIP-based Approach to Text-to-Image Retrieval in Marine Monitoring 🐢 
This work represents our Master's Degree Thesis in **Applied Computer Science (Machine Learning and Big Data)**, the titles of each one is:

A CLIP-based Approach to Text-to-Image Retrieval in Marine Monitoring

Deep-learning based sea-turtles monitoring

## 📋 Table of Contents

1. [Introduction and Project Goals](#1-introduction-and-project-goals)
    * [Specific Objectives](#specific-objectives)
2. [Proposed Method: CLIP Fine-Tuning](#2-proposed-method-clip-fine-tuning)
    * [CLIP Architecture](#clip-architecture)
    * [Combined Loss Strategy](#combined-loss-strategy)
    * [Efficient Fine-Tuning (LoRA)](#efficient-fine-tuning-lora)
3. [Experiments and Evaluation](#3-experiments-and-evaluation)
    * [Datasets](#datasets)
    * [Dataset Captioning Pipeline](#dataset-captioning-pipeline)
    * [Experimental Setup](#experimental-setup)
    * [Evaluation Metrics](#evaluation-metrics)
4. [Results and Conclusions](#4-results-and-conclusions)
5. [Future Developments](#5-future-developments)

---

## 1. Introduction and Project Goals

This project applies Artificial Intelligence (AI) techniques to support marine biodiversity monitoring, aligning with the **UN 2030 Agenda Sustainable Development Goal (SDG) 14**: "**Life below water**" ("Conserve and sustainably use the oceans, seas and marine resources for sustainable development").

The primary goal was the **design and implementation of a specialized Text-to-Image Retrieval system** focusing on the **sea turtle domain**.

### Specific Objectives
* **Align visual and textual representations** in a shared embedding space.
* Enable **querying unstructured datasets using natural language textual descriptions** (e.g., *“turtle with a net on its shell”*) to retrieve the most relevant images.
* Address the critical and complex task of **monitoring the presence of sea turtles** and distinguishing them from potentially harmful *distractors*, such as marine debris or other animals.

---

## 2. Proposed Method: CLIP Fine-Tuning

The core of the system is based on **CLIP (Contrastive Language–Image Pretraining)** by OpenAI (2021), a model designed to learn joint visual and textual representations in a shared semantic space.

### CLIP Architecture
* **Image Encoder (ViT-B/32):** Divides the image into patches, using a special `[CLS]` token to represent the global image feature.
* **Text Encoder (Standard Transformer Encoder):** Uses the output of the final token (e.g., `[EOS]`) as the global text representation.
* The representations from both encoders are **projected into a common embedding space** and normalized.

### Combined Loss Strategy
To specialize the model while preserving its generalization capabilities, a combined loss function was adopted:
$$L_{Loss} = \frac{L_{BiC} + L_{CLIP}}{2}$$

* **$L_{CLIP}$ (Symmetric Cross-Entropy Loss):** The original InfoNCE-based loss, which maximizes cosine similarity for aligned image-text pairs and minimizes it for mismatched pairs.
* **$L_{BiC}$ (Bilinear Unified Contrastive Learning Loss):** Extends the contrastive loss to a **multi-positive case**, allowing more than one matching sample per anchor. This is crucial for maintaining separation between categories and preserving the intra-class variability of the sea turtles.

### Efficient Fine-Tuning (LoRA)
**LoRA (Low-Rank Adaptation)** was utilized for efficient fine-tuning. This technique adds a small number of trainable weights while freezing the majority of the pre-trained model, making CLIP adaptable without prohibitive computational costs. LoRA was applied to the attention layers of both the text and visual encoders.

---

## 3. Experiments and Evaluation

### Datasets
The study utilized approximately **35,100 images** from four datasets:
* **Sea Turtles:** Proprietary Dataset, Dataset Roboflow, SeaTurtleID2022.
* **Distractors:** COCO Validation Set.

The dataset was split 70-20-10 (Train-Validation-Test), and the batch sampling strategy was designed to ensure the presence of the "turtle" class in every batch.

### Dataset Captioning Pipeline
As most images lacked captions, an automatic caption generation pipeline was implemented:
1.  **Automatic Generation:** Using **BLIP** (Image → Caption).
2.  **Correction and Paraphrasing:** Post-processing with **LLaMA** to remove semantic errors, address repetition, and enhance lexical diversity.
3.  **Manual Validation:** Final manual review to ensure the captions are accurate, unique, and semantically consistent.

### Experimental Setup
* **Hardware:** NVIDIA RTX 4090 (24 GB).
* **Base Models:** `laion/CLIP-ViT-B-32-laion2B-s34B-b79K` and `openai/clip-vit-base-patch32`.
* **Optimization:** AdamW optimizer, learning rate $3e^{-4}$, trained for 50 epochs.

### Evaluation Metrics
Evaluation was performed at two levels:

* **Category-level retrieval:** The result is considered correct if the retrieved image belongs to the same general category as the query (ability to distinguish between classes, e.g., turtle vs. distractor).
    * *Metrics:* Recall@K (R@1, R@5, R@10) and Mean Rank for all categories (*cat\_all*).
* **Exact instance-level retrieval:** The result is correct only if the retrieved image is the exact one associated with the caption (ability to distinguish between very similar examples, applied to turtles only).
    * *Metrics:* Recall@K (R@1, R@5, R@10) and Mean Rank for the *turtle* class.

---

## 4. Results and Conclusions

The experimental results show that the **fine-tuning of CLIP** significantly improves performance compared to the base models, especially for the exact instance retrieval of turtles.

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

* The **`CLIP_OpenAI_tuned`** model achieved the **best overall results**, drastically reducing the Mean Rank for turtle retrieval (from 126 to **52**).

**Conclusions:** The project demonstrated promising results in sea turtle recognition through:
* The construction of a **specialized and curated dataset** (cropping + automatic captions + manual review).
* The **efficient fine-tuning of CLIP** using a combination of *CLIP Loss* and *UniCL*.
* The definition of a targeted **evaluation protocol**.

---

## 5. Future Developments

Future research directions include:
* **Dataset Enhancement:** Further enrichment and expansion of the current dataset.
* **Extension to New Classes:** Including other relevant elements for marine monitoring (e.g., debris, dolphins).
* **Generative Models:** Leveraging generative models to enrich data.
* **Advanced Architectures:** Exploring new multimodal architectures and advanced loss strategies.
* **Multimodal Integration:** Going beyond text/images (e.g., integration of voice data).
