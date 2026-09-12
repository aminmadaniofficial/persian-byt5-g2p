# Persian ByT5 G2P (Grapheme-to-Phoneme)

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-AminMadani%2Fpersian--byt5--g2p-blue)](https://huggingface.co/AminMadani/persian-byt5-g2p)
[![Dataset](https://img.shields.io/badge/Dataset-KaamelDict-green)](https://huggingface.co/datasets/MahtaFetrat/KaamelDict)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A lightweight, robust **Grapheme-to-Phoneme (G2P)** model for Persian, fine-tuned on the comprehensive [KaamelDict](https://huggingface.co/datasets/MahtaFetrat/KaamelDict) dictionary (~116,600 phonetic entries) using Google's byte-level `google/byt5-small` architecture.

---

## 🚀 Why Byte-Level G2P for Persian?

Traditional word-piece / subword tokenizers struggle with Persian orthography due to rich morphology (prefixes, suffixes, compound words) and informal spacing. 

**ByT5 operates directly on raw UTF-8 bytes**, providing key advantages:
- **Zero Out-of-Vocabulary (OOV):** Transcribes completely unseen words, loanwords, proper nouns, and neologisms.
- **Implicit Diacritic & Short Vowel Inference:** Accurately discovers unwritten short vowels (*harakat* like *a*, *e*, *o*), the grammatical connector (*ezafe*), and gemination (*tashdid*).
- **Historical Orthography Resolution:** Corrects anomalies such as `خوش` $\to$ `/x o S/` (pronounced with *o* instead of *v*).
- **Lightweight & Production Ready:** Only ~300M parameters, ideal for real-time TTS synthesis pipelines (e.g., Piper TTS, VITS) on CPU and consumer GPUs.

---

## 📊 Benchmark & Training Metrics

- **Base Architecture:** `google/byt5-small` (300M parameters)
- **Dataset:** [MahtaFetrat/KaamelDict](https://huggingface.co/datasets/MahtaFetrat/KaamelDict) (~116k entries, 95% train / 5% test split)
- **Precision:** Full Precision (FP32)
- **Batch Size:** 16 per device with 2 gradient accumulation steps (effective batch size = 32)
- **Learning Rate:** `5e-4` with AdamW
- **Training Epochs:** 3 epochs (~10,386 steps)
- **Final Validation Loss:** **`0.03755`**

### Sample Transcriptions

| Persian Orthography | Model Output (Phonemes) | Note |
| :--- | :--- | :--- |
| **دانشگاه** | `d A n e S g A h` | Inferred short vowel *kasreh* (`e`) |
| **کامپیوتر** | `k A m p i y u t e r` | English loanword phonetic mapping |
| **خوش‌آمدید** | `x o S A m a d i d` | Orthographic rule: `خو` $\to$ `/xo/` |
| **هوش مصنوعی** | `h u S e m a s n u ? i` | *Ezafe* connector (`e`) + glottal stop (`?`) |
| **تلفظ** | `t a l a f f o z` | Inferred gemination / *tashdid* (`ff`) |
| **واترپولو** | `v A t e r p o l o` | Complex compound loanword |
| **دربازکن** | `d a r b A z k o n` | Persian compound with short vowels |

---

## 🛠️ Installation

```bash
git clone https://github.com/aminmadaniofficial/persian-byt5-g2p.git
cd persian-byt5-g2p
pip install -r requirements.txt
```

---

## 💻 Usage

### 1. Python API

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

MODEL_ID = "AminMadani/persian-byt5-g2p"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

def text_to_phonemes(text: str) -> str:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=128,
            num_beams=2,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(text_to_phonemes("دانشگاه"))
# Output: d A n e S g A h
```

Or using the built-in helper class:

```python
from inference import PersianG2P

g2p = PersianG2P()
print(g2p.predict(["دانشگاه", "کامپیوتر"]))
# Output: ['d A n e S g A h', 'k A m p i y u t e r']
```

### 2. Interactive CLI

```bash
# Run interactive transcription shell
python inference.py --interactive

# Transcribe a single word
python inference.py --text "دانشگاه"
```

---

## 📓 Reproduce Training

The complete training script with Google Drive export and Hugging Face Hub integration is provided in [notebook.ipynb](notebook.ipynb). You can run it directly on Google Colab's free T4 GPU tier.

---

## 📖 Citation

If you use this model or the KaamelDict dataset, please cite the original papers:

```bibtex
@inproceedings{qharabagh2025llm,
  title={LLM-Powered Grapheme-to-Phoneme Conversion: Benchmark and Case Study},
  author={Qharabagh, Mahta Fetrat and Dehghanian, Zahra and Rabiee, Hamid R},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2025},
  organization={IEEE}
}

@article{xue2022byt5,
  title={ByT5: Towards a token-free future with pre-trained byte-to-byte models},
  author={Xue, Linting and Barua, Aditya and Constant, Noah and Al-Rfou, Rami and Narang, Sharan and Kale, Mihir and Roberts, Adam and Raffel, Colin},
  journal={Transactions of the Association for Computational Linguistics},
  volume={10},
  pages={291--306},
  year={2022}
}
```

---

## 📜 License

This project is released under the **GNU General Public License v3.0 (GPL-3.0)**, consistent with the KaamelDict source dataset.
