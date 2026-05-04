<div align="center">

<br/>

# 🎙️ EmoVoice AI

### *Understand Emotions. Amplify Empathy.*

<br/>

> **"The most powerful thing you can do is make someone feel heard."**

<br/>

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Whisper](https://img.shields.io/badge/OpenAI-Whisper-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com/research/whisper)
[![DistilBERT](https://img.shields.io/badge/HuggingFace-DistilBERT-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/distilbert-base-uncased)
[![gTTS](https://img.shields.io/badge/gTTS-Text--to--Speech-4285F4?style=flat-square&logo=google&logoColor=white)](https://pypi.org/project/gTTS/)
[![torchaudio](https://img.shields.io/badge/PyTorch-torchaudio-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/audio/)
[![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)](https://www.kaggle.com/datasets/uldisvalainis/audio-emotions)
[![Colab](https://img.shields.io/badge/Google-Colab%20Ready-F9AB00?style=flat-square&logo=googlecolab&logoColor=black)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

<br/>

[![GitHub Repo](https://img.shields.io/badge/📂%20Repository-ibtesaamaslam%2FEmoVoice--AI-181717?style=for-the-badge&logo=github)](https://github.com/ibtesaamaslam/EmoVoice-AI)

<br/>

---

</div>

## Table of Contents

1. [Introduction](#1-introduction)
2. [How It Works — System Overview](#2-how-it-works--system-overview)
3. [Emotion Classes & Empathetic Responses](#3-emotion-classes--empathetic-responses)
4. [Features](#4-features)
5. [Tech Stack](#5-tech-stack)
6. [Dataset](#6-dataset)
   - 6.1 [Source & Structure](#61-source--structure)
   - 6.2 [Kaggle API Setup](#62-kaggle-api-setup)
7. [Project Structure](#7-project-structure)
8. [Installation & Setup](#8-installation--setup)
   - 8.1 [Google Colab Setup](#81-google-colab-setup)
   - 8.2 [Local Setup](#82-local-setup)
9. [Usage](#9-usage)
   - 9.1 [Run Main Analysis Script](#91-run-main-analysis-script)
   - 9.2 [Process a Random Audio File](#92-process-a-random-audio-file)
   - 9.3 [Upload Custom Audio](#93-upload-custom-audio)
10. [Model Architecture & Design Decisions](#10-model-architecture--design-decisions)
    - 10.1 [Why OpenAI Whisper](#101-why-openai-whisper)
    - 10.2 [Why DistilBERT](#102-why-distilbert)
    - 10.3 [Why gTTS for Feedback](#103-why-gtts-for-feedback)
    - 10.4 [Pipeline Design Philosophy](#104-pipeline-design-philosophy)
11. [Sample Output](#11-sample-output)
12. [Real-World Applications](#12-real-world-applications)
13. [Limitations](#13-limitations)
14. [Roadmap — Future Improvements](#14-roadmap--future-improvements)
15. [Contributing](#15-contributing)
16. [License](#16-license)
17. [Author](#17-author)
18. [Acknowledgments](#18-acknowledgments)

---

## 1. Introduction

**EmoVoice AI** is an audio sentiment analysis system that listens to a human voice, understands what was said, detects the underlying emotional tone, and responds with an empathetic spoken reply — all powered by state-of-the-art AI models.

In a world increasingly mediated by voice interfaces, the ability for machines to not just understand *what* we say but *how* we feel when we say it represents a meaningful leap forward in human-computer interaction. EmoVoice AI is a practical demonstration of this capability — combining three powerful AI systems into a single, coherent emotional intelligence pipeline:

1. **Speech Recognition** — OpenAI Whisper converts raw audio into text
2. **Sentiment Classification** — Fine-tuned DistilBERT detects emotional polarity (Positive / Negative / Neutral)
3. **Empathetic Voice Response** — gTTS generates and plays back an emotion-appropriate spoken reply

**Why does this matter?**

| Context | The Problem | EmoVoice AI's Contribution |
|---|---|---|
| **Mental Health** | People in distress need to feel heard, not just processed | Responds with empathy, not just information |
| **Customer Support** | Frustrated callers escalate when they feel dismissed | Detects frustration early and adapts tone |
| **Education** | Anxious students disengage from cold, mechanical feedback | Delivers encouragement calibrated to emotional state |
| **Research** | Affective computing needs practical baselines | Provides a clean, modular reference implementation |

EmoVoice AI is optimised for **Google Colab** but runs cleanly on any local Python environment with `ffmpeg` installed.

---

## 2. How It Works — System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INPUT                                      │
│         .wav Audio File (Dataset sample OR custom upload)           │
│         Format: Mono, 16kHz, PCM WAV                                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  STAGE 1 — SPEECH-TO-TEXT                           │
│                  OpenAI Whisper (via Transformers)                  │
│                                                                     │
│  • Loads audio with torchaudio                                      │
│  • Resamples to 16kHz if needed                                     │
│  • Whisper transcribes speech → plain text string                   │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼  transcribed text
┌─────────────────────────────────────────────────────────────────────┐
│                  STAGE 2 — SENTIMENT CLASSIFICATION                 │
│                  DistilBERT (fine-tuned on SST-2)                   │
│                                                                     │
│  • Tokenises the transcribed text                                   │
│  • Runs forward pass through DistilBERT                             │
│  • Outputs: label (POSITIVE / NEGATIVE / NEUTRAL)                   │
│             + confidence score (0.0 – 1.0)                          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼  sentiment label + score
┌─────────────────────────────────────────────────────────────────────┐
│                  STAGE 3 — EMPATHETIC RESPONSE                      │
│                  Rule-based Response Selection + gTTS               │
│                                                                     │
│  • Maps sentiment label → empathy response template                 │
│  • gTTS converts response text → MP3 audio                          │
│  • IPython plays audio inline in the notebook                       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Emotion Classes & Empathetic Responses

EmoVoice AI classifies transcribed speech into three sentiment categories and generates tailored empathetic responses for each:

| Sentiment | Confidence Threshold | Example Trigger Phrase | Example Response |
|---|---|---|---|
| **POSITIVE** | > 0.80 | *"I'm so thrilled today!"* | *"That's awesome! Keep shining — your energy is contagious!"* |
| **NEGATIVE** | > 0.80 | *"I feel really low and tired."* | *"I'm sorry you're feeling this way. You're not alone, and I'm here to support you."* |
| **NEUTRAL** | < 0.80 (either) | *"I went to the store today."* | *"Thank you for sharing. I'm here if you'd like to talk more about how you're feeling."* |

**Dataset Emotion Labels:**

The Audio Emotions dataset includes speech samples across a wider set of emotional labels including `Happy`, `Sad`, `Angry`, `Fearful`, `Disgusted`, `Surprised`, and `Neutral`. EmoVoice AI maps these to the three-class sentiment system for clean, actionable output.

---

## 4. Features

**🎧 Speech-to-Text Transcription**
OpenAI Whisper — one of the most accurate open-source ASR (Automatic Speech Recognition) models available — converts spoken audio to text with high accuracy across accents, recording qualities, and noise levels.

**💬 Three-Class Sentiment Detection**
A fine-tuned DistilBERT model classifies the transcribed text as Positive, Negative, or Neutral and outputs a calibrated confidence score between 0 and 1.

**🤗 Empathetic Voice Response**
The system does not just label the emotion — it *responds* to it. Google Text-to-Speech (gTTS) converts an emotion-appropriate text response into spoken audio, played back inline in the notebook.

**📊 Audio Emotions Dataset Integration**
Auto-downloads the `uldisvalainis/audio-emotions` Kaggle dataset via `kagglehub`. Selects a random audio file from the dataset for each run — allowing broad coverage of emotional speech styles without manual file management.

**🎲 Random File Selection**
`random_voice.py` selects a random audio file from anywhere in the dataset on every run — ensuring you test across different speakers, emotions, and recording conditions without cherry-picking.

**⬆️ Custom Audio Upload**
Upload your own `.wav` file (mono, 16kHz) named `input.wav` and EmoVoice AI will prioritise it over the dataset sample — making it trivial to test with real recordings.

**🧪 Google Colab Optimised**
The entire pipeline is designed to run without modification in Google Colab — including inline audio playback via `IPython.display.Audio`, `ffmpeg` installation, and `kaggle.json`-based dataset authentication.

---

## 5. Tech Stack

| Technology | Version | Purpose |
|---|---|---|
| [Python](https://www.python.org/) | 3.8+ | Core language |
| [OpenAI Whisper](https://openai.com/research/whisper) (via Transformers) | Latest | Speech-to-text transcription |
| [DistilBERT](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) | Latest | Sentiment classification |
| [gTTS](https://pypi.org/project/gTTS/) | Latest | Text-to-speech audio response generation |
| [torchaudio](https://pytorch.org/audio/) | Latest | Loading, resampling, and processing `.wav` audio |
| [Transformers](https://huggingface.co/docs/transformers) | Latest | Model loading and inference pipeline |
| [scipy](https://scipy.org/) | Latest | Audio file I/O and signal processing utilities |
| [pydub](https://github.com/jiaaro/pydub) | Latest | Audio format conversion (MP3 → WAV) |
| [kagglehub](https://github.com/Kaggle/kagglehub) | Latest | Programmatic dataset download and caching |
| [IPython](https://ipython.org/) | Latest | Inline audio playback in Colab notebooks |
| [ffmpeg](https://ffmpeg.org/) | System | Audio codec backend for pydub and torchaudio |

---

## 6. Dataset

### 6.1 Source & Structure

**Source:** [Kaggle — `uldisvalainis/audio-emotions`](https://www.kaggle.com/datasets/uldisvalainis/audio-emotions)

**Format:** `.wav` files, labelled by emotion category

**Folder Structure:**

```
audio-emotions/
│
├── Happy/
│   ├── audio1.wav
│   ├── audio2.wav
│   └── ...
├── Sad/
│   ├── audio1.wav
│   └── ...
├── Angry/
│   └── ...
├── Fearful/
│   └── ...
├── Disgusted/
│   └── ...
├── Surprised/
│   └── ...
└── Neutral/
    └── ...
```

**Emotion-to-Sentiment Mapping:**

| Dataset Label | Mapped Sentiment |
|---|---|
| Happy, Surprised | POSITIVE |
| Sad, Angry, Fearful, Disgusted | NEGATIVE |
| Neutral | NEUTRAL |

---

### 6.2 Kaggle API Setup

**Step 1 — Generate your API token:**
1. Log in to [kaggle.com](https://www.kaggle.com)
2. Go to **Account → API → Create New Token**
3. `kaggle.json` downloads automatically

**Step 2 — Upload to Google Colab:**

In Colab's left sidebar, click the **Files** icon → **Upload** → select `kaggle.json`. Then run:

```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

**Step 3 — Download the Dataset:**

```python
import kagglehub
dataset_path = kagglehub.dataset_download("uldisvalainis/audio-emotions")
print("Dataset path:", dataset_path)
```

> ✅ In Kaggle notebooks, authentication is automatic — no `kaggle.json` upload needed.

---

## 7. Project Structure

```
EmoVoice-AI/
│
├── emovoice_ai_main.py       # Main pipeline — setup, model init, full analysis
│   ├── Dataset download       # kagglehub auto-download
│   ├── Model initialisation   # Whisper + DistilBERT loading
│   ├── Audio loading          # torchaudio + resampling
│   ├── Transcription          # Whisper speech-to-text
│   ├── Sentiment analysis     # DistilBERT classification
│   └── Empathetic response    # gTTS generation + IPython playback
│
├── random_voice.py           # Randomly selects & processes a dataset audio file
│   ├── Random file selection  # os.walk + random.choice across all emotion folders
│   └── Full pipeline call     # Reuses main pipeline with selected file
│
├── requirements.txt          # All pip dependencies (one per line)
├── README.md                 # This file
└── LICENSE                   # MIT License
```

---

## 8. Installation & Setup

### 8.1 Google Colab Setup

**Step 1 — Open a new Colab notebook** at [colab.research.google.com](https://colab.research.google.com)

**Step 2 — Install system dependencies:**

```bash
!apt-get update && apt-get install -y ffmpeg
```

**Step 3 — Install Python libraries:**

```bash
!pip install torchaudio transformers gtts pydub scipy kagglehub
```

**Step 4 — Configure Kaggle credentials** (see Section 6.2)

**Step 5 — Clone the repository:**

```bash
!git clone https://github.com/ibtesaamaslam/EmoVoice-AI.git
%cd EmoVoice-AI
```

**Step 6 — Run the main script:**

```bash
!python emovoice_ai_main.py
```

---

### 8.2 Local Setup

**Step 1 — Prerequisites:**

| Requirement | Notes |
|---|---|
| Python 3.8+ | `python --version` to verify |
| ffmpeg | Install via `brew install ffmpeg` (Mac) or `sudo apt install ffmpeg` (Linux) or [ffmpeg.org](https://ffmpeg.org/) (Windows) |
| Microphone/audio file | `.wav` format, mono, 16kHz recommended |
| Kaggle account | For dataset access |

**Step 2 — Clone & Install:**

```bash
git clone https://github.com/ibtesaamaslam/EmoVoice-AI.git
cd EmoVoice-AI

# Optional: create virtual environment
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

pip install -r requirements.txt
```

**Step 3 — Configure Kaggle credentials:**

```bash
mkdir -p ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

**Step 4 — Run:**

```bash
python emovoice_ai_main.py
```

---

## 9. Usage

### 9.1 Run Main Analysis Script

```bash
python emovoice_ai_main.py
```

**What happens:**

1. Kaggle dataset is downloaded (or loaded from cache if previously downloaded)
2. OpenAI Whisper model is initialised
3. DistilBERT sentiment classifier is loaded
4. A random audio file from the dataset is selected
5. Whisper transcribes the audio to text
6. DistilBERT classifies the sentiment with a confidence score
7. An empathetic response is generated and spoken aloud via gTTS
8. Audio plays inline in the Colab notebook

**Example output:**

```
Downloading Audio Emotions dataset...
✓ Dataset loaded: /root/.cache/kagglehub/datasets/uldisvalainis/audio-emotions

Selected file: Happy/audio2.wav

🎧 Transcribing audio...
User said: "I'm so thrilled today! Everything is going so well."

💬 Analysing sentiment...
Detected Sentiment: POSITIVE (confidence: 0.93)

🤗 Coach says: "That's awesome! Keep shining — your energy is contagious!"

🔊 Playing audio response...
```

---

### 9.2 Process a Random Audio File

```bash
python random_voice.py
```

This script traverses all emotion folders in the dataset and selects a completely random `.wav` file — ensuring coverage across Happy, Sad, Angry, Neutral, and other categories on every run.

**Example output:**

```
Selected file: Sad/audio1.wav

🎧 Transcribing audio...
User said: "I feel really low and I don't know what to do."

💬 Analysing sentiment...
Detected Sentiment: NEGATIVE (confidence: 0.96)

🤗 Coach says: "I'm sorry you're feeling this way. You're not alone,
               and I'm here to support you every step of the way."

🔊 Playing audio response...
```

---

### 9.3 Upload Custom Audio

To test with your own voice or a custom recording:

1. Record or prepare a `.wav` audio file in **mono, 16kHz** format
2. Rename it to `input.wav`
3. Upload it to your Colab session (Files sidebar → Upload)
4. Run the main script — it will automatically detect and prioritise `input.wav` over dataset samples

**Converting audio to the correct format (ffmpeg):**

```bash
ffmpeg -i your_recording.mp3 -ac 1 -ar 16000 input.wav
```

| Flag | Meaning |
|---|---|
| `-ac 1` | Convert to mono (1 audio channel) |
| `-ar 16000` | Resample to 16,000 Hz |

---

## 10. Model Architecture & Design Decisions

### 10.1 Why OpenAI Whisper

Whisper was selected as the ASR backbone for three reasons:

**Accuracy across accents and noise:** Whisper was trained on 680,000 hours of multilingual audio, making it significantly more robust to different accents, speaking styles, and background noise than alternatives like Google Speech-to-Text API or DeepSpeech.

**No API key required:** Whisper runs entirely locally (or in Colab) without any external API calls — keeping the system self-contained and free to run.

**Transformers integration:** The `openai-whisper` model is available directly through Hugging Face Transformers (`pipeline("automatic-speech-recognition")`), keeping the dependency stack unified and the code concise.

---

### 10.2 Why DistilBERT

**Efficiency:** DistilBERT is 40% smaller and 60% faster than BERT while retaining 97% of BERT's language understanding performance on downstream NLP tasks. This makes it ideal for a pipeline where inference speed matters for a smooth user experience.

**Fine-tuning on SST-2:** The model used (`distilbert-base-uncased-finetuned-sst-2-english`) was fine-tuned on the Stanford Sentiment Treebank — a benchmark dataset of sentiment-labelled sentences. This makes it immediately applicable to short transcribed speech without any additional training.

**Confidence scores:** DistilBERT's `softmax` output layer provides calibrated probability scores alongside the predicted label — allowing EmoVoice AI to differentiate between a confident `NEGATIVE (0.96)` and an uncertain `NEGATIVE (0.61)`, and adjust the response accordingly in future versions.

---

### 10.3 Why gTTS for Feedback

**Simplicity:** Google Text-to-Speech requires no API key, no account setup, and no local model download — a single `pip install gtts` is sufficient.

**Natural voice quality:** gTTS produces significantly more natural-sounding speech than alternatives like `pyttsx3` (which uses system TTS engines that often sound robotic), making the empathetic response feel more genuine and human-like.

**Colab compatibility:** gTTS generates an MP3 file that IPython can play inline — a clean solution for notebook-based audio output without any display drivers or audio device requirements.

---

### 10.4 Pipeline Design Philosophy

The three-stage pipeline was designed with a strict separation of concerns:

```
Stage 1 (Whisper)    →  What did the person say?
Stage 2 (DistilBERT) →  How do they feel?
Stage 3 (gTTS)       →  What should we say back?
```

Each stage is independently swappable — Whisper can be replaced with a faster model, DistilBERT can be replaced with a more nuanced emotion classifier, and gTTS can be replaced with a neural TTS system like Coqui TTS or ElevenLabs — without touching the other stages. This modularity makes EmoVoice AI a clean foundation for more ambitious emotion-aware voice assistant systems.

---

## 11. Sample Output

**Happy/audio2.wav:**

```
User said:        "I'm so thrilled today! Everything is going so well."
Sentiment:        POSITIVE (0.93)
Coach says:       "That's awesome! Keep shining — your energy is contagious!"
```

---

**Sad/audio1.wav:**

```
User said:        "I feel really low and I don't know what to do."
Sentiment:        NEGATIVE (0.96)
Coach says:       "I'm sorry you're feeling this way. You're not alone,
                   and I'm here to support you every step of the way."
```

---

**Neutral/audio3.wav:**

```
User said:        "I went to the store and bought some groceries."
Sentiment:        NEUTRAL (0.71)
Coach says:       "Thank you for sharing. I'm here if you'd like to talk
                   more about how you're feeling."
```

---

*Note: Exact transcription and confidence scores vary based on audio quality, speaker accent, and background noise.*

---

## 12. Real-World Applications

| Domain | Use Case | How EmoVoice AI Helps |
|---|---|---|
| 🧠 **Mental Health** | Voice journaling companion, therapy support tool | Detects distress signals in speech and responds with empathy rather than information overload |
| 🛎️ **Customer Service** | Emotionally intelligent voice bots and IVR systems | Identifies frustrated customers early and routes or responds accordingly |
| 📚 **Education** | AI tutors and e-learning platforms | Detects anxiety or confusion in a student's voice and adjusts feedback tone |
| 🤝 **HR & Coaching** | Interview analysis, employee wellness tools | Analyses voice sentiment in structured conversations for wellbeing insights |
| 🔬 **Research** | Affective computing, HCI studies, emotion recognition benchmarks | Provides a clean, modular, reproducible baseline for sentiment-from-speech research |
| 🎮 **Gaming & XR** | Emotionally responsive game characters and VR companions | NPCs that react to the emotional tone of the player's voice |

---

## 13. Limitations

| Limitation | Explanation |
|---|---|
| **Three-class sentiment only** | Positive / Negative / Neutral does not capture nuanced emotions like anger, fear, or surprise distinctly |
| **Text-based sentiment on voice data** | DistilBERT classifies the *transcription*, not the audio signal itself — prosody, tone, and pitch are not considered |
| **English only** | Both Whisper (in this configuration) and the DistilBERT fine-tune are English-focused |
| **Requires 16kHz mono WAV** | Custom audio must be pre-converted to the correct format; raw microphone recordings may need conversion |
| **gTTS requires internet** | Text-to-speech synthesis requires an active internet connection to reach Google's TTS API |
| **No conversation memory** | Each audio file is processed independently; the system has no memory of previous exchanges |
| **Short utterances may be misclassified** | Very short phrases ("yes", "no", "okay") provide little textual signal for sentiment classification |

---

## 14. Roadmap — Future Improvements

**v1.1 — Nuanced Emotion Classification**
Replace the three-class DistilBERT model with a fine-tuned classifier trained on a multi-class emotion dataset (e.g., GoEmotions) — enabling detection of `joy`, `anger`, `fear`, `sadness`, `surprise`, and `disgust` as distinct classes.

**v1.2 — Prosody & Audio Feature Analysis**
Add paralinguistic feature extraction (pitch, speaking rate, energy) using `librosa` — enabling the system to detect emotion from *how* something is said, not just *what* was said.

**v1.3 — Multilingual Support**
Leverage Whisper's multilingual capabilities to transcribe speech in languages beyond English, paired with a multilingual sentiment model (e.g., XLM-RoBERTa).

**v1.4 — Gradio Web Interface**
Build an interactive browser UI using Gradio — allowing users to record directly from a microphone in the browser and receive real-time empathetic feedback without any code.

**v1.5 — Streamlit Dashboard**
A full dashboard showing the audio waveform, transcription, sentiment confidence bar chart, and response — all in a clean, shareable web app deployable to Streamlit Community Cloud.

**v2.0 — Conversational Memory**
Add multi-turn conversation memory — tracking emotional trajectory across multiple exchanges and adapting the response strategy based on whether the user's sentiment is improving or deteriorating over time.

**v2.1 — Neural TTS Voice**
Replace gTTS with a neural text-to-speech system (Coqui TTS or ElevenLabs API) for a significantly more expressive, human-like empathetic voice.

**v2.2 — Live Microphone Mode**
Enable real-time microphone recording and processing — removing the requirement for a pre-recorded `.wav` file and enabling true conversational interaction.

---

## 15. Contributing

Contributions are welcome — whether you're improving the emotion classifier, adding a new language, building a web interface, or fixing a bug.

**How to Contribute:**

1. **Fork the repository**

2. **Create a feature branch**
   ```bash
   git checkout -b feature/multilingual-support
   ```

3. **Make your changes** — follow PEP8 and document all functions with docstrings

4. **Test in both Colab and local environments** before submitting

5. **Commit with a meaningful message**
   ```bash
   git commit -m "feat: add multilingual Whisper support with XLM-RoBERTa"
   ```

6. **Push and open a Pull Request**
   ```bash
   git push origin feature/multilingual-support
   ```

**Good First Issues:**

- Add a `--file` CLI argument to specify custom audio without renaming to `input.wav`
- Implement a confidence threshold below which the system asks the user to repeat themselves
- Add a `requirements.txt` with pinned versions for reproducibility
- Write a unit test for the sentiment → response mapping logic
- Add support for `.flac` and `.mp3` input formats via automatic conversion

---

## 16. License

```
MIT License

Copyright (c) 2025 Ibtesaam Aslam

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 17. Author

<div align="center">

### **Ibtesaam Aslam**
*Full-Stack Developer · ML Engineer · Emotion-Aware AI Builder*

[![GitHub](https://img.shields.io/badge/GitHub-%40ibtesaamaslam-181717?style=for-the-badge&logo=github)](https://github.com/ibtesaamaslam)
[![Repository](https://img.shields.io/badge/Repo-EmoVoice--AI-0070f3?style=for-the-badge&logo=github)](https://github.com/ibtesaamaslam/EmoVoice-AI)

</div>

---

## 18. Acknowledgments

| Resource | Contribution |
|---|---|
| [OpenAI Whisper](https://openai.com/research/whisper) | State-of-the-art open-source ASR model used for speech transcription |
| [Hugging Face Transformers](https://huggingface.co/transformers) | Model hub and pipeline interface for Whisper and DistilBERT |
| [DistilBERT — Hugging Face](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) | Fine-tuned sentiment classifier used for emotion detection |
| [Google Text-to-Speech (gTTS)](https://pypi.org/project/gTTS/) | Text-to-speech engine powering the empathetic voice responses |
| [Uldis Valainis — Kaggle](https://www.kaggle.com/datasets/uldisvalainis/audio-emotions) | The Audio Emotions dataset providing labelled emotional speech samples |
| [PyTorch / torchaudio](https://pytorch.org/audio/) | Audio loading, resampling, and tensor processing |
| [Google Colab](https://colab.research.google.com/) | Cloud notebook environment the project is optimised for |

---

<div align="center">

---

*Made with ❤️ by [Ibtesaam Aslam](https://github.com/ibtesaamaslam)*

*Where voice meets empathy — powered by intelligence.*

[![Star on GitHub](https://img.shields.io/github/stars/ibtesaamaslam/EmoVoice-AI?style=social)](https://github.com/ibtesaamaslam/EmoVoice-AI)

</div>
