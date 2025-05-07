
# 🎙️ EmoVoice AI

Understand Emotions, Amplify Empathy

EmoVoice AI is a cutting-edge audio sentiment analysis tool that transcribes spoken words, detects emotional tone, and delivers empathetic, spoken feedback. By fusing state-of-the-art AI models in voice recognition and NLP, it enables emotionally intelligent human-AI interactions — ideal for use cases in mental health, virtual coaching, customer support, and education.

---

## 🚀 Features

* 🎧 Voice Transcription: Converts speech to text using OpenAI’s Whisper model.
* 💬 Sentiment Detection: Classifies emotions (Positive, Negative, Neutral) via a fine-tuned DistilBERT model.
* 🤗 Empathetic Feedback: Generates voice responses tailored to the speaker’s emotional state.
* 📊 Dataset Integration: Uses the Audio Emotions dataset with diverse emotional speech samples.
* 🧪 Colab-Friendly: Fully optimized for use in Google Colab, with random dataset file selection and audio playback.

---

## 🧠 Tech Stack

* Python: Core programming language
* OpenAI Whisper (via Transformers): Speech-to-text transcription
* DistilBERT (fine-tuned): Sentiment classification
* gTTS: Converts text responses to speech
* torchaudio: For loading and processing .wav audio
* kagglehub: Dataset downloading
* IPython: In-notebook audio playback

---

## 📦 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/emovoice-ai.git
cd emovoice-ai
```

### 2. Set Up in Google Colab

In a new Colab notebook, run the following:

```python
!pip install torchaudio transformers gtts pydub scipy kagglehub
!apt-get update && apt-get install -y ffmpeg
```

### 3. Configure Kaggle API

1. Go to your Kaggle account > Account > Create API Token
2. Upload kaggle.json to Colab’s sidebar file system
3. Run in Colab:

```python
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
```

---

## 💡 Usage

### ▶️ Run Main Script

1. Open emovoice\_ai\_main.py in Colab
2. Execute the script to:

   * Download the Audio Emotions dataset
   * Initialize Whisper and DistilBERT
   * Process an audio file

Sample Output:

```
Downloading Audio Emotions dataset...
Selected file: Happy/audio2.wav
User said: I’m so thrilled today!
Detected Sentiment: POSITIVE (0.93)
Coach says: That's awesome! Keep shining!
```

🔊 Audio feedback is played inline in the notebook.

---

### 🔁 Process Random Audio

Run random\_voice.py to process a random audio file from the dataset:

```
Selected file: Sad/audio1.wav
User said: I feel really low.
Detected Sentiment: NEGATIVE (0.96)
Coach says: I'm sorry you're feeling this way. You're not alone, and I'm here to support you.
```

---

### ⬆️ Upload Custom Audio (Optional)

Upload a mono, 16kHz .wav file named input.wav. The script will prioritize this file over dataset samples.

---

## 📁 Project Structure

```
emovoice-ai/
├── emovoice_ai_main.py    # Main script for setup and analysis
├── random_voice.py        # Script to process random dataset audio
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
```

---

## 🗃️ Dataset

🎧 Audio Emotions Dataset (Kaggle)

* Emotion-labeled .wav files (e.g., Sad, Happy, Angry)
* Auto-downloaded via:

```python
kagglehub.dataset_download("uldisvalainis/audio-emotions")
```

---

## 🌍 Applications

* 🧠 Mental Health: Emotionally aware voice companion for support and journaling
* 🛎️ Customer Service: Empathy-driven chatbots and voice agents
* 📚 Education: Interactive learning tools with emotional feedback
* 🔬 Research: For affective computing, emotion recognition, and HCI studies

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repo
2. Create a feature branch:

   ```bash
   git checkout -b feature/YourFeature
   ```
3. Commit your changes:

   ```bash
   git commit -m "Add YourFeature"
   ```
4. Push and open a Pull Request

---

## 📄 License

This project is licensed under the MIT License. See LICENSE for details.

---

## 📬 Contact

For issues or collaborations, open a GitHub Issue or reach out via the repository.

> EmoVoice AI: Where voice meets empathy — powered by intelligence.

