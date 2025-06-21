
import soundfile as sf
import streamlit as st

import librosa
import torch

from transformers import AutoModelForAudioClassification, AutoProcessor
import requests
from transformers import AutoFeatureExtractor

from moviepy.editor import VideoFileClip
import torchaudio
from speechbrain.pretrained.interfaces import foreign_class
import os
import soundfile as sf
import streamlit as st

import subprocess
import uuid


classifier = foreign_class(source="Jzuluaga/accent-id-commonaccent_xlsr-en-english", pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
class Agent_classifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classifier = foreign_class(
            source="Jzuluaga/accent-id-commonaccent_xlsr-en-english",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier"
        )
        self.sampling_rate = 16000

    def predict(self, audio_path):
        return self.classifier.classify_file(audio_path)

    def predict_whole(self, audio_path):
        """Predict directly on the full audio file"""
        out_prob, score, index, text_lab = self.predict(audio_path)
        return {
          "chunk": os.path.basename(audio_path),
          "label": text_lab,         # <-- use text_lab as the readable label
          "score": score.item() if hasattr(score, 'item') else score,
          "index": index,
          "out_prob": out_prob.tolist()  # Optional: convert tensor to list if needed
          }

    def split_in_half(self, audio_path):
        """Split audio into two equal-length parts"""
        waveform, sr = librosa.load(audio_path, sr=self.sampling_rate)
        midpoint = len(waveform) // 2

        base_dir = os.path.dirname(audio_path)
        filename = os.path.splitext(os.path.basename(audio_path))[0]
        half_dir = os.path.join(base_dir, f"{filename}_halves")
        os.makedirs(half_dir, exist_ok=True)

        paths = []
        for i, (start, end) in enumerate([(0, midpoint), (midpoint, len(waveform))]):
            half = waveform[start:end]
            half_path = os.path.join(half_dir, f"{filename}_half_{i}.wav")
            sf.write(half_path, half, sr)
            paths.append(half_path)

        return paths

    def predict_halves(self, audio_path):
        """Run prediction on two halves of the audio"""
        halves = self.split_in_half(audio_path)
        results = []
        for path in halves:
            out_prob, score, index, text_lab = self.predict(path)
            results.append({
               "chunk": os.path.basename(audio_path),
              "out_prob": out_prob,
              "score": score,
              "index":index,
              "text_lab":text_lab
            })
        return results
def download_audio(url, output_dir="downloaded_url_videos"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")
    command = f'yt-dlp -x --audio-format wav "{url}" -o "{output_path}"'
    subprocess.run(command, shell=True)
    return output_path
#url = "https://youtube.com/shorts/pv0kvjXObfg?si=brXaI54-9ksJoS55" 
#url_sound=download_audio(url)
#agent=Agent_classifier()
#results_o = agent.predict_whole(url_sound)
#print(len(results_o))
#print(type(results_o))
#result = agent.predict_whole(url_sound)
#print(results_o)
#print(results_o.values())




st.title("English Accent Classifier 🎙️🌍")
st.markdown("Upload a video link (e.g. YouTube, Loom, etc.) and get accent prediction.")

url = st.text_input("Enter public video URL:")

if url:
    with st.spinner("Downloading and analyzing..."):
        audio_path = download_audio(url)
        agent = Agent_classifier()
        result = agent.predict_whole(audio_path)

    st.success("Done!")
    st.markdown(f"**Accent:** `{result['label']}`")
    st.markdown(f"**Confidence Score:** `{round(result['score']*100, 2)}%`")

    st.markdown("**Optional Explanation:**")
    st.markdown(
        f"The model detected this as a `{result['label']}` English accent "
        f"with a confidence of `{round(result['score']*100, 2)}%`. "
        "It uses a fine-tuned wav2vec2 model trained on common English accents."
    )
