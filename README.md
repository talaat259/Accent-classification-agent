# English Accent Classifier 🎙️🌍

A powerful web application that analyzes English accents from video URLs using advanced machine learning models. The application downloads audio from video links, transcribes the speech, classifies the English accent, and provides intelligent responses to user queries about the content.

## Features

- **Accent Classification**: Identifies different English accents using a fine-tuned wav2vec2 model
- **Audio Transcription**: Converts speech to text using OpenAI's Whisper model
- **Video URL Support**: Downloads audio from various video platforms (YouTube, Loom, etc.)
- **Intelligent Q&A**: Answer questions about the transcribed content
- **Text Summarization**: Automatically summarize longer audio content
- **Web Interface**: Clean, user-friendly Flask web application
- **Public Access**: Uses ngrok for public URL sharing

## Technology Stack

- **Machine Learning**: 
  - SpeechBrain for accent classification
  - OpenAI Whisper for transcription
  - Transformers (HuggingFace) for Q&A and summarization
- **Audio Processing**: librosa, torchaudio, soundfile
- **Web Framework**: Flask with ngrok tunneling
- **Video Processing**: yt-dlp for downloading, moviepy for editing
- **Frontend**: HTML with Jinja2 templating

## Installation

### Prerequisites
- Python 3.7+
- CUDA-compatible GPU (optional, for faster processing)
- FFmpeg (automatically installed in the script)

### Setup


1. **Install dependencies**
```bash
pip install speechbrain==0.5.15
pip install -U yt-dlp
pip install flask pyngrok
pip install streamlit
pip install git+https://github.com/openai/whisper.git
pip install librosa torch torchaudio soundfile moviepy transformers
```

2. **Install system dependencies**
```bash
sudo apt update && sudo apt install -y ffmpeg
```

3. **Set up ngrok authentication**
   - Sign up at [ngrok.com](https://ngrok.com)
   - Replace the auth token in the code with your own:
   ```python
   conf.get_default().auth_token = "YOUR_NGROK_TOKEN_HERE"
   ```

## Usage

### Running the Application

1. **Start the Flask server**
```bash
python accent_clean_code.py
```

2. **Access the application**
   - The script will automatically generate a public ngrok URL
   - Open the displayed URL in your browser

### Using the Web Interface

1. **Enter a video URL** in the input field (YouTube, Loom, etc.)
2. **Optional**: Ask a question about the content or request a summary
3. **Click "Analyze"** to process the video

### Example Queries
- "Summarize the main points"
- "What is the speaker talking about?"
- "What are the key topics discussed?"

## Core Components

### Agent_classifier Class
- **Purpose**: Handles accent classification using SpeechBrain
- **Methods**:
  - `predict()`: Classify accent from audio file
  - `predict_whole()`: Full audio analysis
  - `split_in_half()`: Split audio for detailed analysis
  - `predict_halves()`: Analyze audio segments separately

### transcrib Class
- **Purpose**: Audio-to-text transcription using Whisper
- **Method**: `audio_2_text()`: Convert audio to text

### Utility Functions
- `download_audio()`: Download and convert video to audio
- Flask routes for web interface handling

## Supported Accents

The model is trained to recognize common English accents including:
- American English
- British English
- Australian English
- Canadian English
- And other regional variations

## Output Format

The application provides:
```json
{
  "label": "Detected accent name",
  "score": 0.95,
  "confidence": "95%",
  "transcription": "Full text transcription",
  "llm_response": "Answer to user query"
}


## Configuration

### Model Settings
- **Sampling Rate**: 16kHz (automatically handled)
- **Audio Format**: WAV (converted automatically)
- **Processing**: GPU acceleration when available

### Web Server
- **Port**: 5000 (default)
- **Public Access**: Via ngrok tunnel
- **Debug Mode**: Enabled for development

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**
   - The application automatically falls back to CPU if CUDA is unavailable
   - For faster processing, ensure CUDA is properly installed

2. **Video Download Failures**
   - Check if the video URL is publicly accessible
   - Some platforms may have restrictions on automated downloads

3. **Audio Processing Errors**
   - Ensure FFmpeg is properly installed
   - Check audio file format compatibility

4. **Model Loading Issues**
   - First run may take longer due to model downloads
   - Ensure stable internet connection for model fetching

### Performance Optimization
- Use GPU acceleration when available
- Consider audio length for processing time
- Batch processing for multiple files

## Dependencies

Core packages listed in requirements.txt:
- speechbrain==0.5.15
- torch, torchaudio
- transformers
- librosa
- whisper
- flask
- pyngrok
- yt-dlp
- moviepy
- soundfile

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project uses various open-source models and libraries. Please check individual component licenses for commercial use.

## Acknowledgments

- **SpeechBrain** team for the accent classification model
- **OpenAI** for the Whisper transcription model
- **HuggingFace** for the transformer models
- **Jzuluaga** for the pre-trained accent classification model

## Support

For issues, questions, or contributions, please:
1. Check the troubleshooting section
2. Review existing issues
3. Create a new issue with detailed information

---

**Note**: This application is designed for educational and research purposes. Ensure you have appropriate permissions for any content you analyze.
