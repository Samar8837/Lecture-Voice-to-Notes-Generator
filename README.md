# 🎙️ Lecture Voice-to-Notes Generator

This application transforms spoken lectures from audio files into structured, easy-to-study notes. Using Streamlit for the user interface and OpenAI's powerful AI models for transcription and content generation, students can quickly get summaries, flashcards, and quizzes from any lecture.

## ✨ Features

- **Audio Upload & Recording**: Upload common audio formats (`.mp3`, `.wav`, `.m4a`) or record audio directly in the browser.
- **Accurate Transcription**: Powered by OpenAI's Whisper model for fast and accurate speech-to-text.
- **Smart Content Generation**: Utilizes GPT models to generate multiple study formats:
    - **Concise Notes**: Hierarchical notes with titles, summaries, and key bullet points.
    - **Flashcards**: Question/Answer pairs for active recall.
    - **Quizzes**: Multiple-choice questions to test understanding.
- **Timestamps**: Each note is mapped back to its corresponding segment in the audio lecture.
- **Flexible & Modular**: Easy-to-understand code structure, ready for extension.
- **Containerized**: Includes a `Dockerfile` for easy deployment.

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- An OpenAI API Key

### Installation & Setup

Follow the steps to set up and run this project locally.

### Running the Application

1. Install dependencies: `pip install -r requirements.txt`
2. Set your API Key in a `.env` file.
3. Run the app: `streamlit run app.py`

## 🐳 Docker Deployment

To run with Docker:
1. Build the image: `docker build -t lecture-notes-generator .`

2. Run the container: `docker run -p 8501:8501 -e OPENAI_API_KEY="your_api_key_here" lecture-notes-generator`  
