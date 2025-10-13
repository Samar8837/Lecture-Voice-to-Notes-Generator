import os
from typing import Dict, Any, Literal
from openai import OpenAI
import streamlit as st

# --- Transcription Functions ---

def transcribe_with_openai_api(
    client: OpenAI, 
    audio_file_path: str,
    temp_file_path: str # Used to write bytes from UploadedFile
) -> Dict[str, Any]:
    """
    Transcribes audio using the OpenAI Whisper API with word-level timestamps.

    Args:
        client (OpenAI): The OpenAI API client.
        audio_file_path (str): The path to the audio file.
        temp_file_path (str): Path to the temporary file for API upload.

    Returns:
        Dict[str, Any]: The full transcription response object from OpenAI,
                        including text and segments with timestamps.
    """
    try:
        with open(temp_file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        # The API returns a Pydantic model, convert it to a dict for consistency
        return transcript.model_dump()
    except Exception as e:
        st.error(f"Error during OpenAI API transcription: {e}")
        return {}


def transcribe_with_local_whisper(audio_file_path: str) -> Dict[str, Any]:
    """
    Transcribes audio using the local 'openai-whisper' library.
    NOTE: This requires 'openai-whisper' and 'torch' to be installed,
    and 'ffmpeg' to be available on the system PATH.

    Args:
        audio_file_path (str): The path to the audio file.

    Returns:
        Dict[str, Any]: A dictionary matching the OpenAI API's verbose_json format.
    """
    try:
        import whisper
    except ImportError:
        st.error(
            "Local Whisper provider selected, but the required packages are not installed. "
            "Please run: pip install openai-whisper torch"
        )
        return {}

    try:
        # Using a smaller model for faster local processing.
        # Options: "tiny", "base", "small", "medium", "large"
        model = whisper.load_model("base") 
        result = model.transcribe(audio_file_path)
        return result
    except Exception as e:
        st.error(f"Error during local Whisper transcription: {e}")
        st.info("Please ensure you have `ffmpeg` installed on your system. It's a required dependency for local Whisper.")
        return {}

def get_transcription(
    provider: Literal['openai_api', 'local_whisper'],
    client: OpenAI,
    uploaded_file: Any
) -> Dict[str, Any]:
    """
    Main transcription handler that routes to the correct provider.
    
    It saves the uploaded file temporarily to disk because both local whisper
    and the OpenAI client work most reliably with file paths.

    Args:
        provider (str): The chosen transcription provider.
        client (OpenAI): The OpenAI API client.
        uploaded_file (Any): The file-like object from Streamlit's uploader.

    Returns:
        Dict[str, Any]: The transcription result.
    """
    temp_dir = "temp_audio"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    try:
        # Write the uploaded file bytes to a temporary file
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if provider == 'openai_api':
            result = transcribe_with_openai_api(client, uploaded_file.name, temp_file_path)
        elif provider == 'local_whisper':
            result = transcribe_with_local_whisper(temp_file_path)
        else:
            st.error(f"Unknown transcription provider: {provider}")
            result = {}
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
    
    return result