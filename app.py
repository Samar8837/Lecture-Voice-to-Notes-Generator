import os
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from utils import chunk_transcript_segments
from transcribe import get_transcription
from summarize import process_chunks_for_content
from streamlit_mic_recorder import mic_recorder

# --- Page Configuration ---
st.set_page_config(
    page_title="Lecture Voice-to-Notes Generator",
    page_icon="🎙️",
    layout="wide",
)

# --- Load Environment Variables and API Key ---
load_dotenv()
API_KEY_ENV = os.getenv("OPENAI_API_KEY")

# --- UI Functions ---

def display_sidebar():
    """Displays the sidebar with API key input and settings."""
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key Input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=API_KEY_ENV or "",
            help="Enter your OpenAI API key. It's recommended to set this as an environment variable `OPENAI_API_KEY`."
        )

        st.info("💡 **Tip**: Don't have an API key? Get one from [OpenAI](https://platform.openai.com/account/api-keys).")
        
        provider = "openai_api" # Hardcode for simplicity

        # Content Generation Options
        st.header("📝 Output Options")
        content_options = st.multiselect(
            "Select the content to generate:",
            ["Notes", "Flashcards", "Quiz"],
            default=["Notes", "Flashcards"]
        )
        
        return api_key, provider, [opt.lower() for opt in content_options]

def display_audio_input():
    """Displays audio input options: file upload and in-browser recording."""
    st.header("1. Provide Lecture Audio")
    
    # Using columns for a cleaner layout
    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Upload an audio file (.mp3, .wav, .m4a)",
            type=["mp3", "wav", "m4a", "m4b", "mpeg"]
        )

    with col2:
        st.markdown("<div style='text-align: center;'>OR</div>", unsafe_allow_html=True)
        # In-browser audio recording
        st.write("Record audio directly:")
        audio_bytes = mic_recorder(
            start_prompt="Start Recording",
            stop_prompt="Stop Recording",
            key='recorder'
        )

    # Logic to decide which audio source to use
    if audio_bytes:
        # Create a file-like object for the recorded audio
        from io import BytesIO
        recorded_file = BytesIO(audio_bytes['bytes'])
        recorded_file.name = "recording.wav" # Give it a name for processing
        return recorded_file
    
    return uploaded_file

def display_generated_content(content: dict):
    """Renders the generated notes, flashcards, and quiz in separate tabs."""
    st.header("🎉 Your Study Materials")

    # Create tabs for each content type that was generated
    tab_titles = [key.capitalize() for key in content.keys() if content[key]]
    if not tab_titles:
        st.warning("No content was generated. Please check the logs or try again.")
        return

    tabs = st.tabs(tab_titles)
    
    content_map = {
        "Notes": ("notes", display_notes),
        "Flashcards": ("flashcards", display_flashcards),
        "Quiz": ("quiz", display_quiz),
    }

    for i, title in enumerate(tab_titles):
        with tabs[i]:
            content_key, display_func = content_map.get(title, (None, None))
            if content_key and display_func:
                display_func(content[content_key])

def display_notes(notes: list):
    """Displays structured notes."""
    for note in notes:
        with st.container(border=True):
            st.subheader(note.get('title', 'Untitled Section'))
            st.markdown(f"**Timestamp:** `{note.get('timestamp', 'N/A')}`")
            st.markdown(f"**Summary:** {note.get('summary', 'No summary available.')}")
            st.markdown("**Key Points:**")
            for point in note.get('key_points', []):
                st.markdown(f"- {point}")

def display_flashcards(flashcards: list):
    """Displays flashcards with a toggle for the answer."""
    for i, card in enumerate(flashcards):
        with st.expander(f"**Flashcard {i+1}:** {card.get('question', 'No question.')}"):
            st.write(card.get('answer', 'No answer.'))

def display_quiz(quiz: list):
    """Displays a multiple-choice quiz and provides feedback."""
    for i, q in enumerate(quiz):
        st.markdown(f"**Question {i+1}:** {q.get('question', '')}")
        user_answer = st.radio(
            "Select your answer:",
            options=q.get('options', []),
            key=f"quiz_{i}",
            label_visibility="collapsed"
        )
        if st.button("Check Answer", key=f"check_{i}"):
            if user_answer == q.get('correct_answer'):
                st.success(f"Correct! The answer is {q.get('correct_answer')}.")
            else:
                st.error(f"Incorrect. The correct answer is {q.get('correct_answer')}.")


# --- Main Application Logic ---

def main():
    """Main function to run the Streamlit app."""
    st.title("🎙️ Lecture Voice-to-Notes Generator")
    st.markdown("Transform your lecture audio into organized study notes, flashcards, and quizzes effortlessly.")

    api_key, provider, content_options = display_sidebar()

    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar to proceed.")
        st.stop()
    
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        st.stop()

    audio_file = display_audio_input()

    if audio_file and st.button("✨ Generate Notes", type="primary"):
        if not content_options:
            st.error("Please select at least one output option in the sidebar.")
            st.stop()

        with st.status("Processing your lecture... This may take a few minutes.", expanded=True) as status:
            # 1. Transcription
            st.write("Transcribing audio...")
            transcript_data = get_transcription(provider, client, audio_file)
            
            if not transcript_data or "segments" not in transcript_data:
                status.update(label="Transcription failed. Please try a different audio file or check the API key.", state="error")
                st.stop()
            
            status.update(label="Transcription complete!")
            
            # 2. Chunking
            st.write("Chunking transcript for analysis...")
            # Set chunk size based on a buffer for the prompt itself
            transcript_chunks = chunk_transcript_segments(transcript_data["segments"], max_tokens_per_chunk=3000)

            # 3. Content Generation
            st.write("Generating selected content...")
            generated_content = process_chunks_for_content(client, transcript_chunks, content_options)
            
            status.update(label="Processing complete!", state="complete")
        
        # 4. Display Results
        display_generated_content(generated_content)

if __name__ == "__main__":
    main()