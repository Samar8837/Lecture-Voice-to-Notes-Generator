import json
from openai import OpenAI
from typing import List, Dict, Any, Literal, Generator
import streamlit as st

from utils import DEFAULT_MODEL, format_timestamp

# --- PROMPT DEFINITIONS ---

def get_notes_prompt(chunk_text: str, start_time: float, end_time: float) -> str:
    """Generates the prompt for creating hierarchical notes from a text chunk."""
    return f"""
    You are an expert academic assistant. Your task is to create detailed, structured study notes 
    from a transcript chunk of a lecture. The provided chunk is from {format_timestamp(start_time)} to {format_timestamp(end_time)}.

    Analyze the following text and generate notes in valid JSON format with the following structure:
    {{
      "title": "A concise, descriptive title for this section (5-10 words)",
      "summary": "A 2-4 sentence summary of the main points in this chunk.",
      "key_points": [
        "A list of key takeaways, concepts, or important facts as bullet points.",
        "Each point should be a complete sentence.",
        "Extract at least 3-5 key points."
      ]
    }}

    Transcript Chunk:
    ---
    {chunk_text}
    ---

    Ensure the output is only the JSON object, without any surrounding text or markdown.
    """

def get_flashcards_prompt(chunk_text: str) -> str:
    """Generates the prompt for creating flashcards."""
    return f"""
    You are a study aid tool. Based on the following lecture transcript chunk, generate exactly 10 flashcards
    to help a student learn the material.

    For each flashcard, provide a clear, concise question and a direct answer.
    Return the output as a valid JSON object containing a single key "flashcards" which is a list of objects,
    where each object has a "question" and "answer" key.

    Example format:
    {{
      "flashcards": [
        {{
          "question": "What is the capital of France?",
          "answer": "Paris."
        }},
        {{
          "question": "What is the formula for water?",
          "answer": "H2O."
        }}
      ]
    }}

    Transcript Chunk:
    ---
    {chunk_text}
    ---

    Generate exactly 10 flashcards. Ensure the output is only the JSON object.
    """

def get_quiz_prompt(chunk_text: str) -> str:
    """Generates the prompt for creating a multiple-choice quiz."""
    return f"""
    You are an expert quiz creator. From the following lecture transcript chunk, create a multiple-choice quiz
    with exactly 5 questions.

    For each question, provide:
    - The question text.
    - A list of 4 options (one correct, three plausible distractors).
    - The correct answer.

    Return the output as a valid JSON object containing a single key "quiz" which is a list of objects.
    Each object must have "question", "options", and "correct_answer" keys.

    Example format:
    {{
      "quiz": [
        {{
          "question": "Which planet is known as the Red Planet?",
          "options": ["Earth", "Mars", "Jupiter", "Venus"],
          "correct_answer": "Mars"
        }}
      ]
    }}
    
    Transcript Chunk:
    ---
    {chunk_text}
    ---

    Generate exactly 5 questions. Ensure the output is only the JSON object.
    """


# --- CONTENT GENERATION ---

def generate_content_for_chunk(
    client: OpenAI,
    chunk: Dict[str, Any],
    content_type: Literal['notes', 'flashcards', 'quiz'],
    model: str = DEFAULT_MODEL
) -> Dict[str, Any]:
    """
    Calls the OpenAI API to generate a specific type of content for a single text chunk.

    Args:
        client (OpenAI): The OpenAI API client.
        chunk (Dict[str, Any]): A dictionary containing the chunk text and timestamps.
        content_type (str): The type of content to generate ('notes', 'flashcards', 'quiz').
        model (str): The GPT model to use for generation.

    Returns:
        Dict[str, Any]: The parsed JSON response from the API.
    """
    prompt_map = {
        'notes': get_notes_prompt(chunk['text'], chunk['start_time'], chunk['end_time']),
        'flashcards': get_flashcards_prompt(chunk['text']),
        'quiz': get_quiz_prompt(chunk['text']),
    }
    
    prompt = prompt_map.get(content_type)
    if not prompt:
        raise ValueError(f"Invalid content type: {content_type}")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful academic assistant that outputs structured JSON."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3, # Lower temperature for more deterministic, factual output
        )
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        st.error(f"Error generating {content_type} for a chunk: {e}")
        return {}


def process_chunks_for_content(
    client: OpenAI,
    chunks: Generator[Dict[str, Any], None, None],
    content_types: List[str]
) -> Dict[str, List[Any]]:
    """
    Iterates through text chunks and generates all requested content types.

    Args:
        client (OpenAI): The OpenAI API client.
        chunks (Generator): A generator yielding text chunks.
        content_types (List[str]): A list of content types to generate (e.g., ['notes']).

    Returns:
        Dict[str, List[Any]]: A dictionary where keys are content types and values are
                               lists of generated content items.
    """
    generated_content = {ctype: [] for ctype in content_types}
    
    # Convert generator to list to show progress
    chunk_list = list(chunks)
    progress_bar = st.progress(0, text="Generating content from lecture chunks...")

    for i, chunk in enumerate(chunk_list):
        for content_type in content_types:
            # Generate one type of content per API call to keep prompts focused
            result = generate_content_for_chunk(client, chunk, content_type)
            
            if result:
                # For notes, add the timestamp directly
                if content_type == 'notes':
                    result['timestamp'] = f"{format_timestamp(chunk['start_time'])} - {format_timestamp(chunk['end_time'])}"
                    generated_content['notes'].append(result)
                # For flashcards and quizzes, the result is a dict with a key like "flashcards"
                # We extend the main list with the items from the chunk
                elif content_type in ['flashcards', 'quiz']:
                    if content_type in result and isinstance(result[content_type], list):
                        generated_content[content_type].extend(result[content_type])
        
        # Update progress bar
        progress_bar.progress((i + 1) / len(chunk_list), text=f"Processing chunk {i+1}/{len(chunk_list)}...")

    progress_bar.empty() # Clear the progress bar
    return generated_content