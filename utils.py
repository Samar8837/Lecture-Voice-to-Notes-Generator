import tiktoken
from typing import List, Dict, Any, Generator

# --- COST & TOKEN MANAGEMENT ---
# Recommended model for a balance of cost, speed, and quality.
# Other options: "gpt-4-turbo", "gpt-3.5-turbo"
DEFAULT_MODEL = "gpt-4o-mini"
TOKEN_LIMITS = {
    "gpt-4o-mini": 128000,
    "gpt-4-turbo": 128000,
    "gpt-3.5-turbo": 16385,
}

def get_token_limit(model: str = DEFAULT_MODEL) -> int:
    """Returns the token limit for a given model."""
    return TOKEN_LIMITS.get(model, 4096) # Default to 4096 if model not found

def estimate_token_count(text: str, model: str = DEFAULT_MODEL) -> int:
    """
    Estimates the number of tokens a string will occupy for a given model.
    
    Args:
        text (str): The input text.
        model (str): The model name to estimate tokens for.

    Returns:
        int: The estimated number of tokens.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: Model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(text))

def chunk_transcript_segments(
    segments: List[Dict[str, Any]], 
    max_tokens_per_chunk: int = 4000
) -> Generator[Dict[str, Any], None, None]:
    """
    Chunks transcript segments into manageable sizes based on a token limit.
    This is crucial for processing long lectures without exceeding API context windows.

    Args:
        segments (List[Dict[str, Any]]): A list of whisper transcript segments.
        max_tokens_per_chunk (int): The maximum number of tokens allowed per chunk.

    Yields:
        Generator[Dict[str, Any], None, None]: A generator of chunks, each containing
        'text', 'start_time', and 'end_time'.
    """
    current_chunk_text = ""
    current_chunk_start_time = 0
    chunk_segments = []

    if not segments:
        return

    current_chunk_start_time = segments[0]['start']

    for segment in segments:
        segment_text = segment['text']
        
        # Estimate tokens for the current chunk if this new segment is added
        potential_chunk_text = current_chunk_text + " " + segment_text
        if estimate_token_count(potential_chunk_text) > max_tokens_per_chunk:
            # Yield the current chunk if it's not empty
            if current_chunk_text:
                yield {
                    "text": current_chunk_text.strip(),
                    "start_time": current_chunk_start_time,
                    "end_time": chunk_segments[-1]['end']
                }
            
            # Start a new chunk with the current segment
            current_chunk_text = segment_text
            chunk_segments = [segment]
            current_chunk_start_time = segment['start']
        else:
            # Add the segment to the current chunk
            current_chunk_text = potential_chunk_text
            chunk_segments.append(segment)

    # Yield the last remaining chunk
    if current_chunk_text:
        yield {
            "text": current_chunk_text.strip(),
            "start_time": current_chunk_start_time,
            "end_time": chunk_segments[-1]['end']
        }

def format_timestamp(seconds: float) -> str:
    """
    Formats a time in seconds into a more readable MM:SS or HH:MM:SS format.

    Args:
        seconds (float): The time in seconds.

    Returns:
        str: The formatted timestamp string.
    """
    if seconds < 0:
        return "00:00"
    
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    else:
        return f"{minutes:02d}:{seconds:02d}"