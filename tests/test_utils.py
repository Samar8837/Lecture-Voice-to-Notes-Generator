import pytest
from unittest.mock import patch
from .. import utils  # Use relative import for local testing

# --- Test format_timestamp ---

def test_format_timestamp_seconds():
    """Test formatting for times less than a minute."""
    assert utils.format_timestamp(35.6) == "00:35"

def test_format_timestamp_minutes():
    """Test formatting for times less than an hour."""
    assert utils.format_timestamp(150) == "02:30"
    assert utils.format_timestamp(3599) == "59:59"

def test_format_timestamp_hours():
    """Test formatting for times greater than an hour."""
    assert utils.format_timestamp(3661) == "01:01:01"

def test_format_timestamp_zero():
    """Test formatting for zero seconds."""
    assert utils.format_timestamp(0) == "00:00"
    
def test_format_timestamp_negative():
    """Test formatting for negative input."""
    assert utils.format_timestamp(-10) == "00:00"

# --- Test chunk_transcript_segments ---

# Sample segments data structure similar to Whisper's output
MOCK_SEGMENTS = [
    {'text': ' This is the first sentence.', 'start': 0.0, 'end': 2.0},
    {'text': ' And this is a much longer second sentence designed to take up more tokens.', 'start': 2.5, 'end': 7.0},
    {'text': ' A third one.', 'start': 7.5, 'end': 8.5},
    {'text': ' Fourth segment here.', 'start': 9.0, 'end': 10.0},
    {'text': ' The final segment is quite long as well, pushing the token limit.', 'start': 10.5, 'end': 15.0},
]

@patch('utils.estimate_token_count')
def test_chunking_with_token_limit(mock_estimate_tokens):
    """
    Test that segments are chunked correctly based on a mocked token count.
    """
    # Assign a token count to each sentence text
    token_counts = {
        ' This is the first sentence.': 10,
        ' And this is a much longer second sentence designed to take up more tokens.': 30,
        ' A third one.': 5,
        ' Fourth segment here.': 5,
        ' The final segment is quite long as well, pushing the token limit.': 25
    }
    # Mock the estimator to return our predefined counts
    mock_estimate_tokens.side_effect = lambda text: sum(
        token_counts[s] for s in token_counts if s in text
    )

    # Set a limit that forces a split, expecting 2 chunks.
    max_tokens = 40 
    chunks = list(utils.chunk_transcript_segments(MOCK_SEGMENTS, max_tokens_per_chunk=max_tokens))
    
    assert len(chunks) == 2
    assert "second sentence" in chunks[0]['text']
    assert "A third one" not in chunks[0]['text']
    assert chunks[0]['end_time'] == 7.0
    assert "final segment" in chunks[1]['text']
    assert chunks[1]['start_time'] == 7.5

def test_chunking_empty_segments():
    """Test that the chunker handles empty input gracefully."""
    chunks = list(utils.chunk_transcript_segments([]))
    assert len(chunks) == 0

@patch('utils.estimate_token_count', return_value=5)
def test_chunking_no_split_needed(mock_estimate_tokens):
    """Test that if total tokens are under the limit, only one chunk is created."""
    chunks = list(utils.chunk_transcript_segments(MOCK_SEGMENTS, max_tokens_per_chunk=1000))
    assert len(chunks) == 1
    assert chunks[0]['end_time'] == 15.0