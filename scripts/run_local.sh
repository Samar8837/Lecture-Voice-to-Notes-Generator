#!/bin/bash

# This script helps run the Streamlit application locally.
# It ensures environment variables from a .env file are loaded.

# Check if .env file exists
if [ -f .env ]; then
  # Use set -a to export all variables created by the source command
  set -a
  source .env
  set +a
else
  echo "Warning: .env file not found. Make sure to set the OPENAI_API_KEY environment variable."
fi

# Run the Streamlit application
echo "Starting Lecture Voice-to-Notes Generator..."
streamlit run app.py