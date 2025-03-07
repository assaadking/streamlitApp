import uuid

def assign_unique_ids(list_of_strings):
    """Assigns unique UUIDs to a list of strings.
    Args:
        list_of_strings: A list of strings.
    Returns:
        A list of tuples, where each tuple contains a string and its unique UUID.
    """

    unique_ids = []
    for stin in list_of_strings:
        unique_id = str(uuid.uuid4())
        unique_ids.append(unique_id)

    return unique_ids

import re

def clean_for_database_filename(text):
    """
    Cleans a text string for use as a database file name.
    """
    # Remove special characters
    cleaned_text = re.sub(r'[<>:"/\|?* ]', '', text)

    # Remove leading and trailing spaces
    cleaned_text = cleaned_text.strip()

    # Limit length (adjust as needed)
    max_length = 255  # Example maximum length
    cleaned_text = cleaned_text[:max_length]

    return cleaned_text

import os

def find_faiss_files(directory):
    faiss_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
              if file.endswith(".faiss"):
                faiss_files.append(os.path.join(root, file))
    return faiss_files