import os
import tempfile

import pdfplumber
import pytesseract
from pdf2image import convert_from_path

from src.parsing.text_cleaner import clean_text


def parse_pdf(filepath) -> tuple:
    """
    Hybrid PDF parser: pdfplumber (primary) + OCR (fallback).

    Args:
        filepath: Path string or file-like object (e.g. Streamlit UploadedFile).

    Returns:
        (extracted_text, error_message) — one of them will be None.
    """
    text = ""
    temp_path = None

    try:
        # Handle file-like objects (Streamlit UploadedFile)
        if hasattr(filepath, "read"):
            suffix = ".pdf"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(filepath.read())
            tmp.close()
            temp_path = tmp.name
            filepath_str = temp_path
            # Reset file pointer so caller can re-read if needed
            filepath.seek(0)
        else:
            filepath_str = str(filepath)

        # Stage 1: Try pdfplumber first
        with pdfplumber.open(filepath_str) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                text += page_text + "\n"

        # Stage 2: If pdfplumber output is too short, fallback to OCR
        if len(text.strip()) < 50:
            text = ""
            images = convert_from_path(filepath_str, dpi=300)
            for img in images:
                ocr_text = pytesseract.image_to_string(
                    img, lang='ind+eng',
                    config='--psm 6'
                )
                text += ocr_text + "\n"

        # Stage 3: Clean the text
        cleaned_text = clean_text(text)

        if len(cleaned_text.strip()) < 30:
            return None, f"Teks terlalu pendek setelah cleaning ({len(cleaned_text)} chars)"

        return cleaned_text, None

    except Exception as e:
        return None, f"Error parsing {filepath}: {str(e)}"

    finally:
        # Clean up temp file
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
