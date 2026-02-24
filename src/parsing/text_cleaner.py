import re
import unicodedata


def clean_text(text: str) -> str:
    """Bersihkan teks hasil parsing PDF."""
    if not text:
        return ""

    # 1. Normalisasi Unicode
    text = unicodedata.normalize("NFKD", text)

    # 2. Hapus karakter non-printable (keep Latin + extended Latin + newlines)
    text = re.sub(r'[^\x20-\x7E\u00C0-\u024F\u1E00-\u1EFF\n]', ' ', text)

    # 3. Hapus header/footer berulang (nomor halaman, "Page X of Y")
    text = re.sub(r'Page\s+\d+\s+(of|dari)\s+\d+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # 4. Normalisasi whitespace
    text = re.sub(r'[ \t]+', ' ', text)        # Multiple spaces/tabs -> single space
    text = re.sub(r'\n{3,}', '\n\n', text)      # Multiple newlines -> double newline

    # 5. Trim setiap baris
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()
