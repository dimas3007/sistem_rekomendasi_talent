"""Tests for PDF parsing and text cleaning modules."""

import pytest
from src.parsing.text_cleaner import clean_text


class TestTextCleaner:
    def test_empty_input(self):
        assert clean_text("") == ""
        assert clean_text(None) == ""

    def test_basic_cleaning(self):
        text = "Hello   World\t\tTest"
        result = clean_text(text)
        assert "  " not in result
        assert "\t" not in result

    def test_page_number_removal(self):
        text = "Some content\nPage 1 of 5\nMore content"
        result = clean_text(text)
        assert "Page 1 of 5" not in result

    def test_page_number_removal_indonesian(self):
        text = "Some content\nPage 2 dari 10\nMore content"
        result = clean_text(text)
        assert "Page 2 dari 10" not in result

    def test_multiple_newlines(self):
        text = "Line 1\n\n\n\n\nLine 2"
        result = clean_text(text)
        assert "\n\n\n" not in result

    def test_unicode_normalization(self):
        text = "caf\u00e9 r\u00e9sum\u00e9"
        result = clean_text(text)
        assert len(result) > 0

    def test_strip_lines(self):
        text = "  hello  \n  world  "
        result = clean_text(text)
        lines = result.split("\n")
        for line in lines:
            assert line == line.strip()

    def test_preserves_content(self):
        text = "Muhammad Rizki, S.Kom\nPython Developer\nJakarta"
        result = clean_text(text)
        assert "Muhammad Rizki" in result
        assert "Python Developer" in result
        assert "Jakarta" in result
