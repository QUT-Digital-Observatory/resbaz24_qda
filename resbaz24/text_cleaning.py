import re


def truncate_words(text: str, words: int) -> str:
    split_text = text.split()
    if len(split_text) > words:
        return " ".join(split_text[:words])
    return text


def strip_whitespace(text: str) -> str:
    return " ".join(text.split())


def strip_possessives(text):
    # Replace U+2019 with a standard apostrophe
    standardized_text = text.replace("\u2019", "'")

    # Regex to remove possessive forms ('s or just ')
    # The pattern targets 's and ' at the end of words
    cleaned_text = re.sub(r"'s?\b", "", standardized_text)
    return cleaned_text


def remove_urls(text) -> str:
    # This regex pattern is designed to match most URLs
    url_pattern = (
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\'(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )
    return re.sub(url_pattern, "", text)

def eliminate_multi_spaces(text: str):
    return re.sub(r'\s+', ' ', text)
