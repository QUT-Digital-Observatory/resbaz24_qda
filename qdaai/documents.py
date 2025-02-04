from typing import NamedTuple, Optional
from qdaai.text_cleaning import truncate_words, strip_whitespace
import random

class PromptDocument(NamedTuple):
    prompt: str
    idlist: list[str]


class SimpleDocument(NamedTuple):
    id: str
    text: str


def documents_to_prompts(
    data: list[SimpleDocument],
    prompt: str,
    max_words: int,
    max_comment_words: Optional[int] = None,
    shuffle: bool = False,
) -> list[PromptDocument]:
    """
    Cram comments into documents with a prompt, ensuring the total word count doesn't exceed max_words.
    The comments are truncated to max_comment_words if specified, and optionally shuffled within each document.

    Args:
    data: A list of SimpleDocument objects containing comment id and text.
    prompt: The prompt to prepend to each document.
    max_words: The maximum number of words allowed in each document.
    max_comment_words: The number of words to truncate each comment to.
    shuffle: Whether to shuffle the comments within each document.
    """

    def create_and_add_document(
        documents: list[PromptDocument], prompt: str, items: list[SimpleDocument]
    ):
        if shuffle:
            random.shuffle(items)

        document_content = "\n".join(
            f"{i}. {item.text}" for i, item in enumerate(items, 1)
        )
        document_prompt = f"{prompt}\n{document_content}"
        documents.append(PromptDocument(document_prompt, [item.id for item in items]))

    documents: list[PromptDocument] = []
    current_document_items: list[SimpleDocument] = []
    total_words = 0
    prompt_words = len(prompt.split())

    for document in data:
        comment_text = document.text
        if max_comment_words:
            comment_text = truncate_words(comment_text, max_comment_words)
        stripped_text = strip_whitespace(comment_text)
        comment_words = len(stripped_text.split())

        if (
            total_words + comment_words + prompt_words + len(current_document_items)
            > max_words
        ):
            if current_document_items:
                create_and_add_document(documents, prompt, current_document_items)
            current_document_items = []
            total_words = 0

        current_document_items.append(SimpleDocument(document.id, stripped_text))
        total_words += comment_words

    if current_document_items:
        create_and_add_document(documents, prompt, current_document_items)

    return documents
