from collections import defaultdict
from qdaai.text_cleaning import strip_whitespace
from typing import TypedDict, Optional

# This code is intended to create markdown representations of threaded comments. 
# Typically you will want to adapt this to your own comment data structures, AI can help with that :)
# The salient thing is that each comment item must have an 'id', 'body', 'reply_to' field, whatever they are called.
# The 'reply_to' field should be None for top-level comments.
# The 'id' field should be unique for each comment.
# The 'date' field should be a Unix timestamp in seconds.

# An example MiniComment is given below.

class MiniComment(TypedDict):
    id: str
    body: str
    date: int  # Unix timestamp in seconds
    reply_to: Optional[str]
    score: int
    discussion_id: str

def process_discussion_markdown(comments: list[MiniComment], max_depth: int = 3) -> tuple[str, dict[int, str]]:
    """
    Process comments into a threaded markdown representation.
    Returns a tuple of (markdown, id_mapping) where id_mapping maps comment numbers to comment IDs.
    """
    # Create a threaded structure
    threads = defaultdict(list)
    root_comments = []
    
    for comment in sorted(comments, key=lambda x: x['date']):
        if comment['reply_to'] is None:
            root_comments.append(comment)
        else:
            threads[comment['reply_to']].append(comment)
    
    comment_number = 1
    id_mapping = {}

    def generate_markdown(comment: MiniComment, depth: int) -> str:
        nonlocal comment_number
        
        if depth > max_depth:
            return ""
        
        indent = "  " * (depth - 1)
        markdown = f"{indent}* [{comment_number}] {strip_whitespace(comment['body'])}\n"
        
        id_mapping[comment_number] = comment['id']
        comment_number += 1
        
        if comment['id'] in threads and depth < max_depth:
            for reply in threads[comment['id']]:
                markdown += generate_markdown(reply, depth + 1)
        
        return markdown
    
    result = ""
    for root_comment in root_comments:
        result += generate_markdown(root_comment, 1)
    
    return result.strip(), id_mapping

class ChunkResult:
    def __init__(self, markdown: str, id_mapping: dict[int, str], comment_count: int):
        self.markdown = markdown
        self.id_mapping = id_mapping
        self.comment_count = comment_count

def process_discussion_markdown_chunked(
    comments: list[MiniComment], 
    max_items: int = 50,
    max_depth: int = 3
) -> list[tuple[str, dict[int, str]]]:
    """
    Process comments into a threaded markdown representation. This version  splits the comments into chunks of max_items each. 
    Returns a list of tuples of (markdown, id_mapping) where id_mapping maps comment numbers to comment IDs.
    It attempts to preserve the sub-threads in each chunk.
    """
    # Create a threaded structure
    threads = defaultdict(list)
    root_comments = []
    
    # Create a set of valid comment IDs first
    valid_ids = {comment['id'] for comment in comments}
    
    for comment in sorted(comments, key=lambda x: x['date']):
        # If reply_to is None or references a non-existent comment, treat as root
        if comment['reply_to'] is None or comment['reply_to'] not in valid_ids:
            root_comments.append(comment)
        else:
            threads[comment['reply_to']].append(comment)
    
    chunks: list[ChunkResult] = []
    current_chunk = ChunkResult("", {}, 0)
    comment_number = 1

    def process_comment_tree(comment: MiniComment, depth: int) -> str:
        nonlocal comment_number
        
        if depth > max_depth:
            return ""
        
        if current_chunk.comment_count >= max_items:
            return ""
        
        indent = "  " * (depth - 1)
        markdown = f"{indent}* [{comment_number}] {strip_whitespace(comment['body'])}\n"
        
        current_chunk.id_mapping[comment_number] = comment['id']
        comment_number += 1
        current_chunk.comment_count += 1
        
        if comment['id'] in threads and depth < max_depth:
            for reply in threads[comment['id']]:
                markdown += process_comment_tree(reply, depth + 1)
        
        return markdown

    # Process root comments one at a time
    for root_comment in root_comments:
        if current_chunk.comment_count >= max_items:
            if current_chunk.markdown:
                chunks.append(current_chunk)
                current_chunk = ChunkResult("", {}, 0)
        
        result = process_comment_tree(root_comment, 1)
        current_chunk.markdown += result

    # Append the final chunk if it has content
    if current_chunk.markdown:
        chunks.append(current_chunk)

    return [(chunk.markdown.strip(), chunk.id_mapping) for chunk in chunks]

def process_flat_comments(
    comments: list[MiniComment], 
    max_items: int = 50
) -> list[tuple[str, dict[int, str]]]:
    """
    Process comments into flat numbered chunks, without threading.
    Returns a list of (markdown, id_mapping) tuples.
    """
    # Sort comments by date
    sorted_comments = sorted(comments, key=lambda x: x['date'])
    
    chunks: list[tuple[str, dict[int, str]]] = []
    current_markdown = []
    current_mapping = {}
    comment_number = 1
    
    for comment in sorted_comments:
        # If we've hit the chunk limit, save current chunk and start new one
        if len(current_markdown) >= max_items:
            chunks.append(
                ('\n'.join(current_markdown),
                 current_mapping.copy())
            )
            current_markdown = []
            current_mapping = {}
        
        # Format the current comment
        formatted_comment = f"{comment_number}. {strip_whitespace(comment['body'])}"
        current_markdown.append(formatted_comment)
        current_mapping[comment_number] = comment['id']
        comment_number += 1
    
    # Add the final chunk if it has content
    if current_markdown:
        chunks.append(
            ('\n'.join(current_markdown),
             current_mapping)
        )
    
    return chunks
