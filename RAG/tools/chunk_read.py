"""Tool to read full content of documentation chunks."""

import json
from .data_loader import load_docs_data


def chunk_read(chunk_ids: list) -> str:
    """Read full content of specific chunks.

    Args:
        chunk_ids: List of chunk IDs to read

    Returns:
        JSON string with full chunk contents
    """
    print(f"  [TOOL EXECUTED] chunk_read(chunk_ids={chunk_ids})")

    docs = load_docs_data()
    chunks = []

    for chunk_id in chunk_ids:
        if 0 <= chunk_id < len(docs):
            chunk = docs[chunk_id]
            chunks.append({
                "chunk_id": chunk_id,
                "chunk_link": chunk.get('chunk_link', ''),
                "chunk_heading": chunk.get('chunk_heading', ''),
                "text": chunk.get('text', '')
            })
        else:
            chunks.append({
                "chunk_id": chunk_id,
                "error": f"Chunk ID {chunk_id} out of range (0-{len(docs)-1})"
            })

    result = json.dumps(chunks, indent=2)
    print(f"  [TOOL RESULT] Read {len(chunks)} chunks")
    return result
