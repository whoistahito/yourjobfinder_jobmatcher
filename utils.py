import re


def merge_requirements(a, b):
    """Union two Requirements objects (dedupe + sorted for determinism)."""
    from api_schema import Requirements

    return Requirements(
        skills=sorted(set((a.skills or [])) | set((b.skills or []))),
        experiences=sorted(set((a.experiences or [])) | set((b.experiences or []))),
        qualifications=sorted(set((a.qualifications or [])) | set((b.qualifications or []))),
    )


def chunk_markdown(markdown_text, chunk_size):
    chars_per_chunk = chunk_size * 4
    chunks = re.split(r'(#{1,6}\s+.*?\n)', markdown_text)
    result_chunks = []
    current_chunk = ""
    for chunk in chunks:
        if len(current_chunk) + len(chunk) < chars_per_chunk:
            current_chunk += chunk
        else:
            if current_chunk:
                result_chunks.append(current_chunk)
            current_chunk = chunk
    if current_chunk:
        result_chunks.append(current_chunk)
    if not result_chunks or min(len(c) for c in result_chunks) > chars_per_chunk:
        result_chunks = [markdown_text[i:i + chars_per_chunk] for i in range(0, len(markdown_text), chars_per_chunk)]
    return result_chunks
