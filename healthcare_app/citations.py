"""Row-level evidence snippets for transparency / rubric traceability."""


def evidence_snippet(text: str, max_len: int = 320) -> str:
    """Return a bounded verbatim excerpt from facility unstructured text."""
    t = (text or "").strip().replace("\n", " ")
    if len(t) <= max_len:
        return t
    return t[: max_len - 1] + "…"
