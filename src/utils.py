def normalize_label(s: str) -> str:
    """Normalizes label: strip whitespace and convert to lower case."""
    if s is None:
        return ""
    return s.strip().lower()
