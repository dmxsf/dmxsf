"""Data engineering: build and audit a dedup filter.

Wiki pages
  Chinese: https://dmxsf.com/wiki/pretraining/data-engineering/
  English: https://dmxsf.com/en/wiki/pretraining/data-engineering/
"""


def shingles(text: str, width: int) -> set[tuple[str, ...]]:
    """Return case-insensitive consecutive word shingles."""
    if width < 1:
        raise ValueError("width must be positive")
    words = text.casefold().split()
    return {tuple(words[i : i + width]) for i in range(len(words) - width + 1)}


def jaccard(left: set, right: set) -> float:
    """Intersection over union for nonempty shingle sets."""
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def banding_probability(similarity: float, bands: int, rows: int) -> float:
    """Probability of any matching band under independent MinHash rows."""
    return 1 - (1 - similarity**rows) ** bands
