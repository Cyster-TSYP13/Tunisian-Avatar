import re
from pathlib import Path
from typing import Set

try:
    import pandas as pd
except Exception:
    pd = None


def load_bad_words(xlsx_path: str = "T-HSAB.xlsx", sheet_name: str | None = None, column_name: str | None = None) -> Set[str]:
    """Load bad/abusive words from an Excel file into a set of normalized words.

    Args:
        xlsx_path: Path to the Excel file.
        sheet_name: Optional sheet name to read.
        column_name: Optional column name containing words.

    Returns:
        A set of normalized lowercase words (may be empty if file not found or pandas missing).
    """
    path = Path(xlsx_path)
    if pd is None:
        return set()
    if not path.exists():
        return set()

    try:
        df = pd.read_excel(path, sheet_name=sheet_name)
        # Prefer explicit column if provided
        if column_name and column_name in df.columns:
            series = df[column_name]
        else:
            # Choose the first textual column
            text_cols = [c for c in df.columns if df[c].dtype == object]
            if text_cols:
                series = df[text_cols[0]]
            else:
                series = df.iloc[:, 0]

        words = series.dropna().astype(str).str.strip().str.lower().tolist()
        cleaned = set()
        for w in words:
            # basic clean: remove punctuation and extra whitespace
            w2 = re.sub(r"[\p{P}\p{S}]", "", w) if False else re.sub(r"[\W_]+", " ", w)
            w2 = w2.strip()
            if w2:
                cleaned.add(w2.lower())

        return cleaned
    except Exception:
        return set()


def sanitize_text(text: str, bad_words: Set[str], placeholder: str = "[filtered]") -> str:
    """Replace occurrences of any bad word with a placeholder.

    This uses simple substring matching (case-insensitive), sorted by length
    so longer phrases are replaced first to avoid partial masking.
    """
    if not bad_words:
        return text

    # Sort by length to replace longer phrases first
    for w in sorted(bad_words, key=len, reverse=True):
        if not w:
            continue
        try:
            text = re.sub(re.escape(w), placeholder, text, flags=re.IGNORECASE)
        except re.error:
            # fallback: plain replace
            text = text.replace(w, placeholder)

    return text
