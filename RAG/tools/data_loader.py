"""Data loader for documentation chunks."""

import json
import os

# Global variable to cache documentation data
_DOCS_DATA = None


def load_docs_data():
    """Load documentation data from JSON file (cached)."""
    global _DOCS_DATA
    if _DOCS_DATA is None:
        # Get path relative to tools directory
        data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'anthropic_docs.json')
        with open(data_path, 'r', encoding='utf-8') as f:
            _DOCS_DATA = json.load(f)
    return _DOCS_DATA
