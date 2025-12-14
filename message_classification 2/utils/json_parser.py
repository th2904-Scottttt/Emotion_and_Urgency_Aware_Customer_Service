"""
JSON parsing utility.
"""

import json
import re


def extract_json(raw):
    """Extract JSON from model output."""
    try:
        return json.loads(raw)
    except:
        match = re.search(r"\{.*?\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Failed to extract JSON from: {raw}")