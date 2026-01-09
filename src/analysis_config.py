import json
import os

CONFIG_FILE = "analysis_queries.json"

def load_queries():
    """Returns a list of dicts: {'name': str, 'tags': [str]}"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except:
            return []
    return []

def save_queries(queries):
    """Saves list of queries to JSON."""
    with open(CONFIG_FILE, "w") as f:
        json.dump(queries, f, indent=4)
