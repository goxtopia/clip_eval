import streamlit as st
import json
import os
import numpy as np
from src.label_mapping import LabelMapper
from src.data import DataLoader

@st.cache_data
def load_data(data_dir, mapping, filter_path, text_path):
    mapper = LabelMapper(mapping)
    loader = DataLoader(data_dir, mapper, filter_json_path=filter_path, text_json_path=text_path)
    return loader.load()

def save_run(results, history_dir="history"):
    ts = results["timestamp"]
    model = results.get("model", "unknown").replace("/", "_").replace(" ", "_")
    tag = results.get("pretrained", "unknown").replace("/", "_").replace(" ", "_")
    fname = f"run_{model}_{tag}_{ts}.json"
    path = os.path.join(history_dir, fname)
    with open(path, "w") as f:
        # We need to handle np.nan for JSON dump
        # json.dump doesn't like NaN by default or it dumps as NaN which might be valid in some parsers but standard says no.
        # Python's json dump will dump NaN as NaN, but strictly it's not valid JSON.
        # However, for internal read back it might be fine, or we can convert to null.
        # Let's clean NaN to None (null) before saving.

        def clean_nans(obj):
            if isinstance(obj, list):
                return [clean_nans(x) for x in obj]
            if isinstance(obj, dict):
                return {k: clean_nans(v) for k, v in obj.items()}
            if isinstance(obj, float) and np.isnan(obj):
                return None
            return obj

        json.dump(clean_nans(results), f, indent=4)
    return path
