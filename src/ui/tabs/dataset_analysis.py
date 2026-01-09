import streamlit as st
import os
import pandas as pd
from src.app_utils import load_data

def render_dataset_analysis_tab(config):
    st.header("Dataset Analysis")
    st.write("Analyze the distribution of tags in your dataset.")

    dataset_dir = config["dataset_dir"]
    mapping_path = config["mapping_path"]
    filter_json_path = config["filter_json_path"]
    text_json_path = config["text_json_path"]

    if st.button("Load & Analyze Dataset", key="btn_analyze"):
         if not os.path.exists(dataset_dir):
            st.error("Dataset directory not found.")
         else:
            items = load_data(dataset_dir, mapping_path, filter_json_path, text_json_path)
            st.success(f"Loaded {len(items)} items.")

            # Aggregate Tags
            tag_counts = {}
            for item in items:
                for k, v in item.attributes.items():
                    if k == 'filename': continue
                    if k not in tag_counts: tag_counts[k] = {}
                    
                    vals = v if isinstance(v, list) else [v]
                    for val in vals:
                        if val not in tag_counts[k]: tag_counts[k][val] = 0
                        tag_counts[k][val] += 1

            # Display
            if not tag_counts:
                st.warning("No tags found. Run Auto-labeling first.")
            else:
                for cat in sorted(tag_counts.keys()):
                    st.subheader(f"Attribute: {cat}")
                    data = tag_counts[cat]
                    df = pd.DataFrame(list(data.items()), columns=["Value", "Count"])
                    df = df.sort_values(by="Count", ascending=False)

                    c1, c2 = st.columns([1, 2])
                    with c1:
                        st.dataframe(df, hide_index=True)
                    with c2:
                        st.bar_chart(df.set_index("Value"))
