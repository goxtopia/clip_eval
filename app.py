import streamlit as st
import os
import json
import torch
import pandas as pd
import shutil
from datetime import datetime
from src.label_mapping import LabelMapper
from src.data import DataLoader, get_unique_texts, DatasetItem
from src.model import CLIPModel
from src.metrics import compute_metrics, compute_t2i_metrics
from src.utils import normalize_label
import subprocess

# --- Config ---
HISTORY_DIR = "history"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

st.set_page_config(page_title="CLIP Eval Tool", layout="wide")

# --- Sidebar ---
st.sidebar.title("Configuration")

dataset_dir = st.sidebar.text_input("Dataset Directory", "data")
model_name = st.sidebar.text_input("Model Name/ID", "MobileCLIP-S2")
pretrained_tag = st.sidebar.text_input("Pretrained Tag", "openai")
filter_json_path = st.sidebar.text_input("Filter JSON Path", "filter_attributes.json")
mapping_path = st.sidebar.text_input("Mapping JSON Path", "mapping.json")

st.sidebar.markdown("### Auto-labeling Config")
api_url = st.sidebar.text_input("API URL", "http://localhost:8000/v1")
api_key = st.sidebar.text_input("API Key", "sk-...", type="password")
vlm_model = st.sidebar.text_input("VLM Model Name", "gpt-4o-mini")

# --- Functions ---

@st.cache_data
def load_data(data_dir, mapping, filter_path):
    mapper = LabelMapper(mapping)
    loader = DataLoader(data_dir, mapper, filter_json_path=filter_path)
    return loader.load()

def run_evaluation(items, model_name, pretrained, device, selected_filters):
    # Filter items
    filtered_items = []
    for item in items:
        match = True
        for key, desired_values in selected_filters.items():
            if not desired_values: continue

            val = item.attributes.get(key, "Unknown")
            if val not in desired_values:
                match = False
                break
        if match:
            filtered_items.append(item)

    if not filtered_items:
        return None, "No items match the selected filters."

    # Prepare logic (similar to main.py)
    unique_texts = get_unique_texts(filtered_items)

    # Load Model
    model = CLIPModel(model_name, pretrained_tag if pretrained_tag else None, None, device, "eval_cache.pt")
    model.load()

    # Encode
    image_paths = [item.image_path for item in filtered_items]
    image_features = model.encode_images(image_paths, 32)

    templates = ["{}"] # Simple template for now
    text_features = model.encode_texts(unique_texts, 32, templates)

    image_features = image_features.to(device)
    text_features = text_features.to(device)

    gt_class_sets = [item.gt_class_set for item in filtered_items]

    # I2T
    sims_i2t = image_features @ text_features.T
    _, top_idx = sims_i2t.topk(k=5, dim=-1)
    pred_classes = top_idx.cpu()

    class_to_images = {}
    for idx, item in enumerate(filtered_items):
        if not item.representative_text: continue
        rep_lbl = normalize_label(item.representative_text)
        if rep_lbl not in class_to_images: class_to_images[rep_lbl] = []
        class_to_images[rep_lbl].append(idx)

    metrics_i2t, per_class_i2t, bad_cases_i2t = compute_metrics(
        pred_classes, gt_class_sets, unique_texts, class_to_images
    )

    # Matrix Breakdown
    # We want to know accuracy per filter attribute value
    # e.g. Time: Day -> Acc, Night -> Acc
    matrix_results = {}

    # Attributes to check
    # We get all keys present in attributes of filtered_items
    all_attr_keys = set()
    for item in filtered_items:
        all_attr_keys.update(item.attributes.keys())

    correct_top1 = [] # Need to reconstruct this list matching filtered_items indices
    candidate_norms = [normalize_label(t) for t in unique_texts]

    # Re-calculate per-item correctness for matrix grouping
    for i in range(len(filtered_items)):
        gset = gt_class_sets[i]
        preds_idx = pred_classes[i].tolist()
        pred_norms = [candidate_norms[p] for p in preds_idx]

        hit1 = 1.0 if (gset and pred_norms[0] in gset) else 0.0
        correct_top1.append(hit1)

    for attr_key in all_attr_keys:
        matrix_results[attr_key] = {}
        # Group by value
        groups = {}
        for idx, item in enumerate(filtered_items):
            val = item.attributes.get(attr_key, "Unknown")
            if val not in groups: groups[val] = []
            groups[val].append(correct_top1[idx])

        for val, scores in groups.items():
            avg = sum(scores) / len(scores)
            matrix_results[attr_key][val] = {
                "top1": avg,
                "count": len(scores)
            }

    results = {
        "metrics_i2t": metrics_i2t,
        "matrix_results": matrix_results,
        "n_samples": len(filtered_items),
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "filters": selected_filters
    }

    return results, None

def save_run(results):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"run_{ts}.json"
    path = os.path.join(HISTORY_DIR, fname)
    with open(path, "w") as f:
        json.dump(results, f, indent=4)
    return path

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["Run Evaluation", "Auto-labeling", "History & Compare"])

with tab1:
    st.header("Run Evaluation")

    if st.button("Load Data"):
        if not os.path.exists(dataset_dir):
            st.error("Dataset directory not found.")
        else:
            items = load_data(dataset_dir, mapping_path, filter_json_path)
            st.session_state["items"] = items
            st.success(f"Loaded {len(items)} items.")

    if "items" in st.session_state:
        items = st.session_state["items"]

        # Extract available filter options
        filter_options = {}
        for item in items:
            for k, v in item.attributes.items():
                if k not in filter_options: filter_options[k] = set()
                filter_options[k].add(v)

        # Display Filters
        st.subheader("Filters")
        selected_filters = {}
        cols = st.columns(len(filter_options) if filter_options else 1)
        for i, (k, vals) in enumerate(filter_options.items()):
            with cols[i % len(cols)]:
                selected_vals = st.multiselect(f"Filter by {k}", sorted(list(vals)))
                selected_filters[k] = selected_vals

        if st.button("Run"):
            with st.spinner("Running evaluation..."):
                device = "cuda" if torch.cuda.is_available() else "cpu"
                results, error = run_evaluation(items, model_name, pretrained_tag, device, selected_filters)

                if error:
                    st.error(error)
                else:
                    st.session_state["last_results"] = results
                    save_path = save_run(results)
                    st.success(f"Run complete! Saved to {save_path}")

        if "last_results" in st.session_state:
            res = st.session_state["last_results"]
            st.subheader("Results")
            st.metric("Global Top-1", f"{res['metrics_i2t']['global_top1']*100:.2f}%")

            st.subheader("Matrix Results")
            for attr, vals in res["matrix_results"].items():
                st.write(f"**{attr}**")
                # Create a dataframe for display
                df_data = []
                for val_name, stats in vals.items():
                    df_data.append({
                        "Value": val_name,
                        "Top-1 Acc": f"{stats['top1']*100:.2f}%",
                        "Count": stats['count']
                    })
                st.dataframe(pd.DataFrame(df_data))

with tab2:
    st.header("Auto-labeling")
    st.write("This tool will label images with attributes: Person Size, Time of Day, Blurry, Resolution.")

    if st.button("Start Auto-labeling"):
        cmd = [
            sys.executable, "src/autolabel.py",
            "--dataset", dataset_dir,
            "--output", filter_json_path,
            "--api_url", api_url,
            "--api_key", api_key,
            "--model", vlm_model
        ]

        with st.spinner("Labeling... check terminal for progress logs."):
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                st.success("Auto-labeling complete!")
                st.text_area("Output", stdout)
            else:
                st.error("Auto-labeling failed.")
                st.text_area("Error", stderr)

with tab3:
    st.header("History & Compare")

    history_files = sorted([f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")], reverse=True)

    selected_runs = st.multiselect("Select runs to compare", history_files)

    if st.button("Generate Comparison Report"):
        if not selected_runs:
            st.warning("Select at least one run.")
        else:
            # Aggregate data
            comparison_data = []
            for fname in selected_runs:
                with open(os.path.join(HISTORY_DIR, fname)) as f:
                    data = json.load(f)
                    data["filename"] = fname
                    comparison_data.append(data)

            # Create HTML
            html_content = "<html><head><title>Comparison Report</title>"
            html_content += "<style>table {border-collapse: collapse; width: 100%;} th, td {border: 1px solid black; padding: 8px; text-align: left;} th {background-color: #f2f2f2;}</style>"
            html_content += "</head><body><h1>Comparison Report</h1>"

            # Summary Table
            html_content += "<h2>Summary</h2><table><tr><th>Run</th><th>Model</th><th>Samples</th><th>Global Top-1</th><th>Filters</th></tr>"
            for d in comparison_data:
                filters_str = ", ".join([f"{k}:{v}" for k,v in d.get("filters", {}).items() if v])
                html_content += f"<tr><td>{d['filename']}</td><td>{d['model']}</td><td>{d['n_samples']}</td><td>{d['metrics_i2t']['global_top1']*100:.2f}%</td><td>{filters_str}</td></tr>"
            html_content += "</table>"

            # Detailed Matrix Comparison
            # We assume all runs have "matrix_results"
            html_content += "<h2>Matrix Breakdown</h2>"

            # Collect all attribute keys seen across runs
            all_attrs = set()
            for d in comparison_data:
                all_attrs.update(d.get("matrix_results", {}).keys())

            for attr in sorted(list(all_attrs)):
                html_content += f"<h3>{attr}</h3><table><tr><th>Value</th>"
                for d in comparison_data:
                    html_content += f"<th>{d['filename']} (Acc / Count)</th>"
                html_content += "</tr>"

                # Collect all values for this attr
                all_vals = set()
                for d in comparison_data:
                    vals = d.get("matrix_results", {}).get(attr, {}).keys()
                    all_vals.update(vals)

                for val in sorted(list(all_vals)):
                    html_content += f"<tr><td>{val}</td>"
                    for d in comparison_data:
                        stats = d.get("matrix_results", {}).get(attr, {}).get(val)
                        if stats:
                            html_content += f"<td>{stats['top1']*100:.2f}% / {stats['count']}</td>"
                        else:
                            html_content += "<td>N/A</td>"
                    html_content += "</tr>"
                html_content += "</table>"

            html_content += "</body></html>"

            report_path = "comparison_report.html"
            with open(report_path, "w") as f:
                f.write(html_content)

            st.success(f"Report generated: {report_path}")
            st.markdown(f"[Download Report](./{report_path})", unsafe_allow_html=True)

            # Display basic comparison chart
            comp_df = []
            for d in comparison_data:
                comp_df.append({
                    "Run": d["filename"],
                    "Top-1": d['metrics_i2t']['global_top1']
                })
            st.bar_chart(pd.DataFrame(comp_df).set_index("Run"))
