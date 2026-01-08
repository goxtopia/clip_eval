import sys
import streamlit as st
import os
import json
import torch
import pandas as pd
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from src.label_mapping import LabelMapper
from src.data import DataLoader, get_unique_texts, DatasetItem
from src.model import CLIPModel
from src.metrics import compute_metrics, compute_t2i_metrics
from src.utils import normalize_label
import subprocess
import numpy as np

# --- Config ---
HISTORY_DIR = "history"
DEBUG_ROOT = "debug"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)
if not os.path.exists(DEBUG_ROOT):
    os.makedirs(DEBUG_ROOT)

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

def save_debug_cases(items, bad_cases, mode, timestamp):
    """
    Saves debug cases to disk.
    """
    run_debug_dir = os.path.join(DEBUG_ROOT, f"{mode}_{timestamp}")
    if os.path.exists(run_debug_dir):
        shutil.rmtree(run_debug_dir)
    os.makedirs(run_debug_dir)

    if mode == "i2t":
        # bad_cases: {idx: {'preds': [], 'gt': []}}
        for idx, details in bad_cases.items():
            item = items[idx]
            rep_text = item.representative_text
            if not rep_text: continue

            # Organize by GT label
            class_folder = normalize_label(rep_text)
            class_dir = os.path.join(run_debug_dir, class_folder)
            os.makedirs(class_dir, exist_ok=True)

            src = item.image_path
            basename = os.path.basename(src)
            dst_img = os.path.join(class_dir, basename)
            dst_txt = os.path.join(class_dir, basename + ".debug.txt")

            try:
                shutil.copy2(src, dst_img)
                with open(dst_txt, "w", encoding="utf-8") as f:
                    f.write(f"I2T Failure\nRep Text: {rep_text}\n")
                    f.write("-" * 20 + "\nGT:\n")
                    for g in details["gt"]: f.write(f" - {g}\n")
                    f.write("-" * 20 + "\nPreds:\n")
                    for i, p in enumerate(details["preds"]): f.write(f" {i+1}. {p}\n")
            except Exception as e:
                print(f"Error saving debug case: {e}")

    elif mode == "t2i":
        # bad_cases: {query_idx: {'query': str, 'top_images_indices': [], 'top_images_gts': []}}
        for idx, details in bad_cases.items():
            query_text = details["query"]
            class_folder = normalize_label(query_text)

            # Avoid too deep nesting or long names
            class_folder = class_folder[:50]
            class_dir = os.path.join(run_debug_dir, class_folder)
            os.makedirs(class_dir, exist_ok=True)

            top_indices = details["top_images_indices"]
            top_gts = details["top_images_gts"]

            for rank, (img_idx, img_gt) in enumerate(zip(top_indices, top_gts)):
                item = items[img_idx]
                src = item.image_path
                basename = f"rank{rank+1}_" + os.path.basename(src)
                dst_img = os.path.join(class_dir, basename)
                dst_txt = os.path.join(class_dir, basename + ".debug.txt")
                try:
                    shutil.copy2(src, dst_img)
                    with open(dst_txt, "w", encoding="utf-8") as f:
                        f.write(f"T2I Failure\nQuery: {query_text}\nRank: {rank+1}\n")
                        f.write("-" * 20 + "\nRetrieved Image GT:\n")
                        for g in img_gt: f.write(f" - {g}\n")
                except Exception as e:
                    print(f"Error saving debug case: {e}")

    return run_debug_dir

def run_evaluation(items, model_name, pretrained, device, selected_filters, mode="i2t", debug_mode=False):
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

    # Prepare logic
    unique_texts = get_unique_texts(filtered_items)

    # Load Model
    model = CLIPModel(model_name, pretrained_tag if pretrained_tag else None, None, device, "eval_cache.pt")
    model.load()

    # Encode
    image_paths = [item.image_path for item in filtered_items]
    image_features = model.encode_images(image_paths, 32)

    templates = ["{}"]
    text_features = model.encode_texts(unique_texts, 32, templates)

    image_features = image_features.to(device)
    text_features = text_features.to(device)

    gt_class_sets = [item.gt_class_set for item in filtered_items]

    metrics = {}
    bad_cases = {}

    # Store per-item (or per-query) hit status for matrix breakdown
    # format: list of tuples (is_top1, is_top5) corresponding to 'filtered_items' or 'unique_texts'
    performance_records = []

    text_to_item_indices = {}
    for idx, item in enumerate(filtered_items):
        t = item.representative_text
        if t:
            if t not in text_to_item_indices: text_to_item_indices[t] = []
            text_to_item_indices[t].append(idx)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    debug_dir = None

    if mode == "i2t":
        sims_i2t = image_features @ text_features.T
        k = min(5, sims_i2t.size(1))
        _, top_idx = sims_i2t.topk(k=k, dim=-1)
        pred_classes = top_idx.cpu()

        class_to_images = {}
        for idx, item in enumerate(filtered_items):
            if not item.representative_text: continue
            rep_lbl = normalize_label(item.representative_text)
            if rep_lbl not in class_to_images: class_to_images[rep_lbl] = []
            class_to_images[rep_lbl].append(idx)

        metrics, per_class, bad_cases = compute_metrics(
            pred_classes, gt_class_sets, unique_texts, class_to_images
        )

        # Reconstruct per-item performance for matrix
        candidate_norms = [normalize_label(t) for t in unique_texts]
        for i in range(len(filtered_items)):
            gset = gt_class_sets[i]
            preds_idx = pred_classes[i].tolist()
            pred_norms = [candidate_norms[p] for p in preds_idx]

            if not gset:
                performance_records.append((0.0, 0.0))
                continue

            hit1 = 1.0 if pred_norms[0] in gset else 0.0
            hit5 = 1.0 if any(p in gset for p in pred_norms) else 0.0
            performance_records.append((hit1, hit5))

    else: # T2I
        sims_t2i = text_features @ image_features.T
        k = min(5, sims_t2i.size(1))
        _, top_idx = sims_t2i.topk(k=k, dim=-1)
        pred_images = top_idx.cpu()

        metrics, per_class, bad_cases = compute_t2i_metrics(
            pred_images, gt_class_sets, unique_texts
        )

        for stats in per_class:
            performance_records.append((stats["top1"], stats["top5"]))

    # Save Debug if enabled
    if debug_mode and bad_cases:
        debug_dir = save_debug_cases(filtered_items, bad_cases, mode, timestamp)

    # --- Cross-Attribute Matrix Calculation ---
    sample_tags = []
    all_tags = set()

    if mode == "i2t":
        for item in filtered_items:
            tags = set()
            for k, v in item.attributes.items():
                tag = f"{k}: {v}"
                tags.add(tag)
                all_tags.add(tag)
            sample_tags.append(tags)
    else:
        for idx, text in enumerate(unique_texts):
            origin_indices = text_to_item_indices.get(text, [])
            tags = set()
            for oid in origin_indices:
                item = filtered_items[oid]
                for k, v in item.attributes.items():
                    tag = f"{k}: {v}"
                    tags.add(tag)
                    all_tags.add(tag)
            sample_tags.append(tags)

    sorted_tags = sorted(list(all_tags))

    cross_matrix_data = {} # Key: (tag_i, tag_j) -> {sum1, sum5, count}

    for i in range(len(performance_records)):
        p_rec = performance_records[i]
        tags = sample_tags[i]

        tags_list = list(tags)
        for t1 in tags_list:
            for t2 in tags_list:
                key = (t1, t2)
                if key not in cross_matrix_data:
                    cross_matrix_data[key] = {"sum1": 0.0, "sum5": 0.0, "count": 0}

                cross_matrix_data[key]["sum1"] += p_rec[0]
                cross_matrix_data[key]["sum5"] += p_rec[1]
                cross_matrix_data[key]["count"] += 1

    matrix_top1 = []
    matrix_top5 = []

    for t1 in sorted_tags:
        row1 = []
        row5 = []
        for t2 in sorted_tags:
            key = (t1, t2)
            if key in cross_matrix_data:
                d = cross_matrix_data[key]
                avg1 = d["sum1"] / d["count"]
                avg5 = d["sum5"] / d["count"]
                row1.append(avg1)
                row5.append(avg5)
            else:
                row1.append(np.nan) # Use np.nan instead of None
                row5.append(np.nan)
        matrix_top1.append(row1)
        matrix_top5.append(row5)

    cross_results = {
        "tags": sorted_tags,
        "matrix_top1": matrix_top1,
        "matrix_top5": matrix_top5
    }

    # --- Save Matrix Plots ---
    plot_paths = {}
    if sorted_tags:
        try:
            # Helper to save plot
            def save_heatmap(matrix, tags, title, suffix):
                fig, ax = plt.subplots(figsize=(12, 10))

                # Convert to float/nan safely
                data = np.array(matrix, dtype=object)
                data[data == None] = np.nan
                data = data.astype(float)

                sns.heatmap(
                    data, annot=True, fmt=".1%",
                    xticklabels=tags, yticklabels=tags,
                    cmap="Blues", ax=ax, vmin=0, vmax=1
                )
                plt.xticks(rotation=45, ha="right")
                plt.title(title)
                plt.tight_layout()

                fname = f"run_{timestamp}_matrix_{suffix}.png"
                fpath = os.path.join(HISTORY_DIR, fname)
                plt.savefig(fpath)
                plt.close(fig)
                return fpath

            plot_paths["top1"] = save_heatmap(matrix_top1, sorted_tags, f"Top-1 Accuracy Matrix ({mode})", "top1")
            plot_paths["top5"] = save_heatmap(matrix_top5, sorted_tags, f"Top-5 Accuracy Matrix ({mode})", "top5")
        except Exception as e:
            print(f"Error generating plots: {e}")

    # --- Restore Matrix Results (Per-Attribute) ---
    matrix_results = {} # attr -> val -> {top1, top5, count}

    # We can reconstruct this from cross_matrix_data diagonal
    # (tag, tag) is the performance for that tag alone
    for tag in sorted_tags:
        key = (tag, tag)
        if key in cross_matrix_data:
            d = cross_matrix_data[key]
            # split tag "Key: Value"
            if ": " in tag:
                k, v = tag.split(": ", 1)
                if k not in matrix_results: matrix_results[k] = {}
                matrix_results[k][v] = {
                    "top1": d["sum1"] / d["count"],
                    "top5": d["sum5"] / d["count"],
                    "count": d["count"]
                }
            else:
                # Fallback for weird tags
                if "Misc" not in matrix_results: matrix_results["Misc"] = {}
                matrix_results["Misc"][tag] = {
                    "top1": d["sum1"] / d["count"],
                    "top5": d["sum5"] / d["count"],
                    "count": d["count"]
                }

    results = {
        "metrics": metrics,
        "cross_results": cross_results,
        "matrix_results": matrix_results, # Restored
        "plot_paths": plot_paths, # Added
        "n_samples": len(filtered_items) if mode == "i2t" else len(unique_texts),
        "timestamp": timestamp,
        "model": model_name,
        "filters": selected_filters,
        "mode": mode,
        "debug_dir": debug_dir
    }

    return results, None

def save_run(results):
    ts = results["timestamp"]
    fname = f"run_{ts}.json"
    path = os.path.join(HISTORY_DIR, fname)
    with open(path, "w") as f:
        # JSON dump cleanup
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

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Run Evaluation", "Debug View", "Auto-labeling", "History & Compare"])

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

        # Config Cols
        c1, c2, c3 = st.columns(3)
        with c1:
            eval_mode = st.selectbox("Evaluation Mode", ["i2t", "t2i"])
        with c2:
            enable_debug = st.checkbox("Enable Debug Mode (Save Bad Cases)")

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

        if st.button("Run Evaluation"):
            with st.spinner(f"Running {eval_mode.upper()} evaluation..."):
                device = "cuda" if torch.cuda.is_available() else "cpu"
                results, error = run_evaluation(items, model_name, pretrained_tag, device, selected_filters, mode=eval_mode, debug_mode=enable_debug)

                if error:
                    st.error(error)
                else:
                    st.session_state["last_results"] = results
                    save_path = save_run(results)
                    st.success(f"Run complete! Saved to {save_path}")
                    if results.get("debug_dir"):
                        st.info(f"Debug cases saved to {results['debug_dir']}")

        if "last_results" in st.session_state:
            res = st.session_state["last_results"]
            st.subheader("Results")

            m = res["metrics"]
            c1, c2 = st.columns(2)
            c1.metric("Global Top-1", f"{m['global_top1']*100:.2f}%")
            c2.metric("Global Top-5", f"{m['global_top5']*100:.2f}%")

            # Unified Filter Tag Performance Table
            st.subheader("Filter Tags Performance")
            if "matrix_results" in res:
                unified_data = []
                # Flatten the matrix_results into one list
                # matrix_results structure: {Attr: {Val: {stats}}}
                for attr, vals in res["matrix_results"].items():
                    for val_name, stats in vals.items():
                        tag_display = f"{attr}: {val_name}"
                        unified_data.append({
                            "Tag": tag_display,
                            "Top-1": stats['top1'],
                            "Top-5": stats['top5'],
                            "Count": stats['count']
                        })

                # Sort by Tag name
                unified_data.sort(key=lambda x: x["Tag"])

                # Format for display
                df_display = pd.DataFrame(unified_data)
                if not df_display.empty:
                    # Apply formatting
                    df_display["Top-1"] = df_display["Top-1"].apply(lambda x: f"{x*100:.2f}%")
                    df_display["Top-5"] = df_display["Top-5"].apply(lambda x: f"{x*100:.2f}%")
                    st.dataframe(df_display, use_container_width=True)

            st.subheader("Cross-Filter Accuracy Matrix")

            # Use saved plots if available (preferred for performance)
            plot_paths = res.get("plot_paths", {})
            if plot_paths.get("top1"):
                mtab1, mtab2 = st.tabs(["Top-1 Accuracy", "Top-5 Accuracy"])
                with mtab1:
                    if os.path.exists(plot_paths["top1"]):
                        st.image(plot_paths["top1"], caption="Top-1 Accuracy Matrix", use_container_width=True)
                    else:
                        st.warning("Plot file not found.")
                with mtab2:
                    if os.path.exists(plot_paths["top5"]):
                        st.image(plot_paths["top5"], caption="Top-5 Accuracy Matrix", use_container_width=True)
                    else:
                        st.warning("Plot file not found.")
            else:
                # Fallback to dynamic plotting
                cross = res.get("cross_results", {})
                tags = cross.get("tags", [])
                m1 = cross.get("matrix_top1", [])
                m5 = cross.get("matrix_top5", [])

                if tags:
                    def to_float_array(arr):
                        narr = np.array(arr, dtype=object)
                        narr[narr == None] = np.nan
                        return narr.astype(float)

                    m1_arr = to_float_array(m1)
                    m5_arr = to_float_array(m5)

                    mtab1, mtab2 = st.tabs(["Top-1 Accuracy", "Top-5 Accuracy"])
                    with mtab1:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(m1_arr, annot=True, fmt=".1%", xticklabels=tags, yticklabels=tags, cmap="Blues", ax=ax, vmin=0, vmax=1)
                        plt.xticks(rotation=45, ha="right")
                        st.pyplot(fig)
                    with mtab2:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        sns.heatmap(m5_arr, annot=True, fmt=".1%", xticklabels=tags, yticklabels=tags, cmap="Blues", ax=ax, vmin=0, vmax=1)
                        plt.xticks(rotation=45, ha="right")
                        st.pyplot(fig)

with tab2:
    st.header("Debug View")

    # List all directories in debug root
    debug_runs = sorted([d for d in os.listdir(DEBUG_ROOT) if os.path.isdir(os.path.join(DEBUG_ROOT, d))], reverse=True)

    selected_debug_run = st.selectbox("Select Debug Run", [""] + debug_runs)

    if selected_debug_run:
        run_path = os.path.join(DEBUG_ROOT, selected_debug_run)
        # Structure: run_path / class_folder / images...

        classes = sorted([d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))])

        selected_class = st.selectbox("Select Class/Query", [""] + classes)

        if selected_class:
            class_path = os.path.join(run_path, selected_class)
            files = sorted([f for f in os.listdir(class_path) if not f.endswith(".txt")])

            st.write(f"Found {len(files)} cases.")

            # Pagination or grid
            for f in files:
                col1, col2 = st.columns([1, 2])
                img_p = os.path.join(class_path, f)
                txt_p = img_p + ".debug.txt"

                with col1:
                    st.image(img_p, use_container_width=True)
                with col2:
                    if os.path.exists(txt_p):
                        with open(txt_p, "r") as tf:
                            content = tf.read()
                        st.text_area(f"Details for {f}", content, height=200)
                    else:
                        st.write("No debug info file.")
                st.divider()

with tab3:
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

with tab4:
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
            html_content += "<h2>Summary</h2><table><tr><th>Run</th><th>Mode</th><th>Model</th><th>Samples</th><th>Global Top-1</th><th>Global Top-5</th><th>Filters</th></tr>"
            for d in comparison_data:
                filters_str = ", ".join([f"{k}:{v}" for k,v in d.get("filters", {}).items() if v])
                mode_str = d.get("mode", "i2t") # default for old runs

                # Handle old runs structure where metrics might be flattened or inside 'metrics' dict
                # The old code had metrics_i2t at root. The new code has metrics dict.
                if "metrics" in d:
                    g_top1 = d["metrics"]["global_top1"]
                    g_top5 = d["metrics"]["global_top5"]
                elif "metrics_i2t" in d:
                    g_top1 = d["metrics_i2t"]["global_top1"]
                    # Old code didn't save global top 5 explicitly in the dict root,
                    # but computed it inside compute_metrics.
                    # We might not have it in old json.
                    g_top5 = 0.0
                else:
                    g_top1 = 0.0
                    g_top5 = 0.0

                html_content += f"<tr><td>{d['filename']}</td><td>{mode_str}</td><td>{d['model']}</td><td>{d['n_samples']}</td><td>{g_top1*100:.2f}%</td><td>{g_top5*100:.2f}%</td><td>{filters_str}</td></tr>"
            html_content += "</table>"

            # Detailed Matrix Comparison (Per-Attribute)
            html_content += "<h2>Attribute Performance</h2>"

            # Collect all unique tags across runs (using restored matrix_results)
            all_attr_keys = set()
            for d in comparison_data:
                if "matrix_results" in d:
                    all_attr_keys.update(d["matrix_results"].keys())
                # If only cross_results exist (intermediate version), we might miss this.
                # But we just restored it.

            # Also handle older logic where matrix_results existed but structure was different?
            # Old structure: matrix_results[attr][val] = {top1, count}.
            # New structure: matrix_results[attr][val] = {top1, top5, count}.
            # Compatible.

            for attr in sorted(list(all_attr_keys)):
                html_content += f"<h3>{attr}</h3><table><tr><th>Value</th>"
                for d in comparison_data:
                    html_content += f"<th>{d['filename']} (Top1 / Top5)</th>"
                html_content += "</tr>"

                all_vals = set()
                for d in comparison_data:
                    vals = d.get("matrix_results", {}).get(attr, {}).keys()
                    all_vals.update(vals)

                for val in sorted(list(all_vals)):
                    html_content += f"<tr><td>{val}</td>"
                    for d in comparison_data:
                        stats = d.get("matrix_results", {}).get(attr, {}).get(val)
                        if stats:
                            t1 = stats.get('top1', 0)
                            t5 = stats.get('top5', 0)
                            cnt = stats.get('count', 0)
                            html_content += f"<td>{t1*100:.1f}% / {t5*100:.1f}% (n={cnt})</td>"
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
