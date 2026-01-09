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
filter_json_path = st.sidebar.text_input("Tag JSON Path", "filter_attributes.json")
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
        return None, "No items match the selected tags."

    # Prepare logic
    unique_texts = get_unique_texts(filtered_items)

    # Load Model
    model = CLIPModel(model_name, None, pretrained_tag, device, "eval_cache.pt")
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
                if k != 'filename':
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
                    if k != 'filename':
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
    matrix_counts = [] # Added for mixing heatmap

    # This loop generates the FULL matrix (all vs all)
    # We will still compute it for completeness in 'cross_results' if needed,
    # or we can just compute the sub-matrices for display.
    # The existing code computes the full NxN.
    # We can keep it to support the "old" view if needed, or just for data structure consistency.
    
    print(sorted_tags)
    
    for t1 in sorted_tags:
        row1 = []
        row5 = []
        row_c = []
        for t2 in sorted_tags:
            key = (t1, t2)
            if key in cross_matrix_data:
                d = cross_matrix_data[key]
                avg1 = d["sum1"] / d["count"]
                avg5 = d["sum5"] / d["count"]
                row1.append(avg1)
                row5.append(avg5)
                row_c.append(d["count"])
            else:
                row1.append(np.nan) 
                row5.append(np.nan)
                row_c.append(0)
        matrix_top1.append(row1)
        matrix_top5.append(row5)
        matrix_counts.append(row_c)

    # Collect per-tag stats (diagonal)
    tag_stats = []
    for t in sorted_tags:
        key = (t, t)
        if key in cross_matrix_data:
            d = cross_matrix_data[key]
            acc1 = d["sum1"] / d["count"]
            acc5 = d["sum5"] / d["count"]
            tag_stats.append({
                "Tag": t,
                "Count": d["count"],
                "ACC1": acc1,
                "ACC5": acc5
            })

    cross_results = {
        "tags": sorted_tags,
        "matrix_top1": matrix_top1,
        "matrix_top5": matrix_top5,
        "matrix_counts": matrix_counts, 
        "tag_stats": tag_stats 
    }

    # Identify Tag Groups
    # Extract keys from "Key: Value" tags
    tag_groups = {}
    for t in sorted_tags:
        if ": " in t:
            k, v = t.split(": ", 1)
            if k not in tag_groups: tag_groups[k] = []
            tag_groups[k].append(t)
    
    cross_results["tag_groups"] = tag_groups

    results = {
        "metrics": metrics,
        "cross_results": cross_results,
        "n_samples": len(filtered_items) if mode == "i2t" else len(unique_texts),
        "timestamp": timestamp,
        "model": model_name,
        "pretrained": pretrained,
        "filters": selected_filters,
        "mode": mode,
        "debug_dir": debug_dir
    }

    return results, None

def save_run(results):
    ts = results["timestamp"]
    model = results.get("model", "unknown").replace("/", "_").replace(" ", "_")
    tag = results.get("pretrained", "unknown").replace("/", "_").replace(" ", "_")
    fname = f"run_{model}_{tag}_{ts}.json"
    path = os.path.join(HISTORY_DIR, fname)
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
        with c3:
            show_support = st.checkbox("Show Support (Count)")

        # Extract available filter options
        filter_options = {}
        for item in items:
            for k, v in item.attributes.items():
                if k not in filter_options: filter_options[k] = set()
                filter_options[k].add(v)

        # Display Filters
        st.subheader("Tags")
        selected_filters = {}
        cols = st.columns(len(filter_options) if filter_options else 1)
        for i, (k, vals) in enumerate(filter_options.items()):
            with cols[i % len(cols)]:
                selected_vals = st.multiselect(f"Tag by {k}", sorted(list(vals)))
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

            # --- Tag Performance Table ---
            cross = res.get("cross_results", {})
            tag_stats = cross.get("tag_stats", [])
            if tag_stats:
                st.subheader("Tag Performance")
                df_stats = pd.DataFrame(tag_stats)
                # Format percentages
                df_stats["ACC1"] = df_stats["ACC1"].apply(lambda x: f"{x:.2%}")
                df_stats["ACC5"] = df_stats["ACC5"].apply(lambda x: f"{x:.2%}")
                st.dataframe(df_stats)

            # --- Tag Mixing Matrix ---
            st.subheader("Tag Mixing Matrix (Accuracy)")
            
            tags = cross.get("tags", [])
            tag_groups = cross.get("tag_groups", {})
            print(f"Tag Groups Found: {tag_groups.keys()}")
            m1 = cross.get("matrix_top1", [])
            m5 = cross.get("matrix_top5", [])
            ts_str = res.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))

            if tags and m1:
                def to_float_array(arr):
                    narr = np.array(arr, dtype=object) 
                    narr[narr == None] = np.nan
                    return narr.astype(float)

                m1_arr = to_float_array(m1)
                m5_arr = to_float_array(m5)

                counts_arr = np.array(cross.get("matrix_counts", []))

                # Iterate over groups to create sub-matrices
                # If no groups found (no ": " in tags), fallback to full matrix?
                # The prompt implies structured tags. If simple tags, maybe just one group.
                
                # Ensure "All" is always present and first
                groups_to_plot = ["All"] + sorted(list(tag_groups.keys()))
                
                for grp in groups_to_plot:
                    print(f"Processing group: {grp}")
                    st.markdown(f"#### Group: {grp}")
                    
                    if grp == "All":
                         row_indices = range(len(tags))
                         col_indices = range(len(tags))
                         sub_tags_row = tags
                         sub_tags_col = tags
                    else:
                        # Row indices: tags in this group
                        row_indices = [i for i, t in enumerate(tags) if t in tag_groups[grp]]
                        # Col indices: tags NOT in this group
                        col_indices = [i for i, t in enumerate(tags) if t not in tag_groups[grp]]
                        
                        sub_tags_row = [tags[i] for i in row_indices]
                        sub_tags_col = [tags[i] for i in col_indices]
                    
                    if not row_indices or not col_indices:
                        st.write("No interactions with outside groups.")
                        continue

                    # Extract sub-arrays
                    # m1_arr[row_indices, :][:, col_indices]
                    # numpy advanced indexing
                    sub_m1 = m1_arr[np.ix_(row_indices, col_indices)]
                    sub_m5 = m5_arr[np.ix_(row_indices, col_indices)]
                    sub_counts = counts_arr[np.ix_(row_indices, col_indices)]
                    
                    # Prepare annotations
                    if show_support:
                        # Create annotation matrix
                        annot_m1 = np.empty(sub_m1.shape, dtype=object)
                        annot_m5 = np.empty(sub_m5.shape, dtype=object)
                        for r in range(sub_m1.shape[0]):
                            for c in range(sub_m1.shape[1]):
                                count = int(sub_counts[r, c])
                                val1 = sub_m1[r, c]
                                val5 = sub_m5[r, c]

                                if np.isnan(val1):
                                    annot_m1[r, c] = ""
                                else:
                                    annot_m1[r, c] = f"{val1:.1%}\n({count})"

                                if np.isnan(val5):
                                    annot_m5[r, c] = ""
                                else:
                                    annot_m5[r, c] = f"{val5:.1%}\n({count})"

                        annot1 = annot_m1
                        annot5 = annot_m5
                        fmt = "" # format is handled in annot strings
                    else:
                        annot1 = True
                        annot5 = True
                        fmt = ".1%"

                    # Create Tabs for this group
                    mtab1, mtab2 = st.tabs([f"{grp} Top-1", f"{grp} Top-5"])
                    
                    # Generate and Save Top-1
                    fig1, ax1 = plt.subplots(figsize=(10, len(sub_tags_row)*0.5 + 2))
                    sns.heatmap(
                        sub_m1,
                        annot=annot1,
                        fmt=fmt,
                        xticklabels=sub_tags_col,
                        yticklabels=sub_tags_row,
                        cmap="Blues",
                        ax=ax1,
                        vmin=0, vmax=1
                    )
                    plt.xticks(rotation=45, ha="right")
                    plt.title(f"{grp} Interaction Top-1")
                    
                    safe_grp = grp.replace(" ", "_").replace("/", "_")
                    path_top1 = os.path.join(HISTORY_DIR, f"heatmap_{safe_grp}_top1_{ts_str}.png")
                    plt.savefig(path_top1, bbox_inches="tight")
                    
                    # Generate and Save Top-5
                    fig2, ax2 = plt.subplots(figsize=(10, len(sub_tags_row)*0.5 + 2))
                    sns.heatmap(
                        sub_m5,
                        annot=annot5,
                        fmt=fmt,
                        xticklabels=sub_tags_col,
                        yticklabels=sub_tags_row,
                        cmap="Blues",
                        ax=ax2,
                        vmin=0, vmax=1
                    )
                    plt.xticks(rotation=45, ha="right")
                    plt.title(f"{grp} Interaction Top-5")
                    
                    path_top5 = os.path.join(HISTORY_DIR, f"heatmap_{safe_grp}_top5_{ts_str}.png")
                    plt.savefig(path_top5, bbox_inches="tight")
                    
                    with mtab1:
                        st.image(path_top1)
                    with mtab2:
                        st.image(path_top5)

                    plt.close(fig1)
                    plt.close(fig2)
                    print(f"Finished group: {grp}")


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
    baseline_run = st.selectbox("Select Baseline Run", selected_runs) if selected_runs else None

    if st.button("Generate Comparison Report"):
        if not selected_runs:
            st.warning("Select at least one run.")
        else:
            # Aggregate data
            comparison_data = []
            baseline_data = None

            for fname in selected_runs:
                with open(os.path.join(HISTORY_DIR, fname)) as f:
                    data = json.load(f)
                    data["filename"] = fname
                    comparison_data.append(data)
                    if fname == baseline_run:
                        baseline_data = data

            # Create HTML
            html_content = "<html><head><title>Comparison Report</title>"
            html_content += "<style>table {border-collapse: collapse; width: 100%;} th, td {border: 1px solid black; padding: 8px; text-align: left;} th {background-color: #f2f2f2;}</style>"
            html_content += "</head><body><h1>Comparison Report</h1>"

            # Summary Table
            html_content += "<h2>Summary</h2><table><tr><th>Run</th><th>Mode</th><th>Model</th><th>Samples</th><th>Global Top-1</th><th>Global Top-5</th><th>Active Tags</th></tr>"
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

            # Detailed Matrix Comparison
            html_content += "<h2>Attribute Performance (Derived from Cross-Matrix)</h2>"

            # Collect all unique tags across runs
            all_tags_report = set()
            for d in comparison_data:
                if "cross_results" in d:
                    all_tags_report.update(d["cross_results"].get("tags", []))
                elif "matrix_results" in d:
                     for attr, vals in d["matrix_results"].items():
                         for val in vals:
                             all_tags_report.add(f"{attr}: {val}")

            html_content += "<table><tr><th>Tag</th>"
            for d in comparison_data:
                html_content += f"<th>{d['filename']} (Top1 / Top5)</th>"
            html_content += "</tr>"

            for tag in sorted(list(all_tags_report)):
                html_content += f"<tr><td>{tag}</td>"
                for d in comparison_data:
                    # Try to find stats
                    t1, t5, cnt = None, None, None

                    if "cross_results" in d:
                        # Find index of tag
                        tags = d["cross_results"].get("tags", [])
                        if tag in tags:
                            idx = tags.index(tag)
                            # Diagonal
                            # Handle potential None/NaN if data came from json or in memory
                            # The json loader loads null as None.
                            m1_val = d["cross_results"]["matrix_top1"][idx][idx]
                            m5_val = d["cross_results"]["matrix_top5"][idx][idx]

                            if m1_val is not None:
                                t1 = float(m1_val)
                                t5 = float(m5_val) if m5_val is not None else 0.0

                    elif "matrix_results" in d:
                        # Parse tag "Attr: Val"
                        if ": " in tag:
                            k, v = tag.split(": ", 1)
                            stats = d["matrix_results"].get(k, {}).get(v)
                            if stats:
                                t1 = stats.get("top1")
                                t5 = stats.get("top5", 0)
                                cnt = stats.get("count")

                    if t1 is not None:
                        html_content += f"<td>{t1*100:.1f}% / {t5*100:.1f}%</td>"
                    else:
                        html_content += "<td>N/A</td>"
                html_content += "</tr>"
            html_content += "</table>"

            # --- Comparison Heatmap (Delta) ---
            if baseline_data and len(comparison_data) > 1:
                html_content += "<h2>Comparison Heatmaps (Delta: Comparison - Baseline)</h2>"

                # Custom Colormap: Yellow (Neg) -> White (0) -> Blue (Pos)
                from matplotlib.colors import LinearSegmentedColormap
                colors = ["#F7D02C", "#FFFFFF", "#1E90FF"] # Yellow, White, DodgerBlue
                cmap_delta = LinearSegmentedColormap.from_list("custom_delta", colors)

                def get_matrix_map(data):
                    """Extracts tags and Top-1 matrix from run data."""
                    if "cross_results" in data:
                        tags = data["cross_results"]["tags"]
                        mat = np.array(data["cross_results"]["matrix_top1"], dtype=float)
                        return tags, mat
                    return None, None

                base_tags, base_mat = get_matrix_map(baseline_data)

                if base_tags is not None:
                    for d in comparison_data:
                        if d["filename"] == baseline_data["filename"]:
                            continue

                        comp_tags, comp_mat = get_matrix_map(d)
                        if comp_tags is None: continue

                        # Union of tags
                        all_comparison_tags = sorted(list(set(base_tags) | set(comp_tags)))

                        # Reconstruct aligned matrices
                        def align_matrix(tags, mat, all_tags):
                            size = len(all_tags)
                            aligned = np.full((size, size), np.nan)

                            # Create mapping from old idx to new idx
                            old_to_new = {}
                            for i, t in enumerate(tags):
                                if t in all_tags:
                                    old_to_new[i] = all_tags.index(t)

                            for r in range(mat.shape[0]):
                                for c in range(mat.shape[1]):
                                    if r in old_to_new and c in old_to_new:
                                        aligned[old_to_new[r], old_to_new[c]] = mat[r, c]
                            return aligned

                        m_base_aligned = align_matrix(base_tags, base_mat, all_comparison_tags)
                        m_comp_aligned = align_matrix(comp_tags, comp_mat, all_comparison_tags)

                        # Compute Delta
                        delta_mat = m_comp_aligned - m_base_aligned

                        # Plot
                        fig, ax = plt.subplots(figsize=(10, len(all_comparison_tags)*0.5 + 2))
                        sns.heatmap(
                            delta_mat,
                            annot=True,
                            fmt=".1%",
                            xticklabels=all_comparison_tags,
                            yticklabels=all_comparison_tags,
                            cmap=cmap_delta,
                            center=0,
                            vmin=-0.5, vmax=0.5, # Fixed range for consistency or auto? let's constrain a bit
                            ax=ax
                        )
                        plt.title(f"Delta: {d['filename']} - Baseline ({baseline_data['filename']})")
                        plt.xticks(rotation=45, ha="right")

                        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        safe_name = d['filename'].replace(".json", "")
                        delta_path = os.path.join(HISTORY_DIR, f"delta_{safe_name}_vs_base_{ts_str}.png")
                        plt.savefig(delta_path, bbox_inches="tight")
                        plt.close(fig)

                        st.write(f"### Comparison: {d['filename']} vs Baseline")
                        st.image(delta_path)

                        html_content += f"<h3>{d['filename']} vs Baseline</h3>"
                        html_content += f"<img src='{os.path.basename(delta_path)}' style='max-width:100%;'><br>"

            html_content += "</body></html>"

            report_path = "comparison_report.html"
            with open(report_path, "w") as f:
                f.write(html_content)

            st.success(f"Report generated: {report_path}")
            st.markdown(f"[Download Report](./{report_path})", unsafe_allow_html=True)
