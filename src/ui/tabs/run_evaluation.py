import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os
import torch
from src.app_utils import load_data, save_run
from src.evaluation_core import run_evaluation

# Config
HISTORY_DIR = "history"
if not os.path.exists(HISTORY_DIR):
    os.makedirs(HISTORY_DIR)

def render_run_evaluation_tab(config):
    st.header("Run Evaluation")

    dataset_dir = config["dataset_dir"]
    mapping_path = config["mapping_path"]
    filter_json_path = config["filter_json_path"]
    text_json_path = config["text_json_path"]
    model_name = config["model_name"]
    pretrained_tag = config["pretrained_tag"]

    if st.button("Load Data"):
        if not os.path.exists(dataset_dir):
            st.error("Dataset directory not found.")
        else:
            items = load_data(dataset_dir, mapping_path, filter_json_path, text_json_path)
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
                if isinstance(v, list):
                    for sub_v in v:
                        filter_options[k].add(sub_v)
                else:
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
                    save_path = save_run(results, HISTORY_DIR)
                    st.success(f"Run complete! Saved to {save_path}")
                    if results.get("debug_dir"):
                        st.info(f"Debug cases saved to {results['debug_dir']}")

        if "last_results" in st.session_state:
            res = st.session_state["last_results"]
            render_results(res, show_support)

def render_results(res, show_support):
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
    
    ts_str = res.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    
    hm_tab1, hm_tab2 = st.tabs(["Image Tags Heatmap", "Text Tags Heatmap"])
    with hm_tab1:
        plot_matrices("img", "Image", cross, show_support, ts_str)
    with hm_tab2:
        plot_matrices("txt", "Text", cross, show_support, ts_str)


    # --- Joint Tag Analysis ---
    st.divider()
    with st.expander("Joint Tag Analysis (Intersection)"):
        # Collect all available tags from this run
        all_avail_tags = set()
        
        # Check cross results for tags
        for cat in ["img", "txt"]:
            if cat in cross:
                all_avail_tags.update(cross[cat].get("tags", []))
        
        sorted_avail = sorted(list(all_avail_tags))
        sel_joint_tags = st.multiselect("Select Tags to Combine (Intersection)", sorted_avail, key="joint_res")
        
        if st.button("Calculate Joint Accuracy", key="btn_joint_res"):
            if not sel_joint_tags:
                st.warning("Select at least one tag.")
            else:
                # Process per_sample_results
                per_sample = res.get("per_sample_results", [])
                matches = []
                for s in per_sample:
                    attrs = s.get("attributes", {})
                    # Check if ALL selected tags are present
                    # Attributes dict has {key: val} or {key: [vals]}
                    # Tag format is "Key: Value"
                    
                    match_all = True
                    for t in sel_joint_tags:
                        if ": " not in t: 
                            match_all = False
                            break
                        tk, tv = t.split(": ", 1)
                        
                        if tk not in attrs:
                            match_all = False
                            break
                        
                        av = attrs[tk]
                        if isinstance(av, list):
                            if tv not in av:
                                match_all = False
                                break
                        else:
                            if str(av) != tv:
                                match_all = False
                                break
                    
                    if match_all:
                        matches.append(s)
                
                if not matches:
                    st.warning("No samples match ALL selected tags.")
                else:
                    cnt = len(matches)
                    acc1 = sum([m["hit1"] for m in matches]) / cnt
                    acc5 = sum([m["hit5"] for m in matches]) / cnt
                    st.metric("Joint Count", cnt)
                    st.metric("Joint Top-1", f"{acc1:.2%}")
                    st.metric("Joint Top-5", f"{acc5:.2%}")

def plot_matrices(cat_key, title_prefix, cross, show_support, ts_str):
    cat_data = cross.get(cat_key, {})
    tags = cat_data.get("tags", [])
    tag_groups = cat_data.get("tag_groups", {})
    m1 = cat_data.get("matrix_top1", [])
    m5 = cat_data.get("matrix_top5", [])
    counts = cat_data.get("matrix_counts", [])
    
    if not tags or not m1:
        st.info(f"No {title_prefix} tags interaction data.")
        return

    # Convert to numpy
    def to_float_array(arr):
        narr = np.array(arr, dtype=object) 
        narr[narr == None] = np.nan
        return narr.astype(float)

    m1_arr = to_float_array(m1)
    m5_arr = to_float_array(m5)
    counts_arr = np.array(counts)

    groups_to_plot = ["All"] + sorted(list(tag_groups.keys()))
    
    for grp in groups_to_plot:
        st.markdown(f"#### {title_prefix} - Group: {grp}")
        
        if grp == "All":
                row_indices = range(len(tags))
                col_indices = range(len(tags))
                sub_tags_row = tags
                sub_tags_col = tags
        else:
            row_indices = [i for i, t in enumerate(tags) if t in tag_groups[grp]]
            col_indices = [i for i, t in enumerate(tags) if t not in tag_groups[grp]]
            
            sub_tags_row = [tags[i] for i in row_indices]
            sub_tags_col = [tags[i] for i in col_indices]
        
        if not row_indices or not col_indices:
            st.write("No interactions with outside groups.")
            continue

        sub_m1 = m1_arr[np.ix_(row_indices, col_indices)]
        sub_m5 = m5_arr[np.ix_(row_indices, col_indices)]
        sub_counts = counts_arr[np.ix_(row_indices, col_indices)]
        
        if show_support:
            annot_m1 = np.empty(sub_m1.shape, dtype=object)
            annot_m5 = np.empty(sub_m5.shape, dtype=object)
            for r in range(sub_m1.shape[0]):
                for c in range(sub_m1.shape[1]):
                    count = int(sub_counts[r, c])
                    val1 = sub_m1[r, c]
                    val5 = sub_m5[r, c]
                    annot_m1[r, c] = "" if np.isnan(val1) else f"{val1:.1%}\n({count})"
                    annot_m5[r, c] = "" if np.isnan(val5) else f"{val5:.1%}\n({count})"
            annot1, annot5, fmt = annot_m1, annot_m5, ""
        else:
            annot1, annot5, fmt = True, True, ".1%"

        mtab1, mtab2 = st.tabs([f"{grp} Top-1", f"{grp} Top-5"])
        
        # Plotting helper
        def plot_hm(data, annot, ax, title):
            sns.heatmap(data, annot=annot, fmt=fmt, xticklabels=sub_tags_col, yticklabels=sub_tags_row, cmap="Blues", ax=ax, vmin=0, vmax=1)
            ax.set_title(title)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

        fig1, ax1 = plt.subplots(figsize=(10, len(sub_tags_row)*0.5 + 2))
        plot_hm(sub_m1, annot1, ax1, f"{title_prefix} {grp} Top-1")
        safe_grp = grp.replace(" ", "_").replace("/", "_")
        path_top1 = os.path.join(HISTORY_DIR, f"hm_{title_prefix}_{safe_grp}_1_{ts_str}.png")
        plt.savefig(path_top1, bbox_inches="tight")
        plt.close(fig1)

        fig2, ax2 = plt.subplots(figsize=(10, len(sub_tags_row)*0.5 + 2))
        plot_hm(sub_m5, annot5, ax2, f"{title_prefix} {grp} Top-5")
        path_top5 = os.path.join(HISTORY_DIR, f"hm_{title_prefix}_{safe_grp}_5_{ts_str}.png")
        plt.savefig(path_top5, bbox_inches="tight")
        plt.close(fig2)

        with mtab1: st.image(path_top1)
        with mtab2: st.image(path_top5)
