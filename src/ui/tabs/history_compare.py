import streamlit as st
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

HISTORY_DIR = "history"

def render_history_compare_tab():
    st.header("History & Compare")

    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)

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

                if "metrics" in d:
                    g_top1 = d["metrics"]["global_top1"]
                    g_top5 = d["metrics"]["global_top5"]
                elif "metrics_i2t" in d:
                    g_top1 = d["metrics_i2t"]["global_top1"]
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

                def get_matrix_map(data, cat_key):
                    """Extracts tags, Top-1, Top-5, and Count matrices."""
                    if "cross_results" not in data: return None, None, None, None
                    cr = data["cross_results"]
                    
                    # New Structure
                    if cat_key in cr and "tags" in cr[cat_key]:
                        d = cr[cat_key]
                        return (
                            d["tags"], 
                            np.array(d["matrix_top1"], dtype=float),
                            np.array(d["matrix_top5"], dtype=float),
                            np.array(d["matrix_count"], dtype=int)
                        )
                    
                    # Legacy fallback
                    if cat_key == "img" and "tags" in cr:
                        return (
                            cr["tags"], 
                            np.array(cr["matrix_top1"], dtype=float),
                            np.array(cr.get("matrix_top5", cr["matrix_top1"]), dtype=float), # fallback if missing
                            np.array(cr.get("matrix_count", []), dtype=int) if "matrix_count" in cr else None 
                        )

                    return None, None, None, None

                # UI Controls for Heatmap
                hm_metric = st.radio("Heatmap Metric", ["Top-1 Accuracy", "Top-5 Accuracy"], horizontal=True)
                show_support = st.checkbox("Show Support (Count) in Heatmap", value=True)

                # Iterate for both Image and Text matrices
                for cat in ["img", "txt"]:
                    base_tags, base_m1, base_m5, base_cnt = get_matrix_map(baseline_data, cat)
                    if base_tags is None: continue
                    
                    # Select Base Metric
                    base_mat = base_m5 if hm_metric == "Top-5 Accuracy" else base_m1

                    section_title = "Image Tag Interaction" if cat == "img" else "Text Tag Interaction"
                    st.markdown(f"#### {section_title} - Delta Heatmaps")
                    html_content += f"<h3>{section_title} ({hm_metric})</h3>"

                    for d in comparison_data:
                        if d["filename"] == baseline_data["filename"]:
                            continue

                        comp_tags, comp_m1, comp_m5, comp_cnt = get_matrix_map(d, cat)
                        if comp_tags is None: continue

                        # Select Comp Metric
                        comp_mat = comp_m5 if hm_metric == "Top-5 Accuracy" else comp_m1

                        # Union of tags
                        all_comparison_tags = sorted(list(set(base_tags) | set(comp_tags)))

                        # Reconstruct aligned matrices
                        def align_matrix(tags, mat, all_tags, fill_val=np.nan):
                            size = len(all_tags)
                            aligned = np.full((size, size), fill_val)
                            
                            # If source matrix is None (legacy case), return NaNs
                            if mat is None: return aligned

                            # Create mapping from old idx to new idx
                            old_to_new = {}
                            for i, t in enumerate(tags):
                                if t in all_tags:
                                    old_to_new[i] = all_tags.index(t)

                            for r in range(mat.shape[0]):
                                for c in range(mat.shape[1]):
                                    if r in old_to_new and c in old_to_new:
                                        # boundary check
                                        if r < mat.shape[0] and c < mat.shape[1]:
                                            aligned[old_to_new[r], old_to_new[c]] = mat[r, c]
                            return aligned

                        m_base_aligned = align_matrix(base_tags, base_mat, all_comparison_tags)
                        m_comp_aligned = align_matrix(comp_tags, comp_mat, all_comparison_tags)
                        
                        # Align counts for annotation
                        cnt_aligned = align_matrix(comp_tags, comp_cnt, all_comparison_tags, fill_val=0)

                        # Compute Delta
                        delta_mat = m_comp_aligned - m_base_aligned

                        # Prepare Annotations
                        annot_labels = []
                        for r in range(len(all_comparison_tags)):
                            row_labels = []
                            for c in range(len(all_comparison_tags)):
                                val = delta_mat[r, c]
                                if np.isnan(val):
                                    row_labels.append("")
                                else:
                                    lbl = f"{val:+.1%}"
                                    if show_support:
                                        count = int(cnt_aligned[r, c])
                                        lbl += f"\n({count})"
                                    row_labels.append(lbl)
                            annot_labels.append(row_labels)
                        annot_labels = np.array(annot_labels)


                        # Plot
                        fig, ax = plt.subplots(figsize=(10, len(all_comparison_tags)*0.5 + 2))
                        sns.heatmap(
                            delta_mat,
                            annot=annot_labels,
                            fmt="",
                            xticklabels=all_comparison_tags,
                            yticklabels=all_comparison_tags,
                            cmap=cmap_delta,
                            center=0,
                            vmin=-0.2, vmax=0.2, # Adjust range for better visibility
                            ax=ax
                        )
                        metric_short = "Top5" if hm_metric == "Top-5 Accuracy" else "Top1"
                        plt.title(f"Delta {metric_short} ({cat}): {d['filename']} - Baseline")
                        plt.xticks(rotation=45, ha="right")

                        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        safe_name = d['filename'].replace(".json", "")
                        delta_path = os.path.join(HISTORY_DIR, f"delta_{cat}_{metric_short}_{safe_name}_vs_base_{ts_str}.png")
                        plt.savefig(delta_path, bbox_inches="tight")
                        plt.close(fig)

                        st.write(f"**{d['filename']}** vs Baseline")
                        st.image(delta_path)

                        html_content += f"<h4>{d['filename']} vs Baseline</h4>"
                        html_content += f"<img src='{os.path.basename(delta_path)}' style='max-width:100%;'><br>"

            # --- Saved Queries Analysis (History) ---
            st.divider()
            st.subheader("Saved Queries Analysis")
            
            from src.analysis_config import load_queries, save_queries

            # Add New Query UI (History Context)
            with st.expander("Add New Query", expanded=False):
                # Reuse all_tags_report collected earlier for consistency
                sorted_avail_history = sorted(list(all_tags_report))
                
                # Load existing for name checking
                if "saved_queries" not in st.session_state:
                     st.session_state["saved_queries"] = load_queries()
                current_saved = st.session_state["saved_queries"]

                c1, c2, c3 = st.columns([2, 2, 1])
                with c1:
                    new_tags = st.multiselect("Select Tags", sorted_avail_history, key="hist_new_query_tags")
                with c2:
                    default_name = f"Query {len(current_saved)+1}"
                    new_name = st.text_input("Query Name", value=default_name, key="hist_new_query_name")
                with c3:
                    st.write("") 
                    st.write("") 
                    if st.button("Add Query", key="hist_add_query_btn"):
                        if not new_tags:
                            st.warning("Select tags first.")
                        else:
                            if any(q['name'] == new_name for q in current_saved):
                                st.warning("Query name already exists.")
                            else:
                                current_saved.append({"name": new_name, "tags": new_tags})
                                save_queries(current_saved)
                                st.session_state["saved_queries"] = current_saved
                                st.success(f"Added '{new_name}'")
                                st.rerun()

            saved_queries = load_queries()

            if not saved_queries:
                st.info("No saved queries found. Create them in 'Run Evaluation' tab.")
            else:
                st.write("Comparing saved queries across selected runs. (Metric: Top-1 (Delta) / Top-5 (Delta) (Count))")

                # We want a table: Rows = Queries, Columns = Runs
                # Or Rows = Runs, Columns = Queries? Rows=Queries seems better for "Feature X performance across models"
                
                table_data = []

                for q in saved_queries:
                    q_name = q["name"]
                    q_tags = q["tags"]
                    
                    row = {"Query": q_name, "Tags": ", ".join(sorted(q_tags))}

                    # First pass: Collect stats for all runs for this query
                    query_stats = {} # run_name -> (acc1, acc5, cnt)
                    
                    for d in comparison_data:
                        run_name = d["filename"]
                        per_sample = d.get("per_sample_results", [])
                        
                        cnt = 0
                        acc1 = 0.0
                        acc5 = 0.0
                        
                        if per_sample:
                            matches = []
                            for s in per_sample:
                                attrs = s.get("attributes", {})
                                match_all = True
                                for t in q_tags:
                                    if ": " not in t: 
                                        match_all = False; break
                                    tk, tv = t.split(": ", 1)
                                    if tk not in attrs:
                                        match_all = False; break
                                    av = attrs[tk]
                                    if isinstance(av, list):
                                        if tv not in av: match_all = False; break
                                    else:
                                        if str(av) != tv: match_all = False; break
                                
                                if match_all: matches.append(s)
                            
                            if matches:
                                cnt = len(matches)
                                acc1 = sum([m["hit1"] for m in matches]) / cnt
                                acc5 = sum([m["hit5"] for m in matches]) / cnt
                        
                        query_stats[run_name] = (acc1, acc5, cnt)

                    # Get Baseline Stats
                    base_stats = query_stats.get(baseline_run)

                    # Second pass: Format Output
                    for d in comparison_data:
                        run_name = d["filename"]
                        acc1, acc5, cnt = query_stats[run_name]
                        
                        # Base Format: "85.2% / 92.0% (120)"
                        val_str = f"{acc1:.1%} / {acc5:.1%} ({cnt})"

                        # Add Delta if baseline exists and this is not baseline
                        if base_stats and run_name != baseline_run:
                            base_acc1, base_acc5, base_cnt = base_stats
                            # Only show delta if baseline has samples? 
                            if base_cnt > 0:
                                d1 = acc1 - base_acc1
                                d5 = acc5 - base_acc5
                                # Enhanced Format: "85.2% (+2.1%) / 92.0% (-1.0%) (120)"
                                val_str = f"{acc1:.1%} ({d1:+.1%}) / {acc5:.1%} ({d5:+.1%}) ({cnt})"
                        
                        row[run_name] = val_str

                    table_data.append(row)

                st.dataframe(pd.DataFrame(table_data))

            html_content += "</body></html>"

            report_path = "comparison_report.html"
            with open(report_path, "w") as f:
                f.write(html_content)

            st.success(f"Report generated: {report_path}")
            st.markdown(f"[Download Report](./{report_path})", unsafe_allow_html=True)
