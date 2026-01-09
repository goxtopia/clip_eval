from datetime import datetime
import numpy as np
import torch
from src.data import get_unique_texts
from src.model import CLIPModel
from src.metrics import compute_metrics, compute_t2i_metrics
from src.utils import normalize_label
from src.debug_utils import save_debug_cases

def run_evaluation(items, model_name, pretrained_tag, device, selected_filters, mode="i2t", debug_mode=False, debug_root="debug"):
    # Filter items
    filtered_items = []
    for item in items:
        match = True
        for key, desired_values in selected_filters.items():
            if not desired_values: continue

            val = item.attributes.get(key, "Unknown")
            
            # Helper to check membership
            def check_match(val, desired):
                if isinstance(val, list):
                    # If val is list, match if ANY element is in desired_values
                    return any(v in desired for v in val)
                else:
                    return val in desired

            if not check_match(val, desired_values):
                match = False
                break
        if match:
            filtered_items.append(item)

    if not filtered_items:
        return None, "No items match the selected tags."

    # Prepare logic
    # In I2T: classes are unique texts.
    # In T2I: queries are item texts (duplicates allowed to preserve attribute mapping).
    unique_texts = get_unique_texts(filtered_items)

    if mode == "t2i":
        # For T2I, we use the text of each item as a query.
        queries = [item.representative_text for item in filtered_items]
        # We need to filter out items with no text? DataLoader usually ensures this or empty string.
        # But let's be safe.
        valid_indices = [i for i, q in enumerate(queries) if q]
        queries = [queries[i] for i in valid_indices]
        # We might need to filter filtered_items too if we rely on index alignment!
        # Actually, filtered_items contains the attributes we need.
        # If an item has no text, it can't be a T2I query.
        filtered_items = [filtered_items[i] for i in valid_indices]

        target_texts = queries # For T2I, this list aligns with filtered_items
    else:
        target_texts = unique_texts # For I2T, we classify into unique labels

    # Load Model
    model = CLIPModel(model_name, None, pretrained_tag, device, "eval_cache.pt")
    model.load()

    # Encode
    image_paths = [item.image_path for item in filtered_items] # Search space for T2I, Input for I2T
    image_features = model.encode_images(image_paths, 32)

    templates = ["{}"]
    text_features = model.encode_texts(target_texts, 32, templates)

    image_features = image_features.to(device)
    text_features = text_features.to(device)

    gt_class_sets = [item.gt_class_set for item in filtered_items]

    metrics = {}
    bad_cases = {}

    # Store per-item (or per-query) hit status for matrix breakdown
    # format: list of tuples (is_top1, is_top5) corresponding to 'filtered_items'
    performance_records = []

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
        # text_features is [N_queries, D] (aligned with filtered_items)
        # image_features is [N_items, D] (search space)
        sims_t2i = text_features @ image_features.T
        k = min(5, sims_t2i.size(1))
        _, top_idx = sims_t2i.topk(k=k, dim=-1)
        pred_images = top_idx.cpu()

        metrics, per_class, bad_cases = compute_t2i_metrics(
            pred_images, gt_class_sets, target_texts
        )

        # In T2I now, queries align 1-to-1 with filtered_items.
        # So we can just iterate the queries to get performance records.
        # But compute_t2i_metrics aggregates per-class stats?
        # No, wait. compute_t2i_metrics returns global metrics and per-class stats.
        # We need per-QUERY hit status for the matrix.
        # The function internal logic calculates this.
        # I should have modified compute_t2i_metrics to return per-query status or recalculate it here.
        # Since I didn't change the signature to return raw hits, I will recalculate quickly here.
        # Or I can just trust the order?

        # Let's recalculate simply here to be safe and avoid API changes.
        query_norms = [normalize_label(t) for t in target_texts]

        for i in range(len(target_texts)):
            q_norm = query_norms[i]
            preds_idx = pred_images[i].tolist()

            is_relevant = []
            for img_idx in preds_idx:
                gset = gt_class_sets[img_idx]
                rel = 1.0 if q_norm in gset else 0.0
                is_relevant.append(rel)

            hit1 = is_relevant[0]
            hit5 = 1.0 if sum(is_relevant) > 0 else 0.0
            performance_records.append((hit1, hit5))

    # Save Debug if enabled
    if debug_mode and bad_cases:
        debug_dir = save_debug_cases(filtered_items, bad_cases, mode, timestamp, debug_root)

    # --- Cross-Attribute Matrix Calculation ---
    sample_tags_img = []
    sample_tags_txt = []
    
    all_tags_img = set()
    all_tags_txt = set()

    for item in filtered_items:
        tags_img = set()
        tags_txt = set()
        for k, v in item.attributes.items():
            if k == 'filename': continue
            
            vals = v if isinstance(v, list) else [v]
            for val in vals:
                tag = f"{k}: {val}"
                if k.startswith("Text "):
                    tags_txt.add(tag)
                    all_tags_txt.add(tag)
                else:
                    tags_img.add(tag)
                    all_tags_img.add(tag)
        
        sample_tags_img.append(tags_img)
        sample_tags_txt.append(tags_txt)

    def compute_matrix(all_tags, sample_tags, perf_records):
        sorted_tags = sorted(list(all_tags))
        matrix_data = {}
        for i in range(len(perf_records)):
            p_rec = perf_records[i]
            tags = sample_tags[i]
            tags_list = list(tags)
            for t1 in tags_list:
                for t2 in tags_list:
                    if ": " in t1 and ": " in t2:
                        k1, v1 = t1.split(": ", 1)
                        k2, v2 = t2.split(": ", 1)
                        if k1 == k2 and v1 != v2: continue
                    key = (t1, t2)
                    if key not in matrix_data: matrix_data[key] = {"sum1": 0.0, "sum5": 0.0, "count": 0}
                    matrix_data[key]["sum1"] += p_rec[0]
                    matrix_data[key]["sum5"] += p_rec[1]
                    matrix_data[key]["count"] += 1
        return sorted_tags, matrix_data

    # Compute Image Matrix
    img_sorted, img_data = compute_matrix(all_tags_img, sample_tags_img, performance_records)
    # Compute Text Matrix
    txt_sorted, txt_data = compute_matrix(all_tags_txt, sample_tags_txt, performance_records)

    def build_grid(sorted_tags, matrix_data):
        m1, m5, mc = [], [], []
        for t1 in sorted_tags:
            r1, r5, rc = [], [], []
            for t2 in sorted_tags:
                key = (t1, t2)
                if key in matrix_data:
                    d = matrix_data[key]
                    r1.append(d["sum1"] / d["count"])
                    r5.append(d["sum5"] / d["count"])
                    rc.append(d["count"])
                else:
                    r1.append(np.nan)
                    r5.append(np.nan)
                    rc.append(0)
            m1.append(r1)
            m5.append(r5)
            mc.append(rc)
        return m1, m5, mc

    img_m1, img_m5, img_mc = build_grid(img_sorted, img_data)
    txt_m1, txt_m5, txt_mc = build_grid(txt_sorted, txt_data)

    # Collect per-tag stats (merged for the table)
    tag_stats = []
    for tags, data in [(img_sorted, img_data), (txt_sorted, txt_data)]:
        for t in tags:
            key = (t, t)
            if key in data:
                d = data[key]
                tag_stats.append({
                    "Tag": t,
                    "Count": d["count"],
                    "ACC1": d["sum1"] / d["count"],
                    "ACC5": d["sum5"] / d["count"]
                })

    def get_groups(tags):
        groups = {}
        for t in tags:
            if ": " in t:
                k, _ = t.split(": ", 1)
                if k not in groups: groups[k] = []
                groups[k].append(t)
        return groups

    cross_results = {
        "tag_stats": tag_stats,
        "img": {
            "tags": img_sorted, "matrix_top1": img_m1, "matrix_top5": img_m5, "matrix_counts": img_mc,
            "tag_groups": get_groups(img_sorted)
        },
        "txt": {
            "tags": txt_sorted, "matrix_top1": txt_m1, "matrix_top5": txt_m5, "matrix_counts": txt_mc,
            "tag_groups": get_groups(txt_sorted)
        }
    }

    # Prepare Per-Sample Results for Joint Analysis
    per_sample_results = []
    # Using filtered_items loop again
    for i, item in enumerate(filtered_items):
        rec = performance_records[i]
        # Flatten attributes for JSON
        attrs_flat = {}
        for k, v in item.attributes.items():
                if k != 'filename':
                    attrs_flat[k] = v # Can be list or scalar
        
        per_sample_results.append({
            "id": i,
            "hit1": rec[0],
            "hit5": rec[1],
            "attributes": attrs_flat
        })

    results = {
        "metrics": metrics,
        "cross_results": cross_results,
        "per_sample_results": per_sample_results, # NEW
        "n_samples": len(filtered_items),
        "timestamp": timestamp,
        "model": model_name,
        "pretrained": pretrained_tag,
        "filters": selected_filters,
        "mode": mode,
        "debug_dir": debug_dir
    }

    return results, None
