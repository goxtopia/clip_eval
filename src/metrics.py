import torch
from typing import List, Dict, Set, Tuple, Optional, Any
from .utils import normalize_label

def compute_metrics(
    pred_classes: torch.Tensor, # [N, k] - indices into unique_texts
    gt_class_sets: List[Set[str]], # [N] - Set of expanded labels for each image
    candidate_texts: List[str],    # List of unique texts corresponding to pred_classes indices
    class_to_images: Dict[str, List[int]] # Maps representative label (str) to list of image indices
):
    """
    Computes top-k metrics for Image-to-Text (Classification).
    
    Returns:
        metrics: Dict with global and aggregated stats.
        per_class_stats: List of per-class stats.
        bad_cases: Dict[int, Dict[str, Any]]. Key is image index. Value contains 'preds' (list of strings) and 'gt' (list of strings).
    """
    N, k = pred_classes.shape
    
    # Global stats
    correct_top1 = []
    correct_top5 = []
    bad_cases = {}
    
    candidate_norms = [normalize_label(t) for t in candidate_texts]
    
    for i in range(N):
        gset = gt_class_sets[i]
        preds_idx = pred_classes[i].tolist()
        
        # Get the predicted strings (normalized for check)
        pred_norms = [candidate_norms[p] for p in preds_idx]
        
        # Get raw predicted strings for reporting
        pred_raws = [candidate_texts[p] for p in preds_idx]
        
        if not gset:
            correct_top1.append(0.0)
            correct_top5.append(0.0)
            bad_cases[i] = {
                "preds": pred_raws,
                "gt": list(gset)
            }
            continue
            
        # Hit logic: Is the predicted label in the expanded GT set?
        hit1 = 1.0 if pred_norms[0] in gset else 0.0
        hit5 = 1.0 if any(p in gset for p in pred_norms) else 0.0
        
        correct_top1.append(hit1)
        correct_top5.append(hit5)
        
        if hit5 == 0.0:
            bad_cases[i] = {
                "preds": pred_raws,
                "gt": list(gset)
            }
        
    correct_top1_t = torch.tensor(correct_top1, dtype=torch.float32)
    correct_top5_t = torch.tensor(correct_top5, dtype=torch.float32)
    
    global_top1 = correct_top1_t.mean().item()
    global_top5 = correct_top5_t.mean().item()
    
    # Per-class stats
    per_class_stats = []
    per_class_top1_vals = []
    per_class_top5_vals = []
    per_class_counts = []
    
    sorted_classes = sorted(class_to_images.keys())
    
    for cls_lbl in sorted_classes:
        indices = class_to_images[cls_lbl]
        count = len(indices)
        if count == 0:
            continue
            
        c_top1 = correct_top1_t[indices].mean().item()
        c_top5 = correct_top5_t[indices].mean().item()
        
        per_class_stats.append({
            "label": cls_lbl,
            "count": count,
            "top1": c_top1,
            "top5": c_top5
        })
        
        per_class_top1_vals.append(c_top1)
        per_class_top5_vals.append(c_top5)
        per_class_counts.append(count)
        
    if per_class_top1_vals:
        per_class_top1_t = torch.tensor(per_class_top1_vals)
        per_class_top5_t = torch.tensor(per_class_top5_vals)
        counts_t = torch.tensor(per_class_counts, dtype=torch.float32)
        
        macro_top1 = per_class_top1_t.mean().item()
        macro_top5 = per_class_top5_t.mean().item()
        
        total_count = counts_t.sum().item()
        weighted_top1 = (per_class_top1_t * counts_t).sum().item() / total_count
        weighted_top5 = (per_class_top5_t * counts_t).sum().item() / total_count
    else:
        macro_top1 = macro_top5 = weighted_top1 = weighted_top5 = 0.0
        
    metrics = {
        "global_top1": global_top1,
        "global_top5": global_top5,
        "macro_top1": macro_top1,
        "macro_top5": macro_top5,
        "weighted_top1": weighted_top1,
        "weighted_top5": weighted_top5,
    }
    
    return metrics, per_class_stats, bad_cases

def compute_t2i_metrics(
    pred_images: torch.Tensor, # [N_queries, k] - indices into images
    gt_class_sets: List[Set[str]], # [N_images] - GT sets for all images
    query_texts: List[str],        # [N_queries] - The text queries (might contain duplicates)
):
    """
    Computes top-k metrics for Text-to-Image (Search/Retrieval).
    
    For each query text, we retrieve k images.
    A retrieved image is 'relevant' if the query text is present in the image's GT set.
    """
    N_queries, k = pred_images.shape
    
    correct_top1 = []
    correct_top5 = []
    bad_cases = {} # Key: query index. Value: info about top-5 retrieved images.
    
    query_norms = [normalize_label(t) for t in query_texts]
    
    for i in range(N_queries):
        q_norm = query_norms[i]
        q_raw = query_texts[i]
        
        preds_idx = pred_images[i].tolist() # Indices of images
        
        # Check relevance for each retrieved image
        # relevant if q_norm in gt_class_sets[img_idx]
        is_relevant = []
        retrieved_gts = []
        
        for img_idx in preds_idx:
            gset = gt_class_sets[img_idx]
            rel = 1.0 if q_norm in gset else 0.0
            is_relevant.append(rel)
            retrieved_gts.append(list(gset))
            
        hit1 = is_relevant[0]
        hit5 = 1.0 if sum(is_relevant) > 0 else 0.0
        
        correct_top1.append(hit1)
        correct_top5.append(hit5)
        
        if hit5 == 0.0:
            bad_cases[i] = {
                "query": q_raw,
                "top_images_indices": preds_idx,
                "top_images_gts": retrieved_gts
            }
            
    correct_top1_t = torch.tensor(correct_top1, dtype=torch.float32)
    correct_top5_t = torch.tensor(correct_top5, dtype=torch.float32)
    
    global_top1 = correct_top1_t.mean().item()
    global_top5 = correct_top5_t.mean().item()
    
    # Per-class stats (Aggregating identical queries)
    # We want to group stats by the query text to show "Class: cat" performance.

    stats_map = {} # label -> {sum1, sum5, count}
    
    for i in range(N_queries):
        lbl = query_texts[i]
        if lbl not in stats_map:
            stats_map[lbl] = {"sum1": 0.0, "sum5": 0.0, "count": 0}

        stats_map[lbl]["sum1"] += correct_top1[i]
        stats_map[lbl]["sum5"] += correct_top5[i]
        stats_map[lbl]["count"] += 1

    per_class_stats = []
    sorted_labels = sorted(stats_map.keys())

    for lbl in sorted_labels:
        d = stats_map[lbl]
        c = d["count"]
        per_class_stats.append({
            "label": lbl,
            "count": c,
            "top1": d["sum1"] / c,
            "top5": d["sum5"] / c
        })
        
    # Macro/Weighted are same since count is 1
    metrics = {
        "global_top1": global_top1,
        "global_top5": global_top5,
        "macro_top1": global_top1,
        "macro_top5": global_top5,
        "weighted_top1": global_top1,
        "weighted_top5": global_top5,
    }
    
    return metrics, per_class_stats, bad_cases
