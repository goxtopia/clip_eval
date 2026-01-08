import argparse
import random
import torch
import os
import shutil
from src.label_mapping import LabelMapper
from src.data import DataLoader, get_unique_texts
from src.model import CLIPModel
from src.metrics import compute_metrics, compute_t2i_metrics
from src.report import generate_html_report
from src.utils import normalize_label

SIGLIP_TEMPLATES = [
    "a photo of a {}.",
    "a picture of a {}.",
    "a photo of the {}.",
    "a close-up photo of a {}."
]

def main():
    parser = argparse.ArgumentParser(description="CLIP Evaluation Tool")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to image/txt directory")
    parser.add_argument("--model", type=str, default="MobileCLIP-S2", help="Model architecture name or HF hub ID")
    parser.add_argument("--pretrained", type=str, default=None, help="Pretrained tag (e.g. 'openai', 'laion400m_e32')")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint (optional)")
    parser.add_argument("--mapping", type=str, default="mapping.json", help="Path to label mapping JSON")
    parser.add_argument("--output_prefix", type=str, default="report", help="Prefix for output HTML reports (e.g., 'report' -> 'report_i2t.html')")
    parser.add_argument("--cache", type=str, default="eval_image_features.pt", help="Path to image feature cache")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--debug_dir", type=str, default="debug", help="Directory to save bad cases")
    parser.add_argument("--text_template", type=str, default=None, help="Optional single template. If unset, uses defaults based on model.")

    args = parser.parse_args()

    random.seed(args.seed)
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"[Info] Using device: {device}")

    # Determine templates
    if args.text_template:
        templates = [args.text_template]
        print(f"[Info] Using user-provided template: {templates}")
    elif "siglip" in args.model.lower():
        templates = SIGLIP_TEMPLATES
        print(f"[Info] Using default SigLIP templates: {templates}")
    else:
        templates = ["{}"]
        print(f"[Info] Using default raw template: {templates}")

    # 1. Load Mappings
    mapper = LabelMapper(args.mapping)

    # 2. Load Data
    print(f"[Info] Scanning data from {args.data_dir}...")
    loader = DataLoader(args.data_dir, mapper, seed=args.seed)
    items = loader.load()
    
    if not items:
        return

    # 3. Prepare Retrieval Candidates (Unique Texts)
    unique_texts = get_unique_texts(items)
    print(f"[Info] Unique candidate texts: {len(unique_texts)}")
    
    # 4. Load Model
    model = CLIPModel(args.model, args.pretrained, args.checkpoint, device, args.cache)
    model.load()

    # 5. Encode Images
    image_paths = [item.image_path for item in items]
    image_features = model.encode_images(image_paths, args.batch_size) # [N, D]

    # 6. Encode Texts
    # model.encode_texts now handles the conditional logic (short texts get ensemble, long texts get raw)
    text_features = model.encode_texts(unique_texts, args.batch_size, templates) # [L, D]

    # 7. Compute Similarities
    print("[Info] Computing similarities...")
    image_features = image_features.to(device)
    text_features = text_features.to(device)
    
    # Prepare Shared Data
    gt_class_sets = [item.gt_class_set for item in items]
    
    # --- Mode 1: Image-to-Text (Classification) ---
    print("\n=== Running Image-to-Text Evaluation ===")
    with torch.no_grad():
        sims_i2t = image_features @ text_features.T
        k = min(5, sims_i2t.size(1))
        _, top_idx = sims_i2t.topk(k=k, dim=-1)
        pred_classes = top_idx.cpu()

        class_to_images = {}
        for idx, item in enumerate(items):
            if not item.representative_text:
                continue
            rep_lbl = normalize_label(item.representative_text)
            if rep_lbl not in class_to_images:
                class_to_images[rep_lbl] = []
            class_to_images[rep_lbl].append(idx)

        metrics_i2t, per_class_i2t, bad_cases_i2t = compute_metrics(
            pred_classes,
            gt_class_sets,
            unique_texts,
            class_to_images
        )
    
    print(f"I2T Global Top-1: {metrics_i2t['global_top1']*100:.2f}%")
    generate_html_report(f"{args.output_prefix}_i2t.html", metrics_i2t, per_class_i2t, mode="i2t")

    # --- Mode 2: Text-to-Image (Search) ---
    print("\n=== Running Text-to-Image Evaluation ===")
    with torch.no_grad():
        sims_t2i = text_features @ image_features.T
        k = min(5, sims_t2i.size(1))
        _, top_idx = sims_t2i.topk(k=k, dim=-1)
        pred_images = top_idx.cpu()
        
        metrics_t2i, per_class_t2i, bad_cases_t2i = compute_t2i_metrics(
            pred_images,
            gt_class_sets,
            unique_texts
        )

    print(f"T2I Global Top-1: {metrics_t2i['global_top1']*100:.2f}%")
    generate_html_report(f"{args.output_prefix}_t2i.html", metrics_t2i, per_class_t2i, mode="t2i")

    # --- Debug Saving ---
    # We combine logic to save bad cases for both, or separate directories?
    # User just said "output to two HTMLs".
    # For debug, let's separate them.
    
    if os.path.exists(args.debug_dir):
        shutil.rmtree(args.debug_dir)
    os.makedirs(args.debug_dir)

    # Save I2T Bad Cases
    if bad_cases_i2t:
        print(f"[Info] Saving {len(bad_cases_i2t)} I2T bad cases...")
        i2t_debug_dir = os.path.join(args.debug_dir, "i2t")
        os.makedirs(i2t_debug_dir, exist_ok=True)
        count_per_class = {}
        
        for idx, details in bad_cases_i2t.items():
            item = items[idx]
            rep_text = item.representative_text
            if not rep_text: continue
            class_folder = normalize_label(rep_text)
            if class_folder not in count_per_class: count_per_class[class_folder] = 0
            
            if count_per_class[class_folder] < 5:
                class_dir = os.path.join(i2t_debug_dir, class_folder)
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
                    count_per_class[class_folder] += 1
                except Exception: pass

    # Save T2I Bad Cases
    if bad_cases_t2i:
        print(f"[Info] Saving {len(bad_cases_t2i)} T2I bad cases...")
        t2i_debug_dir = os.path.join(args.debug_dir, "t2i")
        os.makedirs(t2i_debug_dir, exist_ok=True)
        count_per_class = {}
        
        for idx, details in bad_cases_t2i.items():
            query_text = details["query"]
            class_folder = normalize_label(query_text)
            if class_folder not in count_per_class: count_per_class[class_folder] = 0
            
            class_dir = os.path.join(t2i_debug_dir, class_folder)
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
                except Exception: pass

if __name__ == "__main__":
    main()
