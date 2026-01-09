import os
import shutil
from src.utils import normalize_label

def save_debug_cases(items, bad_cases, mode, timestamp, debug_root="debug"):
    """
    Saves debug cases to disk.
    """
    run_debug_dir = os.path.join(debug_root, f"{mode}_{timestamp}")
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
