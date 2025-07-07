import os
import csv
import cv2
import pandas as pd
from tqdm import tqdm
from utils.scoring import ImageScorer
from utils.deduplicator import ImageDeduplicator
import config


def get_image_paths(input_dir):
    return [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]


def score_images(image_paths, scorer, log_callback=None, progress_callback=None):
    scored_results = []
    total = len(image_paths)
    for i, path in enumerate(image_paths):
        filename = os.path.basename(path)
        result = scorer.score_image(path)
        result["path"] = path
        scored_results.append(result)

        if log_callback:
            log_callback(f"[{i+1}/{total}] {filename} - {result['rejection_reason'] or 'Accepted'}")
        if progress_callback:
            progress_callback(i + 1)
    return scored_results


def deduplicate_results(scored_results):
    deduplicator = ImageDeduplicator(
        threshold=config.DEDUP_THRESHOLD,
        use_face_clustering=config.DEDUP_USE_FACE_CLUSTERING,
        face_distance_threshold=config.DEDUP_FACE_DISTANCE_THRESHOLD
    )
    groups = []
    for result in scored_results:
        path = result["path"]
        assigned = False
        for group in groups:
            if deduplicator.are_similar(path, group[0]["path"]):
                group.append(result)
                assigned = True
                break
        if not assigned:
            groups.append([result])
    return groups


def select_best_per_group(groups):
    return [
        max(group, key=lambda r: r.get("final_score", 0.0))
        for group in groups
    ]


def filter_and_sort(selected, top_n):
    filtered = [
        r for r in selected
        if r["valid"] and r["final_score"] >= config.FINAL_SCORE_THRESHOLD
    ]
    return sorted(filtered, key=lambda x: -x["final_score"])[:top_n]


def save_csv_log(scored_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, config.CSV_OUTPUT_FILENAME)

    if config.CSV_LOGGING_ENABLED:
        if scored_results:
            keys = list(scored_results[0].keys())
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(scored_results)
          
        else:
            # Write empty CSV with headers if nothing scored
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["path", "final_score", "rejection_reason", "valid"])
                writer.writeheader()
            


def save_selected_images(selected, output_dir, log_callback=None):
    if not selected:
        return
    os.makedirs(output_dir, exist_ok=True)
    for r in selected:
        filename = os.path.basename(r["path"])
        dest = os.path.join(output_dir, filename)
        try:
            img = cv2.imread(r["path"])
            if img is not None:
                cv2.imwrite(dest, img)
            else:
                if log_callback:
                    log_callback(f"[ERROR] Could not read image: {r['path']}")
        except Exception as e:
            if log_callback:
                log_callback(f"[ERROR] Failed to save {filename}: {e}")


def select_best_images(
    input_dir,
    output_dir,
    log_callback=None,
    progress_callback=None,
    show_benchmarks=False
):
    scorer = ImageScorer()
    image_paths = get_image_paths(input_dir)
    scored_results = score_images(image_paths, scorer, log_callback, progress_callback)

    if config.ENABLE_DEDUPLICATION:
        groups = deduplicate_results(scored_results)
    else:
        groups = [[r] for r in scored_results]

    best_per_group = select_best_per_group(groups)
    selected = filter_and_sort(best_per_group, config.TOP_N_IMAGES)

    save_csv_log(scored_results, output_dir)
    save_selected_images(selected, output_dir, log_callback)

    return len(selected)
