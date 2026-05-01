import os
import json
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import iris
from iris.nodes.matcher.hamming_distance_matcher import HammingDistanceMatcher
import kagglehub

# Download latest version
path = kagglehub.dataset_download("sondosaabed/casia-iris-thousand")

# -----------------------------
# Config
# -----------------------------
DATASET_ROOT = path
OUTPUT_ROOT = r".\out_CASIA_Iris_Thousand_MultiSet_L"
CACHE_DIR = r".\cache_templates_CASIA_Iris_Thousand_Baseline"

TARGET_EYE_SIDE = "L"
SUBJECTS_PER_SET = 50
IMAGES_PER_SUBJECT = 10
MAX_SETS = 10  # None = use as many full non-overlapping sets as possible
THRESHOLD = 0.38

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
USE_CACHE = True
OVERWRITE_CACHE = False

FEATURE_COLS = ["hamming", "jaccard", "weighted_euclidean", "pearson"]


# -----------------------------
# Data models
# -----------------------------
@dataclass
class SelectedImageRecord:
    set_id: str
    subject_id: str
    eye_side: str
    image_name: str
    image_path: str
    selection_order: int
    image_label: str
    template_cache_path: str


@dataclass
class FailedImageRecord:
    set_id: str
    subject_id: str
    eye_side: str
    image_name: str
    image_path: str
    failure_stage: str
    failure_reason: str


@dataclass
class SkippedSubjectRecord:
    set_id: str
    subject_id: str
    valid_image_count: int
    required_image_count: int
    reason: str


@dataclass
class CachedTemplate:
    subject_id: str
    eye_side: str
    image_name: str
    image_path: str
    image_label: str
    iris_template_bytes: bytes
    metadata: dict


# -----------------------------
# Basic helpers
# -----------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def count_subject_like_dirs(root_dir: str) -> int:
    """Count child folders that look like subject folders (contain L or R eye dirs)."""
    if not os.path.isdir(root_dir):
        return 0

    count = 0
    for name in os.listdir(root_dir):
        subject_path = os.path.join(root_dir, name)
        if not os.path.isdir(subject_path):
            continue
        if os.path.isdir(os.path.join(subject_path, "L")) or os.path.isdir(os.path.join(subject_path, "R")):
            count += 1
    return count


def resolve_dataset_root(download_root: str) -> str:
    """Resolve the actual subject root when datasets are wrapped in an extra top-level folder."""
    if not os.path.isdir(download_root):
        raise FileNotFoundError(f"Dataset root not found: {download_root}")

    current = download_root
    for _ in range(8):
        score = count_subject_like_dirs(current)
        if score > 0:
            return current

        child_dirs = [
            os.path.join(current, name)
            for name in sorted(os.listdir(current))
            if os.path.isdir(os.path.join(current, name))
        ]

        # Common packaging pattern: one wrapper directory level (sometimes repeated).
        if len(child_dirs) == 1:
            current = child_dirs[0]
            continue

        break

    return current


def list_subject_dirs(dataset_root: str) -> List[str]:
    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    subject_dirs = []
    for name in sorted(os.listdir(dataset_root)):
        path = os.path.join(dataset_root, name)
        if os.path.isdir(path):
            subject_dirs.append(path)
    return subject_dirs


def get_subject_id_from_dir(subject_dir: str) -> str:
    return os.path.basename(subject_dir)


def get_eye_dir(subject_dir: str, eye_side: str) -> Optional[str]:
    eye_dir = os.path.join(subject_dir, eye_side)
    return eye_dir if os.path.isdir(eye_dir) else None


def list_eye_images(eye_dir: str) -> List[str]:
    images = []
    for name in sorted(os.listdir(eye_dir)):
        path = os.path.join(eye_dir, name)
        if os.path.isfile(path) and name.lower().endswith(VALID_EXTENSIONS):
            images.append(path)
    return images


def build_image_label(subject_id: str, eye_side: str, image_name: str) -> str:
    stem = os.path.splitext(image_name)[0]
    return f"{subject_id}_{eye_side}_{stem}"


def cache_path_for_image(image_path: str) -> str:
    import hashlib
    digest = hashlib.md5(image_path.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{digest}.pkl")


def safe_error_message(err_obj: Any) -> str:
    if err_obj is None:
        return "unknown_error"
    if isinstance(err_obj, dict):
        if "message" in err_obj:
            return str(err_obj["message"])
        return json.dumps(err_obj, ensure_ascii=False)
    return str(err_obj)


# -----------------------------
# Template creation / loading
# -----------------------------
def create_template_from_image(
    pipeline: iris.IRISPipeline,
    image_path: str,
    eye_side_for_pipeline: str,
) -> Tuple[Optional[iris.IrisTemplate], Optional[dict], Optional[str], Optional[str]]:
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None, None, "image_read_failed", f"Could not read image: {image_path}"

    try:
        out = pipeline(
            iris.IRImage(
                img_data=img,
                image_id=image_path,
                eye_side=("left" if eye_side_for_pipeline.upper() == "L" else "right"),
            )
        )
    except Exception as e:
        return None, None, "pipeline_exception", str(e)

    if out is None:
        return None, None, "pipeline_failed", "Pipeline returned None"

    err = out.get("error")
    if err is not None:
        return None, out, "pipeline_failed", safe_error_message(err)

    template = out.get("iris_template")
    if template is None:
        return None, out, "template_missing", "Pipeline output did not contain iris_template"

    return template, out, None, None


def save_template_cache(record: CachedTemplate, cache_path: str) -> None:
    ensure_dir(os.path.dirname(cache_path))
    with open(cache_path, "wb") as f:
        pickle.dump(record, f)


def load_template_cache(cache_path: str) -> CachedTemplate:
    with open(cache_path, "rb") as f:
        return pickle.load(f)


def get_or_create_cached_template(
    pipeline: iris.IRISPipeline,
    subject_id: str,
    eye_side: str,
    image_path: str,
) -> Tuple[Optional[CachedTemplate], Optional[str], Optional[str]]:
    image_name = os.path.basename(image_path)
    image_label = build_image_label(subject_id, eye_side, image_name)
    cache_path = cache_path_for_image(image_path)

    if USE_CACHE and os.path.exists(cache_path) and not OVERWRITE_CACHE:
        try:
            rec = load_template_cache(cache_path)
            return rec, None, None
        except Exception as e:
            return None, "cache_load_failed", str(e)

    template, out, failure_stage, failure_reason = create_template_from_image(
        pipeline=pipeline,
        image_path=image_path,
        eye_side_for_pipeline=eye_side,
    )
    if template is None:
        return None, failure_stage, failure_reason

    try:
        template_bytes = pickle.dumps(template)
    except Exception as e:
        return None, "serialization_failed", str(e)

    rec = CachedTemplate(
        subject_id=subject_id,
        eye_side=eye_side,
        image_name=image_name,
        image_path=image_path,
        image_label=image_label,
        iris_template_bytes=template_bytes,
        metadata=out.get("metadata", {}) if out else {},
    )

    if USE_CACHE:
        try:
            save_template_cache(rec, cache_path)
        except Exception as e:
            return None, "cache_save_failed", str(e)

    return rec, None, None


# -----------------------------
# Selection logic
# -----------------------------
def select_subject_images_for_set(
    set_id: str,
    subject_dir: str,
    images_per_subject: int,
    eye_side: str,
    pipeline: iris.IRISPipeline,
) -> Tuple[Optional[List[SelectedImageRecord]], List[FailedImageRecord], Optional[SkippedSubjectRecord], Dict[str, CachedTemplate]]:
    failed_images: List[FailedImageRecord] = []
    selected_templates: Dict[str, CachedTemplate] = {}

    subject_id = get_subject_id_from_dir(subject_dir)
    eye_dir = get_eye_dir(subject_dir, eye_side)
    if eye_dir is None:
        return None, failed_images, SkippedSubjectRecord(set_id, subject_id, 0, images_per_subject, f"Missing eye folder: {eye_side}"), {}

    candidate_images = list_eye_images(eye_dir)
    if not candidate_images:
        return None, failed_images, SkippedSubjectRecord(set_id, subject_id, 0, images_per_subject, "No valid image files found"), {}

    selected_records: List[SelectedImageRecord] = []
    valid_count = 0

    for image_path in candidate_images:
        if valid_count >= images_per_subject:
            break

        cached_tpl, failure_stage, failure_reason = get_or_create_cached_template(
            pipeline=pipeline,
            subject_id=subject_id,
            eye_side=eye_side,
            image_path=image_path,
        )

        if cached_tpl is None:
            failed_images.append(
                FailedImageRecord(
                    set_id=set_id,
                    subject_id=subject_id,
                    eye_side=eye_side,
                    image_name=os.path.basename(image_path),
                    image_path=image_path,
                    failure_stage=failure_stage or "unknown_failure_stage",
                    failure_reason=failure_reason or "unknown_failure_reason",
                )
            )
            continue

        valid_count += 1
        selected_records.append(
            SelectedImageRecord(
                set_id=set_id,
                subject_id=subject_id,
                eye_side=eye_side,
                image_name=cached_tpl.image_name,
                image_path=cached_tpl.image_path,
                selection_order=valid_count,
                image_label=cached_tpl.image_label,
                template_cache_path=cache_path_for_image(image_path),
            )
        )
        selected_templates[cached_tpl.image_label] = cached_tpl

    if valid_count < images_per_subject:
        return None, failed_images, SkippedSubjectRecord(
            set_id=set_id,
            subject_id=subject_id,
            valid_image_count=valid_count,
            required_image_count=images_per_subject,
            reason="Fewer than required valid images after template creation",
        ), {}

    return selected_records, failed_images, None, selected_templates


def build_sets_from_dataset(
    dataset_root: str,
    subjects_per_set: int,
    images_per_subject: int,
    eye_side: str,
    pipeline: iris.IRISPipeline,
    max_sets: Optional[int] = None,
) -> Tuple[List[dict], List[dict]]:
    subject_dirs = list_subject_dirs(dataset_root)
    used_subject_ids = set()
    all_sets: List[dict] = []
    global_failed: List[dict] = []

    set_counter = 1
    idx = 0

    while idx < len(subject_dirs):
        if max_sets is not None and len(all_sets) >= max_sets:
            break

        set_id = f"set_{set_counter:02d}"
        set_selected_images: List[SelectedImageRecord] = []
        set_failed_images: List[FailedImageRecord] = []
        set_skipped_subjects: List[SkippedSubjectRecord] = []
        set_templates_by_label: Dict[str, CachedTemplate] = {}
        set_subjects: List[str] = []

        while idx < len(subject_dirs) and len(set_subjects) < subjects_per_set:
            subject_dir = subject_dirs[idx]
            idx += 1

            subject_id = get_subject_id_from_dir(subject_dir)
            if subject_id in used_subject_ids:
                continue

            selected_records, failed_images, skipped_subject, templates = select_subject_images_for_set(
                set_id=set_id,
                subject_dir=subject_dir,
                images_per_subject=images_per_subject,
                eye_side=eye_side,
                pipeline=pipeline,
            )

            set_failed_images.extend(failed_images)

            if skipped_subject is not None:
                set_skipped_subjects.append(skipped_subject)
                continue

            set_selected_images.extend(selected_records or [])
            set_templates_by_label.update(templates)
            set_subjects.append(subject_id)
            used_subject_ids.add(subject_id)
            print(f"[{set_id}] included subject {subject_id} ({len(set_subjects)}/{subjects_per_set})")

        if len(set_subjects) == subjects_per_set:
            all_sets.append(
                {
                    "set_id": set_id,
                    "subjects": set_subjects,
                    "selected_images": set_selected_images,
                    "failed_images": set_failed_images,
                    "skipped_subjects": set_skipped_subjects,
                    "templates_by_label": set_templates_by_label,
                }
            )
            set_counter += 1
        else:
            # not enough subjects to complete this set; stop cleanly
            global_failed.extend([asdict(x) for x in set_failed_images])
            break

    return all_sets, global_failed


# -----------------------------
# Matching / matrix / metrics
# -----------------------------
def deserialize_template(template_bytes: bytes) -> iris.IrisTemplate:
    return pickle.loads(template_bytes)


def compute_hamming_distance(
    matcher: HammingDistanceMatcher,
    template_a: CachedTemplate,
    template_b: CachedTemplate,
) -> float:
    tpl_a = deserialize_template(template_a.iris_template_bytes)
    tpl_b = deserialize_template(template_b.iris_template_bytes)
    result = matcher.run(tpl_a, tpl_b)

    if isinstance(result, (int, float, np.floating)):
        return float(result)

    if isinstance(result, dict):
        for key in ["hamming_distance", "distance", "score"]:
            if key in result and result[key] is not None:
                return float(result[key])

    for attr in ["hamming_distance", "distance", "score"]:
        if hasattr(result, attr):
            return float(getattr(result, attr))

    raise ValueError(f"Could not parse matcher result: {result}")


def build_distance_matrix(selected_images: List[SelectedImageRecord], templates_by_label: Dict[str, CachedTemplate], matcher: HammingDistanceMatcher) -> pd.DataFrame:
    labels = [rec.image_label for rec in selected_images]
    n = len(labels)
    matrix = np.full((n, n), np.nan, dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            distance = compute_hamming_distance(matcher, templates_by_label[labels[i]], templates_by_label[labels[j]])
            matrix[i, j] = distance
            matrix[j, i] = distance

    return pd.DataFrame(matrix, index=labels, columns=labels)


def label_pair(a: SelectedImageRecord, b: SelectedImageRecord) -> str:
    return "genuine" if a.subject_id == b.subject_id and a.eye_side == b.eye_side else "impostor"


def extract_unique_pair_records(distance_matrix_df: pd.DataFrame, selected_images: List[SelectedImageRecord]) -> pd.DataFrame:
    records = []
    columns = [
        "set_id",
        "img1_label",
        "img2_label",
        "subject1",
        "subject2",
        "eye_side1",
        "eye_side2",
        "image1_name",
        "image2_name",
        "distance",
        "pair_type",
        "true_label",
    ]
    by_label = {rec.image_label: rec for rec in selected_images}
    labels = list(distance_matrix_df.index)

    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            rec_i = by_label[labels[i]]
            rec_j = by_label[labels[j]]
            pair_type = label_pair(rec_i, rec_j)
            records.append(
                {
                    "set_id": rec_i.set_id,
                    "img1_label": rec_i.image_label,
                    "img2_label": rec_j.image_label,
                    "subject1": rec_i.subject_id,
                    "subject2": rec_j.subject_id,
                    "eye_side1": rec_i.eye_side,
                    "eye_side2": rec_j.eye_side,
                    "image1_name": rec_i.image_name,
                    "image2_name": rec_j.image_name,
                    "distance": float(distance_matrix_df.iloc[i, j]),
                    "pair_type": pair_type,
                    "true_label": 1 if pair_type == "genuine" else 0,
                }
            )
    return pd.DataFrame(records, columns=columns)


# -----------------------------
# Multi-score metrics
# -----------------------------

def extract_valid_bits(
    tpl_a: iris.IrisTemplate,
    tpl_b: iris.IrisTemplate,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the bit arrays at positions that are unmasked in both templates."""
    def _flatten(codes, masks):
        if isinstance(codes, np.ndarray):
            return codes.flatten().astype(np.float32), masks.flatten().astype(bool)
        c = np.concatenate([x.flatten() for x in codes]).astype(np.float32)
        m = np.concatenate([x.flatten() for x in masks]).astype(bool)
        return c, m

    a_codes, a_masks = _flatten(tpl_a.iris_codes, tpl_a.mask_codes)
    b_codes, b_masks = _flatten(tpl_b.iris_codes, tpl_b.mask_codes)
    # mask_codes=True means the bit IS valid (unoccluded) in open-iris convention
    valid = a_masks & b_masks
    return a_codes[valid], b_codes[valid]


def compute_extra_scores(
    tpl_a: iris.IrisTemplate,
    tpl_b: iris.IrisTemplate,
) -> Dict[str, float]:
    """Compute Jaccard, Weighted Euclidean, and Pearson distances for a pre-deserialized pair."""
    a, b = extract_valid_bits(tpl_a, tpl_b)
    if len(a) == 0:
        return {"jaccard": np.nan, "weighted_euclidean": np.nan, "pearson": np.nan}

    a_bin, b_bin = a > 0.5, b > 0.5

    # Jaccard (binary)
    inter = float(np.sum(a_bin & b_bin))
    union = float(np.sum(a_bin | b_bin))
    jaccard = 1.0 - (inter / union) if union > 0 else 1.0

    # Weighted Euclidean: L2 distance weighted by per-position activation magnitude.
    # Positions active in neither template (both 0) contribute no weight, making
    # the metric sensitive to the pattern of set bits rather than unset ones.
    weights = np.maximum(a, b) + 1e-6
    diff_sq = (a - b) ** 2
    weighted_euclidean = float(np.sqrt(np.dot(weights, diff_sq) / weights.sum()))

    # Pearson correlation distance
    sa, sb = float(a.std()), float(b.std())
    pearson = 1.0 - float(np.corrcoef(a, b)[0, 1]) if (sa > 0 and sb > 0) else 1.0

    return {"jaccard": jaccard, "weighted_euclidean": weighted_euclidean, "pearson": pearson}


def build_multi_score_pair_df(
    selected_images: List[SelectedImageRecord],
    templates_by_label: Dict[str, CachedTemplate],
    distance_matrix_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build a per-pair DataFrame with hamming + extra distance scores as features."""
    labels = [rec.image_label for rec in selected_images]
    by_label = {rec.image_label: rec for rec in selected_images}

    # Deserialize each template exactly once
    deserialized = {
        lbl: deserialize_template(templates_by_label[lbl].iris_template_bytes)
        for lbl in labels
    }

    records = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            la, lb = labels[i], labels[j]
            rec_i, rec_j = by_label[la], by_label[lb]
            pair_type = label_pair(rec_i, rec_j)
            hamming = float(distance_matrix_df.at[la, lb])
            extra = compute_extra_scores(deserialized[la], deserialized[lb])
            records.append({
                "set_id": rec_i.set_id,
                "img1_label": la,
                "img2_label": lb,
                "pair_type": pair_type,
                "true_label": 1 if pair_type == "genuine" else 0,
                "hamming": hamming,
                **extra,
            })

    return pd.DataFrame(records, columns=["set_id", "img1_label", "img2_label",
                                           "pair_type", "true_label"] + FEATURE_COLS)


def safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) > 0 else np.nan


def safe_std(series: pd.Series) -> float:
    return float(series.std(ddof=0)) if len(series) > 0 else np.nan


def safe_min(series: pd.Series) -> float:
    return float(series.min()) if len(series) > 0 else np.nan


def safe_max(series: pd.Series) -> float:
    return float(series.max()) if len(series) > 0 else np.nan


def compute_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    total = tp + fp + fn + tn
    accuracy = (tp + tn) / total if total > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (2 * precision * recall / (precision + recall)) if pd.notna(precision) and pd.notna(recall) and (precision + recall) > 0 else np.nan
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    balanced_accuracy = (tpr + tnr) / 2 if pd.notna(tpr) and pd.notna(tnr) else np.nan
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def build_confusion_matrix_df(tp: int, fp: int, fn: int, tn: int) -> pd.DataFrame:
    return pd.DataFrame(
        [[tp, fn], [fp, tn]],
        index=["Actual Match", "Actual Non-Match"],
        columns=["Predicted Match", "Predicted Non-Match"],
    )


def compute_comparison_summary(pair_df: pd.DataFrame, selected_images: List[SelectedImageRecord], threshold: float, eye_side: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    genuine_df = pair_df[pair_df["pair_type"] == "genuine"].copy()
    impostor_df = pair_df[pair_df["pair_type"] == "impostor"].copy()

    pair_df = pair_df.copy()
    pair_df["pred_label"] = (pair_df["distance"] <= threshold).astype(int)

    tp = int(((pair_df["true_label"] == 1) & (pair_df["pred_label"] == 1)).sum())
    fp = int(((pair_df["true_label"] == 0) & (pair_df["pred_label"] == 1)).sum())
    fn = int(((pair_df["true_label"] == 1) & (pair_df["pred_label"] == 0)).sum())
    tn = int(((pair_df["true_label"] == 0) & (pair_df["pred_label"] == 0)).sum())

    metrics = compute_metrics(tp, fp, fn, tn)
    subjects_used = sorted({rec.subject_id for rec in selected_images})
    images_per_subject = max(rec.selection_order for rec in selected_images) if selected_images else 0

    summary = {
        "set_id": selected_images[0].set_id if selected_images else "unknown",
        "dataset_name": "CASIA-Iris-Thousand",
        "subjects_used": len(subjects_used),
        "images_per_subject": images_per_subject,
        "eye_side_used": eye_side,
        "total_images": len(selected_images),
        "total_unique_pairs": len(pair_df),
        "genuine_pairs": len(genuine_df),
        "impostor_pairs": len(impostor_df),
        "mean_genuine_distance": safe_mean(genuine_df["distance"]),
        "std_genuine_distance": safe_std(genuine_df["distance"]),
        "min_genuine_distance": safe_min(genuine_df["distance"]),
        "max_genuine_distance": safe_max(genuine_df["distance"]),
        "mean_impostor_distance": safe_mean(impostor_df["distance"]),
        "std_impostor_distance": safe_std(impostor_df["distance"]),
        "min_impostor_distance": safe_min(impostor_df["distance"]),
        "max_impostor_distance": safe_max(impostor_df["distance"]),
        "threshold": threshold,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        **metrics,
    }
    return pd.DataFrame([summary]), pair_df


# -----------------------------
# Saving helpers
# -----------------------------
def save_dataframe(df: pd.DataFrame, out_path: str, index: bool = False) -> None:
    ensure_dir(os.path.dirname(out_path))
    df.to_csv(out_path, index=index)


def save_confusion_matrix_plot(confusion_matrix_df: pd.DataFrame, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    values = confusion_matrix_df.values.astype(int)
    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(values, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(np.arange(confusion_matrix_df.shape[1]))
    ax.set_yticks(np.arange(confusion_matrix_df.shape[0]))
    ax.set_xticklabels(confusion_matrix_df.columns)
    ax.set_yticklabels(confusion_matrix_df.index)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")
    threshold = values.max() / 2.0 if values.size > 0 else 0
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:,}", ha="center", va="center", color="white" if values[i, j] > threshold else "black", fontsize=12, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_set_outputs(
    set_data: dict,
    set_output_dir: str,
    threshold: float,
    eye_side: str,
    matcher: HammingDistanceMatcher,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_dir(set_output_dir)

    selected_images = set_data["selected_images"]
    failed_images = set_data["failed_images"]
    skipped_subjects = set_data["skipped_subjects"]
    templates_by_label = set_data["templates_by_label"]

    # logs
    save_dataframe(pd.DataFrame([asdict(x) for x in selected_images]), os.path.join(set_output_dir, "selected_images.csv"))
    save_dataframe(pd.DataFrame([asdict(x) for x in failed_images]), os.path.join(set_output_dir, "failed_images.csv"))
    save_dataframe(pd.DataFrame([asdict(x) for x in skipped_subjects]), os.path.join(set_output_dir, "skipped_subjects.csv"))

    manifest_df = pd.DataFrame(
        [{
            "set_id": set_data["set_id"],
            "eye_side": eye_side,
            "subject_id": sid,
            "images_per_subject_required": IMAGES_PER_SUBJECT,
            "subjects_per_set_required": SUBJECTS_PER_SET,
        } for sid in set_data["subjects"]]
    )
    save_dataframe(manifest_df, os.path.join(set_output_dir, "set_manifest.csv"))

    # evaluation
    distance_matrix_df = build_distance_matrix(selected_images, templates_by_label, matcher)
    save_dataframe(distance_matrix_df, os.path.join(set_output_dir, "distance_matrix.csv"), index=True)

    pair_df = extract_unique_pair_records(distance_matrix_df, selected_images)
    summary_df, pair_df_with_pred = compute_comparison_summary(pair_df, selected_images, threshold, eye_side)
    save_dataframe(pair_df_with_pred, os.path.join(set_output_dir, "pair_records.csv"))
    save_dataframe(summary_df, os.path.join(set_output_dir, "comparison_summary.csv"))

    row = summary_df.iloc[0]
    confusion_df = build_confusion_matrix_df(int(row["TP"]), int(row["FP"]), int(row["FN"]), int(row["TN"]))
    save_dataframe(confusion_df, os.path.join(set_output_dir, "confusion_matrix.csv"), index=True)
    save_confusion_matrix_plot(confusion_df, os.path.join(set_output_dir, "confusion_matrix.png"))

    # multi-score features for fusion
    multi_score_df = build_multi_score_pair_df(selected_images, templates_by_label, distance_matrix_df)
    save_dataframe(multi_score_df, os.path.join(set_output_dir, "multi_score_features.csv"))

    return summary_df, multi_score_df


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ensure_dir(OUTPUT_ROOT)
    if USE_CACHE:
        ensure_dir(CACHE_DIR)

    resolved_dataset_root = resolve_dataset_root(DATASET_ROOT)

    print("=== CASIA-Iris-Thousand Multi-Set Balanced Evaluation ===")
    print(f"Dataset root        : {DATASET_ROOT}")
    if resolved_dataset_root != DATASET_ROOT:
        print(f"Resolved data root  : {resolved_dataset_root}")
    print(f"Output root         : {OUTPUT_ROOT}")
    print(f"Subjects per set    : {SUBJECTS_PER_SET}")
    print(f"Images per subject  : {IMAGES_PER_SUBJECT}")
    print(f"Eye side            : {TARGET_EYE_SIDE}")
    print(f"Threshold           : {THRESHOLD}")
    print(f"Max sets            : {MAX_SETS}")
    print(f"Use cache           : {USE_CACHE}")
    print()

    pipeline = iris.IRISPipeline()
    matcher = HammingDistanceMatcher()

    all_sets, _ = build_sets_from_dataset(
        dataset_root=resolved_dataset_root,
        subjects_per_set=SUBJECTS_PER_SET,
        images_per_subject=IMAGES_PER_SUBJECT,
        eye_side=TARGET_EYE_SIDE,
        pipeline=pipeline,
        max_sets=MAX_SETS,
    )

    if not all_sets:
        detected_subjects = count_subject_like_dirs(resolved_dataset_root)
        raise RuntimeError(
            "No complete sets could be built with the current policy. "
            f"Detected {detected_subjects} subject-like folders under: {resolved_dataset_root}. "
            "Try reducing SUBJECTS_PER_SET or IMAGES_PER_SUBJECT, or verify TARGET_EYE_SIDE."
        )

    all_summaries: List[pd.DataFrame] = []
    all_multi_score_dfs: List[pd.DataFrame] = []

    for set_data in all_sets:
        subjects = set_data["subjects"]
        subject_tag = f"{subjects[0]}-{subjects[-1]}" if subjects else "unknown"
        set_output_dir = os.path.join(OUTPUT_ROOT, f"{set_data['set_id']}_{TARGET_EYE_SIDE}_{subject_tag}")
        print(f"[run] processing {set_data['set_id']} -> {set_output_dir}")
        summary_df, multi_score_df = save_set_outputs(set_data, set_output_dir, THRESHOLD, TARGET_EYE_SIDE, matcher)
        all_summaries.append(summary_df)
        all_multi_score_dfs.append(multi_score_df)

    all_sets_summary_df = pd.concat(all_summaries, ignore_index=True)
    save_dataframe(all_sets_summary_df, os.path.join(OUTPUT_ROOT, "all_sets_summary.csv"))

    numeric_cols = [
        "total_images", "total_unique_pairs", "genuine_pairs", "impostor_pairs",
        "mean_genuine_distance", "mean_impostor_distance", "TP", "FP", "FN", "TN",
        "accuracy", "precision", "recall", "f1", "balanced_accuracy"
    ]
    aggregate_df = all_sets_summary_df[numeric_cols].agg(["mean", "std", "min", "max"])
    save_dataframe(aggregate_df.reset_index().rename(columns={"index": "stat"}), os.path.join(OUTPUT_ROOT, "aggregate_summary.csv"))

    print("=== DONE ===")
    print(f"Completed sets: {len(all_sets_summary_df)}")
    print(f"Saved all-set summary to: {os.path.join(OUTPUT_ROOT, 'all_sets_summary.csv')}")
    print(f"Saved aggregate summary to: {os.path.join(OUTPUT_ROOT, 'aggregate_summary.csv')}")


if __name__ == "__main__":
    main()
