import os
import json
import glob
import re
import gc
import warnings
import time
import platform
import hashlib
from datetime import datetime, timezone
from collections import Counter

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import nltk
import fitz
from tqdm import tqdm
from scipy import stats

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from bert_score import score as bert_score_fn


os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- CONFIGURATION ---
TARGET_PER_MODEL_PER_TYPE = 50
TYPES_ORDER = ["multiple-choice", "true-false", "fill-in-the-blank", "essay"]

# --- PALETTE MAP ---
PALETTE_MAP = {
    "BERTScore_F1": "Greens",
    "Perplexity": "Reds_r",
    "Distinct_1": "mako",
    "Distinct_2": "mako",
    "Duplicate_Rate": "Greys_r",
    "Latency_Sec_PerQ": "Blues_r",
    "Confidence": "Greys",
}


def log(msg, level="INFO"):
    time_str = datetime.now().strftime("%H:%M:%S")
    print(f"[{time_str}] [{level:<7}] {msg}")


def log_kv(title, d, level="INFO"):
    keys = list(d.keys())
    parts = []
    for k in keys:
        parts.append(f"{k}={d[k]}")
    log(f"{title} | " + " | ".join(parts), level)


class StageTimer:
    def __init__(self, name):
        self.name = name
        self.t0 = None

    def __enter__(self):
        self.t0 = time.time()
        log(f"START {self.name}", "TIMER")
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        log(f"END {self.name} | {dt:.2f}s", "TIMER")


def gpu_info():
    if not torch.cuda.is_available():
        return {"device": "cpu"}
    i = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(i)
    mem_alloc = torch.cuda.memory_allocated(i) / (1024**3)
    mem_reserved = torch.cuda.memory_reserved(i) / (1024**3)
    mem_total = props.total_memory / (1024**3)
    return {
        "device": f"cuda:{i}",
        "name": props.name,
        "mem_alloc_GB": f"{mem_alloc:.2f}",
        "mem_reserved_GB": f"{mem_reserved:.2f}",
        "mem_total_GB": f"{mem_total:.2f}",
    }


def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = str(text)
    text = re.sub(r"-\n", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def cuda_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def to_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return float(default)


def to_float_conf(x):
    if x is None:
        return 0.0
    s = str(x).strip().replace("%", "")
    try:
        v = float(s)
        if v > 1.0:
            v = v / 100.0
        return float(max(0.0, min(v, 1.0)))
    except Exception:
        return 0.0


def compute_f1_score(prediction, ground_truth):
    prediction_tokens = clean_text(prediction).lower().split()
    ground_truth_tokens = clean_text(ground_truth).lower().split()

    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return int(prediction_tokens == ground_truth_tokens)

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def parse_pages(page_str):
    if page_str is None:
        return []
    if isinstance(page_str, (int, float)) and not np.isnan(page_str):
        return [int(page_str)]
    if not isinstance(page_str, str):
        return []
    s = page_str.strip()
    if not s:
        return []
    pages = set()
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                a, b = part.split("-", 1)
                start, end = int(a.strip()), int(b.strip())
                if start > end:
                    start, end = end, start
                pages.update(range(start, end + 1))
            else:
                pages.add(int(part))
        except Exception:
            pass
    return sorted(pages)


def file_md5(path, chunk_size=1024 * 1024):
    try:
        h = hashlib.md5()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk_size)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return ""


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


# --- PATHS ---
BASE_DIR = os.path.abspath(".")
FOLDER_PATH = os.path.join(BASE_DIR, "data")
RUN_ID = os.getenv("OVERRIDE_RUN_ID", datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%SZ"))
RUN_DIR = os.path.join(BASE_DIR, "runs", RUN_ID)
UNIV_DIR = ensure_dir(os.path.join(RUN_DIR, "universal"))
UNIV_TABLES = ensure_dir(os.path.join(UNIV_DIR, "tables"))
UNIV_FIGS = ensure_dir(os.path.join(UNIV_DIR, "figures"))
BYTYPE_DIR = ensure_dir(os.path.join(RUN_DIR, "by_type"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MODELS
EVAL_LM_ID = "aisingapore/Llama-SEA-LION-v2-8B-IT"

PDF_FILENAME = None

log("Checking NLTK resources...", "INIT")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/wordnet.zip")
except LookupError:
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

from nltk.tokenize import word_tokenize


def resolve_pdf_path():
    if PDF_FILENAME:
        p = os.path.join(FOLDER_PATH, PDF_FILENAME)
        if os.path.exists(p):
            return p
    pdfs = glob.glob(os.path.join(FOLDER_PATH, "**", "*.pdf"), recursive=True)
    if not pdfs:
        return None
    if len(pdfs) == 1:
        return pdfs[0]
    pdfs = sorted(pdfs, key=lambda x: os.path.getsize(x), reverse=True)
    return pdfs[0]


pdf_full_path = resolve_pdf_path()
HAS_PDF = pdf_full_path is not None
if HAS_PDF:
    log(f"PDF Reference : FOUND {os.path.basename(pdf_full_path)}", "OK")
else:
    log("PDF Reference : NOT FOUND (Fallback to JSON source_text)", "WARN")

pdf_cache = {}


def get_ref_text_pages(page_str, json_anchor):
    anchor = clean_text(json_anchor)
    if not HAS_PDF:
        return (anchor if anchor else " "), "JSON_Source(NoPDF)"
    if not page_str:
        return (anchor if anchor else " "), "JSON_Source(NoPage)"
    pages = parse_pages(page_str)
    if not pages:
        return (anchor if anchor else " "), "JSON_Source(BadPages)"

    cache_key = f"pages:{page_str}"
    if cache_key in pdf_cache:
        return pdf_cache[cache_key], "PDF_Cache"

    try:
        doc = fitz.open(pdf_full_path)
        extracted = []
        for p in pages:
            i = p - 1
            if 0 <= i < len(doc):
                extracted.append(doc.load_page(i).get_text("text"))
        cleaned = clean_text("\n".join(extracted))

        if len(cleaned) < 50:
            cleaned = anchor
        if not cleaned:
            cleaned = " "

        pdf_cache[cache_key] = cleaned
        return cleaned, "PDF_Extracted"
    except Exception:
        cleaned = anchor if anchor else " "
        return cleaned, "JSON_Fallback(PDF_Error)"


def iter_questions_sonar(data):
    if not isinstance(data, dict):
        return None, []
    session = data.get("session_details", {}) or {}
    logs_ = session.get("logs", []) or []
    out = []
    for le in logs_:
        if not isinstance(le, dict):
            continue
        pages = le.get("pages", None)
        batch_latency = to_float(le.get("duration", 0.0), default=0.0)
        q_count = int(le.get("question_count", 0) or 0)
        qlist = le.get("questions", []) or []
        for q in qlist:
            if not isinstance(q, dict):
                continue
            q2 = dict(q)
            q2["_pages"] = pages
            q2["_log_duration_sec"] = batch_latency
            q2["_log_question_count"] = q_count
            out.append(q2)
    return session, out


def calculate_duplicate_rate(questions_list):
    if not questions_list:
        return 0.0
    normalized = [
        re.sub(r"[^\w\s]", "", (q or "").lower().strip()) for q in questions_list if q
    ]
    if not normalized:
        return 0.0
    return 1.0 - (len(set(normalized)) / len(normalized))


def calculate_distinct_n(texts, n=2):
    if not texts:
        return 0.0
    all_ngrams = []
    for text in texts:
        tokens = word_tokenize((text or "").lower())
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            all_ngrams.append(tuple(tokens[i : i + n]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def normalize_difficulty(x: str) -> str:
    s = clean_text(x).upper()
    if s in ["LOTS", "MOTS", "HOTS"]:
        return s
    return "INVALID"


def add_value_labels(ax, fmt="{:.3f}"):
    for container in ax.containers:
        try:
            labels = []
            for p in container:
                h = p.get_height()
                if h is None or (isinstance(h, float) and np.isnan(h)):
                    labels.append("")
                else:
                    labels.append(fmt.format(h))
            ax.bar_label(container, labels=labels, padding=3, fontsize=8)
        except Exception:
            pass


def format_for_metric(metric):
    if metric in ["Perplexity"]:
        return "{:.2f}"
    if metric in ["Latency_Sec_PerQ"]:
        return "{:.3f}"
    return "{:.4f}"


def build_df_by_model(df):
    return (
        df.groupby("Model")
        .agg(
            Perplexity=("Perplexity", "mean"),
            BERTScore_F1=("BERTScore_F1", "mean"),
            Distinct_1=("Distinct_1", "first"),
            Distinct_2=("Distinct_2", "first"),
            Duplicate_Rate=("Duplicate_Rate", "first"),
            Latency_Sec_PerQ=("Latency_Sec_PerQ", "mean"),
            Confidence=("Confidence", "mean"),
        )
        .reset_index()
    )


# --- PLOT HEATMAP ---
def plot_p_values_heatmap(stat_results, out_dir):
    if not stat_results:
        return

    df_stats = pd.DataFrame(stat_results)

    heatmap_data = []
    labels = []

    for _, row in df_stats.iterrows():
        kw_p = row["KW_P_Value"]
        anova_p = row["ANOVA_P_Value"]
        heatmap_data.append([kw_p, anova_p])
        labels.append([f"{kw_p:.4f}", f"{anova_p:.4f}"])

    heatmap_df = pd.DataFrame(
        heatmap_data, index=df_stats["Metric"], columns=["Kruskal-Wallis P", "ANOVA P"]
    )

    plt.figure(figsize=(9, len(df_stats) * 0.8 + 2))
    sns.set_theme(style="white")

    ax = sns.heatmap(
        heatmap_df,
        annot=np.array(labels),
        fmt="",
        cmap="Greens_r",
        cbar_kws={"label": "P-Value (Darker Green = More Significant)"},
        vmin=0,
        vmax=0.1,
    )

    plt.title(
        "Statistical Significance Test Results (P-Values)\np < 0.05 indicates significant difference",
        fontweight="bold",
    )
    plt.tight_layout()

    out_path = os.path.join(out_dir, "Test_Result_Heatmap.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    log(f"Heatmap Saved: {out_path}", "SUCCESS")


def plot_universal(df_by_model, out_dir):
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
    # Define grid size explicitly
    N_ROWS, N_COLS = 2, 4
    fig, axes = plt.subplots(N_ROWS, N_COLS, figsize=(28, 18))
    plt.subplots_adjust(hspace=0.45, wspace=0.3)

    combined = [
        (0, 0, "BERTScore_F1", "BERTScore F1 (Higher Better)"),
        (0, 1, "Confidence", "Gen. Confidence (Higher Better)"),
        (0, 2, "Perplexity", "Perplexity (Lower Better)"),
        (0, 3, "Latency_Sec_PerQ", "Latency/Question (Lower Better)"),
        (1, 0, "Distinct_1", "Distinct-1 (Higher Better)"),
        (1, 1, "Distinct_2", "Distinct-2 (Higher Better)"),
        (1, 2, "Duplicate_Rate", "Duplicate Rate (Lower Better)"),
    ]

    for r, c, metric, title in combined:
        # Safety check to ensure we don't exceed dimensions defined above
        if r >= N_ROWS or c >= N_COLS:
            continue

        ax = axes[r, c]
        if metric in df_by_model.columns:
            sns.barplot(
                data=df_by_model,
                x="Model",
                y=metric,
                ax=ax,
                palette=PALETTE_MAP.get(metric, "viridis"),
                hue="Model",
                dodge=False,
            )
            ax.set_title(title, fontweight="bold", fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="x", rotation=45, labelsize=9)
            fmt = format_for_metric(metric)
            add_value_labels(ax, fmt=fmt)
            try:
                ax.legend_.remove()
            except Exception:
                pass
        else:
            ax.axis("off")

    # Corrected loop limits to match N_ROWS and N_COLS
    for r in range(N_ROWS):
        for c in range(N_COLS):
            if not any((item[0] == r and item[1] == c) for item in combined):
                axes[r, c].axis("off")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles and labels:
        fig.legend(
            handles,
            labels,
            title="Model",
            loc="upper center",
            ncol=min(8, len(labels)),
            bbox_to_anchor=(0.5, 0.98),
        )

    # --- UPDATED INFO BOX ---
    info_text = (
        f"Evaluation Config:\n"
        f"- PPL Model: {EVAL_LM_ID.split('/')[-1]}\n"
        f"- Stats: Kruskal-Wallis & ANOVA (p<0.05)\n"
        f"- Generated: {datetime.now().strftime('%Y-%m-%d')}"
    )
    plt.figtext(
        0.98,
        0.02,
        info_text,
        fontsize=10,
        color="#333333",
        ha="right",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.5", fc="#f0f0f0", ec="#999999", alpha=0.9),
    )

    combined_path = os.path.join(out_dir, "Combined_Metrics.png")
    plt.savefig(combined_path, dpi=300, bbox_inches="tight")
    plt.close()
    log(f"Chart Saved: {combined_path}", "SUCCESS")

    separated_plots = [
        ("Separated_BERTScore.png", "BERTScore F1 (Higher Better)", "BERTScore_F1"),
        (
            "Separated_Confidence.png",
            "Confidence (Higher Better)",
            "Confidence",
        ),
        ("Separated_Perplexity.png", "Perplexity (Lower Better)", "Perplexity"),
        (
            "Separated_Latency.png",
            "Latency/Question (Lower Better)",
            "Latency_Sec_PerQ",
        ),
        ("Separated_Distinct1.png", "Distinct-1 (Higher Better)", "Distinct_1"),
        ("Separated_Distinct2.png", "Distinct-2 (Higher Better)", "Distinct_2"),
        (
            "Separated_DuplicateRate.png",
            "Duplicate Rate (Lower Better)",
            "Duplicate_Rate",
        ),
    ]

    for fname, title, y_col in separated_plots:
        plt.figure(figsize=(10, 6))
        # Check if column exists before plotting to avoid errors
        if y_col not in df_by_model.columns:
            plt.close()
            continue

        ax = sns.barplot(
            data=df_by_model,
            x="Model",
            y=y_col,
            palette=PALETTE_MAP.get(y_col, "viridis"),
            hue="Model",
            dodge=False,
        )
        plt.title(title, fontweight="bold", fontsize=14, pad=15)
        plt.ylabel(y_col, fontweight="bold", fontsize=12)
        plt.xlabel("AI Model", fontweight="bold", fontsize=12)
        plt.xticks(rotation=45, ha="right")

        fmt = format_for_metric(y_col)
        add_value_labels(ax, fmt=fmt)

        plt.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()

        outp = os.path.join(out_dir, fname)
        plt.savefig(outp, dpi=300, transparent=True)
        plt.close()
        log(f"Chart Saved: {outp}", "SUCCESS")


def plot_type_distribution(df_all, out_dir):
    plt.figure(figsize=(12, 6))
    type_counts = df_all.groupby(["Model", "Type"]).size().reset_index(name="Count")
    total_counts = df_all.groupby("Model")["Type"].count().reset_index(name="Total")
    type_counts = type_counts.merge(total_counts, on="Model")
    type_counts["Percentage"] = (type_counts["Count"] / type_counts["Total"]) * 100

    ax = sns.barplot(
        data=type_counts, x="Model", y="Percentage", hue="Type", palette="viridis"
    )
    plt.title("Distribution of Question Types", fontweight="bold")
    plt.ylabel("Percentage (%)")
    plt.xticks(rotation=45)
    add_value_labels(ax, fmt="{:.1f}")
    plt.legend(title="Question Type", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    dist_path = os.path.join(out_dir, "Distribution.png")
    plt.savefig(dist_path, dpi=300)
    plt.close()
    log(f"Chart Saved: {dist_path}", "SUCCESS")


# --- MAIN EXECUTION FLOW ---
if __name__ == "__main__":
    json_files = glob.glob(os.path.join(FOLDER_PATH, "**", "*.json"), recursive=True)
    json_files = [f for f in json_files if "package" not in f and "tsconfig" not in f]

    if not json_files:
        log("NO JSON FILES FOUND!", "FATAL")
        raise SystemExit(1)

    log(f"JSON Files Found : {len(json_files)} files ready.", "OK")
    log_kv(
        "ENV",
        {"DEVICE": DEVICE, "RUN_DIR": RUN_DIR, "FOLDER_PATH": FOLDER_PATH},
        "INFO",
    )
    log_kv("GPU", gpu_info(), "INFO")

    meta = {
        "run_id": RUN_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": os.path.abspath(FOLDER_PATH),
        "output_root": os.path.abspath(RUN_DIR),
        "device": DEVICE,
        "gpu": gpu_info(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "eval_lm_id": EVAL_LM_ID,
        "n_json_files": len(json_files),
        "json_files": [{"path": fp, "md5": file_md5(fp)} for fp in json_files[:5000]],
        "pdf_reference_path": os.path.abspath(pdf_full_path) if pdf_full_path else None,
        "target_per_model_per_type": TARGET_PER_MODEL_PER_TYPE,
        "types_order": TYPES_ORDER,
    }

    with open(os.path.join(RUN_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print("AQG Evaluation Pipeline (FINAL VALIDATED FOR JOURNAL)")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    rows_all, rows_mcq, rows_tf, rows_fib, rows_essay = [], [], [], [], []
    ref_source_counter = Counter()
    type_counter = Counter()
    file_q_counts = {}

    with StageTimer("LOAD_JSON_AND_BUILD_ROWS"):
        for i, fp in enumerate(json_files):
            fname = os.path.basename(fp)
            print(f"\n>> PROCESSING FILE {i+1}/{len(json_files)}: {fp}")
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception as e:
                log(f"Skip (bad json): {fp} | {e}", "WARN")
                continue

            session, questions = iter_questions_sonar(data)
            if not questions:
                log(f"Skip (no questions): {fname}", "WARN")
                continue

            model_name = (
                session.get("model", fname.split("_")[0])
                if isinstance(session, dict)
                else fname.split("_")[0]
            )
            model_name = model_name.split("/")[-1] if "/" in model_name else model_name
            session_id = session.get("id", "") if isinstance(session, dict) else ""
            file_q_counts[fname] = len(questions)

            for q in questions:
                q_type = (q.get("type", "unknown") or "unknown").strip()
                q_text = clean_text(q.get("question", "")) or " "
                ans = clean_text(q.get("answer", "")) or " "
                expl = clean_text(q.get("explanation", "")) or " "
                anchor = clean_text(q.get("source_text", ""))
                pages = q.get("_pages", "") or ""

                ref_text, ref_source = get_ref_text_pages(pages, anchor)

                pages_raw = pages
                if isinstance(pages_raw, (list, tuple, set)):
                    pages_list = [
                        str(p)
                        for p in pages_raw
                        if p is not None and str(p).strip() != ""
                    ]
                    pages_str = ",".join(pages_list)
                else:
                    pages_str = str(pages_raw) if pages_raw is not None else ""
                pages_csv = f"'{pages_str}" if pages_str else ""

                conf = float(to_float_conf(q.get("confidence", 0)))
                log_dur = float(to_float(q.get("_log_duration_sec", 0.0), default=0.0))
                log_qcount = int(q.get("_log_question_count", 0) or 0)
                latency_batch = log_dur
                latency_per_q = (log_dur / log_qcount) if log_qcount > 0 else 0.0

                raw_diff = (q.get("difficulty", "") or "").strip()
                diff_norm = normalize_difficulty(raw_diff)

                base = {
                    "Run_ID": RUN_ID,
                    "Source_File": fp,
                    "Session_ID": session_id,
                    "Model": model_name,
                    "Type": q_type,
                    "Difficulty": raw_diff if raw_diff else "unknown",
                    "Difficulty_Norm": diff_norm,
                    "Question": q_text,
                    "Answer": ans,
                    "Explanation": expl,
                    "Reference": ref_text if ref_text else " ",
                    "Ref_Source": ref_source if ref_source else "unknown",
                    "Pages": pages_csv,
                    "Confidence": conf,
                    "Latency_Sec_Batch": latency_batch,
                    "Latency_Sec_PerQ": float(latency_per_q),
                    "Distinct_1": 0.0,
                    "Distinct_2": 0.0,
                    "Duplicate_Rate": 0.0,
                    "BERTScore_F1": 0.0,
                    "Perplexity": 0.0,
                }
                rows_all.append(base)
                type_counter[q_type] += 1
                ref_source_counter[ref_source] += 1

    log_kv("FILES_Q_COUNT", file_q_counts, "INFO")
    log_kv("TYPE_COUNT", dict(type_counter), "INFO")
    log_kv("REF_SOURCE_COUNT", dict(ref_source_counter), "INFO")

    df_all = pd.DataFrame(rows_all)
    if df_all.empty:
        log("No data extracted.", "FATAL")
        raise SystemExit(1)

    with StageTimer("DISTINCT_AND_DUPLICATE"):
        for m in df_all["Model"].unique():
            mask = df_all["Model"] == m
            qs = df_all.loc[mask, "Question"].tolist()
            d1 = float(calculate_distinct_n(qs, n=1))
            d2 = float(calculate_distinct_n(qs, n=2))
            dr = float(calculate_duplicate_rate(qs))

            df_all.loc[mask, "Distinct_1"] = d1
            df_all.loc[mask, "Distinct_2"] = d2
            df_all.loc[mask, "Duplicate_Rate"] = dr

            log_kv(
                "MODEL_LEXICAL",
                {
                    "model": m,
                    "Distinct_1": f"{d1:.4f}",
                    "Distinct_2": f"{d2:.4f}",
                    "Duplicate_Rate": f"{dr:.4f}",
                },
                "INFO",
            )

    raw_all_path = os.path.join(UNIV_TABLES, "raw_all_data.csv")
    df_all.to_csv(raw_all_path, index=False, encoding="utf-8-sig")
    log(f"Saved: {raw_all_path}", "OK")

    def save_debug_type(rows, out_tables):
        df = pd.DataFrame(rows)
        if df.empty:
            return None
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        zero_cols = [c for c in numeric_cols if (df[c].fillna(0) == 0).all()]
        if zero_cols:
            df = df.drop(columns=zero_cols)
        p = os.path.join(out_tables, "raw_debug.csv")
        df.to_csv(p, index=False, encoding="utf-8-sig")
        log(f"Saved: {p}", "OK")
        return df, p

    type_map = {
        "multiple-choice": rows_mcq,
        "true-false": rows_tf,
        "fill-in-the-blank": rows_fib,
        "essay": rows_essay,
    }
    bytype_packs = {}
    for tname in TYPES_ORDER:
        rows = type_map[tname]
        tdir = ensure_dir(os.path.join(BYTYPE_DIR, tname))
        pack = save_debug_type(rows, ensure_dir(os.path.join(tdir, "tables")))
        bytype_packs[tname] = {
            "dir": tdir,
            "tables": ensure_dir(os.path.join(tdir, "tables")),
            "figures": ensure_dir(os.path.join(tdir, "figures")),
            "pack": pack,
        }

    with StageTimer("BERTSCORE"):
        try:
            _, _, F1 = bert_score_fn(
                df_all["Question"].tolist(),
                df_all["Reference"].tolist(),
                lang="id",
                verbose=False,
                device=DEVICE,
                batch_size=8,
            )
            df_all["BERTScore_F1"] = F1.cpu().numpy()
        except:
            df_all["BERTScore_F1"] = 0.0
        log_kv(
            "BERTSCORE_STATS", {"mean": f"{df_all['BERTScore_F1'].mean():.4f}"}, "INFO"
        )
        cuda_cleanup()

    with StageTimer("PERPLEXITY"):
        tok = AutoTokenizer.from_pretrained(EVAL_LM_ID)
        lm = AutoModelForCausalLM.from_pretrained(
            EVAL_LM_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        def ppl_compute(text):
            try:
                enc = tok(
                    text, return_tensors="pt", truncation=True, max_length=256
                ).to(lm.device)
                return float(torch.exp(lm(**enc, labels=enc["input_ids"]).loss).item())
            except:
                return 0.0

        df_all["Perplexity"] = [
            ppl_compute(q) for q in tqdm(df_all["Question"], desc="PPL")
        ]
        log_kv("PPL_STATS", {"mean": f"{df_all['Perplexity'].mean():.4f}"}, "INFO")
        del lm, tok
        cuda_cleanup()

    df_all.to_csv(raw_all_path, index=False, encoding="utf-8-sig")

    # --- RESTORED CSV OUTPUTS FOR EACH TYPE ---
    for tname in TYPES_ORDER:
        sub = df_all[df_all["Type"] == tname].copy()
        if sub.empty:
            continue
        out_raw = os.path.join(bytype_packs[tname]["tables"], f"raw_{tname}.csv")
        sub.to_csv(out_raw, index=False, encoding="utf-8-sig")

    with StageTimer("SUMMARY_TABLES"):
        df_summary = (
            df_all.groupby(["Model", "Type"])
            .agg(
                n=("Question", "count"),
                Avg_PPL=("Perplexity", "mean"),
                Avg_BERTScore=("BERTScore_F1", "mean"),
                Distinct_1=("Distinct_1", "first"),
                Distinct_2=("Distinct_2", "first"),
                Duplicate_Rate=("Duplicate_Rate", "first"),
                Avg_Confidence=("Confidence", "mean"),
                Avg_Latency_Batch_Sec=("Latency_Sec_Batch", "mean"),
                Avg_Latency_PerQ_Sec=("Latency_Sec_PerQ", "mean"),
            )
            .reset_index()
        )
        df_summary.to_csv(
            os.path.join(UNIV_TABLES, "summary_by_model_type.csv"),
            index=False,
            encoding="utf-8-sig",
        )
        build_df_by_model(df_all).to_csv(
            os.path.join(UNIV_TABLES, "summary_overall_micro.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    # --- RESTORED DATA QUALITY & COMPLIANCE ---
    with StageTimer("DATA_QUALITY_CHECKS"):
        dq = df_all.groupby(["Model", "Type"]).size().reset_index(name="Total")
        for col in ["Question", "Answer", "Explanation", "Reference", "Difficulty"]:
            miss = (
                df_all.assign(
                    _miss=df_all[col]
                    .astype(str)
                    .str.strip()
                    .isin(["", "nan", "None", " "])
                )
                .groupby(["Model", "Type"])["_miss"]
                .mean()
                .reset_index(name=f"MissingRate_{col}")
            )
            dq = dq.merge(miss, on=["Model", "Type"], how="left")
        dq["InvalidRate_Difficulty"] = (
            (df_all["Difficulty_Norm"] == "INVALID")
            .groupby([df_all["Model"], df_all["Type"]])
            .mean()
            .reset_index(drop=True)
        )

        count_check = df_all.groupby(["Model", "Type"]).size().reset_index(name="n")
        count_check["Count_Compliant"] = (
            count_check["n"] == TARGET_PER_MODEL_PER_TYPE
        ).astype(int)
        dq = dq.merge(
            count_check[["Model", "Type", "Count_Compliant"]],
            on=["Model", "Type"],
            how="left",
        )

        dq.to_csv(
            os.path.join(UNIV_TABLES, "data_quality_checks.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    with StageTimer("PROMPT_COMPLIANCE_DIFFICULTY_MIXED"):
        targets = {"LOTS": 0.40, "MOTS": 0.30, "HOTS": 0.30}
        dist = (
            df_all.groupby(["Model", "Type", "Difficulty_Norm"])
            .size()
            .reset_index(name="Count")
        )
        pivot = dist.pivot_table(
            index=["Model", "Type"],
            columns="Difficulty_Norm",
            values="Count",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()
        for c in ["LOTS", "MOTS", "HOTS", "INVALID"]:
            if c not in pivot.columns:
                pivot[c] = 0

        pivot["Total"] = pivot[["LOTS", "MOTS", "HOTS", "INVALID"]].sum(axis=1)
        pivot["Pct_Invalid"] = np.where(
            pivot["Total"] > 0, pivot["INVALID"] / pivot["Total"], 0.0
        )
        valid_tot = pivot[["LOTS", "MOTS", "HOTS"]].sum(axis=1)

        for k in ["LOTS", "MOTS", "HOTS"]:
            pivot[f"Pct_{k}"] = np.where(valid_tot > 0, pivot[k] / valid_tot, 0.0)
            pivot[f"Dev_{k}"] = (pivot[f"Pct_{k}"] - targets[k]).abs()

        pivot["Compliant_Flag"] = (
            (pivot["Pct_Invalid"] == 0.0)
            & (pivot["Dev_LOTS"] <= 0.1)
            & (pivot["Dev_MOTS"] <= 0.1)
            & (pivot["Dev_HOTS"] <= 0.1)
        ).astype(int)
        pivot.to_csv(
            os.path.join(UNIV_TABLES, "compliance_by_model_type.csv"),
            index=False,
            encoding="utf-8-sig",
        )

    with StageTimer("STATISTICAL_TESTS"):
        target_metrics = [
            "BERTScore_F1",
            "Latency_Sec_PerQ",
            "Perplexity",
        ]
        stat_results = []
        models = df_all["Model"].unique()

        if len(models) > 1:
            for metric in target_metrics:
                if metric not in df_all.columns:
                    continue
                groups = [
                    df_all[df_all["Model"] == m][metric].dropna().values for m in models
                ]
                if len(groups) < 2 or any(len(g) < 2 for g in groups):
                    continue

                try:
                    kw_stat, kw_p = stats.kruskal(*groups)
                except:
                    kw_p = 1.0

                try:
                    anova_stat, anova_p = stats.f_oneway(*groups)
                except:
                    anova_p = 1.0

                stat_results.append(
                    {
                        "Metric": metric,
                        "KW_P_Value": kw_p,
                        "KW_Significant": "**YES**" if kw_p < 0.05 else "No",
                        "ANOVA_P_Value": anova_p,
                        "ANOVA_Significant": "**YES**" if anova_p < 0.05 else "No",
                    }
                )

            pd.DataFrame(stat_results).to_csv(
                os.path.join(UNIV_TABLES, "statistical_tests_results.csv"), index=False
            )
            plot_p_values_heatmap(stat_results, UNIV_FIGS)

    with StageTimer("PLOTS_UNIVERSAL"):
        df_by_model = build_df_by_model(df_all)
        plot_universal(df_by_model, UNIV_FIGS)
        plot_type_distribution(df_all, UNIV_FIGS)

    with StageTimer("PLOTS_BY_TYPE"):
        for tname in TYPES_ORDER:
            sub = df_all[df_all["Type"] == tname].copy()
            if sub.empty:
                continue
            out_f = bytype_packs[tname]["figures"]
            plot_universal(build_df_by_model(sub), out_f)
            plot_type_distribution(sub, out_f)

    log("DONE", "DONE")
    print(os.path.abspath(RUN_DIR))
