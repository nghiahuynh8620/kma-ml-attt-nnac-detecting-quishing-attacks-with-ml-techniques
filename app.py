from __future__ import annotations

from pathlib import Path
from datetime import datetime
import csv
import os
import time
import statistics

import joblib
import numpy as np
import streamlit as st
import qrcode
from PIL import Image

try:
    import pandas as pd
except Exception:
    pd = None

try:
    import psutil
except Exception:
    psutil = None


# =========================================================
# PROJECT CONFIG
# =========================================================
LABEL_NAME_MAP = {0: "benign", 1: "malicious"}

SCHOOL_PARENT = "Ban Cơ yếu Chính phủ"
SCHOOL_NAME = "Học viện Kỹ thuật Mật mã"
PROJECT_TITLE = "Phát hiện tấn công Quishing bằng Học máy"
PROJECT_SUBTITLE = "Ứng dụng nhận diện quishing trực tiếp từ cấu trúc ảnh QR theo pipeline paper-only 10-fold"
TEAM_MEMBERS = [
    "Vũ Thị Diệu Anh - CHAT4P001",
    "Diệp Kim Chi - CHAT4P003",
    "Huỳnh Trọng Nghĩa - CHAT4P011",
    "Võ Minh Nhật - CHAT4P013",
]
ADVISOR_NAME = "TS. Nguyễn An Khương"

SAFE_BENIGN_QR_PAYLOADS = [
    "https://example.com",
    "https://example.org/library",
    "WIFI:T:WPA;S:CampusGuest;P:Safe12345;;",
    "BEGIN:VCARD\nFN:Lab Reception\nTEL:+84000000000\nEND:VCARD",
]

SAFE_SIMULATED_MALICIOUS_QR_PAYLOADS = [
    "http://198.51.100.24/verify",
    "http://203.0.113.7/billing",
    "https://secure-login.example.invalid/update",
    "https://account-check.example.invalid/open",
]

OUTPUT_DIR_CANDIDATES = [
    Path("./outputs_quishing_paper_10fold"),
    Path("outputs_quishing_paper_10fold"),
    Path("/mnt/data/outputs_quishing_paper_10fold"),
    Path("./outputs"),
    Path("outputs"),
]

LOGO_CANDIDATES = [
    Path("./logo_kma.png"),
    Path("./logo_kma.jpg"),
    Path("./logo_kma.jpeg"),
    Path("./logo_kma.webp"),
    Path("./logo.png"),
    Path("./logo.jpg"),
    Path("./assets/logo_kma.png"),
    Path("./assets/logo.png"),
    Path("./assets/KMA_logo.png"),
]

TARGET_SHAPE = (69, 69)
QR_VERSION = 13
QR_ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_L
QR_BOX_SIZE = 1
QR_BORDER = 0


# =========================================================
# PATH / IO
# =========================================================
def resolve_output_dir() -> Path:
    for p in OUTPUT_DIR_CANDIDATES:
        if (p / "models_new").exists():
            return p
    return Path("./outputs_quishing_paper_10fold")


def resolve_logo_path() -> Path | None:
    for p in LOGO_CANDIDATES:
        if p.exists():
            return p
    return None


DEFAULT_OUTPUT_DIR = resolve_output_dir()
DEFAULT_LOG_DIR = DEFAULT_OUTPUT_DIR / "webapp_logs"
DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)

PERF_LOG_FILE = DEFAULT_LOG_DIR / "webapp_perf_log.csv"
BENCHMARK_LOG_FILE = DEFAULT_LOG_DIR / "webapp_benchmark_log.csv"


# =========================================================
# UI STYLE
# =========================================================
def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59,130,246,0.10), transparent 28%),
                radial-gradient(circle at top right, rgba(16,185,129,0.09), transparent 24%),
                linear-gradient(180deg, #09101d 0%, #0f172a 38%, #111827 100%);
            color: #e8eef9;
        }

        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
            max-width: 1380px;
        }

        h1, h2, h3, h4 {
            color: #f8fbff !important;
            letter-spacing: 0.2px;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15,23,42,0.98), rgba(17,24,39,0.98));
            border-right: 1px solid rgba(255,255,255,0.05);
        }

        .top-shell {
            background: linear-gradient(135deg, rgba(37,99,235,0.22), rgba(16,185,129,0.15));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 28px;
            padding: 1.25rem 1.3rem;
            box-shadow: 0 16px 36px rgba(0,0,0,0.24);
            margin-bottom: 1rem;
        }

        .project-eyebrow {
            text-transform: uppercase;
            font-size: 0.8rem;
            color: #bfdbfe;
            letter-spacing: 0.7px;
            margin-bottom: 0.2rem;
            font-weight: 700;
        }

        .project-title {
            font-size: 2.05rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 0.35rem;
            line-height: 1.2;
        }

        .project-subtitle {
            font-size: 1rem;
            color: #dbeafe;
            line-height: 1.65;
            margin-bottom: 0.8rem;
        }

        .chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.5rem;
        }

        .chip {
            border-radius: 999px;
            padding: 0.28rem 0.7rem;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.10);
            color: #f8fafc;
            font-size: 0.82rem;
        }

        .pro-card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 1rem 1rem;
            box-shadow: 0 10px 28px rgba(0,0,0,0.18);
            backdrop-filter: blur(10px);
            margin-bottom: 0.95rem;
        }

        .pro-card-title {
            font-weight: 750;
            font-size: 1.02rem;
            color: #f8fbff;
            margin-bottom: 0.55rem;
        }

        .soft-copy {
            color: #d2dceb;
            line-height: 1.65;
        }

        .info-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
        }

        .info-tile {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 18px;
            padding: 0.8rem 0.85rem;
        }

        .info-label {
            font-size: 0.78rem;
            color: #93c5fd;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: 0.5px;
            margin-bottom: 0.14rem;
        }

        .info-value {
            font-size: 0.96rem;
            color: #ffffff;
            font-weight: 650;
            word-break: break-word;
        }

        .team-list {
            margin: 0;
            padding-left: 1rem;
            color: #ffffff;
            line-height: 1.8;
        }

        .logo-box {
            width: 165px;
            height: 165px;
            margin: auto;
            border-radius: 30px;
            background: linear-gradient(135deg, rgba(37,99,235,0.30), rgba(16,185,129,0.22));
            border: 1px solid rgba(255,255,255,0.10);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            box-shadow: 0 12px 28px rgba(0,0,0,0.22);
        }

        .logo-big {
            color: white;
            font-size: 2.2rem;
            font-weight: 800;
            letter-spacing: 1px;
        }

        .logo-small {
            color: #dbeafe;
            font-size: 0.82rem;
        }

        .section-divider {
            height: 1px;
            background: linear-gradient(90deg, rgba(255,255,255,0), rgba(255,255,255,0.11), rgba(255,255,255,0));
            margin: 0.8rem 0 0.35rem 0;
        }

        .prediction-banner-good {
            background: rgba(16,185,129,0.14);
            border: 1px solid rgba(16,185,129,0.25);
            color: #bbf7d0;
            border-radius: 18px;
            padding: 0.8rem 0.95rem;
            font-weight: 700;
        }

        .prediction-banner-danger {
            background: rgba(239,68,68,0.14);
            border: 1px solid rgba(239,68,68,0.22);
            color: #fecaca;
            border-radius: 18px;
            padding: 0.8rem 0.95rem;
            font-weight: 700;
        }

        .prediction-banner-neutral {
            background: rgba(59,130,246,0.13);
            border: 1px solid rgba(59,130,246,0.22);
            color: #dbeafe;
            border-radius: 18px;
            padding: 0.8rem 0.95rem;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.07);
            border-radius: 18px;
            padding: 0.78rem 0.9rem;
        }

        div[data-testid="stMetricLabel"] {
            color: #cbd5e1;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.45rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.04);
            color: #dbeafe;
            border-radius: 14px 14px 0 0;
            padding: 0.58rem 0.95rem;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(37,99,235,0.18) !important;
            color: white !important;
        }

        code {
            white-space: pre-wrap !important;
            word-break: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def ui_card(title: str, body: str):
    st.markdown(
        f"""
        <div class="pro-card">
            <div class="pro-card-title">{title}</div>
            <div class="soft-copy">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_project_shell(model_count: int):
    st.markdown(
        f"""
        <div class="top-shell">
            <div class="project-eyebrow">{SCHOOL_PARENT} · {SCHOOL_NAME}</div>
            <div class="project-title">{PROJECT_TITLE}</div>
            <div class="project-subtitle">{PROJECT_SUBTITLE}</div>
            <div class="chip-row">
                <span class="chip">QR version 13</span>
                <span class="chip">69 × 69</span>
                <span class="chip">10-fold paper setup</span>
                <span class="chip">{model_count} model files</span>
                <span class="chip">Feature selection supported</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_identity_panel():
    logo_path = resolve_logo_path()
    col1, col2, col3 = st.columns([0.85, 1.3, 1.15], gap="large")

    with col1:
        if logo_path is not None:
            st.image(str(logo_path), use_container_width=True)
            st.caption(f"Logo: {logo_path.name}")
        else:
            st.markdown(
                """
                <div class="logo-box">
                    <div class="logo-big">KMA</div>
                    <div class="logo-small">School logo</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Đặt file logo_kma.png hoặc logo.png cạnh app để hiện logo thật.")

    with col2:
        st.markdown(
            f"""
            <div class="pro-card">
                <div class="pro-card-title">Thông tin đề tài</div>
                <div class="info-grid">
                    <div class="info-tile">
                        <div class="info-label">Đơn vị</div>
                        <div class="info-value">{SCHOOL_NAME}</div>
                    </div>
                    <div class="info-tile">
                        <div class="info-label">Cơ quan</div>
                        <div class="info-value">{SCHOOL_PARENT}</div>
                    </div>
                    <div class="info-tile" style="grid-column: span 2;">
                        <div class="info-label">Chủ đề</div>
                        <div class="info-value">{PROJECT_TITLE}</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        members_html = "".join([f"<li>{m}</li>" for m in TEAM_MEMBERS])
        st.markdown(
            f"""
            <div class="pro-card">
                <div class="pro-card-title">Nhóm thực hiện</div>
                <ul class="team-list">{members_html}</ul>
                <div class="section-divider"></div>
                <div class="info-label">Giảng viên hướng dẫn</div>
                <div class="info-value">{ADVISOR_NAME}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# =========================================================
# APP HELPERS
# =========================================================
def get_safe_qr_payload(qr_type: str = "benign", sample_index: int = 0):
    qr_type = str(qr_type).strip().lower()
    if qr_type in {"benign", "normal", "safe"}:
        pool = SAFE_BENIGN_QR_PAYLOADS
        canonical_type = "benign"
    else:
        pool = SAFE_SIMULATED_MALICIOUS_QR_PAYLOADS
        canonical_type = "malicious"
    sample_index = int(sample_index) % len(pool)
    return canonical_type, pool[sample_index]


def render_qr_to_array_paper(text: str):
    qr = qrcode.QRCode(
        version=QR_VERSION,
        error_correction=QR_ERROR_CORRECTION,
        box_size=QR_BOX_SIZE,
        border=QR_BORDER,
    )
    qr.add_data(text)
    try:
        qr.make(fit=False)
    except Exception as e:
        raise ValueError("Nội dung quá dài hoặc không phù hợp để mã hóa với QR version 13 theo cấu hình paper.") from e

    img = qr.make_image(fill_color="black", back_color="white").convert("L")
    arr = np.asarray(img, dtype=np.uint8)

    if arr.shape != TARGET_SHAPE:
        raise ValueError(f"QR tạo ra có shape {arr.shape}, không khớp shape kỳ vọng {TARGET_SHAPE}.")
    return img, arr


def get_positive_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        return scores
    return model.predict(X)


def load_model_bundle_uncached(path: str):
    bundle = joblib.load(path)
    if isinstance(bundle, dict) and "model" in bundle:
        return bundle
    return {"model": bundle, "metadata": {}}


@st.cache_resource(show_spinner=False)
def load_model_bundle_cached(path: str):
    return load_model_bundle_uncached(path)


@st.cache_data(show_spinner=False)
def load_selected_idx(idx_path: str):
    return np.load(idx_path)


def get_process_stats():
    if psutil is None:
        return {"cpu_percent": None, "ram_mb": None}
    proc = psutil.Process(os.getpid())
    return {
        "cpu_percent": round(psutil.cpu_percent(interval=0.05), 2),
        "ram_mb": round(proc.memory_info().rss / 1024 / 1024, 2),
    }


def append_csv_row(file_path: Path, row: dict):
    file_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not file_path.exists()
    with open(file_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def log_perf(stage: str, elapsed_s: float, extra: dict | None = None):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "stage": stage,
        "elapsed_ms": round(elapsed_s * 1000.0, 3),
    }
    row.update(get_process_stats())
    if extra:
        row.update(extra)
    append_csv_row(PERF_LOG_FILE, row)


def try_read_csv(path: Path):
    if not path.exists():
        return None
    if pd is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def resolve_results_dir_from_model_path(model_path: str) -> Path:
    p = Path(model_path).resolve()
    for parent in [p] + list(p.parents):
        if parent.name == "models_new":
            return parent.parent / "results"
    return DEFAULT_OUTPUT_DIR / "results"


def resolve_selected_idx_path(model_path: str, metadata: dict) -> Path | None:
    selector_name = metadata.get("selector_name")
    stage = str(metadata.get("stage", "")).strip().lower()
    if stage != "feature_selection" and not selector_name:
        return None
    if not selector_name:
        return None
    results_dir = resolve_results_dir_from_model_path(model_path)
    idx_path = results_dir / f"selected_idx_{selector_name}.npy"
    return idx_path if idx_path.exists() else None


def build_input_vector(arr: np.ndarray, selected_idx: np.ndarray | None = None):
    X_input = arr.astype(np.float32).reshape(1, -1)
    original_dim = X_input.shape[1]
    if selected_idx is not None:
        X_input = X_input[:, selected_idx]
    return X_input, original_dim, X_input.shape[1]


def run_single_prediction(model, text: str, qr_source: str, model_path: str, selected_idx: np.ndarray | None, metadata: dict):
    t_total_0 = time.perf_counter()

    t0 = time.perf_counter()
    img, arr = render_qr_to_array_paper(text=text)
    t1 = time.perf_counter()
    qr_generation_s = t1 - t0

    t0 = time.perf_counter()
    X_input, original_dim, used_dim = build_input_vector(arr=arr, selected_idx=selected_idx)
    t1 = time.perf_counter()
    preprocess_s = t1 - t0

    t0 = time.perf_counter()
    pred_label = int(model.predict(X_input)[0])
    score_1 = float(get_positive_scores(model, X_input)[0])
    t1 = time.perf_counter()
    inference_s = t1 - t0

    total_s = time.perf_counter() - t_total_0

    result = {
        "predicted_label": pred_label,
        "predicted_label_name": LABEL_NAME_MAP.get(pred_label, f"class_{pred_label}"),
        "prob_class_1": round(score_1, 6),
        "selected_model": model_path,
        "qr_source": qr_source,
        "input_shape_used": f"{X_input.shape[0]}x{X_input.shape[1]}",
        "original_feature_dim": int(original_dim),
        "used_feature_dim": int(used_dim),
        "model_stage": metadata.get("stage"),
        "selector_name": metadata.get("selector_name"),
        "top_k": metadata.get("top_k"),
        "timing_ms": {
            "qr_generation_ms": round(qr_generation_s * 1000.0, 3),
            "preprocess_ms": round(preprocess_s * 1000.0, 3),
            "inference_ms": round(inference_s * 1000.0, 3),
            "end_to_end_ms": round(total_s * 1000.0, 3),
        },
    }

    log_perf(
        stage="predict",
        elapsed_s=total_s,
        extra={
            "selected_model": model_path,
            "qr_source": qr_source,
            "predicted_label": pred_label,
            "predicted_label_name": result["predicted_label_name"],
            "prob_class_1": result["prob_class_1"],
            "model_stage": metadata.get("stage"),
            "selector_name": metadata.get("selector_name"),
            "top_k": metadata.get("top_k"),
            "original_feature_dim": original_dim,
            "used_feature_dim": used_dim,
            "qr_generation_ms": result["timing_ms"]["qr_generation_ms"],
            "preprocess_ms": result["timing_ms"]["preprocess_ms"],
            "inference_ms": result["timing_ms"]["inference_ms"],
            "end_to_end_ms": result["timing_ms"]["end_to_end_ms"],
        },
    )
    return img, arr, result


def summarize_latency_ms(values):
    values = [float(v) for v in values if v is not None]
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        "n_runs": int(arr.size),
        "mean_ms": round(float(arr.mean()), 4),
        "std_ms": round(float(arr.std()), 4),
        "min_ms": round(float(arr.min()), 4),
        "max_ms": round(float(arr.max()), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
    }


def run_benchmark(model, text: str, qr_source: str, model_path: str, selected_idx: np.ndarray | None, metadata: dict, n_runs: int = 100):
    total_ms = []
    infer_ms = []
    prep_ms = []
    gen_ms = []
    predictions = []

    for _ in range(int(n_runs)):
        t_total_0 = time.perf_counter()

        t0 = time.perf_counter()
        _, arr = render_qr_to_array_paper(text=text)
        t1 = time.perf_counter()
        gen_ms.append((t1 - t0) * 1000.0)

        t0 = time.perf_counter()
        X_input, _, _ = build_input_vector(arr=arr, selected_idx=selected_idx)
        t1 = time.perf_counter()
        prep_ms.append((t1 - t0) * 1000.0)

        t0 = time.perf_counter()
        pred_label = int(model.predict(X_input)[0])
        _ = float(get_positive_scores(model, X_input)[0])
        t1 = time.perf_counter()
        infer_ms.append((t1 - t0) * 1000.0)
        predictions.append(pred_label)

        total_ms.append((time.perf_counter() - t_total_0) * 1000.0)

    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "selected_model": model_path,
        "qr_source": qr_source,
        "n_runs": int(n_runs),
        "majority_prediction": int(round(statistics.mean(predictions))) if predictions else None,
        "model_stage": metadata.get("stage"),
        "selector_name": metadata.get("selector_name"),
        "top_k": metadata.get("top_k"),
        **{f"total_{k}": v for k, v in summarize_latency_ms(total_ms).items()},
        **{f"infer_{k}": v for k, v in summarize_latency_ms(infer_ms).items()},
        **{f"preprocess_{k}": v for k, v in summarize_latency_ms(prep_ms).items()},
        **{f"generate_{k}": v for k, v in summarize_latency_ms(gen_ms).items()},
    }
    summary.update(get_process_stats())
    append_csv_row(BENCHMARK_LOG_FILE, summary)
    return summary


def render_prediction_banner(result: dict):
    label = str(result.get("predicted_label_name", "")).lower()
    score = result.get("prob_class_1", "-")
    if label == "malicious":
        cls = "prediction-banner-danger"
        text = f"Kết quả dự đoán: MALICIOUS · Mức nghi ngờ class 1 = {score}"
    elif label == "benign":
        cls = "prediction-banner-good"
        text = f"Kết quả dự đoán: BENIGN · Mức nghi ngờ class 1 = {score}"
    else:
        cls = "prediction-banner-neutral"
        text = f"Kết quả dự đoán: {label.upper()} · Score class 1 = {score}"

    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def render_registry_card(metadata: dict, model_path: str, selected_idx_path: Path | None, use_cache: bool, load_elapsed_s: float):
    stage_text = str(metadata.get("stage", "unknown"))
    if stage_text == "feature_selection":
        stage_label = "Feature Selection"
    elif stage_text == "cv10_baseline":
        stage_label = "CV10 Baseline"
    else:
        stage_label = stage_text

    st.markdown(
        f"""
        <div class="pro-card">
            <div class="pro-card-title">Model Registry</div>
            <div class="info-grid">
                <div class="info-tile">
                    <div class="info-label">Stage</div>
                    <div class="info-value">{stage_label}</div>
                </div>
                <div class="info-tile">
                    <div class="info-label">Model name</div>
                    <div class="info-value">{metadata.get("model_name", "-")}</div>
                </div>
                <div class="info-tile">
                    <div class="info-label">Fold</div>
                    <div class="info-value">{metadata.get("fold", "-")}</div>
                </div>
                <div class="info-tile">
                    <div class="info-label">Load time</div>
                    <div class="info-value">{round(load_elapsed_s * 1000.0, 3)} ms</div>
                </div>
                <div class="info-tile">
                    <div class="info-label">Selector</div>
                    <div class="info-value">{metadata.get("selector_name", "-")}</div>
                </div>
                <div class="info-tile">
                    <div class="info-label">Top-k</div>
                    <div class="info-value">{metadata.get("top_k", "-")}</div>
                </div>
                <div class="info-tile" style="grid-column: span 2;">
                    <div class="info-label">Model path</div>
                    <div class="info-value">{model_path}</div>
                </div>
                <div class="info-tile" style="grid-column: span 2;">
                    <div class="info-label">Selected idx path</div>
                    <div class="info-value">{str(selected_idx_path) if selected_idx_path else "-"}</div>
                </div>
                <div class="info-tile">
                    <div class="info-label">Cache mode</div>
                    <div class="info-value">{use_cache}</div>
                </div>
                <div class="info-tile">
                    <div class="info-label">Output dir</div>
                    <div class="info-value">{DEFAULT_OUTPUT_DIR}</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# APP START
# =========================================================
st.set_page_config(page_title="QR Quishing Professional App", layout="wide")
inject_css()

model_files = sorted(DEFAULT_OUTPUT_DIR.glob("models_new/**/*.joblib"))
model_count = len(model_files)

render_project_shell(model_count=model_count)
render_identity_panel()

if not model_files:
    st.warning(f"Chưa tìm thấy model .joblib trong {DEFAULT_OUTPUT_DIR / 'models_new'}. Hãy chạy notebook train mới trước.")
    st.stop()

with st.sidebar:
    st.header("Cấu hình hệ thống")
    st.caption("Phiên bản giao diện đã được thiết kế lại theo hướng chuyên nghiệp hơn.")
    st.write({"output_dir": str(DEFAULT_OUTPUT_DIR.resolve())})
    use_cache = st.checkbox("Dùng cache khi load model", value=True)
    benchmark_runs = st.slider("Số lần benchmark", min_value=10, max_value=500, value=100, step=10)
    show_logs = st.checkbox("Hiện log gần nhất", value=True)
    st.markdown("---")
    st.markdown("**QR chuẩn paper**")
    st.write({
        "version": QR_VERSION,
        "shape": TARGET_SHAPE,
        "error_correction": "L",
        "border": QR_BORDER,
        "box_size": QR_BOX_SIZE,
    })

model_path = st.selectbox("Chọn model đã train", options=[str(p) for p in model_files])

load_t0 = time.perf_counter()
bundle = load_model_bundle_cached(model_path) if use_cache else load_model_bundle_uncached(model_path)
load_elapsed_s = time.perf_counter() - load_t0

model = bundle["model"]
metadata = bundle.get("metadata", {}) or {}
selected_idx_path = resolve_selected_idx_path(model_path=model_path, metadata=metadata)
selected_idx = load_selected_idx(str(selected_idx_path)) if selected_idx_path is not None else None

log_perf(
    "load_model",
    load_elapsed_s,
    {
        "selected_model": model_path,
        "use_cache": use_cache,
        "model_stage": metadata.get("stage"),
        "selector_name": metadata.get("selector_name"),
        "top_k": metadata.get("top_k"),
        "selected_idx_path": str(selected_idx_path) if selected_idx_path else None,
    },
)

render_registry_card(
    metadata=metadata,
    model_path=model_path,
    selected_idx_path=selected_idx_path,
    use_cache=use_cache,
    load_elapsed_s=load_elapsed_s,
)

top_left, top_right = st.columns([1.1, 1], gap="large")

with top_left:
    ui_card(
        "Phòng thí nghiệm đầu vào",
        "Chọn QR synthetic hoặc nhập nội dung thủ công. Ứng dụng sẽ sinh QR đúng chuẩn paper rồi đưa vào model đã train."
    )
    input_mode = st.radio("Nguồn QR", ["Synthetic", "Text nhập tay"], horizontal=True)
    is_synthetic_mode = input_mode == "Synthetic"

    c1, c2 = st.columns(2)
    with c1:
        qr_type = st.selectbox("Loại QR synthetic", ["benign", "malicious"], disabled=not is_synthetic_mode)
    with c2:
        sample_index = st.number_input("Sample index", min_value=0, max_value=20, value=0, step=1, disabled=not is_synthetic_mode)

    custom_text = st.text_area(
        "Nội dung QR thủ công",
        value="https://example.com",
        disabled=is_synthetic_mode,
        height=130,
        help="Vì app dùng version 13 cố định, nội dung quá dài có thể không tạo được QR.",
    )

    if is_synthetic_mode:
        st.info("Chế độ đang hoạt động: Synthetic demo.")
    else:
        st.info("Chế độ đang hoạt động: Text nhập tay.")

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        run_predict = st.button("Phân tích QR", type="primary", use_container_width=True)
    with action_col2:
        run_bench = st.button("Chạy benchmark", use_container_width=True)

with top_right:
    ui_card(
        "Mục tiêu của app",
        "Đây là bản demo trình diễn khả năng phát hiện quishing trực tiếp từ cấu trúc ảnh QR. "
        "App đồng thời hỗ trợ đo hiệu năng để phục vụ báo cáo và demo trước hội đồng."
    )

    meta_a, meta_b, meta_c, meta_d = st.columns(4)
    meta_a.metric("Model files", model_count)
    meta_b.metric("Load model", f"{round(load_elapsed_s * 1000.0, 3)} ms")
    meta_c.metric("Top-k", metadata.get("top_k", "-"))
    meta_d.metric("Stage", str(metadata.get("stage", "-")))

if input_mode == "Synthetic":
    qr_type, text = get_safe_qr_payload(qr_type=qr_type, sample_index=sample_index)
else:
    text = custom_text
    qr_type = "custom"

tab_predict, tab_bench, tab_logs, tab_about = st.tabs(
    ["Dự đoán trực tiếp", "Phân tích hiệu năng", "Nhật ký hệ thống", "Tổng quan dự án"]
)

with tab_predict:
    left, right = st.columns([0.9, 1.1], gap="large")
    with left:
        ui_card(
            "QR Preview",
            "Vùng này hiển thị QR được sinh ra từ payload đầu vào và tóm tắt ma trận ảnh trước khi đưa vào mô hình."
        )

    with right:
        ui_card(
            "Prediction Output",
            "Kết quả hiển thị gồm nhãn dự đoán, score class 1, kích thước feature thực dùng và thời gian xử lý từng bước."
        )

    if run_predict:
        try:
            img, arr, result = run_single_prediction(
                model=model,
                text=text,
                qr_source=qr_type,
                model_path=model_path,
                selected_idx=selected_idx,
                metadata=metadata,
            )
            perf_stats = get_process_stats()

            left, right = st.columns([0.9, 1.1], gap="large")
            with left:
                st.image(img, caption=f"QR rendered | source={qr_type}", use_container_width=False)
                st.write("**Payload**")
                st.code(text)
                st.write("**QR matrix summary**")
                st.write(
                    {
                        "shape": arr.shape,
                        "pixel_min": int(arr.min()),
                        "pixel_max": int(arr.max()),
                        "pixel_mean": round(float(arr.mean()), 4),
                    }
                )

            with right:
                render_prediction_banner(result)
                st.write("")
                st.json(result)

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Load model (ms)", f"{round(load_elapsed_s * 1000.0, 3)}")
                m2.metric("QR gen (ms)", f"{result['timing_ms']['qr_generation_ms']}")
                m3.metric("Preprocess (ms)", f"{result['timing_ms']['preprocess_ms']}")
                m4.metric("Inference (ms)", f"{result['timing_ms']['inference_ms']}")

                m5, m6, m7 = st.columns(3)
                m5.metric("End-to-end (ms)", f"{result['timing_ms']['end_to_end_ms']}")
                m6.metric("RAM (MB)", "-" if perf_stats["ram_mb"] is None else str(perf_stats["ram_mb"]))
                m7.metric("CPU (%)", "-" if perf_stats["cpu_percent"] is None else str(perf_stats["cpu_percent"]))
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Bấm **Phân tích QR** để chạy dự đoán trên payload hiện tại.")

with tab_bench:
    ui_card(
        "Benchmark Lab",
        "Benchmark đo riêng thời gian tạo QR, tiền xử lý, suy luận mô hình và tổng end-to-end latency. "
        "Các chỉ số p95, p99 đặc biệt hữu ích khi trình bày hiệu năng ứng dụng."
    )

    if run_bench:
        try:
            with st.spinner("Đang benchmark..."):
                summary = run_benchmark(
                    model=model,
                    text=text,
                    qr_source=qr_type,
                    model_path=model_path,
                    selected_idx=selected_idx,
                    metadata=metadata,
                    n_runs=benchmark_runs,
                )

            st.json(summary)

            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Mean total (ms)", f"{summary.get('total_mean_ms', '-')}")
            b2.metric("P95 total (ms)", f"{summary.get('total_p95_ms', '-')}")
            b3.metric("P99 total (ms)", f"{summary.get('total_p99_ms', '-')}")
            b4.metric("Max total (ms)", f"{summary.get('total_max_ms', '-')}")

            c1, c2, c3 = st.columns(3)
            c1.metric("Mean infer (ms)", f"{summary.get('infer_mean_ms', '-')}")
            c2.metric("Mean preprocess (ms)", f"{summary.get('preprocess_mean_ms', '-')}")
            c3.metric("Mean generate (ms)", f"{summary.get('generate_mean_ms', '-')}")
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Bấm **Chạy benchmark** để tạo báo cáo độ trễ.")

    if pd is not None:
        bench_df = try_read_csv(BENCHMARK_LOG_FILE)
        if bench_df is not None and not bench_df.empty:
            st.markdown("#### Lịch sử benchmark gần đây")
            st.dataframe(bench_df.tail(10), use_container_width=True)

            if "total_mean_ms" in bench_df.columns:
                chart_df = bench_df.tail(20).copy()
                chart_df["run_id"] = range(1, len(chart_df) + 1)
                st.line_chart(chart_df.set_index("run_id")[["total_mean_ms", "infer_mean_ms", "generate_mean_ms"]])

with tab_logs:
    if show_logs:
        perf_df = try_read_csv(PERF_LOG_FILE)
        bench_df = try_read_csv(BENCHMARK_LOG_FILE)

        log_tab1, log_tab2 = st.tabs(["Prediction log", "Benchmark log"])
        with log_tab1:
            if perf_df is not None and not perf_df.empty:
                st.dataframe(perf_df.tail(25), use_container_width=True)
            else:
                st.info("Chưa có prediction log.")
        with log_tab2:
            if bench_df is not None and not bench_df.empty:
                st.dataframe(bench_df.tail(25), use_container_width=True)
            else:
                st.info("Chưa có benchmark log.")
    else:
        st.info("Bật tùy chọn **Hiện log gần nhất** ở sidebar để xem nhật ký.")

with tab_about:
    about_left, about_right = st.columns([1, 1], gap="large")
    with about_left:
        ui_card(
            "Những gì app đã đồng bộ với notebook mới",
            "- Dùng đúng output directory `outputs_quishing_paper_10fold`<br>"
            "- Render QR đúng cấu hình paper: version 13, 69×69, error correction low, border 0, box_size 1<br>"
            "- Dùng trực tiếp vector pixel QR để predict<br>"
            "- Tự phát hiện model thuộc nhánh feature selection và nạp `selected_idx_*.npy`"
        )

    with about_right:
        ui_card(
            "Cách dùng hiệu quả khi demo",
            "- Ưu tiên chọn model tốt nhất trong registry<br>"
            "- Dùng Synthetic để trình diễn ổn định<br>"
            "- Dùng Benchmark để lấy mean, p95, p99 cho báo cáo<br>"
            "- Đặt logo thật bằng file `logo_kma.png` hoặc `logo.png` cạnh app"
        )

    st.markdown("### Thành phần giao diện")
    st.markdown(
        "- Khu vực nhận diện đề tài ở đầu trang<br>"
        "- Panel thông tin trường, nhóm thực hiện, giảng viên hướng dẫn<br>"
        "- Model Registry tách riêng<br>"
        "- Khu vực thao tác chính chia rõ: input / prediction / benchmark / logs / project overview",
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption("Phiên bản này được thiết kế lại toàn bộ theo hướng chuyên nghiệp, phục vụ tốt cho demo và báo cáo.")
