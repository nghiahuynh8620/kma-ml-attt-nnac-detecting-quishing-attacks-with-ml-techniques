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
    import psutil
except Exception:
    psutil = None


LABEL_NAME_MAP = {0: "benign", 1: "malicious"}

SCHOOL_NAME = "Học viện Kỹ thuật Mật mã"
SCHOOL_PARENT = "Ban Cơ yếu Chính phủ"
PROJECT_TITLE = "Phát hiện tấn công Quishing bằng Học máy"
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
    Path("./outputs"),
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
]

TARGET_SHAPE = (69, 69)
QR_VERSION = 13
QR_ERROR_CORRECTION = qrcode.constants.ERROR_CORRECT_L
QR_BOX_SIZE = 1
QR_BORDER = 0


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


def inject_css():
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(59,130,246,0.12), transparent 28%),
                radial-gradient(circle at top right, rgba(16,185,129,0.10), transparent 24%),
                linear-gradient(180deg, #0b1220 0%, #111827 35%, #0f172a 100%);
            color: #e5eefc;
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 1.5rem;
            max-width: 1320px;
        }
        h1, h2, h3, h4 {
            color: #f8fbff !important;
            letter-spacing: 0.2px;
        }
        .hero {
            background: linear-gradient(135deg, rgba(37,99,235,0.24), rgba(16,185,129,0.18));
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 24px;
            padding: 1.25rem 1.35rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.22);
            margin-bottom: 1rem;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 800;
            margin-bottom: 0.25rem;
        }
        .hero-subtitle {
            color: #dbeafe;
            font-size: 1rem;
            line-height: 1.55;
            margin-bottom: 0.75rem;
        }
        .badge-row {
            display: flex;
            gap: 0.55rem;
            flex-wrap: wrap;
            margin-top: 0.5rem;
        }
        .badge {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.10);
            color: #f8fafc;
            border-radius: 999px;
            padding: 0.28rem 0.72rem;
            font-size: 0.84rem;
        }
        .card {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 1rem 1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
            margin-bottom: 0.9rem;
            backdrop-filter: blur(10px);
        }
        .card-title {
            font-weight: 700;
            font-size: 1.03rem;
            margin-bottom: 0.55rem;
            color: #f8fbff;
        }
        .soft-text {
            color: #d1d9e6;
            line-height: 1.6;
        }
        .mini-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.7rem;
        }
        .mini-item {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 0.7rem 0.8rem;
        }
        .mini-label {
            font-size: 0.8rem;
            color: #bfdbfe;
            margin-bottom: 0.15rem;
        }
        .mini-value {
            font-size: 0.95rem;
            color: #ffffff;
            font-weight: 650;
            word-break: break-word;
        }
        .status-good {
            color: #86efac;
            font-weight: 700;
        }
        .status-warn {
            color: #fde68a;
            font-weight: 700;
        }
        .info-panel {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 22px;
            padding: 1rem 1rem;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18);
            min-height: 100%;
        }
        .school-parent {
            color: #bfdbfe;
            font-size: 0.88rem;
            letter-spacing: 0.4px;
            text-transform: uppercase;
            margin-bottom: 0.2rem;
        }
        .school-name {
            font-size: 1.5rem;
            font-weight: 800;
            color: #ffffff;
            margin-bottom: 0.25rem;
        }
        .project-name {
            color: #dbeafe;
            font-size: 1rem;
            line-height: 1.55;
        }
        .section-label {
            font-size: 0.82rem;
            color: #93c5fd;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 0.45rem;
            font-weight: 700;
        }
        .person-list {
            margin: 0;
            padding-left: 1rem;
            color: #f8fafc;
            line-height: 1.75;
        }
        .advisor-name {
            color: #ffffff;
            font-weight: 700;
            font-size: 1rem;
        }
        .logo-fallback {
            width: 150px;
            height: 150px;
            border-radius: 28px;
            background: linear-gradient(135deg, rgba(37,99,235,0.32), rgba(16,185,129,0.24));
            border: 1px solid rgba(255,255,255,0.10);
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            color: white;
            box-shadow: 0 10px 30px rgba(0,0,0,0.22);
            margin: auto;
        }
        .logo-fallback-big {
            font-size: 2rem;
            font-weight: 800;
            letter-spacing: 1px;
        }
        .logo-fallback-small {
            font-size: 0.8rem;
            color: #dbeafe;
            margin-top: 0.15rem;
        }
        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.05);
            border: 1px solid rgba(255,255,255,0.07);
            padding: 0.75rem 0.9rem;
            border-radius: 18px;
        }
        div[data-testid="stMetricLabel"] {
            color: #cbd5e1;
        }
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15,23,42,0.96), rgba(17,24,39,0.98));
            border-right: 1px solid rgba(255,255,255,0.05);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.4rem;
        }
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.04);
            border-radius: 14px 14px 0 0;
            color: #dbeafe;
            padding: 0.55rem 0.95rem;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(37,99,235,0.20) !important;
            color: #ffffff !important;
        }
        code {
            white-space: pre-wrap !important;
            word-break: break-word;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def card(title: str, body: str):
    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <div class="soft-text">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_project_header():
    logo_path = resolve_logo_path()

    col1, col2, col3 = st.columns([0.75, 1.5, 1.25], gap="large")

    with col1:
        if logo_path is not None:
            st.image(str(logo_path), use_container_width=True)
            st.caption(f"Logo: {logo_path.name}")
        else:
            st.markdown(
                """
                <div class="logo-fallback">
                    <div class="logo-fallback-big">KMA</div>
                    <div class="logo-fallback-small">School logo</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Đặt file logo vào cùng thư mục app với tên: logo_kma.png hoặc logo.png")

    with col2:
        st.markdown(
            f"""
            <div class="info-panel">
                <div class="school-parent">{SCHOOL_PARENT}</div>
                <div class="school-name">{SCHOOL_NAME}</div>
                <div class="project-name">{PROJECT_TITLE}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col3:
        members_html = "".join([f"<li>{m}</li>" for m in TEAM_MEMBERS])
        st.markdown(
            f"""
            <div class="info-panel">
                <div class="section-label">Nhóm thực hiện</div>
                <ul class="person-list">{members_html}</ul>
                <div style="height: 0.8rem;"></div>
                <div class="section-label">Giảng viên hướng dẫn</div>
                <div class="advisor-name">{ADVISOR_NAME}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


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
    try:
        import pandas as pd
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


st.set_page_config(page_title="QR Quishing Demo - Paper UI", layout="wide")
inject_css()

model_files = sorted(DEFAULT_OUTPUT_DIR.glob("models_new/**/*.joblib"))
model_count = len(model_files)

st.markdown(
    f"""
    <div class="hero">
        <div class="hero-title">QR Quishing Demo</div>
        <div class="hero-subtitle">
            Giao diện đẹp theo kiểu dashboard của app cũ, nhưng phần xử lý đã khớp notebook train mới.
            App tự hỗ trợ model baseline lẫn feature selection.
        </div>
        <div class="badge-row">
            <span class="badge">QR version 13</span>
            <span class="badge">69×69</span>
            <span class="badge">10-fold paper setup</span>
            <span class="badge">{model_count} model files</span>
            <span class="badge">Feature selection supported</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

render_project_header()

if not model_files:
    st.warning(f"Chưa tìm thấy model .joblib trong {DEFAULT_OUTPUT_DIR / 'models_new'}. Hãy chạy notebook train mới trước.")
    st.stop()

with st.sidebar:
    st.header("Cấu hình hệ thống")
    st.caption("Đã đồng bộ với notebook paper-only 10-fold.")
    st.write({"output_dir": str(DEFAULT_OUTPUT_DIR.resolve())})
    use_cache = st.checkbox("Dùng cache khi load model", value=True)
    benchmark_runs = st.slider("Số lần benchmark", min_value=10, max_value=500, value=100, step=10)
    show_logs = st.checkbox("Hiện log gần nhất", value=True)

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

top_col_1, top_col_2 = st.columns([1.15, 1])

with top_col_1:
    st.markdown('<div class="card-title">Thiết lập đầu vào</div>', unsafe_allow_html=True)
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
        help="App sẽ tạo QR theo đúng version 13 cố định, nên nội dung quá dài có thể không encode được.",
        height=120,
    )

    st.info("Đang dùng QR synthetic." if is_synthetic_mode else "Đang dùng nội dung nhập tay.")

    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        run_predict = st.button("Predict QR", type="primary", use_container_width=True)
    with btn_col2:
        run_bench = st.button("Benchmark hiệu năng", use_container_width=True)

with top_col_2:
    stage_text = str(metadata.get("stage", "unknown"))
    status = '<span class="status-good">Feature Selection</span>' if stage_text == "feature_selection" else (
        '<span class="status-good">CV10 Baseline</span>' if stage_text == "cv10_baseline" else f'<span class="status-warn">{stage_text}</span>'
    )

    st.markdown(
        f"""
        <div class="card">
            <div class="card-title">Thông tin model</div>
            <div class="mini-grid">
                <div class="mini-item"><div class="mini-label">Stage</div><div class="mini-value">{status}</div></div>
                <div class="mini-item"><div class="mini-label">Model</div><div class="mini-value">{metadata.get("model_name", "-")}</div></div>
                <div class="mini-item"><div class="mini-label">Fold</div><div class="mini-value">{metadata.get("fold", "-")}</div></div>
                <div class="mini-item"><div class="mini-label">Load model</div><div class="mini-value">{round(load_elapsed_s * 1000.0, 3)} ms</div></div>
                <div class="mini-item"><div class="mini-label">Selector</div><div class="mini-value">{metadata.get("selector_name", "-")}</div></div>
                <div class="mini-item"><div class="mini-label">Top-k</div><div class="mini-value">{metadata.get("top_k", "-")}</div></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.json({
        "selected_model": model_path,
        "selected_idx_path": str(selected_idx_path) if selected_idx_path else None,
        "cache_enabled": use_cache,
        "output_dir": str(DEFAULT_OUTPUT_DIR),
    })

if input_mode == "Synthetic":
    qr_type, text = get_safe_qr_payload(qr_type=qr_type, sample_index=sample_index)
else:
    text = custom_text
    qr_type = "custom"

tab_predict, tab_bench, tab_logs, tab_guide = st.tabs(["Dự đoán", "Benchmark", "Log hệ thống", "Hướng dẫn"])

with tab_predict:
    card("Mô tả", "Tab này sinh QR đúng chuẩn paper rồi đưa vào model đã train. Nếu model thuộc nhánh feature selection, app sẽ tự nạp selected_idx để cắt feature trước khi dự đoán.")
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

            c1, c2 = st.columns([1, 1.2])
            with c1:
                st.markdown('<div class="card-title">QR preview</div>', unsafe_allow_html=True)
                st.image(img, caption=f"QR rendered | source={qr_type}", use_container_width=False)
                st.write("**Payload**")
                st.code(text)
                st.write("**QR matrix summary**")
                st.write({"shape": arr.shape, "pixel_min": int(arr.min()), "pixel_max": int(arr.max()), "pixel_mean": round(float(arr.mean()), 4)})
            with c2:
                st.markdown('<div class="card-title">Kết quả dự đoán</div>', unsafe_allow_html=True)
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
        st.info("Chọn nguồn QR và bấm Predict QR để xem kết quả.")

with tab_bench:
    card("Benchmark hiệu năng", "Đo generate QR, preprocess, inference và end-to-end latency. Khi báo cáo, nên theo dõi mean, p95, p99.")
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
        except Exception as e:
            st.error(str(e))
    else:
        st.info("Bấm Benchmark hiệu năng để chạy đo độ trễ.")

with tab_logs:
    if show_logs:
        perf_df = try_read_csv(PERF_LOG_FILE)
        bench_df = try_read_csv(BENCHMARK_LOG_FILE)
        t1, t2 = st.tabs(["Prediction log", "Benchmark log"])
        with t1:
            st.dataframe(perf_df.tail(20), use_container_width=True) if perf_df is not None and not perf_df.empty else st.info("Chưa có prediction log.")
        with t2:
            st.dataframe(bench_df.tail(20), use_container_width=True) if bench_df is not None and not bench_df.empty else st.info("Chưa có benchmark log.")
    else:
        st.info("Bật tùy chọn Hiện log gần nhất ở sidebar để xem log.")

with tab_guide:
    st.markdown("### Những gì đã được đồng bộ với notebook mới")
    st.markdown(
        "- Dùng đúng output directory: `outputs_quishing_paper_10fold`\n"
        "- Render QR đúng cấu hình paper: version 13, 69×69, error correction low, border 0, box_size 1\n"
        "- Dùng trực tiếp vector pixel QR để predict\n"
        "- Tự phát hiện model thuộc nhánh `feature_selection` và nạp đúng `selected_idx_*.npy`\n"
        "- Có thêm khu vực trường, nhóm thực hiện và giảng viên hướng dẫn ở đầu app"
    )
    st.markdown("### Gợi ý thêm logo")
    st.markdown(
        "- Đặt file logo vào cùng thư mục app với tên `logo_kma.png` hoặc `logo.png`\n"
        "- Nếu chưa có file logo, app sẽ hiện khối chữ `KMA` thay thế"
    )

st.markdown("---")
st.caption("Bản này đã bổ sung khu vực logo trường, nhóm thực hiện và người hướng dẫn.")
