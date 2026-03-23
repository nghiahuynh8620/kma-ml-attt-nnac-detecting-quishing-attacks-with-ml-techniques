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
except Exception:  # pragma: no cover
    psutil = None

LABEL_NAME_MAP = {0: "benign", 1: "malicious"}
SAFE_BENIGN_QR_PAYLOADS = [
    "https://example.com/news/academic-update-2026",
    "WIFI:T:WPA;S:CampusGuest;P:SafePassword123;;",
    "BEGIN:VCARD\nFN:Lab Reception\nTEL:+84000000000\nEND:VCARD",
    "https://example.org/library/schedule?week=12",
]
SAFE_SIMULATED_MALICIOUS_QR_PAYLOADS = [
    "http://198.51.100.24/verify/session?id=8842",
    "https://secure-login.example.invalid/update?token=reset-now",
    "http://203.0.113.7/billing/review?step=confirm",
    "https://account-check.example.invalid/open?next=signin",
]
DEFAULT_OUTPUT_DIR = Path("outputs")
DEFAULT_LOG_DIR = DEFAULT_OUTPUT_DIR / "webapp_logs"
DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
PERF_LOG_FILE = DEFAULT_LOG_DIR / "webapp_perf_log.csv"
BENCHMARK_LOG_FILE = DEFAULT_LOG_DIR / "webapp_benchmark_log.csv"
TARGET_SHAPE = (69, 69)

TEAM_MEMBERS = [
    ("Vũ Thị Diệu Anh", "CHAT4P001"),
    ("Diệp Kim Chi", "CHAT4P003"),
    ("Huỳnh Trọng Nghĩa", "CHAT4P011"),
    ("Võ Minh Nhật", "CHAT4P013"),
]

INTRO_MD = """
### Tổng quan
Ứng dụng này là **web demo phát hiện QR quishing bằng học máy**. Hệ thống nhận nội dung QR, sinh ảnh QR, tiền xử lý về đúng kích thước đầu vào mô hình và dự đoán nhãn **benign** hoặc **malicious**.

### Chức năng chính
- Dự đoán QR từ **Synthetic** hoặc **Text nhập tay**.
- Đo **benchmark hiệu năng**: QR generation, preprocessing, inference, end-to-end.
- Ghi **prediction log** và **benchmark log** để phục vụ báo cáo.
- Hiển thị thông tin model, metadata và tài nguyên sử dụng.

### Luồng xử lý
1. Chọn model đã train.
2. Chọn nguồn QR và nội dung đầu vào.
3. Render QR thành ảnh xám.
4. Resize về kích thước chuẩn của mô hình.
5. Dự đoán nhãn và score.
6. Ghi log, hiển thị kết quả và chỉ số hiệu năng.
"""


def inject_custom_css():
    st.markdown(
        """
        <style>
            .block-container {
                padding-top: 1.25rem;
                padding-bottom: 2rem;
                max-width: 1300px;
            }
            .main-title {
                font-size: 2.2rem;
                font-weight: 800;
                line-height: 1.15;
                margin-bottom: 0.3rem;
            }
            .subtitle {
                font-size: 1rem;
                color: #b6bfd0;
                margin-bottom: 0.25rem;
            }
            .hero-box {
                padding: 1.2rem 1.25rem;
                border-radius: 22px;
                background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.98));
                border: 1px solid rgba(148,163,184,0.18);
                box-shadow: 0 10px 30px rgba(2,6,23,0.25);
                margin-bottom: 1rem;
            }
            .pill-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.5rem;
                margin-top: 0.8rem;
            }
            .pill {
                display: inline-block;
                padding: 0.38rem 0.8rem;
                border-radius: 999px;
                background: rgba(59,130,246,0.12);
                border: 1px solid rgba(96,165,250,0.26);
                font-size: 0.9rem;
            }
            .section-card {
                padding: 1rem 1.1rem;
                border-radius: 20px;
                background: rgba(15,23,42,0.5);
                border: 1px solid rgba(148,163,184,0.14);
                margin-bottom: 0.85rem;
            }
            .section-card h4 {
                margin-top: 0;
                margin-bottom: 0.45rem;
            }
            .mini-muted {
                color: #94a3b8;
                font-size: 0.92rem;
            }
            div[data-testid="stMetric"] {
                background: rgba(15,23,42,0.55);
                border: 1px solid rgba(148,163,184,0.14);
                padding: 0.8rem 0.7rem;
                border-radius: 18px;
            }
            div.stButton > button {
                border-radius: 14px;
                font-weight: 700;
            }
            div[data-baseweb="tab-list"] button {
                border-radius: 12px 12px 0 0;
                font-weight: 600;
            }
            .team-card {
                padding: 0.9rem 1rem;
                border-radius: 18px;
                background: rgba(30,41,59,0.45);
                border: 1px solid rgba(148,163,184,0.14);
                min-height: 94px;
            }
            .footer-note {
                padding: 0.8rem 1rem;
                border-left: 4px solid #38bdf8;
                background: rgba(8,47,73,0.20);
                border-radius: 12px;
                margin-top: 0.75rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def find_school_logo() -> Path | None:
    candidates = [
        Path("assets/logo_truong.png"),
        Path("assets/logo_truong.jpg"),
        Path("assets/logo_school.png"),
        Path("assets/logo_kma.png"),
        Path("assets/logo_uit.png"),
        Path("logo_truong.png"),
        Path("logo_truong.jpg"),
        Path("logo_school.png"),
        Path("logo_kma.png"),
        Path("logo_uit.png"),
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return p
    return None


def render_header(selected_model: str, use_cache: bool, load_elapsed_s: float):
    logo_path = find_school_logo()
    left, right = st.columns([1.2, 6])
    with left:
        if logo_path is not None:
            st.image(str(logo_path), use_container_width=True)
        else:
            st.markdown(
                """
                <div class="section-card" style="text-align:center; padding:1.1rem 0.8rem;">
                    <div style="font-size:2rem; font-weight:800;">🎓</div>
                    <div class="mini-muted">Chưa có logo trường</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    with right:
        st.markdown(
            f"""
            <div class="hero-box">
                <div class="main-title">QR Quishing Detection Demo</div>
                <div class="subtitle">Web demo minh họa nhận diện QR code lành tính hoặc nghi ngờ độc hại bằng mô hình học máy.</div>
                <div class="pill-row">
                    <span class="pill">Machine Learning</span>
                    <span class="pill">QR Demo</span>
                    <span class="pill">Performance Benchmark</span>
                    <span class="pill">Model Cache: {'ON' if use_cache else 'OFF'}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    c1, c2, c3 = st.columns(3)
    c1.metric("Model đang chọn", Path(selected_model).name)
    c2.metric("Load model (ms)", f"{round(load_elapsed_s * 1000.0, 3)}")
    c3.metric("Cache", "Bật" if use_cache else "Tắt")


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



def render_qr_to_array(text: str, target_shape=(69, 69)):
    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(text)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white").convert("L")
    try:
        resample = Image.Resampling.LANCZOS
    except Exception:
        resample = Image.LANCZOS
    img = img.resize(target_shape[::-1], resample=resample)
    arr = np.asarray(img, dtype=np.float32)
    arr = 1.0 - (arr / 255.0)
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



def run_single_prediction(model, text: str, qr_source: str, model_path: str):
    t_total_0 = time.perf_counter()

    t0 = time.perf_counter()
    img, arr = render_qr_to_array(text=text, target_shape=TARGET_SHAPE)
    t1 = time.perf_counter()
    qr_generation_s = t1 - t0

    t0 = time.perf_counter()
    X_input = arr.reshape(1, -1)
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
        "input_shape": f"{X_input.shape[0]}x{X_input.shape[1]}",
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



def run_benchmark(model, text: str, qr_source: str, model_path: str, n_runs: int = 100):
    total_ms = []
    infer_ms = []
    prep_ms = []
    gen_ms = []
    predictions = []

    for _ in range(int(n_runs)):
        t_total_0 = time.perf_counter()

        t0 = time.perf_counter()
        _, arr = render_qr_to_array(text=text, target_shape=TARGET_SHAPE)
        t1 = time.perf_counter()
        gen_ms.append((t1 - t0) * 1000.0)

        t0 = time.perf_counter()
        X_input = arr.reshape(1, -1)
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
        **{f"total_{k}": v for k, v in summarize_latency_ms(total_ms).items()},
        **{f"infer_{k}": v for k, v in summarize_latency_ms(infer_ms).items()},
        **{f"preprocess_{k}": v for k, v in summarize_latency_ms(prep_ms).items()},
        **{f"generate_{k}": v for k, v in summarize_latency_ms(gen_ms).items()},
    }
    stats = get_process_stats()
    summary.update(stats)
    append_csv_row(BENCHMARK_LOG_FILE, summary)
    return summary



def try_read_csv(path: Path):
    if not path.exists():
        return None
    try:
        import pandas as pd
        return pd.read_csv(path)
    except Exception:
        return None


st.set_page_config(page_title="QR Quishing Demo + Performance", layout="wide")
inject_custom_css()

model_files = sorted(DEFAULT_OUTPUT_DIR.glob("models/**/*.joblib"))
if not model_files:
    st.warning("Chưa tìm thấy model .joblib. Hãy chạy notebook train trước.")
    st.stop()

with st.sidebar:
    st.header("Cấu hình")
    use_cache = st.checkbox("Dùng cache khi load model", value=True)
    benchmark_runs = st.slider("Số lần benchmark", min_value=10, max_value=500, value=100, step=10)
    show_logs = st.checkbox("Hiện log gần nhất", value=True)
    st.markdown("---")
    st.caption("Mẹo: đặt logo trường vào assets/logo_truong.png để app tự hiển thị ở phần đầu trang.")

model_path = st.selectbox("Chọn model đã train", options=[str(p) for p in model_files])

load_t0 = time.perf_counter()
if use_cache:
    bundle = load_model_bundle_cached(model_path)
else:
    bundle = load_model_bundle_uncached(model_path)
load_elapsed_s = time.perf_counter() - load_t0
log_perf("load_model", load_elapsed_s, {"selected_model": model_path, "use_cache": use_cache})
model = bundle["model"]
metadata = bundle.get("metadata", {}) or {}

render_header(selected_model=model_path, use_cache=use_cache, load_elapsed_s=load_elapsed_s)
st.caption("Demo an toàn: các payload malicious trong app chỉ là mẫu mô phỏng để kiểm thử mô hình và hiệu năng webapp.")

intro_tab, demo_tab, logs_tab = st.tabs(["Giới thiệu", "Demo + Benchmark", "Logs"])

with intro_tab:
    left, right = st.columns([1.15, 1])
    with left:
        st.markdown(INTRO_MD)
        st.markdown(
            """
            <div class="footer-note">
                <b>Lưu ý:</b> Kết quả trong webapp mang tính minh họa ứng dụng, không thay thế cho bảng kết quả thực nghiệm chính thức trong notebook hoặc báo cáo.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            """
            <div class="section-card">
                <h4>Thông tin phiên chạy</h4>
                <div class="mini-muted">Thông số kỹ thuật của model đang được nạp trong phiên hiện tại.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(
            {
                "selected_model": model_path,
                "metadata_keys": list(metadata.keys())[:20],
                "model_load_ms": round(load_elapsed_s * 1000.0, 3),
                "cache_enabled": use_cache,
            }
        )

    st.markdown("### Nhóm thực hiện")
    cols = st.columns(len(TEAM_MEMBERS))
    for col, (name, code) in zip(cols, TEAM_MEMBERS):
        with col:
            st.markdown(
                f"""
                <div class="team-card">
                    <div style="font-weight:700; font-size:1rem; margin-bottom:0.25rem;">{name}</div>
                    <div class="mini-muted">Mã học viên: {code}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("**Giảng viên hướng dẫn:** TS. Nguyễn An Khương")

    a, b, c = st.columns(3)
    a.info("""**Đầu vào**

QR synthetic hoặc text nhập tay.""")
    b.success("""**Đầu ra**

Nhãn dự đoán, score lớp 1 và thời gian xử lý.""")
    c.warning("""**Mục đích**

Minh họa ứng dụng, benchmark webapp và hỗ trợ trình bày báo cáo.""")

with demo_tab:
    col_cfg, col_out = st.columns([1, 1.25])
    with col_cfg:
        st.markdown(
            """
            <div class="section-card">
                <h4>Thiết lập đầu vào</h4>
                <div class="mini-muted">Chọn nguồn QR, cấu hình dữ liệu thử nghiệm và thực hiện dự đoán hoặc benchmark.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        input_mode = st.radio("Nguồn QR", ["Synthetic", "Text nhập tay"])
        is_synthetic_mode = input_mode == "Synthetic"

        qr_type = st.selectbox(
            "Loại QR synthetic",
            ["benign", "malicious"],
            disabled=not is_synthetic_mode,
            help="Chỉ bật khi chọn nguồn QR là Synthetic.",
        )
        sample_index = st.number_input(
            "Sample index",
            min_value=0,
            max_value=20,
            value=0,
            step=1,
            disabled=not is_synthetic_mode,
            help="Chỉ bật khi chọn nguồn QR là Synthetic.",
        )
        custom_text = st.text_area(
            "Nội dung QR thủ công",
            value="https://example.com",
            height=140,
            disabled=is_synthetic_mode,
            help="Chỉ bật khi chọn nguồn QR là Text nhập tay.",
        )

        if is_synthetic_mode:
            st.caption("Đang dùng QR synthetic. Phần nhập tay đã được làm mờ.")
        else:
            st.caption("Đang dùng nội dung nhập tay. Tùy chọn QR synthetic đã được làm mờ.")

        run_predict = st.button("Predict QR", type="primary", use_container_width=True)
        run_bench = st.button("Benchmark hiệu năng", use_container_width=True)

    with col_out:
        st.markdown(
            """
            <div class="section-card">
                <h4>Model info</h4>
                <div class="mini-muted">Thông tin model đang dùng cho lần dự đoán hiện tại.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.write(
            {
                "selected_model": model_path,
                "metadata_keys": list(metadata.keys())[:20],
                "model_load_ms": round(load_elapsed_s * 1000.0, 3),
                "cache_enabled": use_cache,
            }
        )

    if input_mode == "Synthetic":
        qr_type, text = get_safe_qr_payload(qr_type=qr_type, sample_index=sample_index)
    else:
        text = custom_text
        qr_type = "custom"

    if run_predict:
        img, arr, result = run_single_prediction(model=model, text=text, qr_source=qr_type, model_path=model_path)
        perf_stats = get_process_stats()

        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.markdown(
                """
                <div class="section-card">
                    <h4>QR đã render</h4>
                    <div class="mini-muted">Ảnh QR sau khi được sinh từ payload đầu vào.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(img, caption=f"QR rendered | source={qr_type}", use_container_width=False)
            st.write("**Payload**")
            st.code(text)
        with c2:
            st.markdown(
                """
                <div class="section-card">
                    <h4>Kết quả dự đoán</h4>
                    <div class="mini-muted">Nhãn dự đoán, score và các chỉ số hiệu năng cho một lượt chạy.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
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

    if run_bench:
        with st.spinner("Đang benchmark..."):
            summary = run_benchmark(model=model, text=text, qr_source=qr_type, model_path=model_path, n_runs=benchmark_runs)
        st.subheader("Benchmark summary")
        st.json(summary)
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Mean total (ms)", f"{summary.get('total_mean_ms', '-')}")
        b2.metric("P95 total (ms)", f"{summary.get('total_p95_ms', '-')}")
        b3.metric("P99 total (ms)", f"{summary.get('total_p99_ms', '-')}")
        b4.metric("Max total (ms)", f"{summary.get('total_max_ms', '-')}")
        st.caption("Khi báo cáo hiệu năng, nên ưu tiên mean, std, p95, p99 và max latency.")

with logs_tab:
    if show_logs:
        st.subheader("Log hiệu năng gần nhất")
        perf_df = try_read_csv(PERF_LOG_FILE)
        bench_df = try_read_csv(BENCHMARK_LOG_FILE)
        tab1, tab2 = st.tabs(["Prediction log", "Benchmark log"])
        with tab1:
            if perf_df is not None and not perf_df.empty:
                st.dataframe(perf_df.tail(20), use_container_width=True)
            else:
                st.info("Chưa có prediction log.")
        with tab2:
            if bench_df is not None and not bench_df.empty:
                st.dataframe(bench_df.tail(20), use_container_width=True)
            else:
                st.info("Chưa có benchmark log.")
    else:
        st.info("Bạn đang tắt hiển thị log trong sidebar.")

    st.markdown("---")
    st.subheader("Gợi ý đánh giá hiệu năng webapp")
    st.markdown("""
- **Startup / model load time**: thời gian mở app và nạp model.
- **QR generation time**: thời gian sinh ảnh QR từ payload.
- **Preprocessing time**: thời gian reshape và chuẩn hóa đầu vào.
- **Inference time**: thời gian mô hình dự đoán.
- **End-to-end latency**: tổng thời gian từ input đến output.
- **RAM / CPU usage**: mức tài nguyên tiêu thụ khi chạy demo.
- **P95 / P99 latency**: độ trễ của các lượt chạy chậm hơn bình thường, hữu ích khi stress test.
""")
