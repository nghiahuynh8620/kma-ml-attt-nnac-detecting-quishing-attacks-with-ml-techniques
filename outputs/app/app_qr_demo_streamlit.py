from pathlib import Path
import joblib
import numpy as np
import streamlit as st
import qrcode
from PIL import Image

LABEL_NAME_MAP = {0: 'benign', 1: 'malicious'}
SAFE_BENIGN_QR_PAYLOADS = ['https://example.com/news/academic-2026', 'WIFI:T:WPA;S:CampusGuest;P:Safe12345;;', 'BEGIN:VCARD\nFN:Lab Reception\nTEL:+84000000000\nEND:VCARD', 'https://example.org/library/week-12']
SAFE_SIMULATED_MALICIOUS_QR_PAYLOADS = ['http://198.51.100.24/verify?id=8842', 'https://login.example.invalid/update?t=reset', 'http://203.0.113.7/billing?r=confirm', 'https://account.example.invalid/open?n=signin']
DEFAULT_OUTPUT_DIR = Path(r"outputs")
QR_VERSION = 13
QR_BORDER = 0
QR_BOX_SIZE = 1
QR_TARGET_SHAPE = (69, 69)

def get_safe_qr_payload(qr_type='benign', sample_index=0):
    qr_type = str(qr_type).strip().lower()
    if qr_type in {'benign', 'normal', 'safe'}:
        pool = SAFE_BENIGN_QR_PAYLOADS
        canonical_type = 'benign'
    else:
        pool = SAFE_SIMULATED_MALICIOUS_QR_PAYLOADS
        canonical_type = 'malicious'
    sample_index = int(sample_index) % len(pool)
    return canonical_type, pool[sample_index]

def render_qr_to_array(text, target_shape=QR_TARGET_SHAPE):
    qr = qrcode.QRCode(
        version=QR_VERSION,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=QR_BOX_SIZE,
        border=QR_BORDER,
    )
    qr.add_data(text)
    qr.make(fit=False)
    img = qr.make_image(fill_color='black', back_color='white').convert('L')
    arr = np.asarray(img, dtype=np.float32)
    if tuple(arr.shape) != tuple(target_shape):
        raise ValueError(f'QR shape {arr.shape} != expected {target_shape}')
    return img, arr

def get_positive_scores(model, X):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X)
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-12)
        return scores
    return model.predict(X)

def load_model_bundle(path):
    bundle = joblib.load(path)
    if isinstance(bundle, dict) and 'model' in bundle:
        return bundle
    return {'model': bundle, 'metadata': {}}

def align_generated_array_to_reference(qr_array, reference_stats):
    ref_min = reference_stats.get('min', 0.0)
    ref_max = reference_stats.get('max', 1.0)
    ref_mean = reference_stats.get('mean', 0.5)
    if ref_max <= 1.0 + 1e-12:
        base = qr_array.astype(np.float32) / 255.0 if qr_array.max() > 1 else qr_array.astype(np.float32)
    else:
        base = qr_array.astype(np.float32) / 255.0 * (ref_max - ref_min) + ref_min
    inv = ref_min + ref_max - base
    chosen = base if abs(float(base.mean()) - ref_mean) <= abs(float(inv.mean()) - ref_mean) else inv
    return chosen.astype(np.float32)

st.set_page_config(page_title='QR Quishing Demo', layout='wide')
st.title('QR Quishing Demo')
st.info('Đây là phần minh họa ứng dụng để thử mô hình với QR synthetic an toàn; không phải đánh giá chính thức của mô hình.')

model_files = sorted(DEFAULT_OUTPUT_DIR.glob('models/**/*.joblib'))
if not model_files:
    st.warning('Chưa tìm thấy model .joblib. Hãy chạy script train trước.')
    st.stop()

model_path = st.selectbox('Chọn model đã train', options=[str(p) for p in model_files])
bundle = load_model_bundle(model_path)
model = bundle['model']
metadata = bundle.get('metadata', {})
reference_stats = metadata.get('reference_image_stats', {'min': 0.0, 'max': 1.0, 'mean': 0.5})

col1, col2 = st.columns(2)
with col1:
    input_mode = st.radio('Nguồn QR', ['Synthetic', 'Text nhập tay'])
    qr_type = st.selectbox('Loại QR synthetic', ['benign', 'malicious'])
    sample_index = st.number_input('Sample index', min_value=0, max_value=20, value=0, step=1)
    custom_text = st.text_area('Nội dung QR thủ công', value='https://example.com')
with col2:
    try:
        if input_mode == 'Synthetic':
            qr_type, text = get_safe_qr_payload(qr_type=qr_type, sample_index=sample_index)
        else:
            text = custom_text
            qr_type = 'custom'

        img, arr = render_qr_to_array(text=text, target_shape=QR_TARGET_SHAPE)
        arr = align_generated_array_to_reference(arr, reference_stats)
        X_input = arr.reshape(1, -1)
        pred_label = int(model.predict(X_input)[0])
        score_1 = float(get_positive_scores(model, X_input)[0])

        st.image(img, caption=f'QR rendered | source={qr_type} | version=13 | 69x69 | EC=LOW | border=0', use_container_width=False)
        st.write('**Payload**')
        st.code(text)
        st.write('**Prediction**')
        st.json({
            'predicted_label': pred_label,
            'predicted_label_name': LABEL_NAME_MAP.get(pred_label, f'class_{pred_label}'),
            'prob_class_1': round(score_1, 6),
            'selected_model': model_path,
        })
    except Exception as e:
        st.error(str(e))

st.caption('Lưu ý: đây là minh họa ứng dụng với QR synthetic an toàn; phần đánh giá chính thức vẫn phải dựa trên test set và cross-validation.')
