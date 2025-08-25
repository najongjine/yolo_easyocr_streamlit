import importlib.util, sys, subprocess, streamlit as st, pkgutil
st.write("python:", sys.version)
st.write("ultralytics spec:", importlib.util.find_spec("ultralytics"))
st.write("site-packages in sys.path?", any("site-packages" in p for p in sys.path))
st.write("Top-20 pip freeze:")
out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"]).decode().splitlines()
st.code("\n".join(out[:20]))


import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

# =========================
# 1) Model loaders
# =========================
@st.cache_resource
def load_yolo(weights_path: str):
    from ultralytics import YOLO
    return YOLO(weights_path)

@st.cache_resource
def load_reader():
    import easyocr
    return easyocr.Reader(['ko','en'], gpu=False)

# =========================
# 2) App UI
# =========================
st.set_page_config(page_title="간단 번호판 OCR", layout="wide")
st.title("번호판 인식 → 크롭(여백) → EasyOCR (no OpenCV)")

weights_path = st.text_input("YOLO 가중치 경로(.pt)", value="carplate_v11_yolo11n_70n.pt")
conf = st.slider("감지 신뢰도(conf)", 0.1, 0.9, 0.5, 0.05)
margin_px = st.slider("크롭 여백(px)", 0, 120, 2, 2)

uploaded = st.file_uploader("이미지 업로드 (jpg/png/bmp)", type=["jpg","jpeg","png","bmp"])

col1, col2 = st.columns(2)

if uploaded is not None and os.path.exists(weights_path):
    # 입력 이미지 (PIL RGB)
    image = Image.open(uploaded).convert("RGB")
    img_rgb = np.array(image)               # (H, W, 3), RGB

    # 모델
    yolo = load_yolo(weights_path)
    reader = load_reader()

    # 3) Detection (Ultralytics는 PIL/numpy RGB도 입력 가능)
    results = yolo.predict(img_rgb, conf=conf, verbose=False)

    annotated_pil = image.copy()
    draw = ImageDraw.Draw(annotated_pil)
    font = ImageFont.load_default()

    crops = []           # [(crop_rgb(np.array), (x1,y1,x2,y2))]
    ocr_rows = []        # [{"plate_index":i, "text":t, "conf":p}, ...]

    if len(results):
        res = results[0]
        boxes = res.boxes.xyxy.cpu().numpy().astype(int) if res.boxes is not None else np.empty((0,4), dtype=int)
        H, W = img_rgb.shape[:2]

        for i, (x1, y1, x2, y2) in enumerate(boxes, start=1):
            # 여백 적용
            x1m = max(0, x1 - margin_px)
            y1m = max(0, y1 - margin_px)
            x2m = min(W, x2 + margin_px)
            y2m = min(H, y2 + margin_px)

            crop_rgb = img_rgb[y1m:y2m, x1m:x2m].copy()
            if crop_rgb.size == 0:
                continue

            # 박스/라벨 그리기 (PIL)
            draw.rectangle([(x1m, y1m), (x2m, y2m)], outline=(0, 255, 0), width=2)
            label = f"plate {i}"
            # 텍스트 배경 살짝 넣고 싶으면 아래 두 줄 사용 (선택)
            # tw, th = draw.textbbox((0,0), label, font=font)[2:]
            # draw.rectangle([(x1m, y1m - th - 4), (x1m + tw + 4, y1m)], fill=(0,255,0))
            draw.text((x1m, max(0, y1m - 16)), label, fill=(0, 255, 0), font=font)

            crops.append((crop_rgb, (x1m, y1m, x2m, y2m)))

            # 4) EasyOCR (RGB 배열 바로 전달)
            ocr_result = reader.readtext(crop_rgb)  # [(bbox, text, conf), ...]
            for (_, text, confv) in ocr_result:
                ocr_rows.append({
                    "plate_index": i,
                    "text": str(text),
                    "conf": float(confv)
                })

    # 5) 출력
    with col1:
        st.subheader("원본 + 감지 박스(여백 반영)")
        st.image(annotated_pil, use_container_width=True)

        if crops:
            st.caption("각 크롭 결과")
            for idx, (crop_rgb, _) in enumerate(crops, start=1):
                st.image(crop_rgb, caption=f"Plate #{idx} crop", use_container_width=True)

    with col2:
        st.subheader("OCR 인식된 문자들 (그대로)")
        if ocr_rows:
            import pandas as pd
            df = pd.DataFrame(ocr_rows)
            st.dataframe(df, use_container_width=True)
            with st.expander("원시 JSON 보기"):
                st.json(ocr_rows)
        else:
            st.info("OCR 결과가 없습니다. conf를 낮추거나 여백/이미지를 조정하세요.")

else:
    if uploaded is None:
        st.info("이미지를 업로드하세요.")
    elif not os.path.exists(weights_path):
        st.warning("가중치 파일 경로가 올바르지 않습니다.")
