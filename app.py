import base64
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Деэтанизации конденсата", page_icon="⚙️", layout="wide")

MODEL_DIR = Path("saved") if Path("saved").exists() else Path(".")
TEMP_MODEL_PATH = MODEL_DIR / "model_temp.pkl"
PRESS_MODEL_PATH = MODEL_DIR / "model_press.pkl"
TEMP_FEATURES_PATH = MODEL_DIR / "features_temp.pkl"
PRESS_FEATURES_PATH = MODEL_DIR / "features_press.pkl"
TEMP_MEDIANS_PATH = MODEL_DIR / "medians_temp.pkl"
PRESS_MEDIANS_PATH = MODEL_DIR / "medians_press.pkl"

DATA_CANDIDATES = [
    Path("preparing/dataset_E-301_Температура.csv"),
    Path("preparing/dataset_Давление_в_E-301.csv"),
    Path("preparing/dataset_Давление_в_Е-301.csv"),
    Path("final_validation.csv"),
]

SCHEME_IMAGE_CANDIDATES = [
    Path("7-3.jpg"),
    Path("7.jpg"),
    Path("reference_for_vizualization.jpg"),
    Path("images/7-3.jpg"),
    Path("assets/7-3.jpg"),
]

METHANE_CANDIDATES = ["Содержание метана", "Метан", "CH4", "Содержание метана в сырье"]
ETHANE_CANDIDATES = ["Содержание этана", "Этан", "C2H6", "Содержание этана в сырье"]

TOP_TEMP = "K-301 Температура верха"
HOT_ZONE = "K-301 Темп-ра в гор. зоне"
COLD_ZONE = "K-301 Темп-ра в хол. зоне"
EXCESS = "Вывод балансового избытка"
KGD_MASS = "Масса КГД из куба колонны"

REMOVED_FROM_UI = {
    "Метан+этан из жидкости в Е-301",
    "Метан+этан из жидкости в E-301",
}


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Не найден файл: {path}")
    return joblib.load(path)


@st.cache_resource
def load_artifacts():
    return {
        "temp_model": load_pickle(TEMP_MODEL_PATH),
        "press_model": load_pickle(PRESS_MODEL_PATH),
        "temp_features": load_pickle(TEMP_FEATURES_PATH),
        "press_features": load_pickle(PRESS_FEATURES_PATH),
        "temp_medians": load_pickle(TEMP_MEDIANS_PATH),
        "press_medians": load_pickle(PRESS_MEDIANS_PATH),
    }


def find_reference_dataset():
    for path in DATA_CANDIDATES:
        if path.exists():
            try:
                df = pd.read_csv(path, sep=";", decimal=".", encoding="utf-8-sig")
                df.columns = [str(c).strip() for c in df.columns]
                for c in df.columns:
                    if df[c].dtype == object:
                        s = df[c].astype(str).str.replace(",", ".", regex=False)
                        num = pd.to_numeric(s, errors="coerce")
                        if num.notna().mean() >= 0.7:
                            df[c] = num
                return df
            except Exception:
                continue
    return None


def find_scheme_image():
    for path in SCHEME_IMAGE_CANDIDATES:
        if path.exists():
            return path
    return None


def image_to_base64(path: Path):
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def build_input_frame(features, medians, user_values):
    row = {}
    for feat in features:
        row[feat] = user_values.get(feat, medians.get(feat, 0.0))
    return pd.DataFrame([row])


def predict_values(artifacts, user_values):
    temp_X = build_input_frame(artifacts["temp_features"], artifacts["temp_medians"], user_values)
    press_X = build_input_frame(artifacts["press_features"], artifacts["press_medians"], user_values)
    temp_pred = float(artifacts["temp_model"].predict(temp_X)[0])
    press_pred = float(artifacts["press_model"].predict(press_X)[0])
    return temp_pred, press_pred


def metric_box(label, value, unit="", color="#1f5131", bg="#ffffff"):
    st.markdown(
        f"""
        <div style='background:{bg};border:1px solid #d7ddd7;border-radius:14px;padding:10px 12px;margin-bottom:10px;box-shadow:0 2px 8px rgba(34,57,34,0.05);'>
            <div style='font-size:12px;color:#677267;margin-bottom:2px;'>{label}</div>
            <div style='font-size:22px;font-weight:700;color:{color};line-height:1.2;'>{value} <span style='font-size:14px;font-weight:600;'>{unit}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def find_first_existing(candidates, all_features):
    for item in candidates:
        if item in all_features:
            return item
    return None


def display_value(user_values, medians, feat):
    return float(user_values.get(feat, medians.get(feat, 0.0))) if feat else 0.0


def render_scheme_with_overlay(
    scheme_path,
    temp_pred,
    press_pred,
    methane_label,
    ethane_label,
    methane_val,
    ethane_val,
    top_val,
    hot_val,
    cold_val,
    kgd_val,
    excess_val,
):
    if scheme_path is None:
        st.warning("Не найдено изображение схемы установки. Положите файл 7-3.jpg рядом с app.py.")
        return

    img_b64 = image_to_base64(scheme_path)

    html = f"""
    <div style="
        width:100%;
        display:flex;
        justify-content:center;
        align-items:flex-start;
        padding:8px 0 18px 0;
        font-family:Arial,sans-serif;
    ">
        <div style="
            width:100%;
            max-width:1260px;
            background:linear-gradient(180deg,#f5f8f3 0%,#eef3ec 100%);
            border:1px solid #cfd8cd;
            border-radius:22px;
            padding:22px 22px 26px 22px;
            box-sizing:border-box;
        ">
            <div style="margin-bottom:18px;">
                <div style="font-size:30px;font-weight:800;color:#203321;">Деэтанизации конденсата</div>
                <div style="font-size:13px;color:#627262;margin-top:4px;">
                    Расчёт рекомендуемых параметров ведется из условия деэтанизации конденсата в рефлюксной емкости
                </div>
            </div>

            <div style="
                position:relative;
                width:100%;
                min-height:760px;
            ">
                <div style="
                    position:absolute;
                    left:50%;
                    top:44%;
                    transform:translate(-50%,-50%);
                    width:58%;
                    max-width:720px;
                    height:430px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    background:#ffffff;
                    border:4px solid #202020;
                    box-sizing:border-box;
                ">
                    <img
                        src="data:image/jpeg;base64,{img_b64}"
                        style="
                            max-width:100%;
                            max-height:100%;
                            object-fit:contain;
                            display:block;
                        "
                    />
                </div>

                <div style="
                    position:absolute;
                    left:2.5%;
                    top:8%;
                    width:170px;
                    min-height:96px;
                    background:#fffffff2;
                    border:3px solid #202020;
                    border-radius:0;
                    padding:10px 10px;
                    box-sizing:border-box;
                ">
                    <div style="font-size:13px;font-weight:700;color:#222;margin-bottom:8px;">Рекомендуемые параметры</div>
                    <div style="font-size:11px;color:#555;">Температура, ℃</div>
                    <div style="font-size:18px;font-weight:800;color:#1c6d34;margin-bottom:6px;">{temp_pred:.2f}</div>
                    <div style="font-size:11px;color:#555;">Давление, МПа</div>
                    <div style="font-size:18px;font-weight:800;color:#1f5fbf;">{press_pred:.3f}</div>
                </div>

                <div style="
                    position:absolute;
                    left:2.5%;
                    top:34%;
                    width:170px;
                    min-height:86px;
                    background:#fffffff2;
                    border:3px solid #202020;
                    border-radius:0;
                    padding:10px 10px;
                    box-sizing:border-box;
                ">
                    <div style="font-size:13px;font-weight:700;color:#222;margin-bottom:8px;">Сырье</div>
                    <div style="font-size:11px;color:#555;">{methane_label}</div>
                    <div style="font-size:16px;font-weight:800;color:#245a2d;margin-bottom:6px;">{methane_val:.4f}</div>
                    <div style="font-size:11px;color:#555;">{ethane_label}</div>
                    <div style="font-size:16px;font-weight:800;color:#245a2d;">{ethane_val:.4f}</div>
                </div>

                <div style="
                    position:absolute;
                    left:2.5%;
                    top:58%;
                    width:170px;
                    min-height:108px;
                    background:#fffffff2;
                    border:3px solid #202020;
                    border-radius:0;
                    padding:10px 10px;
                    box-sizing:border-box;
                ">
                    <div style="font-size:12px;color:#333;">E-301</div>
                    <div style="font-size:13px;font-weight:700;color:#222;margin-bottom:8px;">Рекомендуемые параметры</div>
                    <div style="font-size:11px;color:#555;">Температура, ℃</div>
                    <div style="font-size:16px;font-weight:800;color:#1c6d34;margin-bottom:6px;">{temp_pred:.2f}</div>
                    <div style="font-size:11px;color:#555;">Давление, МПа</div>
                    <div style="font-size:16px;font-weight:800;color:#1f5fbf;">{press_pred:.3f}</div>
                </div>

                <div style="
                    position:absolute;
                    right:2.5%;
                    top:8%;
                    width:170px;
                    min-height:86px;
                    background:#fffffff2;
                    border:3px solid #202020;
                    border-radius:0;
                    padding:10px 10px;
                    box-sizing:border-box;
                ">
                    <div style="font-size:13px;font-weight:700;color:#222;margin-bottom:8px;">K-301</div>
                    <div style="font-size:11px;color:#555;">Температура верха</div>
                    <div style="font-size:15px;font-weight:800;color:#234c28;margin-bottom:4px;">{top_val:.2f} ℃</div>
                    <div style="font-size:11px;color:#555;">Гор. зона</div>
                    <div style="font-size:15px;font-weight:800;color:#234c28;margin-bottom:4px;">{hot_val:.2f} ℃</div>
                    <div style="font-size:11px;color:#555;">Хол. зона</div>
                    <div style="font-size:15px;font-weight:800;color:#234c28;">{cold_val:.2f} ℃</div>
                </div>

                <div style="
                    position:absolute;
                    right:2.5%;
                    top:40%;
                    width:170px;
                    min-height:86px;
                    background:#fffffff2;
                    border:3px solid #202020;
                    border-radius:0;
                    padding:10px 10px;
                    box-sizing:border-box;
                ">
                    <div style="font-size:12px;color:#333;">Между K-301 и ВХ-302</div>
                    <div style="font-size:12px;color:#555;margin-top:8px;">Масса КГД</div>
                    <div style="font-size:20px;font-weight:800;color:#245a2d;">{kgd_val:.2f}</div>
                </div>

                <div style="
                    position:absolute;
                    right:2.5%;
                    top:66%;
                    width:170px;
                    min-height:86px;
                    background:#fffffff2;
                    border:3px solid #202020;
                    border-radius:0;
                    padding:10px 10px;
                    box-sizing:border-box;
                ">
                    <div style="font-size:12px;color:#555;margin-top:8px;">Вывод балансового избытка</div>
                    <div style="font-size:20px;font-weight:800;color:#245a2d;">{excess_val:.2f}</div>
                </div>

                <div style="
                    position:absolute;
                    left:50%;
                    bottom:2%;
                    transform:translateX(-50%);
                    width:240px;
                    min-height:42px;
                    background:#fffffff2;
                    border:3px solid #202020;
                    border-radius:0;
                    padding:8px 12px;
                    box-sizing:border-box;
                    text-align:left;
                ">
                    <div style="font-size:13px;font-weight:700;color:#222;margin-bottom:4px;">Расчет</div>
                    <div style="font-size:11px;color:#555;line-height:1.35;">
                        Расчёт выполнен на основе введенных параметров сырья и режима работы колонны.
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    components.html(html, height=860, scrolling=False)


def main():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 1600px;}
        .stNumberInput label {font-weight: 600;}
        .stApp {background: linear-gradient(180deg, #07111f 0%, #091221 100%);}
        h1, h2, h3, p, label, .stMarkdown, .stCaption {color: #f5f7fb;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    try:
        artifacts = load_artifacts()
    except Exception as e:
        st.error(f"Ошибка загрузки артефактов: {e}")
        st.stop()

    ref_df = find_reference_dataset()
    scheme_path = find_scheme_image()

    temp_features = artifacts["temp_features"]
    press_features = artifacts["press_features"]
    all_features = list(dict.fromkeys(temp_features + press_features))

    methane_feat = find_first_existing(METHANE_CANDIDATES, all_features)
    ethane_feat = find_first_existing(ETHANE_CANDIDATES, all_features)

    medians = {}
    for feat, val in artifacts["temp_medians"].items():
        medians[feat] = val
    for feat, val in artifacts["press_medians"].items():
        medians.setdefault(feat, val)

    important_order = [
        f
        for f in [methane_feat, ethane_feat, TOP_TEMP, HOT_ZONE, COLD_ZONE, EXCESS, KGD_MASS]
        if f and f in all_features and f not in REMOVED_FROM_UI
    ] + ["40-50", "50-60", "60-70", "90-100", "100-110", "110-120"]

    ordered = [f for f in important_order if f in all_features] + [
        f for f in all_features if f not in important_order and f not in REMOVED_FROM_UI
    ]

    with st.sidebar:
        st.subheader("Управление")
        st.write("Изменяйте параметры сырья и режима работы. Рекомендуемые параметры обновляются автоматически.")
        show_more = st.checkbox("Показать расширенный ввод", value=False)
        use_demo = st.button("Заполнить demo-значениями")

    selected_features = ordered[:8] if not show_more else ordered[:16]
    user_values = {}

    selected_features = ordered[:8] if not show_more else ordered[:16]
    user_values = {}

    st.markdown("## Входные данные")
    st.caption("Во входах оставлены отдельные параметры сырья: содержание метана и содержание этана.")

    input_cols = st.columns(4)

    temp_pred, press_pred = predict_values(artifacts, user_values)

    methane_label = methane_feat if methane_feat else "Содержание метана"
    ethane_label = ethane_feat if ethane_feat else "Содержание этана"

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    render_scheme_with_overlay(
        scheme_path=scheme_path,
        temp_pred=temp_pred,
        press_pred=press_pred,
        methane_label=methane_label,
        ethane_label=ethane_label,
        methane_val=display_value(user_values, medians, methane_feat),
        ethane_val=display_value(user_values, medians, ethane_feat),
        top_val=display_value(user_values, medians, TOP_TEMP),
        hot_val=display_value(user_values, medians, HOT_ZONE),
        cold_val=display_value(user_values, medians, COLD_ZONE),
        kgd_val=display_value(user_values, medians, KGD_MASS),
        excess_val=display_value(user_values, medians, EXCESS),
    )

    st.markdown("## Рекомендуемые параметры в рефлюксной емкости")
    m1, m2 = st.columns(2)
    with m1:
        metric_box("Температура, ℃", f"{temp_pred:.2f}", "℃", "#1c6d34", "#f8fcf8")
    with m2:
        metric_box("Давление, МПа", f"{press_pred:.3f}", "МПа", "#1f5fbf", "#f7faff")

    with st.expander("Показать входные значения для модели"):
        preview = pd.DataFrame(
            {
                "feature": all_features,
                "value": [user_values.get(f, medians.get(f, 0.0)) for f in all_features],
            }
        )
        st.dataframe(preview, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
