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

# Добавляем константы для метана и этана
METHANE_FEAT = "Содержание метана"
ETHANE_FEAT = "Содержание этана"

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


def get_kgd_mass_median(column_name):
    """Получить медианное значение массы КГД только из адекватных значений (100-250 тонн/час)"""
    for path in DATA_CANDIDATES:
        if path.exists():
            try:
                df = pd.read_csv(path, sep=";", decimal=".", encoding="utf-8-sig")
                df.columns = [str(c).strip() for c in df.columns]
                
                if column_name in df.columns:
                    series = pd.to_numeric(df[column_name], errors="coerce").dropna()
                    if not series.empty:
                        # Фильтруем значения от 100 до 250 тонн/час
                        valid_series = series[(series >= 100) & (series <= 250)]
                        if not valid_series.empty:
                            return float(valid_series.median())
                        
                        # Если нет значений в диапазоне 100-250, берем все положительные значения > 10
                        positive_series = series[series > 10]
                        if not positive_series.empty:
                            return float(positive_series.median())
            except Exception:
                continue
    
    return 150.0


def get_limits(df, feature, fallback_value):
    if df is not None and feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
        series = pd.to_numeric(df[feature], errors="coerce").dropna()
        if not series.empty:
            lo = float(series.quantile(0.05))
            hi = float(series.quantile(0.95))
            med = float(series.median())

            if feature == COLD_ZONE:
                q1 = float(series.quantile(0.25))
                q3 = float(series.quantile(0.75))
                lo, hi = q1, q3

            if lo == hi:
                lo, hi = float(series.min()), float(series.max())
            if lo == hi:
                lo -= 1.0
                hi += 1.0
            return lo, hi, med

    v = float(fallback_value) if pd.notna(fallback_value) else 0.0
    return v - abs(v) * 0.3 - 1, v + abs(v) * 0.3 + 1, v


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
        flex-direction:column;
        align-items:center;
        padding:8px 0 18px 0;
        font-family:Arial,sans-serif;
    ">
        <div style="
            width:100%;
            max-width:1500px;
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
                display:flex;
                justify-content:space-between;
                align-items:flex-start;
                gap:16px;
                width:100%;
            ">
                <div style="
                    display:flex;
                    flex-direction:column;
                    gap:16px;
                    min-width:200px;
                    max-width:220px;
                    flex-shrink:0;
                ">
                    <div style="
                        background:#ffffff;
                        border:3px solid #202020;
                        padding:14px 16px;
                        box-sizing:border-box;
                    ">
                        <div style="font-size:12px;color:#333;margin-bottom:2px;">E-301</div>
                        <div style="font-size:14px;font-weight:700;color:#222;margin-bottom:10px;">Рекомендуемые параметры</div>
                        <div style="font-size:11px;color:#555;">Температура, ℃</div>
                        <div style="font-size:20px;font-weight:800;color:#1c6d34;margin-bottom:8px;">{temp_pred:.2f}</div>
                        <div style="font-size:11px;color:#555;">Давление, МПа</div>
                        <div style="font-size:20px;font-weight:800;color:#1f5fbf;">{press_pred:.3f}</div>
                    </div>

                    <div style="
                        background:#ffffff;
                        border:3px solid #202020;
                        padding:14px 16px;
                        box-sizing:border-box;
                    ">
                        <div style="font-size:14px;font-weight:700;color:#222;margin-bottom:10px;">Сырье</div>
                        <div style="font-size:11px;color:#555;">Содержание метана</div>
                        <div style="font-size:18px;font-weight:800;color:#245a2d;margin-bottom:8px;">{methane_val:.4f}</div>
                        <div style="font-size:11px;color:#555;">Содержание этана</div>
                        <div style="font-size:18px;font-weight:800;color:#245a2d;margin-bottom:8px;">{ethane_val:.4f}</div>
                        <div style="font-size:10px;color:#666;">массовые доли</div>
                    </div>
                </div>

                <div style="
                    flex:1;
                    display:flex;
                    justify-content:center;
                    align-items:center;
                    min-height:560px;
                ">
                    <div style="
                        width:100%;
                        max-width:900px;
                        background:#ffffff;
                        border:4px solid #202020;
                        box-sizing:border-box;
                        padding:10px;
                        display:flex;
                        align-items:center;
                        justify-content:center;
                    ">
                        <img
                            src="data:image/jpeg;base64,{img_b64}"
                            style="
                                width:100%;
                                height:auto;
                                max-height:540px;
                                object-fit:contain;
                                display:block;
                            "
                        />
                    </div>
                </div>

                <div style="
                    display:flex;
                    flex-direction:column;
                    gap:16px;
                    min-width:200px;
                    max-width:220px;
                    flex-shrink:0;
                ">
                    <div style="
                        background:#ffffff;
                        border:3px solid #202020;
                        padding:14px 16px;
                        box-sizing:border-box;
                    ">
                        <div style="font-size:14px;font-weight:700;color:#222;margin-bottom:10px;">K-301</div>
                        <div style="font-size:11px;color:#555;">Температура верха</div>
                        <div style="font-size:18px;font-weight:800;color:#234c28;margin-bottom:6px;">{top_val:.2f} ℃</div>
                        <div style="font-size:11px;color:#555;">Температура низа</div>
                        <div style="font-size:11px;color:#555;">Гор. зона</div>
                        <div style="font-size:18px;font-weight:800;color:#234c28;margin-bottom:6px;">{hot_val:.2f} ℃</div>
                        <div style="font-size:11px;color:#555;">Хол. зона</div>
                        <div style="font-size:18px;font-weight:800;color:#234c28;">{cold_val:.2f} ℃</div>
                    </div>

                    <div style="
                        background:#ffffff;
                        border:3px solid #202020;
                        padding:14px 16px;
                        box-sizing:border-box;
                    ">
                        <div style="font-size:14px;font-weight:700;color:#222;margin-bottom:6px;">Масса КГД</div>
                        <div style="font-size:22px;font-weight:800;color:#245a2d;">{kgd_val:.2f}</div>
                        <div style="font-size:10px;color:#666;">тонн/час</div>
                    </div>

                    <div style="
                        background:#ffffff;
                        border:3px solid #202020;
                        padding:14px 16px;
                        box-sizing:border-box;
                    ">
                        <div style="font-size:14px;font-weight:700;color:#222;margin-bottom:6px;">Масса выхода</div>
                        <div style="font-size:11px;color:#555;">Вывод балансового избытка</div>
                        <div style="font-size:22px;font-weight:800;color:#245a2d;margin-bottom:4px;">{excess_val:.2f}</div>
                        <div style="font-size:10px;color:#666;">тонн/час</div>
                    </div>
                </div>
            </div>

            <div style="
                display:flex;
                justify-content:center;
                margin-top:20px;
            ">
                <div style="
                    background:#ffffff;
                    border:3px solid #202020;
                    padding:12px 18px;
                    box-sizing:border-box;
                    max-width:380px;
                    text-align:left;
                ">
                    <div style="font-size:14px;font-weight:700;color:#222;margin-bottom:4px;">Расчет</div>
                    <div style="font-size:11px;color:#555;line-height:1.4;">
                        Расчёт выполнен на основе введенных параметров сырья и режима работы колонны.
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    components.html(html, height=820, scrolling=False)


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

    medians = {}
    for feat, val in artifacts["temp_medians"].items():
        medians[feat] = val
    for feat, val in artifacts["press_medians"].items():
        medians.setdefault(feat, val)

    # Устанавливаем значение для массы КГД
    if KGD_MASS in medians:
        medians[KGD_MASS] = get_kgd_mass_median(KGD_MASS)

    # Убираем диапазоны и добавляем только нужные поля
    important_order = [
        METHANE_FEAT,
        ETHANE_FEAT,
        TOP_TEMP,
        HOT_ZONE,
        COLD_ZONE,
        EXCESS,
        KGD_MASS,
    ]

    ordered = [f for f in important_order if f in all_features or f in [METHANE_FEAT, ETHANE_FEAT]] + [
        f for f in all_features if f not in important_order and f not in REMOVED_FROM_UI
    ]

    with st.sidebar:
        st.subheader("Управление")
        st.write("Изменяйте параметры сырья и режима работы. Рекомендуемые параметры обновляются автоматически.")
        show_more = st.checkbox("Показать расширенный ввод", value=False)
        use_demo = st.button("Заполнить demo-значениями")

    selected_features = ordered[:8] if not show_more else ordered[:16]
    user_values = {}

    st.markdown("## Входные данные")
    st.caption("Во входах оставлены отдельные параметры сырья: содержание метана и содержание этана.")

    input_cols = st.columns(4)
    for i, feat in enumerate(selected_features):
        # Для метана и этана используем fallback значения
        if feat == METHANE_FEAT:
            fallback = 0.035
            lo, hi, med = -0.1, 0.5, fallback
        elif feat == ETHANE_FEAT:
            fallback = 0.035
            lo, hi, med = -0.1, 0.5, fallback
        else:
            fallback = medians.get(feat, 0.0)
            lo, hi, med = get_limits(ref_df, feat, fallback)
        
        value = med if use_demo else fallback
        step = max((hi - lo) / 100, 0.0001) if hi != lo else 0.01
        fmt = "%.4f" if abs(hi) < 10 else "%.2f"

        with input_cols[i % 4]:
            user_values[feat] = st.number_input(
                feat,
                value=float(value),
                step=float(step),
                format=fmt,
                key=f"input_{i}_{feat}",
            )

    temp_pred, press_pred = predict_values(artifacts, user_values)

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

    render_scheme_with_overlay(
        scheme_path=scheme_path,
        temp_pred=temp_pred,
        press_pred=press_pred,
        methane_val=user_values.get(METHANE_FEAT, 0.035),
        ethane_val=user_values.get(ETHANE_FEAT, 0.035),
        top_val=display_value(user_values, medians, TOP_TEMP),
        hot_val=display_value(user_values, medians, HOT_ZONE),
        cold_val=display_value(user_values, medians, COLD_ZONE),
        kgd_val=display_value(user_values, medians, KGD_MASS),
        excess_val=display_value(user_values, medians, EXCESS),
    )

    st.markdown("## Рекомендуемые параметры в рефлюксной емкости")
    m1, m2 = st.columns(2)
    with m1:
        metric_box("Температура, ", f"{temp_pred:.2f}", "℃", "#1c6d34", "#f8fcf8")
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
