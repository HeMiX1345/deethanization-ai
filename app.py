import joblib
from pathlib import Path
import pandas as pd
import streamlit as st

st.set_page_config(page_title='Цифровой двойник деэтанизации', page_icon='⚙️', layout='wide')

MODEL_DIR = Path('saved') if Path('saved').exists() else Path('.')
TEMP_MODEL_PATH = MODEL_DIR / 'model_temp.pkl'
PRESS_MODEL_PATH = MODEL_DIR / 'model_press.pkl'
TEMP_FEATURES_PATH = MODEL_DIR / 'features_temp.pkl'
PRESS_FEATURES_PATH = MODEL_DIR / 'features_press.pkl'
TEMP_MEDIANS_PATH = MODEL_DIR / 'medians_temp.pkl'
PRESS_MEDIANS_PATH = MODEL_DIR / 'medians_press.pkl'
DATA_CANDIDATES = [
    Path('preparing/dataset_E-301_Температура.csv'),
    Path('preparing/dataset_Давление_в_E-301.csv'),
    Path('preparing/dataset_Давление_в_Е-301.csv'),
    Path('final_validation.csv')
]


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Не найден файл: {path}')
    return joblib.load(path)


@st.cache_resource
def load_artifacts():
    return {
        'temp_model': load_pickle(TEMP_MODEL_PATH),
        'press_model': load_pickle(PRESS_MODEL_PATH),
        'temp_features': load_pickle(TEMP_FEATURES_PATH),
        'press_features': load_pickle(PRESS_FEATURES_PATH),
        'temp_medians': load_pickle(TEMP_MEDIANS_PATH),
        'press_medians': load_pickle(PRESS_MEDIANS_PATH),
    }


def find_reference_dataset():
    for path in DATA_CANDIDATES:
        if path.exists():
            try:
                df = pd.read_csv(path, sep=';', decimal='.', encoding='utf-8-sig')
                df.columns = [str(c).strip() for c in df.columns]
                for c in df.columns:
                    if df[c].dtype == object:
                        s = df[c].astype(str).str.replace(',', '.', regex=False)
                        num = pd.to_numeric(s, errors='coerce')
                        if num.notna().mean() >= 0.7:
                            df[c] = num
                return df
            except Exception:
                continue
    return None


def get_limits(df, feature, fallback_value):
    if df is not None and feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
        series = pd.to_numeric(df[feature], errors='coerce').dropna()
        if not series.empty:
            lo = float(series.quantile(0.05))
            hi = float(series.quantile(0.95))
            med = float(series.median())
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
    temp_X = build_input_frame(artifacts['temp_features'], artifacts['temp_medians'], user_values)
    press_X = build_input_frame(artifacts['press_features'], artifacts['press_medians'], user_values)
    temp_pred = float(artifacts['temp_model'].predict(temp_X)[0])
    press_pred = float(artifacts['press_model'].predict(press_X)[0])
    return temp_pred, press_pred


def metric_box(label, value, unit='', color='green'):
    st.markdown(
        f"""
        <div style='background:#ffffff;border:1px solid #d7ddd7;border-radius:10px;padding:8px 10px;margin-bottom:8px;'>
            <div style='font-size:11px;color:#727972;'>{label}</div>
            <div style='font-size:20px;font-weight:700;color:{color};'>{value} {unit}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def draw_column_unit(temp_pred, press_pred):
    st.markdown(
        f"""
        <div style='display:flex;justify-content:center;'>
            <div style='width:120px;background:#f9fbf8;border:1px solid #ccd7c9;border-radius:20px;padding:12px;text-align:center;'>
                <div style='margin:0 auto;width:74px;height:190px;border:4px solid #7e907d;border-radius:36px;background:linear-gradient(180deg,#fcfefc 0%,#e2ebe0 100%);position:relative;'>
                    <div style='position:absolute;left:14px;right:14px;top:22px;height:5px;background:#91a391;border-radius:10px;'></div>
                    <div style='position:absolute;left:14px;right:14px;top:58px;height:5px;background:#91a391;border-radius:10px;'></div>
                    <div style='position:absolute;left:14px;right:14px;top:94px;height:5px;background:#91a391;border-radius:10px;'></div>
                    <div style='position:absolute;left:14px;right:14px;top:130px;height:5px;background:#91a391;border-radius:10px;'></div>
                    <div style='position:absolute;top:-16px;left:27px;width:18px;height:16px;background:#7e907d;border-radius:8px 8px 0 0;'></div>
                    <div style='position:absolute;bottom:-16px;left:27px;width:18px;height:16px;background:#7e907d;border-radius:0 0 8px 8px;'></div>
                </div>
                <div style='font-size:16px;font-weight:700;color:#2f3f2f;margin-top:10px;'>E-301</div>
                <div style='font-size:12px;color:#708070;margin-top:2px;'>колонна</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    metric_box('Прогноз температуры', f'{temp_pred:.2f}', '°C', '#198754')
    metric_box('Прогноз давления', f'{press_pred:.2f}', '', '#0d6efd')


def main():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 1450px;}
        .stNumberInput label {font-weight: 500;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title('Цифровой двойник деэтанизации')
    st.caption('Сделано в формате простой мнемосхемы: только ключевые элементы и понятные прогнозные значения.')

    try:
        artifacts = load_artifacts()
    except Exception as e:
        st.error(f'Ошибка загрузки артефактов: {e}')
        st.stop()

    ref_df = find_reference_dataset()
    temp_features = artifacts['temp_features']
    press_features = artifacts['press_features']

    medians = {}
    for feat, val in artifacts['temp_medians'].items():
        medians[feat] = val
    for feat, val in artifacts['press_medians'].items():
        medians.setdefault(feat, val)

    important_order = [
        'K-301 Температура верха',
        'K-301 Темп-ра в гор. зоне',
        'K-301 Темп-ра в хол. зоне',
        'Метан+этан из жидкости в Е-301',
        'Вывод балансового избытка',
        'Масса КГД из куба колонны',
        '40-50', '50-60', '60-70', '90-100', '100-110', '110-120'
    ]

    all_features = list(dict.fromkeys(temp_features + press_features))
    ordered = [f for f in important_order if f in all_features] + [f for f in all_features if f not in important_order]

    with st.sidebar:
        st.subheader('Управление')
        st.write('Изменяйте ключевые параметры процесса. Прогноз обновляется автоматически.')
        show_more = st.checkbox('Показать расширенный ввод', value=False)
        use_demo = st.button('Заполнить demo-значениями')

    selected_features = ordered[:8] if not show_more else ordered[:16]
    user_values = {}

    st.markdown('### Параметры процесса')
    c1, c2 = st.columns(2)
    for i, feat in enumerate(selected_features):
        fallback = medians.get(feat, 0.0)
        lo, hi, med = get_limits(ref_df, feat, fallback)
        value = med if use_demo else fallback
        step = max((hi - lo) / 100, 0.01)
        with (c1 if i % 2 == 0 else c2):
            user_values[feat] = st.number_input(
                feat,
                min_value=float(lo),
                max_value=float(hi),
                value=float(value),
                step=float(step),
                format='%.4f'
            )

    temp_pred, press_pred = predict_values(artifacts, user_values)

    st.markdown('### Визуализация установки')
    st.markdown("""
    <div style='background:#eef2ed;border:1px solid #c9d2c7;border-radius:18px;padding:16px;'>
        <div style='font-size:18px;font-weight:700;color:#243224;margin-bottom:8px;'>Схема узла деэтанизации</div>
    </div>
    """, unsafe_allow_html=True)

    left, center, right = st.columns([1.2, 0.9, 1.2])

    with left:
        st.markdown('#### Подача / K-301')
        metric_box('Температура верха', f"{user_values.get('K-301 Температура верха', medians.get('K-301 Температура верха', 0.0)):.2f}", '°C', '#2e7d32')
        metric_box('Горячая зона', f"{user_values.get('K-301 Темп-ра в гор. зоне', medians.get('K-301 Темп-ра в гор. зоне', 0.0)):.2f}", '°C', '#2e7d32')
        metric_box('Балансовый избыток', f"{user_values.get('Вывод балансового избытка', medians.get('Вывод балансового избытка', 0.0)):.2f}", '', '#2e7d32')

    with center:
        draw_column_unit(temp_pred, press_pred)

    with right:
        st.markdown('#### Выходные потоки')
        metric_box('Метан+этан', f"{user_values.get('Метан+этан из жидкости в Е-301', medians.get('Метан+этан из жидкости в Е-301', 0.0)):.2f}", '', '#2e7d32')
        metric_box('Масса КГД', f"{user_values.get('Масса КГД из куба колонны', medians.get('Масса КГД из куба колонны', 0.0)):.2f}", '', '#2e7d32')
        metric_box('Состояние', 'расчет', '', '#2e7d32')

    st.markdown('### Сводка')
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric('Температура E-301', f'{temp_pred:.2f}')
    with m2:
        st.metric('Давление E-301', f'{press_pred:.2f}')
    with m3:
        st.metric('Активных параметров', len(selected_features))

    with st.expander('Показать входные значения для модели'):
        preview = pd.DataFrame({
            'feature': all_features,
            'value': [user_values.get(f, medians.get(f, 0.0)) for f in all_features]
        })
        st.dataframe(preview, use_container_width=True, hide_index=True)


if __name__ == '__main__':
    main()
