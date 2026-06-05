import json
import unicodedata
from pathlib import Path

import joblib
import numpy as np
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
SAMPLE_COL = 'sample_id'


def load_pickle(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'Не найден файл: {path}')
    return joblib.load(path)


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


def build_input_frame(features, medians, user_values):
    row = {}
    for feat in features:
        row[feat] = user_values.get(feat, medians.get(feat, 0.0))
    return pd.DataFrame([row])


def get_feature_limits(df, feature, fallback_value):
    if df is not None and feature in df.columns and pd.api.types.is_numeric_dtype(df[feature]):
        series = pd.to_numeric(df[feature], errors='coerce').dropna()
        if not series.empty:
            q1 = float(series.quantile(0.05))
            q9 = float(series.quantile(0.95))
            if q1 == q9:
                q1 = float(series.min())
                q9 = float(series.max())
            if q1 == q9:
                q1 -= 1.0
                q9 += 1.0
            return q1, q9, float(series.median())
    val = float(fallback_value) if pd.notna(fallback_value) else 0.0
    return val - abs(val) * 0.3 - 1, val + abs(val) * 0.3 + 1, val


def format_delta(pred, ref):
    delta = pred - ref
    sign = '+' if delta >= 0 else ''
    return f'{sign}{delta:.2f}'


def section_card(title, value, subtitle, color):
    html = f"""
    <div style='background:{color};padding:16px 18px;border-radius:14px;border:1px solid rgba(0,0,0,0.08);'>
        <div style='font-size:14px;color:#5b5b5b;margin-bottom:6px;'>{title}</div>
        <div style='font-size:30px;font-weight:700;color:#1f1f1f;line-height:1.1;'>{value}</div>
        <div style='font-size:13px;color:#6b6b6b;margin-top:6px;'>{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def main():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.2rem; padding-bottom: 1rem; max-width: 1400px;}
        .stNumberInput label {font-weight: 500;}
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('Цифровой двойник деэтанизации')
    st.caption('Простой и понятный интерфейс для прогноза температуры и давления в E-301. Стиль ориентирован на спокойную операторскую панель.')

    try:
        artifacts = load_artifacts()
    except Exception as e:
        st.error(f'Ошибка загрузки моделей: {e}')
        st.stop()

    ref_df = find_reference_dataset()
    temp_features = artifacts['temp_features']
    press_features = artifacts['press_features']
    temp_medians = artifacts['temp_medians']
    press_medians = artifacts['press_medians']

    common_features = []
    seen = set()
    for feat in temp_features + press_features:
        if feat not in seen:
            seen.add(feat)
            common_features.append(feat)

    with st.sidebar:
        st.subheader('Параметры')
        st.write('Введите основные параметры процесса. Остальные признаки будут заполнены автоматически.')
        show_all = st.checkbox('Показать все признаки', value=False)
        use_demo = st.button('Подставить demo-значения')

    defaults = {}
    for feat in common_features:
        defaults[feat] = temp_medians.get(feat, press_medians.get(feat, 0.0))

    if use_demo and ref_df is not None:
        for feat in common_features:
            if feat in ref_df.columns and pd.api.types.is_numeric_dtype(ref_df[feat]):
                defaults[feat] = float(pd.to_numeric(ref_df[feat], errors='coerce').median())

    user_values = {}
    editable_features = common_features if show_all else common_features[:10]

    st.markdown('### Ввод данных')
    col1, col2 = st.columns(2)
    for i, feat in enumerate(editable_features):
        fallback = defaults.get(feat, 0.0)
        min_v, max_v, med_v = get_feature_limits(ref_df, feat, fallback)
        step = max((max_v - min_v) / 100, 0.01)
        target_col = col1 if i % 2 == 0 else col2
        with target_col:
            user_values[feat] = st.number_input(
                feat,
                min_value=float(min_v),
                max_value=float(max_v),
                value=float(med_v if use_demo else fallback),
                step=float(step),
                format='%.4f'
            )

    temp_X = build_input_frame(temp_features, temp_medians, user_values)
    press_X = build_input_frame(press_features, press_medians, user_values)

    temp_pred = float(artifacts['temp_model'].predict(temp_X)[0])
    press_pred = float(artifacts['press_model'].predict(press_X)[0])

    temp_ref = float(np.mean(list(temp_medians.values()))) if temp_medians else temp_pred
    press_ref = float(np.mean(list(press_medians.values()))) if press_medians else press_pred

    st.markdown('### Результаты прогноза')
    r1, r2 = st.columns(2)
    with r1:
        section_card('Температура E-301', f'{temp_pred:.2f}', f'Отклонение от условной базы: {format_delta(temp_pred, temp_ref)}', '#e8f3eb')
    with r2:
        section_card('Давление E-301', f'{press_pred:.2f}', f'Отклонение от условной базы: {format_delta(press_pred, press_ref)}', '#edf3fb')

    st.markdown('### Интерпретация')
    i1, i2 = st.columns(2)
    with i1:
        if temp_pred >= temp_ref:
            st.info('Прогнозная температура выше условного среднего уровня.')
        else:
            st.success('Прогнозная температура находится в спокойной зоне относительно условной базы.')
    with i2:
        if press_pred >= press_ref:
            st.info('Прогнозное давление выше условного среднего уровня.')
        else:
            st.success('Прогнозное давление находится в спокойной зоне относительно условной базы.')

    with st.expander('Показать входной вектор модели'):
        preview_df = pd.DataFrame({
            'feature': common_features,
            'value': [user_values.get(f, defaults.get(f, 0.0)) for f in common_features]
        })
        st.dataframe(preview_df, use_container_width=True, hide_index=True)

    st.markdown('### О приложении')
    st.write('Визуализация сделана в простом и user-friendly стиле: светлый фон, спокойные карточки результатов, минимум перегрузки и быстрый сценарий работы для демонстрации.')


if __name__ == '__main__':
    main()
