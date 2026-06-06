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


def equipment_block(title, metrics, border_color='#d5dfd2'):
    chips = ''
    for label, value, unit, color in metrics:
        chips += f"""
        <div style='display:inline-block;background:#ffffff;border:1px solid #d8ddd8;border-radius:8px;padding:8px 10px;margin:4px;min-width:112px;'>
            <div style='font-size:11px;color:#707870;'>{label}</div>
            <div style='font-size:18px;font-weight:700;color:{color};'>{value} {unit}</div>
        </div>
        """
    return f"""
    <div style='flex:1;min-width:240px;background:#f7faf7;border-radius:16px;padding:16px;border:1px solid {border_color};'>
        <div style='font-size:14px;font-weight:700;color:#304130;margin-bottom:10px;'>{title}</div>
        <div>{chips}</div>
    </div>
    """


def installation_view(temp_pred, press_pred, selected_values):
    feed_temp = selected_values.get('K-301 Температура верха', selected_values.get('K-301 Темп-ра в хол. зоне', 0.0))
    hot_zone = selected_values.get('K-301 Темп-ра в гор. зоне', 0.0)
    methane = selected_values.get('Метан+этан из жидкости в Е-301', 0.0)
    balance = selected_values.get('Вывод балансового избытка', 0.0)
    kgd = selected_values.get('Масса КГД из куба колонны', 0.0)

    left_block = equipment_block('Подача / контур K-301', [
        ('Температура верха', f'{feed_temp:.2f}', '°C', '#2e7d32'),
        ('Горячая зона', f'{hot_zone:.2f}', '°C', '#2e7d32'),
        ('Балансовый избыток', f'{balance:.2f}', '', '#2e7d32'),
    ])

    center_block = f"""
    <div style='width:280px;background:#f9fbf8;border-radius:22px;padding:16px 12px;border:1px solid #ccd7c9;text-align:center;'>
        <div style='margin:0 auto 10px auto;width:92px;height:220px;border:4px solid #738873;border-radius:42px;background:linear-gradient(180deg,#fbfdfb 0%,#dfe9dd 100%);position:relative;'>
            <div style='position:absolute;left:16px;right:16px;top:24px;height:6px;background:#8ea28f;border-radius:10px;'></div>
            <div style='position:absolute;left:16px;right:16px;top:74px;height:6px;background:#8ea28f;border-radius:10px;'></div>
            <div style='position:absolute;left:16px;right:16px;top:124px;height:6px;background:#8ea28f;border-radius:10px;'></div>
            <div style='position:absolute;left:16px;right:16px;top:174px;height:6px;background:#8ea28f;border-radius:10px;'></div>
            <div style='position:absolute;top:-18px;left:36px;width:20px;height:18px;background:#738873;border-radius:8px 8px 0 0;'></div>
            <div style='position:absolute;bottom:-18px;left:36px;width:20px;height:18px;background:#738873;border-radius:0 0 8px 8px;'></div>
        </div>
        <div style='font-size:18px;font-weight:700;color:#253425;'>Колонна E-301</div>
        <div style='display:flex;justify-content:center;gap:8px;flex-wrap:wrap;margin-top:10px;'>
            <div style='display:inline-block;background:#ffffff;border:1px solid #d8ddd8;border-radius:8px;padding:8px 10px;min-width:112px;'>
                <div style='font-size:11px;color:#707870;'>Прогноз температуры</div>
                <div style='font-size:18px;font-weight:700;color:#198754;'>{temp_pred:.2f} °C</div>
            </div>
            <div style='display:inline-block;background:#ffffff;border:1px solid #d8ddd8;border-radius:8px;padding:8px 10px;min-width:112px;'>
                <div style='font-size:11px;color:#707870;'>Прогноз давления</div>
                <div style='font-size:18px;font-weight:700;color:#0d6efd;'>{press_pred:.2f}</div>
            </div>
        </div>
    </div>
    """

    right_block = equipment_block('Выходные потоки', [
        ('Метан+этан', f'{methane:.2f}', '', '#2e7d32'),
        ('Масса КГД', f'{kgd:.2f}', '', '#2e7d32'),
        ('Состояние', 'расчет', '', '#2e7d32'),
    ])

    html = f"""
    <div style='background:#eef2ed;border:1px solid #c9d2c7;border-radius:18px;padding:18px 18px 14px 18px;'>
        <div style='font-size:18px;font-weight:700;color:#243224;margin-bottom:12px;'>Схема узла деэтанизации</div>
        <div style='display:flex;align-items:center;justify-content:space-between;gap:18px;flex-wrap:wrap;'>
            {left_block}
            <div style='width:60px;text-align:center;font-size:34px;color:#5f7661;'>⟶</div>
            {center_block}
            <div style='width:60px;text-align:center;font-size:34px;color:#5f7661;'>⟶</div>
            {right_block}
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def main():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1rem; padding-bottom: 1rem; max-width: 1500px;}
        .stNumberInput label {font-weight: 500;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title('Цифровой двойник деэтанизации')
    st.caption('Упрощенная технологическая схема с ключевыми параметрами установки и прогнозом для E-301.')

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
        st.write('Изменяйте ключевые параметры процесса. Визуализация обновляется автоматически.')
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
    installation_view(temp_pred, press_pred, user_values)

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
