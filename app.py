import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Цифровой двойник деэтанизации", layout="centered")
st.title("🏭 Цифровой двойник процесса деэтанизации")
st.markdown(
    "Введите параметры сырья и режима работы колонного оборудования. "
    "Модель мгновенно рассчитает рекомендуемые параметры в рефлюксной емкости Е-301 "
    "для получения конденсата требуемого качества."
)

@st.cache_resource
def load_artifacts():
    with open('model_temp.pkl', 'rb') as f: model_temp = pickle.load(f)
    with open('model_press.pkl', 'rb') as f: model_press = pickle.load(f)
    with open('features.pkl', 'rb') as f: features = pickle.load(f)
    with open('medians.pkl', 'rb') as f: medians = pickle.load(f)
    return model_temp, model_press, features, medians

model_temp, model_press, features, medians = load_artifacts()

# Маппинг старых названий признаков на новые UI-лейблы
ui_labels = {
    'метан': 'Содержание метана в сырье, в массовых долях',
    'этан': 'Содержание этана в сырье, в массовых долях',
    'K-301 Температура верха': 'Температура верха колонны К-301, °C',
    'K-301 Масса рефлюксной жидкости': 'Масса рефлюксной жидкости колонны К-301, т/ч',
    'Масса КГД из куба колонны': 'Масса КГД, выводимого из куба колонны, т/ч'
}

# Фракционный состав (столбцы вида 40-50, 50-60...)
frac_cols = [c for c in features if '-' in c and c[0].isdigit()]

st.subheader("📥 Ключевые параметры")
inputs = {}

col1, col2 = st.columns(2)
with col1:
    inputs['метан'] = st.number_input(ui_labels['метан'], value=float(medians['метан']), format="%.3f")
    inputs['K-301 Температура верха'] = st.number_input(
        ui_labels['K-301 Температура верха'],
        value=float(medians['K-301 Температура верха']),
        max_value=97.0,
        format="%.2f",
        help="Технологическое ограничение: не более 97°C"
    )
    inputs['Масса КГД из куба колонны'] = st.number_input(
        ui_labels['Масса КГД из куба колонны'],
        value=float(medians['Масса КГД из куба колонны']),
        format="%.3f"
    )
with col2:
    inputs['этан'] = st.number_input(ui_labels['этан'], value=float(medians['этан']), format="%.3f")
    inputs['K-301 Масса рефлюксной жидкости'] = st.number_input(
        ui_labels['K-301 Масса рефлюксной жидкости'],
        value=float(medians['K-301 Масса рефлюксной жидкости']),
        format="%.3f"
    )

with st.expander("📊 Данные о компонентно-фракционном составе сырья, в масс. долях"):
    st.caption("💡 Значения по умолчанию заполнены медианами из исторических данных. Это означает, что если вы не меняете параметр, модель использует типичное (срединное) значение для вашей установки, чтобы расчёт был физически корректным.")
    cols = st.columns(3)
    for i, feat in enumerate(frac_cols):
        default = float(medians[feat])
        with cols[i % 3]:
            inputs[feat] = st.number_input(feat, value=default, format="%.3f")

# Заполняем скрытые параметры медианами или фиксированными значениями
hidden_features = [f for f in features if f not in inputs.keys()]
for feat in hidden_features:
    if feat in ['Метан+этан из жидкости в Е-301', 'Метан и этан в КГД']:
        inputs[feat] = 0.07  # Фиксация ≤ 0.08 по требованию технолога
    else:
        inputs[feat] = float(medians[feat])

if st.button("🔮 Рассчитать рекомендуемые параметры", type="primary", use_container_width=True):
    # Строгий порядок признаков, как при обучении
    input_df = pd.DataFrame([inputs])[features]

    pred_temp = model_temp.predict(input_df)[0]
    pred_press = model_press.predict(input_df)[0]

    st.success("✅ Расчёт завершён!")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("🌡️ Температура в рефлюксной емкости Е-301, °C", f"{pred_temp:.2f}")
    with c2:
        st.metric("📊 Давление в рефлюксной емкости Е-301, МПа", f"{pred_press:.3f}")

    st.info("💡 Модель работает в режиме соблюдения ограничений: содержание метана+этана в жидкости Е-301 и в КГД поддерживается на уровне ≤ 0.08 масс. долей.")
    st.caption("📌 Примечание: Параметры «Вывод балансового избытка», «Масса циркулирующей жидкости» и температуры зон К-301 зафиксированы на медианных значениях, так как в рамках данной модели они не являются варьируемыми управляющими воздействиями.")