import streamlit as st
import joblib
import pandas as pd
import numpy as np

st.set_page_config(page_title="Деэтанизация: Прогноз E-301", layout="centered")
st.title("🏭 Цифровой двойник установки деэтанизации")
st.markdown("Введите параметры сырья и режима работы. Модель мгновенно рассчитает прогноз температуры в **E-301**.")

@st.cache_resource
def load_artifacts():
    model = joblib.load('model.pkl')
    features = joblib.load('features.pkl')
    return model, features

model, features = load_artifacts()

# Загружаем датасет для медианных значений по умолчанию
df = pd.read_csv('baza_final_clean.csv', sep=';', decimal=',')
medians = df[features].median()

# Определяем топ-10 самых важных признаков для быстрого ввода
importances = model.feature_importances_
imp_df = pd.DataFrame({'feature': features, 'importance': importances})
top_features = imp_df.sort_values('importance', ascending=False).head(10)['feature'].tolist()
other_features = [f for f in features if f not in top_features]

st.subheader("📥 Ключевые параметры")
inputs = {}
cols = st.columns(2)
for i, feat in enumerate(top_features):
    default = float(medians[feat])
    with cols[i % 2]:
        inputs[feat] = st.number_input(feat, value=default, format="%.3f")

with st.expander("🔧 Остальные параметры (заполнены медианными значениями)"):
    cols2 = st.columns(3)
    for i, feat in enumerate(other_features):
        default = float(medians[feat])
        with cols2[i % 3]:
            inputs[feat] = st.number_input(feat, value=default, format="%.3f")

if st.button("🔮 Рассчитать температуру", type="primary", use_container_width=True):
    input_df = pd.DataFrame([inputs])[features]  # строгий порядок признаков
    prediction = model.predict(input_df)[0]
    
    st.success(f"✅ Прогноз температуры в E-301: **{prediction:.2f} °C**")
    
    # Визуализация влияния параметров
    st.subheader("📊 Что сильнее всего повлияло на прогноз?")
    top_imp = imp_df.sort_values('importance', ascending=False).head(8)
    st.bar_chart(top_imp.set_index('feature')['importance'])
    
    st.caption("💡 Модель обучена на 68 промышленных пробах. R² = 0.96 на чистых данных.")