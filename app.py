import base64
from pathlib import Path
import streamlit.components.v1 as components

def img_to_base64(path):
    data = Path(path).read_bytes()
    return base64.b64encode(data).decode()

def render_scheme_with_overlay(temp_pred, press_pred, kgd_val, excess_val):
    img_b64 = img_to_base64("7-3.jpg")   # или путь к вашему файлу схемы

    html = f"""
    <div style="position:relative;width:100%;max-width:980px;margin:auto;">
        <img src="data:image/jpeg;base64,{img_b64}" style="width:100%;display:block;border-radius:18px;" />

        <div style="position:absolute; left:68%; top:18%; width:180px;
                    background:#ffffffee; border:1px solid #d7ddd7; border-radius:14px;
                    padding:10px 12px; box-shadow:0 4px 10px rgba(0,0,0,0.08);">
            <div style="font-size:13px;font-weight:700;color:#304530;margin-bottom:8px;">Рекомендуемые параметры</div>
            <div style="font-size:12px;color:#6d776d;">Температура, ℃</div>
            <div style="font-size:22px;font-weight:800;color:#1c6d34;">{temp_pred:.2f}</div>
            <div style="font-size:12px;color:#6d776d;margin-top:6px;">Давление, МПа</div>
            <div style="font-size:22px;font-weight:800;color:#1f5fbf;">{press_pred:.3f}</div>
        </div>

        <div style="position:absolute; left:48%; top:73%; width:150px;
                    background:#ffffffee; border:1px solid #d7ddd7; border-radius:12px;
                    padding:8px 10px;">
            <div style="font-size:11px;color:#6d776d;">Масса КГД</div>
            <div style="font-size:20px;font-weight:800;color:#245a2d;">{kgd_val:.2f}</div>
        </div>

        <div style="position:absolute; left:77%; top:70%; width:170px;
                    background:#ffffffee; border:1px solid #d7ddd7; border-radius:12px;
                    padding:8px 10px;">
            <div style="font-size:11px;color:#6d776d;">Вывод балансового избытка</div>
            <div style="font-size:20px;font-weight:800;color:#245a2d;">{excess_val:.2f}</div>
        </div>
    </div>
    """
    components.html(html, height=720, scrolling=False)
