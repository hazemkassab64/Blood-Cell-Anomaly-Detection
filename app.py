import streamlit as st
import requests
import joblib

# توسيع مساحة العرض في الصفحة
st.set_page_config(layout="wide")

st.title("Blood Cell Anomaly Detection 🩸")
st.write("أدخل بيانات الخلية للفحص:")

# 1. تحميل أسماء الأعمدة
try:
    model_columns = joblib.load('model_columns.pkl')
except FileNotFoundError:
    st.error("ملف model_columns.pkl غير موجود!")
    model_columns = []

# تصفير كل القيم في البداية
input_data = {col: 0.0 for col in model_columns}

# 2. البيانات الأساسية (قصاد بعض)
st.subheader("بيانات المريض الأساسية")
col_age, col_sex = st.columns(2)

with col_age:
    age_group_str = st.selectbox("العمر (Age Group)", ["Pediatric", "Adult", "Elderly"])
    if age_group_str == "Pediatric": input_data["patient_age_group"] = 0
    elif age_group_str == "Adult": input_data["patient_age_group"] = 1
    else: input_data["patient_age_group"] = 2

with col_sex:
    sex_str = st.selectbox("الجنس (Patient Sex)", ["Male", "Female"])
    input_data["patient_sex"] = 0 if sex_str == "Male" else 1

# 3. إعدادات الميكروسكوب بالأسماء الحقيقية
st.subheader("إعدادات الفحص والميكروسكوب")
microscope_options = ['Zeiss_Axio', 'Leica_DM2000', 'Olympus_BX51']
selected_microscope = st.selectbox("نوع الميكروسكوب المستخدم", microscope_options)

# الربط الذكي: لو اختار نوع له عمود هنحط 1، ولو اختار النوع المحذوف هيسيب الباقي 0
expected_microscope_col = f"microscope_model_{selected_microscope}"
if expected_microscope_col in input_data:
    input_data[expected_microscope_col] = 1.0

# 4. القياسات الرقمية (كل خانتين قصاد بعض)
st.subheader("القياسات الرقمية (Numerical Features)")
columns_to_skip = ["patient_age_group", "patient_sex"] + [col for col in model_columns if "microscope_model" in col or "dataset_source" in col or "staining_protocol" in col]

numerical_cols = [col for col in model_columns if col not in columns_to_skip]

# تقسيم الشاشة لعمودين للقياسات
col1, col2 = st.columns(2)
for i, col in enumerate(numerical_cols):
    # تجميل اسم العمود عشان يظهر بشكل أنيق للمستخدم
    display_name = col.replace("_", " ").title()
    
    if i % 2 == 0:
        with col1:
            input_data[col] = st.number_input(f"{display_name}", value=0.0, format="%.4f")
    else:
        with col2:
            input_data[col] = st.number_input(f"{display_name}", value=0.0, format="%.4f")

st.markdown("---")

# 5. زرار التوقع
if st.button("Predict (توقع النتيجة)", type="primary", use_container_width=True):
    api_url = "http://127.0.0.1:8000/predict"
    
    with st.spinner('جاري تحليل البيانات...'):
        response = requests.post(api_url, json={"data": input_data})
    
    if response.status_code == 200:
        result = response.json()["prediction"]
        if result == 1:
            st.error("⚠️ Anomaly Detected (يوجد شذوذ في الخلية)")
        else:
            st.success("✅ Normal Cell (خلية طبيعية)")
    else:
        st.error(f"حدث خطأ في الاتصال بالسيرفر! كود الخطأ: {response.status_code}")