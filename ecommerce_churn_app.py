# import library
import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Judul Utama
st.title('Customer Churn Predictor')
st.text('This web predicts the likelihood of a customer churning')

# Menambahkan sidebar
st.sidebar.header("Please input customer features")

def create_user_input():
    Tenure = st.sidebar.slider('Tenure (in months)', 0, 61, 12)
    WarehouseToHome = st.sidebar.slider('Warehouse To Home (in km)', 5, 126, 20)
    HourSpendOnApp = st.sidebar.slider('Hour Spent on App (per day)', 0.0, 5.0, 1.0)
    NumberOfDeviceRegistered = st.sidebar.slider('Number of Devices Registered', 1, 6, 2)
    OrderAmountHikeFromlastYear = st.sidebar.slider('Order Amount Hike From Last Year (%)', 11, 26, 15)
    CouponUsed = st.sidebar.slider('Coupons Used', 0, 16, 3)
    OrderCount = st.sidebar.slider('Order Count', 1, 16, 5)
    DaySinceLastOrder = st.sidebar.slider('Days Since Last Order', 0, 46, 10)
    CashbackAmount = st.sidebar.slider('Cashback Amount', 0.0, 324.99, 50.0)

    PreferredLoginDevice = st.sidebar.selectbox('Preferred Login Device', ['Mobile Phone', 'Computer'])
    CityTier = st.sidebar.selectbox('City Tier', ['1', '2', '3'])
    PreferredPaymentMode = st.sidebar.selectbox('Preferred Payment Mode', ['Debit Card', 'Credit Card', 'E wallet', 'Cash on Delivery', 'UPI'])
    Gender = st.sidebar.selectbox('Gender', ['Female', 'Male'])
    PreferedOrderCat = st.sidebar.selectbox('Preferred Order Category', ['Mobile Phone', 'Laptop & Accessory', 'Grocery', 'Fashion', 'Others'])
    SatisfactionScore = st.sidebar.selectbox('Satisfaction Score', ['1', '2', '3', '4', '5'])
    MaritalStatus = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    Complain = st.sidebar.radio('Customer Complaint?', [0, 1])
    CountOfAddress = st.sidebar.selectbox('Count of Address', ['1–2', '3', '4–6', '7+'])


    user_data = {
        'Tenure': Tenure,
        'WarehouseToHome': WarehouseToHome,
        'HourSpendOnApp': HourSpendOnApp,
        'NumberOfDeviceRegistered': NumberOfDeviceRegistered,
        'OrderAmountHikeFromlastYear': OrderAmountHikeFromlastYear,
        'CouponUsed': CouponUsed,
        'OrderCount': OrderCount,
        'DaySinceLastOrder': DaySinceLastOrder,
        'CashbackAmount': CashbackAmount,
        'PreferredLoginDevice': PreferredLoginDevice,
        'CityTier': CityTier,
        'PreferredPaymentMode': PreferredPaymentMode,
        'Gender': Gender,
        'PreferedOrderCat': PreferedOrderCat,
        'SatisfactionScore': SatisfactionScore,
        'MaritalStatus': MaritalStatus,
        'Complain': Complain,
        'CountOfAddress': CountOfAddress
    }

    return pd.DataFrame([user_data])

# Ambil data dari input user
data_customer = create_user_input()

# Load model
with open('final_tuned_lightgbm_ros_selectkbest.pkl', 'rb') as f:
    model_loaded = pickle.load(f)

# Cek apakah data input sesuai urutan kolom model
try:
    # (Optional) urutkan kolom input jika model punya .feature_names_in_
    data_customer = data_customer[model_loaded.feature_names_in_]
except AttributeError:
    pass  # model tidak menyimpan nama kolom fitur, lewati saja

# Tampilkan input user
st.subheader("Customer's Features")
st.write(data_customer.transpose())

# Prediksi churn
try:
    kelas = model_loaded.predict(data_customer)
    prob = model_loaded.predict_proba(data_customer)[0]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# Tampilkan hasil prediksi
st.subheader("Prediction Result")
if kelas[0] == 1:
    st.success("Churn: **Yes** – This customer is likely to churn.")
else:
    st.info("Churn: **No** – This customer is likely to stay.")

st.write(f"Probability of Churn: **{prob[1]:.2f}**")
