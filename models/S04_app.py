import streamlit as st
import pickle
import xgboost as xgb
import pandas as pd

# -----------------------------
# Load the model
# -----------------------------
with open("xgb_spw3_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# -----------------------------
# Prediction function
# -----------------------------
def predict(input_data):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)
    return prediction, probability

# -----------------------------
# Streamlit interface
# -----------------------------
st.title("🛒 Online Purchasing Prediction")

st.sidebar.header("Parameters")

# -----------------------------
# Input form
# -----------------------------
def user_input_features():

    months = ['Aug','Dec','Feb','Jul','June','Mar','May','Nov','Oct','Sep']
    selected_month = st.sidebar.selectbox("Select Month", months, index=1)

    month_data = {f'month_{m}':0 for m in months}
    month_data[f'month_{selected_month}'] = 1

    visitor_types = ['New_Visitor','Other','Returning_Visitor']
    selected_visitor = st.sidebar.selectbox("Visitor Type", visitor_types)

    visitor_data = {
        f'visitor_type_{v}':1 if selected_visitor == v else 0
        for v in visitor_types
    }

    os_types = [f'os_{i}' for i in range(1,9)]
    selected_os = st.sidebar.selectbox("Operating System", os_types)

    os_data = {f'os_{i}':1 if selected_os == f'os_{i}' else 0 for i in range(1,9)}

    browsers = [f'browser_{i}' for i in range(1,14)]
    selected_browser = st.sidebar.selectbox("Browser", browsers)

    browser_data = {
        f'browser_{i}':1 if selected_browser == f'browser_{i}' else 0
        for i in range(1,14)
    }

    regions = [f'region_{i}' for i in range(1,10)]
    selected_region = st.sidebar.selectbox("Region", regions)

    region_data = {
        f'region_{i}':1 if selected_region == f'region_{i}' else 0
        for i in range(1,10)
    }

    traffic_types = [f'traffic_type_{i}' for i in range(1,21)]
    selected_traffic = st.sidebar.selectbox("Traffic Type", traffic_types)

    traffic_data = {
        f'traffic_type_{i}':1 if selected_traffic == f'traffic_type_{i}' else 0
        for i in range(1,21)
    }

    bounce_rate_pct = st.sidebar.slider("Bounce Rate (%)",0,20,0)
    exit_rate_pct = st.sidebar.slider("Exit Rate (%)",0,20,1)

    input_data = {

        'admin': st.sidebar.slider("Administrative pages visited",0,27,7),
        'admin_duration': st.sidebar.slider("Administrative duration",0,3400,139),

        'info': st.sidebar.slider("Informational pages visited",0,24,0),
        'info_duration': st.sidebar.slider("Informational duration",0,2600,0),

        'prod_related': st.sidebar.slider("Product pages viewed",0,705,30),
        'prod_related_duration': st.sidebar.slider("Product page duration",0,70000,986),

        'bounce_rate': bounce_rate_pct / 100,
        'exit_rate': exit_rate_pct / 100,

        'page_value': st.sidebar.slider("Page Value",0,362,36),

        'special_day': st.sidebar.selectbox("Special Day",[0,1]),
        'weekend': st.sidebar.selectbox("Weekend",[0,1]),

        **month_data,
        **visitor_data,
        **os_data,
        **browser_data,
        **region_data,
        **traffic_data
    }

    return input_data, selected_month

# -----------------------------
# Get user input
# -----------------------------
user_input, selected_month = user_input_features()

# -----------------------------
# Prediction
# -----------------------------
prediction, probability = predict(user_input)

st.subheader("Prediction")

if probability[0][1] >= 0.58:
    confidence = probability[0][1] * 100
    st.success(f"✅ Purchase predicted — Confidence: {confidence:.1f}%")
else:
    confidence = probability[0][0] * 100
    st.warning(f"❌ No purchase predicted — Confidence: {confidence:.1f}%")

# -----------------------------
# Confidence bar
# -----------------------------
st.subheader("Purchase Confidence")

st.progress(int(probability[0][1] * 100))
st.caption(f"Model confidence in purchase: {probability[0][1]*100:.1f}%")

# -----------------------------
# Session Profiling
# -----------------------------
st.subheader("📊 Session Profile")

prod_views = user_input['prod_related']
bounce = user_input['bounce_rate']
exit_rate = user_input['exit_rate']
page_value = user_input['page_value']

col1,col2,col3 = st.columns(3)

col1.metric("Products Viewed", prod_views)
col2.metric("Bounce Rate", f"{bounce*100:.1f}%")
col3.metric("Page Value", page_value)

st.divider()

# -----------------------------
# Behavioral Insights
# -----------------------------
st.subheader("🧠 Session Insights")

if prod_views > 40:
    st.info("Envoyez une réduction aux acheteurs qui aiment vos articles pour les aider à se décider.")

elif prod_views > 15:
    st.write("👀 Visitor is comparing several products.")

else:
    st.write("⚡ Visitor browsed only a few products.")

if bounce > 0.10:
    st.warning("High bounce rate detected.")

if exit_rate > 0.10:
    st.warning("User may leave soon — consider showing a promotion.")

if page_value > 50:
    st.success("💎 High value visitor.")

# -----------------------------
# Month highlight
# -----------------------------
st.subheader("📅 Month Highlight")

high_season = ["Nov","Dec"]

if selected_month in high_season:
    st.success(f"{selected_month} is a high conversion period (Black Friday / Christmas).")
else:
    st.write(f"Session occurred in **{selected_month}**.")