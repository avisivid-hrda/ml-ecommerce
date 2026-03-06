import streamlit as st
import pickle
import pandas as pd
import xgboost as xgb

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="E-commerce Conversion Predictor",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-commerce Purchase Prediction Engine")
st.caption("AI powered session analysis & marketing insights")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

with open("xgb_spw3_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict(input_data):
    df = pd.DataFrame([input_data])
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    return prediction, probability

# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------

st.sidebar.header("⚙️ Session Configuration")

def user_input_features():

    months = ['Aug','Dec','Feb','Jul','June','Mar','May','Nov','Oct','Sep']
    selected_month = st.sidebar.selectbox("Month of visit", months)

    month_data = {f'month_{m}':0 for m in months}
    month_data[f'month_{selected_month}'] = 1

    visitor_types = ['New_Visitor','Other','Returning_Visitor']
    selected_visitor = st.sidebar.selectbox("Visitor type", visitor_types)

    visitor_data = {
        f'visitor_type_{v}':1 if selected_visitor == v else 0
        for v in visitor_types
    }

    os_types = [f'os_{i}' for i in range(1,9)]
    selected_os = st.sidebar.selectbox("Operating system", os_types)

    os_data = {f'os_{i}':1 if selected_os == f'os_{i}' else 0 for i in range(1,9)}

    browsers = [f'browser_{i}' for i in range(1,14)]
    selected_browser = st.sidebar.selectbox("Browser", browsers)

    browser_data = {f'browser_{i}':1 if selected_browser == f'browser_{i}' else 0 for i in range(1,14)}

    regions = [f'region_{i}' for i in range(1,10)]
    selected_region = st.sidebar.selectbox("Region", regions)

    region_data = {f'region_{i}':1 if selected_region == f'region_{i}' else 0 for i in range(1,10)}

    traffic_types = [f'traffic_type_{i}' for i in range(1,21)]
    selected_traffic = st.sidebar.selectbox("Traffic type", traffic_types)

    traffic_data = {
        f'traffic_type_{i}':1 if selected_traffic == f'traffic_type_{i}' else 0
        for i in range(1,21)
    }

    bounce = st.sidebar.slider("Bounce Rate (%)",0,20,2)
    exit_rate = st.sidebar.slider("Exit Rate (%)",0,20,3)

    input_data = {

        'admin': st.sidebar.slider("Administrative pages",0,27,7),
        'admin_duration': st.sidebar.slider("Administrative duration (sec)",0,3400,139),

        'info': st.sidebar.slider("Informational pages",0,24,0),
        'info_duration': st.sidebar.slider("Informational duration",0,2600,0),

        'prod_related': st.sidebar.slider("Product pages viewed",0,705,30),
        'prod_related_duration': st.sidebar.slider("Product browsing time",0,70000,986),

        'bounce_rate': bounce/100,
        'exit_rate': exit_rate/100,

        'page_value': st.sidebar.slider("Page value",0,362,36),

        'special_day': st.sidebar.selectbox("Special shopping day",[0,1]),
        'weekend': st.sidebar.selectbox("Weekend session",[0,1]),

        **month_data,
        **visitor_data,
        **os_data,
        **browser_data,
        **region_data,
        **traffic_data
    }

    return input_data, selected_month, selected_visitor, selected_browser, selected_os, selected_region, selected_traffic


user_input, month, visitor, browser, os, region, traffic = user_input_features()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

prediction, probability = predict(user_input)

purchase_prob = probability[0][1] * 100

st.divider()
st.subheader("🎯 Purchase Prediction")

col1, col2 = st.columns(2)

with col1:

    if purchase_prob >= 58:
        st.success(f"✅ Purchase Likely")
    else:
        st.warning("❌ Purchase Unlikely")

with col2:
    st.metric("Conversion probability", f"{purchase_prob:.1f}%")

st.progress(int(purchase_prob))

# --------------------------------------------------
# SESSION DASHBOARD
# --------------------------------------------------

st.divider()
st.header("📊 Session Intelligence Dashboard")

prod = user_input['prod_related']
prod_time = user_input['prod_related_duration']
bounce = user_input['bounce_rate']
exit_rate = user_input['exit_rate']
page_value = user_input['page_value']

k1,k2,k3,k4 = st.columns(4)

k1.metric("Products viewed", prod)
k2.metric("Browsing time", prod_time)
k3.metric("Bounce rate", f"{bounce*100:.1f}%")
k4.metric("Page value", page_value)

# --------------------------------------------------
# ENGAGEMENT ANALYSIS
# --------------------------------------------------

st.subheader("🛍 Product Engagement")

if prod > 60:
    st.success("🔥 Visitor is deeply exploring the catalog. Strong buying intent.")

elif prod > 30:
    st.info("💡 Envoyez une réduction aux acheteurs qui aiment vos articles pour les aider à se décider.")

elif prod > 10:
    st.write("👀 Visitor comparing multiple products.")

else:
    st.write("⚡ Quick browsing behaviour.")

# --------------------------------------------------
# TIME ANALYSIS
# --------------------------------------------------

st.subheader("⏱ Engagement Time")

if prod_time > 5000:
    st.success("Visitor is carefully reviewing products.")

elif prod_time > 1000:
    st.write("Moderate product exploration.")

else:
    st.write("Short browsing session.")

# --------------------------------------------------
# NAVIGATION BEHAVIOR
# --------------------------------------------------

st.subheader("🧭 Navigation Behavior")

if bounce > 0.12:
    st.warning("High bounce rate detected.")

elif bounce > 0.05:
    st.write("Moderate bounce behaviour.")

else:
    st.success("Strong engagement — user navigates multiple pages.")

if exit_rate > 0.12:
    st.warning("User likely to exit soon — consider exit promotion.")

# --------------------------------------------------
# VISITOR PROFILE
# --------------------------------------------------

st.subheader("👤 Visitor Profile")

if visitor == "Returning_Visitor":
    st.success("Returning customer — already familiar with the store.")

elif visitor == "New_Visitor":
    st.info("New visitor — onboarding offers could increase conversion.")

else:
    st.write("Unclassified visitor type.")

# --------------------------------------------------
# MARKETING CONTEXT
# --------------------------------------------------

st.subheader("📢 Acquisition Channel")

st.write(f"Traffic source: **{traffic}**")

if "traffic_type_2" in traffic:
    st.write("Traffic likely coming from search engines.")

# --------------------------------------------------
# TECHNICAL CONTEXT
# --------------------------------------------------

st.subheader("💻 Technical Environment")

t1,t2 = st.columns(2)

t1.write(f"Browser: **{browser}**")
t2.write(f"Operating System: **{os}**")

# --------------------------------------------------
# REGION
# --------------------------------------------------

st.subheader("🌍 Geographic Segment")

st.write(f"Visitor region: **{region}**")

# --------------------------------------------------
# SEASONALITY
# --------------------------------------------------

st.subheader("📅 Seasonal Context")

if month in ["Nov","Dec"]:
    st.success("Peak shopping season detected (Black Friday / Christmas).")

elif month in ["May","Oct"]:
    st.write("Moderate seasonal shopping activity.")

else:
    st.write("Regular shopping period.")

# --------------------------------------------------
# FINAL MARKETING RECOMMENDATION
# --------------------------------------------------

st.divider()
st.header("🎯 Marketing Recommendation Engine")

if purchase_prob > 70:
    st.success("Offer premium product recommendations or bundles.")

elif purchase_prob > 50:
    st.info("Offer limited-time discount to close the sale.")

elif purchase_prob > 30:
    st.write("Retarget visitor with personalized ads.")

else:
    st.warning("Focus on engagement: recommend popular products.")