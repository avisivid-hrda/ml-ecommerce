import streamlit as st
import pickle
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="E-commerce Conversion Dashboard",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-commerce Conversion Intelligence")
st.caption("AI-driven session analysis for marketing teams")

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

with open("xgb_spw3_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# --------------------------------------------------
# FEATURE MAX VALUES (from dataset)
# --------------------------------------------------

feature_max = {
    "admin": 27,
    "admin_duration": 3400,
    "info": 24,
    "info_duration": 2600,
    "prod_related": 705,
    "prod_related_duration": 70000,
    "bounce_rate": 0.20,
    "exit_rate": 0.20,
    "page_value": 362
}

# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict(data):
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    probability = model.predict_proba(df)
    return prediction, probability


def percent_of_max(feature, value):
    max_val = feature_max[feature]
    return (value / max_val) * 100


# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------

st.sidebar.header("Session Parameters")

def user_input():

    months = ['Aug','Dec','Feb','Jul','June','Mar','May','Nov','Oct','Sep']
    month = st.sidebar.selectbox("Month", months)

    month_data = {f"month_{m}":0 for m in months}
    month_data[f"month_{month}"] = 1

    visitors = ['New_Visitor','Other','Returning_Visitor']
    visitor = st.sidebar.selectbox("Visitor type", visitors)

    visitor_data = {
        f'visitor_type_{v}':1 if v==visitor else 0
        for v in visitors
    }

    os_types = [f"os_{i}" for i in range(1,9)]
    os = st.sidebar.selectbox("Operating system", os_types)

    os_data = {f"os_{i}":1 if os==f"os_{i}" else 0 for i in range(1,9)}

    browsers = [f"browser_{i}" for i in range(1,14)]
    browser = st.sidebar.selectbox("Browser", browsers)

    browser_data = {f"browser_{i}":1 if browser==f"browser_{i}" else 0 for i in range(1,14)}

    regions = [f"region_{i}" for i in range(1,10)]
    region = st.sidebar.selectbox("Region", regions)

    region_data = {f"region_{i}":1 if region==f"region_{i}" else 0 for i in range(1,10)}

    traffic_types = [f"traffic_type_{i}" for i in range(1,21)]
    traffic = st.sidebar.selectbox("Traffic source", traffic_types)

    traffic_data = {
        f"traffic_type_{i}":1 if traffic==f"traffic_type_{i}" else 0
        for i in range(1,21)
    }

    bounce = st.sidebar.slider("Bounce rate (%)",0,20,3)
    exit_rate = st.sidebar.slider("Exit rate (%)",0,20,4)

    data = {

        'admin': st.sidebar.slider("Admin pages",0,27,7),
        'admin_duration': st.sidebar.slider("Admin duration",0,3400,139),

        'info': st.sidebar.slider("Info pages",0,24,0),
        'info_duration': st.sidebar.slider("Info duration",0,2600,0),

        'prod_related': st.sidebar.slider("Product pages viewed",0,705,30),
        'prod_related_duration': st.sidebar.slider("Product browsing time",0,70000,986),

        'bounce_rate': bounce/100,
        'exit_rate': exit_rate/100,

        'page_value': st.sidebar.slider("Page value",0,362,36),

        'special_day': st.sidebar.selectbox("Special day",[0,1]),
        'weekend': st.sidebar.selectbox("Weekend",[0,1]),

        **month_data,
        **visitor_data,
        **os_data,
        **browser_data,
        **region_data,
        **traffic_data
    }

    return data, month, visitor, browser, os, region, traffic


data, month, visitor, browser, os, region, traffic = user_input()

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

prediction, prob = predict(data)
purchase_prob = prob[0][1] * 100

st.divider()
st.header("📈 Conversion Prediction")

col1,col2 = st.columns([1,2])

with col1:

    if purchase_prob > 60:
        st.success("High purchase probability")

    elif purchase_prob > 40:
        st.info("Medium purchase probability")

    else:
        st.warning("Low purchase probability")

with col2:
    st.metric("Conversion probability", f"{purchase_prob:.1f}%")

st.progress(int(purchase_prob))

# --------------------------------------------------
# SESSION OVERVIEW
# --------------------------------------------------

st.divider()
st.header("📊 Session Overview")

k1,k2,k3,k4 = st.columns(4)

k1.metric("Products viewed", data["prod_related"])
k2.metric("Browsing time", data["prod_related_duration"])
k3.metric("Bounce rate", f"{data['bounce_rate']*100:.1f}%")
k4.metric("Page value", data["page_value"])

# --------------------------------------------------
# FEATURE ANALYTICS
# --------------------------------------------------

st.divider()
st.header("📊 Feature Performance Analysis")

features_to_analyze = [
    "admin",
    "admin_duration",
    "prod_related",
    "prod_related_duration",
    "bounce_rate",
    "exit_rate",
    "page_value"
]

feature_data = []

for f in features_to_analyze:

    value = data[f]
    percent = percent_of_max(f, value)

    temp = data.copy()
    temp[f] = feature_max[f]

    _, high_prob = predict(temp)

    conversion_effect = high_prob[0][1] * 100

    feature_data.append({
        "Feature": f,
        "Value": value,
        "% of Max": percent,
        "Potential Conversion %": conversion_effect
    })

df_features = pd.DataFrame(feature_data)

# --------------------------------------------------
# PLOT INPUT VS MAX
# --------------------------------------------------

fig1 = px.bar(
    df_features,
    x="Feature",
    y="% of Max",
    title="Input Value vs Dataset Maximum (%)"
)

st.plotly_chart(fig1, use_container_width=True)

# --------------------------------------------------
# PLOT FEATURE CONVERSION IMPACT
# --------------------------------------------------

fig2 = px.bar(
    df_features,
    x="Feature",
    y="Potential Conversion %",
    title="Conversion Probability if Feature is Maximized"
)

st.plotly_chart(fig2, use_container_width=True)

# --------------------------------------------------
# TABLE VIEW
# --------------------------------------------------

st.subheader("Feature Impact Table")

st.dataframe(
    df_features.style.format({
        "% of Max": "{:.1f}%",
        "Potential Conversion %": "{:.1f}%"
    })
)