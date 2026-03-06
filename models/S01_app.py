import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load the model
# -----------------------------
with open("xgb_spw3_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

MODEL_FEATURES = model.get_booster().feature_names


# -----------------------------
# Prediction function
# -----------------------------
def predict(input_data):

    df = pd.DataFrame([input_data])

    # ensure correct feature order
    df = df.reindex(columns=MODEL_FEATURES, fill_value=0)

    prediction = model.predict(df)
    probability = model.predict_proba(df)

    return prediction, probability


# -----------------------------
# Streamlit UI
# -----------------------------
st.title('Online Purchasing Prediction')

st.sidebar.header('Parameters')


# -----------------------------
# Input Form
# -----------------------------
def user_input_features():

    months = ['Aug','Dec','Feb','Jul','June','Mar','May','Nov','Oct','Sep']
    selected_month = st.sidebar.selectbox('Select Month', months)

    month_data = {f'month_{m}':0 for m in months}
    month_data[f'month_{selected_month}'] = 1


    visitor_types = ['New_Visitor','Other','Returning_Visitor']
    selected_visitor = st.sidebar.selectbox('Select Visitor Type', visitor_types)

    visitor_data = {f'visitor_type_{v}':1 if v==selected_visitor else 0 for v in visitor_types}


    os_types = [f'os_{i}' for i in range(1,9)]
    selected_os = st.sidebar.selectbox('Select OS', os_types)

    os_data = {f'os_{i}':1 if selected_os == f'os_{i}' else 0 for i in range(1,9)}


    browsers = [f'browser_{i}' for i in range(1,14)]
    selected_browser = st.sidebar.selectbox('Select Browser', browsers)

    browser_data = {f'browser_{i}':1 if selected_browser == f'browser_{i}' else 0 for i in range(1,14)}


    regions = [f'region_{i}' for i in range(1,10)]
    selected_region = st.sidebar.selectbox('Select Region', regions)

    region_data = {f'region_{i}':1 if selected_region == f'region_{i}' else 0 for i in range(1,10)}


    traffic_types = [f'traffic_type_{i}' for i in range(1,21)]
    selected_traffic = st.sidebar.selectbox('Select Traffic Type', traffic_types)

    traffic_data = {f'traffic_type_{i}':1 if selected_traffic == f'traffic_type_{i}' else 0 for i in range(1,21)}


    bounce_rate_pct = st.sidebar.slider('Bounce Rate (%)',0,20,0)
    exit_rate_pct = st.sidebar.slider('Exit Rate (%)',0,20,1)


    input_data = {

        'admin': st.sidebar.slider('Administrative pages visited',0,27,7),
        'admin_duration': st.sidebar.slider('Administrative duration',0,3400,139),

        'info': st.sidebar.slider('Informational pages visited',0,24,0),
        'info_duration': st.sidebar.slider('Informational duration',0,2600,0),

        'prod_related': st.sidebar.slider('Product pages visited',0,705,30),
        'prod_related_duration': st.sidebar.slider('Product pages duration',0,70000,986),

        'bounce_rate': bounce_rate_pct/100,
        'exit_rate': exit_rate_pct/100,

        'page_value': st.sidebar.slider('Page Value',0,362,36),

        'special_day': st.sidebar.selectbox('Special Day',[0,1]),
        'weekend': st.sidebar.selectbox('Weekend',[0,1]),

        **month_data,
        **visitor_data,
        **os_data,
        **browser_data,
        **region_data,
        **traffic_data
    }

    return input_data


user_input = user_input_features()


# -----------------------------
# Prediction
# -----------------------------
prediction, probability = predict(user_input)


if probability[0][1] >= 0.58:

    confidence = probability[0][1]*100
    st.success(f"✅ Purchase Made! — Confidence: {confidence:.1f}%")

else:

    confidence = probability[0][0]*100
    st.warning(f"❌ No Purchase Made — Confidence: {confidence:.1f}%")


st.subheader('Purchase Confidence')

st.progress(int(probability[0][1]*100))

st.caption(f"Model confidence in purchase: {probability[0][1]*100:.1f}%")


# -----------------------------
# Behavioral Profiling Section
# -----------------------------
st.header("Marketing Behavioral Profiling")


baseline = user_input.copy()


def plot_behavior(feature, values):

    probs = []

    for v in values:

        temp = baseline.copy()
        temp[feature] = v

        df = pd.DataFrame([temp])
        df = df.reindex(columns=MODEL_FEATURES, fill_value=0)

        prob = model.predict_proba(df)[0][1]

        probs.append(prob)

    fig, ax = plt.subplots()

    ax.plot(values, probs)

    ax.set_xlabel(feature)
    ax.set_ylabel("Purchase Probability")

    st.pyplot(fig)


# -----------------------------
# Individual Behavioral Plots
# -----------------------------
st.subheader("Page Value Impact")
plot_behavior("page_value", np.linspace(0,200,40))


st.subheader("Bounce Rate Impact")
plot_behavior("bounce_rate", np.linspace(0,0.2,40))


st.subheader("Exit Rate Impact")
plot_behavior("exit_rate", np.linspace(0,0.2,40))


st.subheader("Product Page Duration Impact")
plot_behavior("prod_related_duration", np.linspace(0,20000,40))


st.subheader("Product Pages Viewed Impact")
plot_behavior("prod_related", np.linspace(0,200,40))


st.subheader("Administrative Pages Impact")
plot_behavior("admin", np.linspace(0,20,40))