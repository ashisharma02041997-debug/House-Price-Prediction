import streamlit as st
import pandas as pd
import numpy as np
import joblib


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)


# --------------------------------------------------
# LOAD MODEL AND SAVED CITY LIST
# --------------------------------------------------
@st.cache_resource
def load_model():
    model = joblib.load("house_price_model.pkl")
    common_cities = joblib.load("common_cities.pkl")
    return model, common_cities


model, common_cities = load_model()


# --------------------------------------------------
# APP HEADER
# --------------------------------------------------
st.title("🏠 House Price Prediction")

st.write(
    "Enter the property details below to estimate the house price "
    "using the trained XGBoost regression model."
)

st.divider()


# --------------------------------------------------
# PROPERTY DETAILS
# --------------------------------------------------
st.subheader("Property Details")

posted_by = st.selectbox(
    "Posted By",
    ["Owner", "Dealer", "Builder"]
)

bhk_no = st.number_input(
    "Number of BHK",
    min_value=1,
    max_value=20,
    value=2,
    step=1
)

square_ft = st.number_input(
    "Property Area (Square Feet)",
    min_value=1.0,
    value=1000.0,
    step=50.0
)

city = st.selectbox(
    "City",
    sorted(common_cities)
)


# --------------------------------------------------
# PROPERTY STATUS
# --------------------------------------------------
st.subheader("Property Status")

under_construction = st.selectbox(
    "Under Construction?",
    ["No", "Yes"]
)

rera = st.selectbox(
    "RERA Approved?",
    ["No", "Yes"]
)

resale = st.selectbox(
    "Resale Property?",
    ["No", "Yes"]
)


# --------------------------------------------------
# LOCATION DETAILS
# --------------------------------------------------
st.subheader("Location Details")

latitude = st.number_input(
    "Latitude",
    min_value=6.0,
    max_value=38.0,
    value=12.9716,
    format="%.6f"
)

longitude = st.number_input(
    "Longitude",
    min_value=68.0,
    max_value=98.0,
    value=77.5946,
    format="%.6f"
)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
st.divider()

if st.button("Predict House Price", type="primary", use_container_width=True):

    # Convert Yes/No values to 1/0
    under_construction_value = 1 if under_construction == "Yes" else 0
    rera_value = 1 if rera == "Yes" else 0
    resale_value = 1 if resale == "Yes" else 0

    # Apply same transformation used during model training
    log_square_ft = np.log1p(square_ft)

    # Handle rare/unseen cities
    city_for_model = city if city in common_cities else "Other"

    # Create input dataframe in EXACT same structure as training data
    input_data = pd.DataFrame({
        "POSTED_BY": [posted_by],
        "UNDER_CONSTRUCTION": [under_construction_value],
        "RERA": [rera_value],
        "BHK_NO.": [bhk_no],
        "RESALE": [resale_value],
        "LOG_SQUARE_FT": [log_square_ft],
        "CITY": [city_for_model],
        "LONGITUDE": [longitude],
        "LATITUDE": [latitude]
    })

    try:
        # Model predicts LOG_PRICE
        predicted_log_price = model.predict(input_data)[0]

        # Convert log price back to original price scale
        predicted_price = np.expm1(predicted_log_price)

        st.success("Prediction Completed Successfully!")

        st.metric(
            label="Estimated House Price",
            value=f"₹ {predicted_price:,.2f} Lakhs"
        )

        # Additional display in crores for expensive properties
        if predicted_price >= 100:
            predicted_crore = predicted_price / 100

            st.info(
                f"Approximately ₹ {predicted_crore:,.2f} Crore"
            )

    except Exception as e:
        st.error(f"Unable to make prediction: {e}")


# --------------------------------------------------
# MODEL INFORMATION
# --------------------------------------------------
st.divider()

with st.expander("About the Model"):
    st.write(
        """
        This application predicts residential property prices using
        an XGBoost Regression model.

        The model uses property characteristics such as BHK,
        square footage, city, geographic coordinates, construction
        status, RERA status, resale status and posting category.

        The target property price was log-transformed during model
        development to reduce skewness. The predicted value is
        transformed back to the original price scale before being
        displayed.

        The final XGBoost model achieved a Test R² of approximately
        0.867 on the held-out test dataset.
        """
    )

st.caption(
    "This prediction is generated by a machine learning model and "
    "should be treated as an estimate rather than an actual property valuation."
)