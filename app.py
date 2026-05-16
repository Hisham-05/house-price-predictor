import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import joblib
import shap

matplotlib.use('Agg')

# Page Configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    # layout="wide"
)

# Load Model
@st.cache_resource
def load_model():
    loaded_pipeline = joblib.load('models/house_price_pipeline.pkl')
    loaded_numerical_col = joblib.load('models/numerical_cols.pkl')
    loaded_categorical_col = joblib.load('models/categorical_cols.pkl')
    loaded_feature_cols = joblib.load('models/feature_columns.pkl')
    return loaded_pipeline, loaded_numerical_col, loaded_categorical_col, loaded_feature_cols

pipeline, numerical_col, categorical_col, feature_columns = load_model()

# Feature Engineering

def engineer_features(df):
    df = df.copy()
    df['TotalSF'] = (
            df['TotalBsmtSF'].fillna(0) +
            df['1stFlrSF'].fillna(0) +
            df['2ndFlrSF'].fillna(0)
    )

    df['TotalBathrooms'] = (
            df['FullBath'].fillna(0) +
            df['HalfBath'].fillna(0) * 0.5 +
            df['BsmtFullBath'].fillna(0) +
            df['BsmtHalfBath'].fillna(0) * 0.5
    )

    df['TotalPorchSF'] = (
            df['OpenPorchSF'].fillna(0) +
            df['EnclosedPorch'].fillna(0) +
            df['3SsnPorch'].fillna(0) +
            df['ScreenPorch'].fillna(0)
    )

    df['HouseAge'] = df['YrSold'] - df['YearBuilt']

    df['YearsSinceRemodel'] = df['YrSold'] - df['YearRemodAdd']

    df['IsRemodeled'] = (df['YearRemodAdd'] != df['YearBuilt']).astype(int)

    df['HasBasement'] = (df['TotalBsmtSF'].fillna(0) > 0).astype(int)

    df['Has1stFloor'] = (df['1stFlrSF'].fillna(0) > 0).astype(int)

    df['Has2ndFloor'] = (df['2ndFlrSF'].fillna(0) > 0).astype(int)

    df['HasGarage'] = (df['GarageArea'].fillna(0) > 0).astype(int)

    df['IsNew'] = (df['HouseAge'] <= 2).astype(int)

    return df

# Header
st.title("🏠 House Price Predictor")
with st.sidebar:
    st.header("About")

    st.write("""
    This application predicts house prices using an optimized XGBoost regression model.

    Features:
    - sklearn Pipeline
    - Feature Engineering
    - Cross Validation
    - SHAP Explainability
    - Streamlit Deployment
    """)

    st.divider()

    st.subheader("Model")
    st.write("XGBoost Regressor")

    st.subheader("Tech Stack")
    st.write("""
    Python  
    scikit-learn  
    XGBoost  
    SHAP  
    Streamlit
    """)

    st.divider()

    st.markdown(
        "[GitHub Repository](https://github.com/Hisham-05/house-price-predictor)"
    )
st.markdown(
    "Enter the details of a house below to get an "
    "estimated price and understand what's driving it."
)
st.divider()

# Input Form

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Size & Space")
    gr_liv_area = st.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=6000, value=1500)
    total_bsmt_sf = st.number_input("Basement Area (sq ft)", min_value=0, max_value=6000, value=800)
    first_flr_sf = st.number_input("First Floor Area (sq ft)", min_value=300, max_value=5000, value=900)
    second_flr_sf = st.number_input("Second Floor Area (sq ft)", min_value=0, max_value=3000, value=0)
    garage_area = st.number_input("Garage Area (sq ft)", min_value=0, max_value=1500, value=400)
    lot_area = st.number_input("Lot Area (sq ft)", min_value=1000, max_value=200000, value=8000)

with col2:
    st.subheader("Quality & Condition")
    overall_qual = st.slider("Overall Quality (1-10)", min_value=1,  max_value=10, value=5)
    overall_cond = st.slider("Overall Condition (1-10)", min_value=1,  max_value=10, value=5)
    full_bath = st.number_input("Full Bathrooms", min_value=0, max_value=5, value=2)
    half_bath = st.number_input("Half Bathrooms", min_value=0, max_value=5, value=2)
    bedroom = st.number_input("Bedrooms Above Ground", min_value=1, max_value=10, value=3)
    kitchen = st.number_input("Kitchens", min_value=1, max_value=3, value=1)

with col3:
    st.subheader("Age & Location")
    year_built = st.number_input("Year Built", min_value=1800, max_value=2025, value=2000)
    year_remod = st.number_input("Year Remodeled", min_value=1800, max_value=2025, value=2000)
    yr_sold = st.number_input("Year Sold", min_value=2006, max_value=2025, value=2010)
    ms_zoning = st.selectbox("Zoning", ["RL", "RM", "FV", "RH", "C (all)"])
    neighborhood = st.selectbox("Neighbourhood", [
        "CollgCr", "Veenker", "Crawfor", "NoRidge", "Mitchel",
        "Somerst", "NWAmes", "OldTown", "BrkSide", "Sawyer",
        "NridgHt", "NAmes", "SawyerW", "IDOTRR", "MeadowV",
        "Edwards", "Timber", "Gilbert", "StoneBr", "ClearCr",
        "NPkVill", "Blmngtn", "BrDale", "SWISU", "Blueste"
    ])
    house_style = st.selectbox("House Style", [
        "1Story", "2Story", "1.5Fin", "SFoyer", "SLvl"
    ])

st.divider()

# Prediction Button
if st.button("Predict Price", type="primary", use_container_width=True):
    with st.spinner("Predicting house price..."):
        # Validation
        errors = []
        if yr_sold < year_built:
            errors.append("Year sold cannot be before year built.")
        if year_remod < year_built:
            errors.append("Year remodeled cannot be before year built.")
        if gr_liv_area < first_flr_sf:
            errors.append("Total living area cannot be less than first floor area.")

        if errors:
            for error in errors:
                st.error(f"⚠️ {error}")
            st.stop()

    # Input Dictionary

        input_data = {
            'MSSubClass':    60,
            'MSZoning':      ms_zoning,
            'LotArea':       lot_area,
            'LotShape':      'Reg',
            'LandContour':   'Lvl',
            'Neighborhood':  neighborhood,
            'HouseStyle':    house_style,
            'OverallQual':   overall_qual,
            'OverallCond':   overall_cond,
            'YearBuilt':     year_built,
            'YearRemodAdd':  year_remod,
            'TotalBsmtSF':   float(total_bsmt_sf),
            '1stFlrSF':      first_flr_sf,
            '2ndFlrSF':      second_flr_sf,
            'GrLivArea':     gr_liv_area,
            'FullBath':      full_bath,
            'HalfBath':      half_bath,
            'BsmtFullBath':  0.0,
            'BsmtHalfBath':  0.0,
            'BedroomAbvGr':  bedroom,
            'KitchenAbvGr':  kitchen,
            'TotRmsAbvGrd':  bedroom + kitchen + 1,
            'GarageArea':    float(garage_area),
            'OpenPorchSF':   0,
            'EnclosedPorch': 0,
            '3SsnPorch':     0,
            'ScreenPorch':   0,
            'YrSold':        yr_sold,
            'MoSold':        6,
    }

    # Predict

        try:
            # Convert user input into DataFrame for pipeline inference
            input_df = pd.DataFrame([input_data])

            # # Align input schema with training features
            input_df = input_df.reindex(columns=feature_columns, fill_value=np.nan)
            # Feature engineering
            input_engineer = engineer_features(input_df)

            # Pipeline preprocesses and predict in one line
            log_pred = pipeline.predict(input_engineer)[0] # [0] because the result is an array with one value

            # Convert log scale back to dollars
            price = np.expm1(log_pred)

            # Display Prediction
            st.success(f"### Estimated Price: ${price:,.0f}")
            # average error of 10%
            margin = price * 0.10
            st.info(f"Estimated range: **${price - margin:,.0f} - ${price + margin:,.0f}** _(based on model's typical error margin)_")

            st.divider()

            # SHAP
            preprocessor_step = pipeline.named_steps['preprocessor']
            xgb_model = pipeline.named_steps['model']

            # Transform input into the fitted preprocessor
            input_transformed = preprocessor_step.transform(input_engineer)

            # Rebuild feature name after encoding
            cat_feature_names = (
                preprocessor_step
                .named_transformers_['cat']['encoder']
                .get_feature_names_out(categorical_col)
                .tolist()
            )
            all_feature_names = numerical_col + cat_feature_names

            # Compute SHAP values for this one house
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(input_transformed)

            st.subheader("Prediction Explanation")
            # Waterfall plot
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.waterfall_plot(
                shap.Explanation(
                    values = shap_values[0],
                    base_values = explainer.expected_value,
                    data = input_transformed[0],
                    feature_names = all_feature_names
                ),
                max_display = 12,
                show = False
            )
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

            st.divider()
            st.markdown("**How to read this chart:**")
            st.markdown("- Red bars pushed the price **up**")
            st.markdown("- Blue bars pushed the price **down**")
            st.markdown("- The starting point is the average predicted price across all houses in the training data")

        except Exception as e:
            st.error(f"Something went wrong: {str(e)}")
            st.markdown("Please check your inputs and try again.")

# Footer
st.divider()
st.markdown("Built and deployed using Streamlit, XGBoost, sklearn Pipelines, and SHAP · [GitHub](https://github.com/Hisham-05/house-price-predictor)")

