import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from lightgbm import LGBMClassifier

# --- Faylları yüklə ---
df_cleaned = pd.read_csv("cleaned_airbnb_2.csv")        # Clean edilmiş fayl
df_original = pd.read_csv("airbnb_data.csv")            # Orijinal fayl

# --- Streamlit başlıq ---
st.title("Airbnb Qiymət və Lisenziya Proqnozu")

# --- Şəhər, otaq növü və rayonlar üçün seçimlər ---
unique_cities = sorted(df_original['city'].dropna().unique().tolist())
room_types = sorted(df_original['room_type'].dropna().unique().tolist())

# --- Proqnoz seçimi ---
option = st.radio("Proqnoz növünü seç:", ["Qiymət Proqnozu", "Lisenziya Proqnozu"])

# --------------------------------
# PRICE PROQNOZU
# --------------------------------
if option == "Qiymət Proqnozu":
    st.header("Qiymət Proqnozu")

    city = st.selectbox("Şəhər seçin", unique_cities, key="city_price")
    filtered_neighs = df_original[df_original['city'] == city]['neighbourhood'].dropna().unique().tolist()
    neighbourhood = st.selectbox("Küçəni seçin", sorted(filtered_neighs), key="neigh_price")
    room_type = st.selectbox("Otaq növü seçin", room_types, key="room_price")
    min_nights = st.slider("Minimum gecə sayı", 1, 30, value=2)
    availability = st.slider("İldə neçə gün əlçatandır?", 1, 365, value=180)
    host_listings = st.number_input("Ev sahibinin elan sayı", min_value=1, max_value=1000, step=1, value=1)

    input_df = pd.DataFrame({
        'city': [city],
        'neighbourhood': [neighbourhood],
        'room_type': [room_type],
        'minimum_nights': [min_nights],
        'availability_365': [availability],
        'calculated_host_listings_count': [host_listings]
    })

    feature_cols = ['city', 'neighbourhood', 'room_type', 'minimum_nights', 'availability_365', 'calculated_host_listings_count']
    df_price = df_original.dropna(subset=feature_cols + ['price'])
    X = df_price[feature_cols]
    y = df_price['price']

    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['city', 'neighbourhood', 'room_type'])],
        remainder='passthrough'
    )
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', XGBRegressor(enable_categorical=False, verbosity=0))
    ])
    model.fit(X, y)

    if st.button("Qiyməti Hesabla"):
        prediction = model.predict(input_df)
        st.success(f"Təxmini qiymət: €{prediction[0]:.2f}")

# --------------------------------
# LICENSE PROQNOZU
# --------------------------------
else:
    st.header("Lisenziya Proqnozu")

    df_original['license_binary'] = df_original['license'].notnull().astype(int)
    license_df = df_original.dropna(subset=['city', 'neighbourhood', 'room_type'])

    city = st.selectbox("Şəhər seçin", unique_cities, key="city_license")
    filtered_neighs = license_df[license_df['city'] == city]['neighbourhood'].dropna().unique().tolist()
    neighbourhood = st.selectbox("Küçəni seçin", sorted(filtered_neighs), key="neigh_license")
    room_type = st.selectbox("Otaq növü seçin", room_types, key="room_license")
    availability = st.slider("İldə neçə gün əlçatandır?", 1, 365, key="avail_license")
    number_of_reviews_ltm = st.number_input("Son 12 ayda olan rəy sayı", min_value=0, max_value=1000, step=1)
    host_listings = st.number_input("Ev sahibinin elan sayı", min_value=1, max_value=1000, step=1)

    input_df = pd.DataFrame({
        'city': [city],
        'neighbourhood': [neighbourhood],
        'room_type': [room_type],
        'availability_365': [availability],
        'number_of_reviews_ltm': [number_of_reviews_ltm],
        'calculated_host_listings_count': [host_listings]
    })

    feature_cols = ['city', 'neighbourhood', 'room_type', 'availability_365', 'number_of_reviews_ltm', 'calculated_host_listings_count']
    df_license = license_df.dropna(subset=feature_cols)
    X = df_license[feature_cols]
    y = df_license['license_binary']

    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), ['city', 'neighbourhood', 'room_type'])],
        remainder='passthrough'
    )
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier())
    ])
    model.fit(X, y)

    if st.button("Lisenziya Proqnozlaşdır"):
        prediction = model.predict(input_df)
        result = "VAR" if prediction[0] == 1 else "YOXDUR"
        st.success(f"Təxmini lisenziya vəziyyəti: {result}")
