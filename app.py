import streamlit as st
import pandas as pd
import joblib
import gzip

st.set_page_config(page_title="Video Game Sales Predictor", layout="centered")

st.title("🎮 Video Game Sales Predictor")
st.write("Predict global video game sales using Machine Learning")

# Load model
with gzip.open("game_sales_model.pkl.gz", "rb") as f:
    model = joblib.load(f)

features = joblib.load("features.pkl")

st.subheader("Enter Game Information")

year = st.slider("Release Year", 1985, 2016, 2010)

platform = st.selectbox(
    "Platform",
    ["PS2","PS3","PS4","X360","Xbox","PC","Wii","DS"]
)

genre = st.selectbox(
    "Genre",
    ["Action","Sports","Shooter","Role-Playing","Racing","Adventure","Strategy"]
)

publisher = st.selectbox(
    "Publisher",
    ["Nintendo","Electronic Arts","Activision","Ubisoft","Sony Computer Entertainment"]
)

# Prepare input
input_dict = {"Year": year}
input_df = pd.DataFrame([input_dict])

for col in features:
    if col.startswith("Platform_"):
        input_df[col] = 1 if col == f"Platform_{platform}" else 0
    elif col.startswith("Genre_"):
        input_df[col] = 1 if col == f"Genre_{genre}" else 0
    elif col.startswith("Publisher_"):
        input_df[col] = 1 if col == f"Publisher_{publisher}" else 0
    else:
        if col not in input_df:
            input_df[col] = 0

input_df = input_df[features]

# Prediction
if st.button("Predict Sales"):
    prediction = model.predict(input_df)
    st.success(f"🌍 Predicted Global Sales: {prediction[0]:.2f} million copies")
