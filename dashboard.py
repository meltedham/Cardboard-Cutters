import streamlit as st
import pandas as pd
from pathlib import Path
from src.data_preprocess import preprocess_reviews
from training.training import train_review_classifier
from training.predict_and_filter_valid import predict_and_filter_valid


st.title("Tiktok TechJam 2025")

# --- Train model on startup if not already trained in this session ---
if "model_trained" not in st.session_state:
    st.session_state["model_trained"] = False

if not st.session_state["model_trained"]:
    with st.spinner("Training AI model... (this may take a while the first time)"):
        train_review_classifier(
            data_path="data/reviews_labeled.csv",
            model_save_path="training/results/final_model",
            num_train_epochs=3,
            batch_size=8
        )
    st.session_state["model_trained"] = True
    st.success("Model trained and ready to use!")

# --- Main dashboard logic ---
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Save uploaded CSV temporarily
    temp_input = Path("data/temp_uploaded.csv")
    temp_input.parent.mkdir(exist_ok=True)
    with open(temp_input, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Step 1: Clean the data
    temp_cleaned = Path("data/temp_cleaned.csv")
    with st.spinner("Cleaning CSV..."):
        cleaned_df = preprocess_reviews(temp_input, temp_cleaned)
    st.success("CSV cleaned successfully!")

    # Step 2: Predict and filter valid reviews
    temp_valid = Path("data/temp_valid.csv")
    with st.spinner("Filtering valid reviews with AI model..."):
        valid_df = predict_and_filter_valid(
            input_csv=str(temp_cleaned),
            output_csv=str(temp_valid),
            model_dir="training/results/final_model"
        )
    st.success("Valid reviews filtered successfully!")

    # Step 3: Show and offer download of valid reviews
    st.subheader("Valid Reviews Only")
    st.dataframe(valid_df)

    csv = valid_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Valid Reviews CSV",
        data=csv,
        file_name="valid_reviews.csv",
        mime='text/csv'
    )
else:
    st.info("Please upload a CSV file to start.")
