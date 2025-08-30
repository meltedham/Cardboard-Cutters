import streamlit as st
import pandas as pd
from pathlib import Path
from src.data_preprocess import preprocess_reviews  # import from src folder

st.title("Tiktok TechJam 2025")

# --- File uploader ---
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")

if uploaded_file is not None:
    # Save uploaded CSV temporarily
    temp_input = Path("data/temp_uploaded.csv")
    temp_input.parent.mkdir(exist_ok=True)
    with open(temp_input, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load original CSV
    df = pd.read_csv(temp_input)
    st.subheader("Original CSV")
    st.dataframe(df)

    # --- Button to clean CSV ---
    if st.button("Clean CSV"):
        temp_output = Path("data/temp_cleaned.csv")
        with st.spinner("Cleaning CSV..."):
            # Call the cleaning function from src/data_preprocess.py
            cleaned_df = preprocess_reviews(temp_input, temp_output)
        st.success("CSV cleaned successfully!")

        df = cleaned_df  # use cleaned data for dashboard

    # --- Sorting ---
    st.subheader("Sort Data")
    sort_column = st.selectbox("Choose a column to sort by", df.columns)
    ascending = st.radio("Sort order", ("Ascending", "Descending")) == "Ascending"
    sorted_df = df.sort_values(by=sort_column, ascending=ascending)
    st.dataframe(sorted_df)

    # --- Filtering ---
    st.subheader("Filter Data")
    filter_column = st.selectbox("Choose a column to filter", df.columns)

    if df[filter_column].dtype == 'object':
        search_text = st.text_input(f"Enter text to filter '{filter_column}' column")
        if search_text:
            filtered_df = df[df[filter_column].str.contains(search_text, case=False, na=False)]
        else:
            filtered_df = df
    else:
        min_val = float(df[filter_column].min())
        max_val = float(df[filter_column].max())
        selected_range = st.slider(f"Select range for '{filter_column}'", min_val, max_val, (min_val, max_val))
        filtered_df = df[df[filter_column].between(*selected_range)]

    st.dataframe(filtered_df)

    # --- Visualization ---
    st.subheader("Visualize Numeric Columns")
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        chart_col = st.selectbox("Select column to visualize", numeric_cols)
        chart_type = st.radio("Chart type", ("Line Chart", "Bar Chart"))
        if chart_type == "Line Chart":
            st.line_chart(df[chart_col])
        else:
            st.bar_chart(df[chart_col])
    else:
        st.write("No numeric columns available for visualization.")

    # --- Summary ---
    st.subheader("Summary Statistics")
    st.write(df.describe())

    # --- Download filtered CSV ---
    st.subheader("Download Filtered CSV")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download filtered CSV",
        data=csv,
        file_name="filtered_uploaded.csv",
        mime='text/csv'
    )

else:
    st.info("Please upload a CSV file to start.")
