
import streamlit as st

st.set_page_config(
    page_title="Earth System Trend Detective",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Earth System Trend Detective")

st.write(
    "Investigate changes in Earth's environment "
    "using NASA satellite observations."
)

st.divider()

st.header("1. Select an environmental variable")

variables = st.multiselect(
    "What do you want to investigate?",
    [
        "Vegetation",
        "Land Surface Temperature (LST)"
    ],
    default=[
        "Vegetation",
        "Land Surface Temperature (LST)"
    ]
)

st.header("2. Select your geographical region")

region = st.selectbox(
    "Choose a region",
    [
        "Bahrain",
        "Middle East",
        "Sudan",
        "Africa",
        "Global"
    ]
)

st.header("3. Select your time period")

col1, col2 = st.columns(2)

with col1:
    start_year = st.number_input(
        "Starting year",
        min_value=2000,
        max_value=2025,
        value=2015
    )

with col2:
    end_year = st.number_input(
        "Ending year",
        min_value=2001,
        max_value=2026,
        value=2025
    )

st.divider()

if st.button("🔍 Investigate Earth Trends"):
    if start_year >= end_year:
        st.error("Your ending year must be after your starting year.")
    else:
        st.success("Your investigation settings are ready!")

        st.write("**Variables:**", ", ".join(variables))
        st.write("**Region:**", region)
        st.write("**Time period:**", start_year, "-", end_year)

        st.info(
            "NASA data analysis will be displayed here "
            "once the data connection is added."
        )
