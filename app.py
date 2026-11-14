
import streamlit as st
import xarray as xr
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Earth System Trend Detective",
    page_icon="🌍",
    layout="wide"
)

DATA_DIR = Path(__file__).parent / "data"

LST_FILE = DATA_DIR / "MOD11A2-061-Statistics.csv"
VEG_FILE = DATA_DIR / "MOD13Q1-061-Statistics.csv"

METHANE_URL = (
    "https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_mm_gl.csv"
)

st.title("🌍 Earth System Trend Detective")

st.write(
    "Investigate changes in Earth's environment "
    "using NASA satellite observations and atmospheric "
    "methane measurements."
)

st.divider()


# --------------------------------------------------
# REGION DEFINITIONS
# --------------------------------------------------

REGIONS = {
    "Bahrain": (25.5, 26.5, 50.2, 50.9),
 
}


# --------------------------------------------------
# NASA DATA PROCESSING
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def load_nasa_data(file_path, category):

    df = pd.read_csv(file_path)

    # Prepare the date and measurement columns
    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Mean"] = pd.to_numeric(
        df["Mean"],
        errors="coerce"
    )

    df["Count"] = pd.to_numeric(
        df["Count"],
        errors="coerce"
    )

    # VEGETATION DATA (NDVI)
    if category == "Vegetation":

        df = df[
            df["Dataset"].str.contains(
                "NDVI",
                case=False,
                na=False
            )
        ].copy()

        df["Value"] = df["Mean"]
        df["Series"] = "Vegetation (NDVI)"

    # LAND SURFACE TEMPERATURE DATA
    elif category == "Land Surface Temperature (LST)":

        df = df[
            df["Dataset"].str.contains(
                "LST_",
                case=False,
                na=False
            )
        ].copy()

        # Convert Kelvin to Celsius
        df["Value"] = df["Mean"] - 273.15

        # Separate daytime and nighttime readings
        df["Series"] = (
            df["Dataset"]
            .str.replace("LST_", "", regex=False)
            .str.replace("_1km", "", regex=False)
            .str.title()
        )

    # Remove invalid values
    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.dropna(
        subset=["Date", "Value"]
    )

    return df[["Date", "Value", "Series", "Count"]]

# --------------------------------------------------
# METHANE DATA
# NOAA GLOBAL MONTHLY MEASUREMENTS
# --------------------------------------------------

@st.cache_data(ttl=86400, show_spinner=False)
def load_methane_data():

    df = pd.read_csv(
        METHANE_URL,
        comment="#"
    )

    df.columns = [
        str(column).strip().lower().replace(" ", "_")
        for column in df.columns
    ]

    required = {"year", "month", "average"}

    if not required.issubset(df.columns):

        df = pd.read_csv(
            METHANE_URL,
            comment="#",
            header=None,
            sep=r"[\s,]+",
            engine="python"
        )

        if df.shape[1] < 4:
            raise ValueError(
                "Unexpected NOAA methane data format."
            )

        df = df.iloc[:, :7]

        df.columns = [
            "year",
            "month",
            "decimal_date",
            "average",
            "average_unc",
            "trend",
            "trend_unc"
        ]

    for column in ["year", "month", "average", "trend"]:

        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df["Date"] = pd.to_datetime(
        dict(
            year=df["year"],
            month=df["month"],
            day=1
        ),
        errors="coerce"
    )

    df = df.replace(
        [-999.99, -999.9, -9999],
        np.nan
    )

    df = df.dropna(
        subset=["Date", "average"]
    )

    return df


# --------------------------------------------------
# 1. SELECT ENVIRONMENTAL VARIABLES
# --------------------------------------------------

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


# --------------------------------------------------
# 2. SELECT REGION
# --------------------------------------------------

st.header("2. Select your geographical region")

region = st.selectbox(
    "Choose a region",
    list(REGIONS.keys())
)


# --------------------------------------------------
# 3. SELECT TIME PERIOD
# --------------------------------------------------

st.header("3. Select your time period")

col1, col2 = st.columns(2)

with col1:

    start_year = st.number_input(
        "Starting year",
        min_value=2000,
        max_value=2026,
        value=2015
    )

with col2:

    end_year = st.number_input(
        "Ending year",
        min_value=2001,
        max_value=2026,
        value=2025
    )


# --------------------------------------------------
# INVESTIGATE BUTTON
# --------------------------------------------------

st.divider()

if st.button(
    "🔍 Investigate Earth Trends",
    type="primary",
    use_container_width=True
):

    if start_year >= end_year:

        st.error(
            "Your ending year must be after your starting year."
        )

    elif not variables:

        st.warning(
            "Please select at least one environmental variable."
        )

    else:

        st.header("📊 Your Environmental Investigation")

        st.write(
            f"**Region:** {region}  \n"
            f"**Period:** {start_year}–{end_year}"
        )

        for category in variables:

            st.subheader(category)

            if category == "Vegetation":
                file_path = VEG_FILE
            else:
                file_path = LST_FILE

            if not file_path.exists():

                st.error(
                    f"Dataset not found: {file_path.name}"
                )
                continue

           
            try:

                with st.spinner(
                    f"Processing {category} NASA data..."
                ):

                    df = load_nasa_data(
                        str(file_path),
                        category
                    )

                # Filter observations by selected years
                df = df[
                    (df["Date"].dt.year >= start_year) &
                    (df["Date"].dt.year <= end_year)
                ].copy()

                if df.empty:
                    st.warning(
                        "No observations were found for "
                        "the selected period."
                    )
                    continue

                st.success(
                    f"Successfully processed {category}"
                )

                st.caption(
                    "The graph represents the geographical area "
                    "contained in the uploaded NASA dataset. "
                    "The region selector does not spatially filter "
                    "these CSV observations."
                )

                if category == "Vegetation":
                    y_label = "Vegetation Index (NDVI)"
                else:
                    y_label = "Land Surface Temperature (°C)"

                fig = px.line(
                    df,
                    x="Date",
                    y="Value",
                    color="Series",
                    title=f"{category} Over Time",
                    markers=True
                )

                fig.update_layout(
                    xaxis_title="Year",
                    yaxis_title=y_label,
                    hovermode="x unified"
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                col_a, col_b, col_c = st.columns(3)

                col_a.metric(
                    "Average",
                    f"{df['Value'].mean():.2f}"
                )

                col_b.metric(
                    "Minimum",
                    f"{df['Value'].min():.2f}"
                )

                col_c.metric(
                    "Maximum",
                    f"{df['Value'].max():.2f}"
                )

                st.write("**Data preview**")

                st.dataframe(
                    df,
                    use_container_width=True
                )


            except Exception as error:

                st.error(
                    f"Could not process {category}: {error}"
                )

        # --------------------------------------------------
        # METHANE INVESTIGATION
        # --------------------------------------------------

        st.divider()

        st.header("🌡️ 4. Investigating Atmospheric Methane")

        st.write(
            "Methane (CH₄) is a greenhouse gas that absorbs "
            "outgoing infrared radiation, contributing to "
            "the warming of Earth's atmosphere."
        )

        st.info(
            "Methane data are obtained separately from NOAA "
            "Global Monitoring Laboratory. These are global "
            "atmospheric measurements, not local measurements "
            "of Bahrain or satellite measurements from MODIS."
        )

        try:

            with st.spinner(
                "Loading atmospheric methane measurements..."
            ):

                methane = load_methane_data()

            methane = methane[
                (methane["Date"].dt.year >= start_year) &
                (methane["Date"].dt.year <= end_year)
            ]

            if methane.empty:

                st.warning(
                    "No methane measurements are available "
                    "for the selected period."
                )

            else:

                methane_fig = px.line(
                    methane,
                    x="Date",
                    y="average",
                    title="Global Atmospheric Methane Concentration",
                    labels={
                        "Date": "Year",
                        "average": "Methane Concentration (ppb)"
                    }
                )

                methane_fig.update_layout(
                    hovermode="x unified"
                )

                st.plotly_chart(
                    methane_fig,
                    use_container_width=True
                )

                latest = methane.iloc[-1]

                st.metric(
                    "Latest available methane measurement",
                    f"{latest['average']:.2f} ppb",
                    help="Global monthly mean from NOAA."
                )

                if "trend" in methane.columns:

                    trend_data = methane.dropna(
                        subset=["trend"]
                    )

                    if not trend_data.empty:

                        trend_fig = px.line(
                            trend_data,
                            x="Date",
                            y="trend",
                            title=f"Smoothed Atmospheric Methane Trend ({start_year}–{end_year})",
                            labels={
                                "Date": "Year",
                                "trend": "Methane Trend (ppb)"
                            }
                        )

                        st.plotly_chart(
                            trend_fig,
                            use_container_width=True
                        )

                
                # Automatic methane trend interpretation
                methane = methane.sort_values("Date")

                first_reading = methane.iloc[0]
                last_reading = methane.iloc[-1]

                change = (
                    last_reading["average"] -
                    first_reading["average"]
                )

                st.subheader("What Does the Methane Data Show?")

                st.write(
                    f"Between "
                    f"{first_reading['Date'].strftime('%b %Y')} "
                    f"and "
                    f"{last_reading['Date'].strftime('%b %Y')}, "
                    f"global methane concentration changed from "
                    f"{first_reading['average']:.2f} ppb to "
                    f"{last_reading['average']:.2f} ppb."
                )

                if change > 0:
                    st.success(
                        f"Methane concentration increased by "
                        f"{change:.2f} ppb during the selected period."
                    )

                elif change < 0:
                    st.info(
                        f"Methane concentration decreased by "
                        f"{abs(change):.2f} ppb during the selected period."
                    )

                else:
                    st.info(
                        "Methane concentration remained unchanged "
                        "between the first and last readings."
                    )


                st.subheader("Why Is Methane Important?")
                st.write("""
                Methane (CH₄) is a greenhouse gas that contributes to
                global warming. It absorbs outgoing infrared radiation
                from Earth's surface, trapping heat in the atmosphere.
                
                As atmospheric methane concentrations increase,
                more heat can be retained in the climate system,
                contributing to rising global temperatures.
                
                The graph allows us to observe how global atmospheric
                methane concentrations change over time.
                """)

                st.caption(
                    "Source: NOAA Global Monitoring Laboratory. "
                    "Recent measurements may be preliminary."
                )

        except Exception as error:

            st.error(
                f"Could not load methane measurements: {error}"
            )

        st.success(
            "Your environmental investigation is complete!"
        )


# --------------------------------------------------
# 5. UNDERSTANDING METHANE
# --------------------------------------------------

st.divider()

st.header("5. How Does Methane Contribute to Global Warming?")

st.write(
    "Methane is a greenhouse gas. It absorbs some of the "
    "infrared radiation emitted by Earth's surface and "
    "reduces the amount of heat escaping directly into space."
)

st.write(
    "As atmospheric methane concentrations increase, "
    "its contribution to the greenhouse effect increases."
)

st.write(
    "Methane emissions come from several sources, including "
    "agriculture, livestock, fossil fuel production, "
    "waste disposal and natural wetlands."
)

st.subheader("How methane changes over time")

st.write(
    "The methane time-series graph allows you to investigate "
    "whether atmospheric concentrations are increasing, "
    "decreasing or remaining relatively stable over a "
    "selected period."
)


# --------------------------------------------------
# 6. UNDERSTANDING COOLING STRATEGIES
# --------------------------------------------------

st.divider()

st.header("6. Understanding Cooling Strategies")

st.write(
    "Different strategies can help reduce urban heat "
    "and improve thermal comfort."
)

with st.expander("🌳 Green Infrastructure"):

    st.write(
        "Trees, parks and vegetation provide shade and "
        "contribute to cooling through evapotranspiration."
    )

with st.expander("🏠 Cool Roofs"):

    st.write(
        "Reflective roofing materials reduce the amount "
        "of solar energy absorbed by buildings."
    )

with st.expander("🌿 Urban Shading"):

    st.write(
        "Trees and shading structures reduce direct "
        "solar exposure in urban spaces."
    )

with st.expander("❄️ District Cooling"):

    st.write(
        "Centralised cooling systems distribute chilled "
        "water to multiple buildings, helping manage "
        "cooling demand."
    )
    
    st.subheader("1. Reducing Methane Emissions")
    
    st.write("""
    Reducing methane emissions is an important way to slow
    global warming. Methane comes from sources such as livestock,
    landfills, rice cultivation and fossil fuel production.
    
    Strategies include detecting and repairing gas leaks,
    capturing methane from landfills and improving waste
    management practices.
    """)
    
    st.subheader("2. Cooling Urban Environments")
    
    st.write("""
    Cities can become hotter than surrounding areas because
    buildings, roads and other surfaces absorb and retain heat.
    This is known as the urban heat island effect.
    
    Cooling strategies include:
    - Planting more trees and expanding green spaces.
    - Using reflective or light-coloured roofs.
    - Creating shaded pedestrian areas.
    - Designing buildings to improve natural ventilation.
    """)
    
    st.subheader("3. Adapting to Rising Temperatures")
    
    st.write("""
    Communities can prepare for higher temperatures by
    developing heat action plans, providing shaded public
    spaces, improving access to drinking water and creating
    cooling centres during periods of extreme heat.
    
    These measures help reduce people's exposure to heat.
    """)
    
    st.subheader("Mitigation vs Adaptation")
    
    st.write("""
    Mitigation focuses on addressing the causes of climate
    change, such as reducing methane emissions.
    
    Adaptation focuses on dealing with its effects, such as
    creating cooler urban environments and preparing for
    heatwaves.
    
    Both approaches are important for responding to
    climate change.
    """)
    
    st.subheader("4. Cooling Strategies in Bahrain")
    
    st.write("""
    Bahrain experiences very hot summers, making urban cooling
    an important part of climate adaptation. Several strategies
    can help reduce heat exposure in residential areas and
    public spaces.
    
    - **Reflective roofs:** Light-coloured and reflective
    materials can reduce the amount of solar energy absorbed
    by buildings.
    
    - **Urban greenery:** Planting suitable trees and expanding
    green spaces can provide shade and cool surrounding areas
    through evapotranspiration.
    
    - **Shaded walkways:** Covered pedestrian paths can reduce
    direct exposure to sunlight and make outdoor spaces
    more comfortable.
    
    - **Building design:** Improved insulation, ventilation
    and energy-efficient cooling systems can help manage
    indoor temperatures.
    
    These strategies are particularly relevant to Bahrain's
    hot climate and growing urban environments.
    """)

# --------------------------------------------------
# DATA SOURCES
# --------------------------------------------------

st.divider()

st.header("📚 Data Sources")

st.markdown(
    """
    - NASA MODIS MOD11A2: Land Surface Temperature.
    - NASA MODIS MOD13Q1: Vegetation Indices.
    - NOAA Global Monitoring Laboratory: Atmospheric Methane.
    
    NASA and NOAA datasets are presented separately because
    they measure different environmental variables.
    """
)
