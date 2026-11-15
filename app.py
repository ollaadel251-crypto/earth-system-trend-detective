
import streamlit as st
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

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_DIR = Path(__file__).parent
DATA_DIR = PROJECT_DIR / "data"
ASSETS_DIR = PROJECT_DIR / "assets"

def find_dataset_file(keyword):
    matches = [
        file for file in DATA_DIR.rglob("*.csv")
        if keyword.lower() in file.name.lower()
    ]
    return matches[0] if matches else None


LST_FILE = find_dataset_file("MOD11A2")
VEG_FILE = find_dataset_file("MOD13Q1")

METHANE_URL = (
    "https://gml.noaa.gov/webdata/ccgg/trends/ch4/ch4_mm_gl.csv"
)

# --------------------------------------------------
# PROJECT LOGOS
# --------------------------------------------------

col1, col2, col3 = st.columns([1, 2, 1])

school_logo = ASSETS_DIR / "shgs_logo.png"
agency_logo = ASSETS_DIR / "bsa_logo.png"

with col1:
    if school_logo.exists():
        st.image(str(school_logo), width=180)
    else:
        st.warning("School logo not found.")

with col3:
    if agency_logo.exists():
        st.image(str(agency_logo), width=180)
    else:
        st.warning("Bahrain Space Agency logo not found.")

st.divider()

# --------------------------------------------------
# INTRODUCTION
# --------------------------------------------------

st.title("🌍 Earth System Trend Detective")

st.write(
    "Bahrain faces environmental challenges from extreme heat, "
    "limited vegetation and landfill waste. Our project uses "
    "satellite and environmental data to track changes in "
    "temperature, vegetation and methane, turning complex "
    "information into clear evidence for better sustainability decisions."
)

# --------------------------------------------------
# ENVIRONMENTAL PROBLEM
# --------------------------------------------------

st.header("⚠️ The Environmental Challenge")

st.markdown("""
- Rising population produces more waste, increasing landfill accumulation.
- Decomposing organic waste releases methane, contributing to global warming.
- Bahrain's natural gas fields also pose a risk of methane leaks.
- We use satellite technology to identify risks and help protect our community and environment.
""")

st.subheader("Our Challenge")

st.write(
    "Our challenge is to use satellite technology to detect "
    "methane emissions, identify their sources and protect "
    "our communities through early detection and informed action."
)

# --------------------------------------------------
# THE PROPOSED SOLUTION
# --------------------------------------------------

st.header("💡 Our Proposed Solution")

st.write(
    "We are developing an interactive platform focused on Bahrain. "
    "It is designed to let users explore environmental variables, "
    "view maps and time-series graphs, and interpret the direction "
    "and rate of change. The platform connects three related but "
    "distinct lines of investigation:"
)

st.markdown("""
| **Investigation** | **Data source** | **What it tells us** |
|---|---|---|
| **Land surface temperature (LST)** | NASA MODIS MOD11A2 | How the temperature of the land surface changes over time and across locations (every 8 days at 1 km resolution). |
| **Vegetation** | NASA MODIS MOD13Q1 (NDVI/EVI) | How vegetation greenness and condition vary across Bahrain (16 days at 250 m resolution). |
| **Atmospheric methane / large plumes** | Sentinel-5P TROPOMI and UNEP MARS | Atmospheric methane concentrations or individual large methane plume events; these are not the same as direct site-level emission measurements. |
""")

st.info(
    "**Important distinction:** MODIS measures land-surface temperature "
    "and vegetation indices; it does not measure methane. Methane "
    "information must come from methane-specific satellite products "
    "or emission inventories. The website should label each dataset "
    "and unit clearly."
)

# --------------------------------------------------
# BAHRAIN HEATSHIELD VIDEO
# --------------------------------------------------

st.subheader("🎥 Bahrain HeatShield")

st.write(
    "The following video introduces Bahrain HeatShield "
    "and demonstrates an approach to addressing the "
    "challenges of rising temperatures in Bahrain."
)

video_path = ASSETS_DIR / "bahrain_heatshield.mp4"

if video_path.exists():
    st.video(str(video_path))
else:
    st.warning("Bahrain HeatShield video not found.")

# --------------------------------------------------
# BAHRAIN GEOGRAPHICAL FOCUS
# --------------------------------------------------

st.header("🇧🇭 Geographical Focus: Bahrain")

st.write(
    "This investigation focuses exclusively on Bahrain, "
    "an island nation facing extreme heat, limited vegetation "
    "and waste-management challenges."
)

st.success("Currently investigating: Bahrain")

# --------------------------------------------------
# NASA DATA PROCESSING
# --------------------------------------------------

@st.cache_data(show_spinner=False)
def load_nasa_data(file_path, category):

    df = pd.read_csv(file_path)

    df.columns = [str(c).strip() for c in df.columns]

    required = ["Date", "Mean", "Count", "Dataset"]

    for column in required:
        if column not in df.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

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

    df["Dataset"] = df["Dataset"].astype(str)

    # VEGETATION DATA
    if category == "Vegetation":

        df = df[
            df["Dataset"].str.contains(
                "NDVI",
                case=False,
                na=False
            )
        ].copy()

        df["Value"] = df["Mean"]

        if not df.empty and df["Value"].abs().median() > 1:
            df["Value"] *= 0.0001

        df["Series"] = "Vegetation (NDVI)"

    # LAND SURFACE TEMPERATURE
    elif category == "Land Surface Temperature (LST)":

        df = df[
            df["Dataset"].str.contains(
                "LST_",
                case=False,
                na=False
            )
        ].copy()

        if not df.empty and df["Mean"].median() > 1000:
            df["Mean"] *= 0.02

        if not df.empty and df["Mean"].median() > 100:
            df["Value"] = df["Mean"] - 273.15
        else:
            df["Value"] = df["Mean"]

        df["Series"] = (
            df["Dataset"]
            .str.replace("LST_", "", regex=False)
            .str.replace("_1km", "", regex=False)
            .str.title()
        )

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
# 2. GEOGRAPHICAL AREA
# --------------------------------------------------

st.header("2. Select your geographical area")

st.write("Current geographical focus: **Bahrain 🇧🇭**")

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
            f"**Country:** Bahrain 🇧🇭  \n"
            f"**Period:** {start_year}–{end_year}"
        )

        for category in variables:

            st.subheader(category)

            if category == "Vegetation":
                file_path = VEG_FILE
            else:
                file_path = LST_FILE

            if file_path is None or not file_path.exists():

                st.error(
                    f"NASA dataset for {category} was not found "
                    "inside the data folder."
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

                # GRAPH EXPLANATION
                if category == "Vegetation":

                    st.markdown("""
                    **Graph explanation:**

                    This graph shows changes in vegetation over the
                    selected period using the Normalized Difference
                    Vegetation Index (NDVI).

                    NDVI measures vegetation greenness. Higher values
                    generally indicate denser or healthier green
                    vegetation, while lower values may indicate sparse
                    vegetation or less green cover.

                    Observe whether the graph increases, decreases
                    or remains relatively stable over time.
                    """)

                else:

                    st.markdown("""
                    **Graph explanation:**

                    This graph shows changes in land surface
                    temperature over the selected period.

                    The horizontal axis represents time, while the
                    vertical axis represents land surface temperature
                    in degrees Celsius.

                    An increasing trend may indicate that the land
                    surface is becoming warmer, while a decreasing
                    trend indicates cooling during the observed period.

                    Look for seasonal patterns and long-term changes.
                    """)

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
            ].copy()

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

                st.markdown("""
                **Graph explanation:**

                This graph displays global atmospheric methane
                concentrations measured over time by NOAA.

                The horizontal axis represents the date, and the
                vertical axis represents methane concentration
                in parts per billion (ppb).

                An upward trend indicates that the amount of methane
                in the global atmosphere is increasing.
                """)

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
                            title=(
                                f"Smoothed Atmospheric Methane "
                                f"Trend ({start_year}–{end_year})"
                            ),
                            labels={
                                "Date": "Year",
                                "trend": "Methane Trend (ppb)"
                            }
                        )

                        st.plotly_chart(
                            trend_fig,
                            use_container_width=True
                        )

                        st.markdown("""
                        **Smoothed trend explanation:**

                        The smoothed trend reduces short-term
                        fluctuations and makes the longer-term
                        direction of atmospheric methane easier
                        to observe.
                        """)

                # AUTOMATIC METHANE TREND INTERPRETATION

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

                st.subheader(
                    "Historical Changes in Atmospheric Methane (2015–2025)"
                )

                st.write("""
                According to the project's research:

                - **2015–2019:** Atmospheric methane concentrations
                  increased steadily.
                - **2020–2022:** Methane concentrations increased
                  more rapidly.
                - **2023–2025:** Concentrations continued rising,
                  although the rate of increase became slower.

                The research reports that atmospheric methane
                increased from approximately 1,834 ppb in 2015
                to 1,935–1,945 ppb by late 2025.

                This historical pattern helps us understand how
                methane concentrations have changed over time.
                """)

                st.info(
                    "The historical figures and phases above are "
                    "reported in the project research document. "
                    "The NOAA graph displays the actual monthly "
                    "measurements available in the selected period."
                )

        except Exception as error:

            st.error(
                f"Could not load methane measurements: {error}"
            )

        # --------------------------------------------------
        # BAHRAIN-ONLY METHANE DATA
        # --------------------------------------------------

        st.divider()

        st.header("🇧🇭 Methane Plumes in Bahrain")

        st.subheader("Awali Oil & Gas")

        st.write(
            "Satellite monitoring identified two large methane "
            "plume events at Awali Oil & Gas in Bahrain. "
            "The measurements below show the reported emission "
            "rates on each detection date."
        )

        awali_data = pd.DataFrame({
            "Date": [
                "12 July 2024",
                "18 May 2025"
            ],
            "Methane Emission Rate (kg/h)": [
                10018,
                5578
            ]
        })

        awali_fig = px.bar(
            awali_data,
            x="Date",
            y="Methane Emission Rate (kg/h)",
            title="Reported Methane Plume Rates at Awali",
            text="Methane Emission Rate (kg/h)"
        )

        awali_fig.update_traces(
            texttemplate="%{text:,.0f} kg/h",
            textposition="outside"
        )

        awali_fig.update_layout(
            xaxis_title="Detection Date",
            yaxis_title="Methane Emission Rate (kg/h)",
            yaxis_range=[0, 11500]
        )

        st.plotly_chart(
            awali_fig,
            use_container_width=True
        )

        st.markdown("""
        **Graph explanation:**

        This bar chart compares two reported methane plume
        emission rates detected at Awali Oil & Gas.

        The first detection recorded 10,018 kg/h on 12 July 2024,
        while the second recorded 5,578 kg/h on 18 May 2025.

        The chart shows that the reported rate was lower in the
        second detection. These two observations alone do not
        establish a long-term trend.
        """)

        st.caption(
            "Source: UNEP MARS figures as reported in the "
            "project research document. These are individual "
            "plume detections, not annual emissions."
        )

        # --------------------------------------------------
        # WASTE-SECTOR METHANE
        # --------------------------------------------------

        st.divider()

        st.subheader(
            "Methane Emissions from Bahrain's Waste Sector"
        )

        st.write(
            "The project research reports an increase in annual "
            "waste-sector methane emissions between 2015 and 2022."
        )

        waste_data = pd.DataFrame({
            "Year": [
                2015, 2016, 2017, 2018,
                2019, 2020, 2021, 2022
            ],
            "Methane Emissions (Metric Tons)": [
                32200, 33700, 34800, 35800,
                36900, 37800, 38900, 39900
            ]
        })

        waste_fig = px.line(
            waste_data,
            x="Year",
            y="Methane Emissions (Metric Tons)",
            title="Annual Waste-Sector Methane Emissions (2015–2022)",
            markers=True
        )

        waste_fig.update_layout(
            xaxis_title="Year",
            yaxis_title="Methane Emissions (Metric Tons)",
            hovermode="x unified"
        )

        st.plotly_chart(
            waste_fig,
            use_container_width=True
        )

        st.markdown("""
        **Graph explanation:**

        This line graph shows the reported annual methane
        emissions from Bahrain's waste sector between 2015
        and 2022.

        The graph indicates an overall increase across the
        selected years. This suggests that waste-sector methane
        emissions became higher over the period represented.

        The figures are based on the project's research sources.
        """)

        st.metric(
            "Increase from 2015 to 2022",
            "7,700 metric tons",
            delta="+23.9%"
        )

        st.info(
            "The research document estimates waste-sector "
            "methane emissions of approximately 40,970–43,110 "
            "metric tons for 2023–2025. These are estimates, "
            "not confirmed annual measurements."
        )

        st.caption(
            "Source: Climate Watch / Charting the Globe and "
            "Bahrain's First Biennial Transparency Report to "
            "the UNFCCC, as referenced in the project research."
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
    "Different strategies can help reduce urban heat, "
    "manage energy consumption and improve thermal comfort."
)

with st.expander("🌳 Green Infrastructure"):

    st.write("""
    Trees, parks and vegetation provide shade and contribute
    to cooling through evapotranspiration. Expanding green
    spaces can help reduce temperatures in urban areas.
    """)

with st.expander("🏠 Cool Roofs"):

    st.write("""
    Reflective roofing materials reduce the amount of solar
    energy absorbed by buildings. This can lower indoor
    temperatures and reduce cooling demand.
    """)

with st.expander("🌿 Urban Shading"):

    st.write("""
    Trees, shaded walkways and shading structures reduce
    direct solar exposure in urban spaces, making outdoor
    environments more comfortable.
    """)

with st.expander("❄️ District Cooling"):

    st.write("""
    Centralised cooling systems distribute chilled water
    to multiple buildings through a network of pipes.
    This can improve cooling efficiency and help manage
    electricity demand compared with separate cooling
    systems in individual buildings.
    """)

st.subheader("Reducing Methane Emissions")

st.write("""
Reducing methane emissions is an important way to slow
global warming. Methane comes from sources such as livestock,
landfills, rice cultivation and fossil fuel production.

Strategies include detecting and repairing gas leaks,
capturing methane from landfills and improving waste
management practices.
""")

st.subheader("Understanding the Urban Heat Island Effect")

st.write("""
Cities can become hotter than surrounding areas because
buildings, roads and other surfaces absorb and retain heat.
This is known as the urban heat island effect.

Increasing vegetation, using reflective materials and
creating shaded public spaces can help reduce this effect.
""")

st.subheader("Adapting to Rising Temperatures")

st.write("""
Communities can prepare for higher temperatures by
developing heat action plans, providing shaded public
spaces, improving access to drinking water and creating
cooling centres during periods of extreme heat.

These measures help reduce people's exposure to heat.
""")

st.subheader("Cooling Strategies in Bahrain")

st.write("""
Bahrain experiences very hot summers, making urban cooling
an important part of climate adaptation.

Examples of relevant strategies include:

- **Reflective roofs:** Light-coloured materials reduce
  the amount of solar energy absorbed by buildings.

- **Urban greenery:** Suitable trees and green spaces
  provide shade and cooling through evapotranspiration.

- **Shaded walkways:** Covered pedestrian paths reduce
  direct exposure to sunlight.

- **Building design:** Improved insulation and ventilation
  can help manage indoor temperatures.

These strategies are relevant to Bahrain's hot climate
and growing urban environments.
""")

st.subheader("Mitigation vs Adaptation")

st.write("""
**Mitigation** focuses on addressing the causes of climate
change, such as reducing methane emissions.

**Adaptation** focuses on dealing with its effects, such as
creating cooler urban environments and preparing for heatwaves.

Both approaches are important for responding to climate change.
""")

# --------------------------------------------------
# 7. MONTROSE VERIPLUME DEVICE
# --------------------------------------------------

st.divider()

st.header("7. Montrose VeriPlume: Detecting Methane Emissions")

st.write("""
While NASA satellite observations help identify environmental
changes over large areas, ground-based monitoring devices
can help investigate individual methane emission sources.
""")

with st.expander(
    "🔬 Discover the Montrose VeriPlume Capture System",
    expanded=True
):

    st.subheader("What is the VeriPlume device?")

    st.write("""
    The Montrose VeriPlume Capture System is a methane
    monitoring technology developed by Montrose Environmental
    Group. It is designed to capture and measure emissions
    from high-volume methane vents, particularly in the
    oil and gas industry.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Main target", "Methane (CH₄)")
        st.metric("Main application", "Oil & Gas")

    with col2:
        st.metric("Monitoring approach", "Ground-based")
        st.metric("Main purpose", "Emission detection")

    st.subheader("How does it work?")

    st.write("""
    - Uses a lightweight carbon-fibre telescoping pole.
    - Allows technicians to sample emissions from ground level.
    - Captures and measures methane released from high-volume vents.
    - Helps operators identify and prioritize significant
      emission sources for repair.
    """)

    st.subheader("How could Bahrain benefit?")

    st.write("""
    Bahrain could explore using similar monitoring technology
    around oil and gas facilities to identify significant
    methane leaks and prioritize maintenance.

    This could complement satellite observations by providing
    more detailed information about specific emission sources.
    """)

    st.info("""
    **Important distinction:** NASA satellite observations
    identify broad environmental patterns, while VeriPlume
    is a ground-based emission sampling system. It does not
    replace satellite monitoring or continuous monitoring
    systems, and its use in Bahrain is a proposed application,
    not an existing deployment.
    """)

# --------------------------------------------------
# 8. PROPOSED SUSTAINABLE SOLUTIONS FOR BAHRAIN
# --------------------------------------------------

st.divider()

st.header("8. Proposed Sustainable Solutions for Bahrain")

st.write("""
Based on the environmental challenges explored in this app,
the following solutions could help Bahrain reduce methane
emissions, manage rising temperatures and improve vegetation.
""")

solutions = [
    {
        "title": "🌳 1. Urban Greening",
        "problem": "Limited vegetation and increasing urban heat.",
        "solution": "Plant drought-resistant trees and expand green spaces in identified heat hotspots.",
        "impact": "Provides shade, supports vegetation and helps reduce local surface heating.",
        "sdgs": "SDG 11 and SDG 15"
    },
    {
        "title": "🚌 2. Shaded Public Spaces",
        "problem": "High outdoor temperatures and direct solar exposure.",
        "solution": "Introduce shaded walkways, bus stops and public seating areas.",
        "impact": "Reduces direct exposure to sunlight and improves outdoor thermal comfort.",
        "sdgs": "SDG 3 and SDG 11"
    },
    {
        "title": "🏠 3. Cool Roofs and Reflective Pavements",
        "problem": "Buildings and roads absorb and retain solar heat.",
        "solution": "Use reflective roofing materials and suitable cool pavement technologies.",
        "impact": "Reduces heat absorption and may lower cooling demand in buildings.",
        "sdgs": "SDG 11 and SDG 13"
    },
    {
        "title": "♻️ 4. Methane Recovery at Landfills",
        "problem": "Decomposing organic waste releases methane.",
        "solution": "Introduce methane capture and recovery systems at suitable landfill sites, including Askar.",
        "impact": "Captures methane that would otherwise escape and allows potential energy recovery.",
        "sdgs": "SDG 7 and SDG 13"
    },
    {
        "title": "🌱 5. Organic Waste Management",
        "problem": "Organic waste sent to landfills contributes to methane generation.",
        "solution": "Separate food and garden waste and introduce composting programmes.",
        "impact": "Reduces the amount of biodegradable waste sent to landfills.",
        "sdgs": "SDG 12 and SDG 13"
    },
    {
        "title": "💧 6. Water-Efficient Landscaping",
        "problem": "Maintaining vegetation in Bahrain's hot, dry climate requires careful water management.",
        "solution": "Use drought-tolerant plants and efficient irrigation systems.",
        "impact": "Supports sustainable green spaces while reducing unnecessary water consumption.",
        "sdgs": "SDG 11 and SDG 15"
    }
]

for solution in solutions:

    with st.expander(solution["title"]):

        st.markdown(
            f"**Environmental challenge:** {solution['problem']}"
        )

        st.markdown(
            f"**Proposed solution:** {solution['solution']}"
        )

        st.markdown(
            f"**Expected environmental benefit:** {solution['impact']}"
        )

        st.success(
            f"Related goals: {solution['sdgs']}"
        )

# --------------------------------------------------
# 9. CONNECTIONS TO THE UNITED NATIONS SDGs
# --------------------------------------------------

st.divider()

st.header("9. Connections to the Sustainable Development Goals")

st.write("""
Our project supports several United Nations Sustainable
Development Goals (SDGs). By combining satellite observations,
methane monitoring and proposed environmental solutions,
we can connect scientific investigation to sustainable action.
""")

sdgs = [
    {
        "number": 3,
        "name": "Good Health and Well-being",
        "description": "Reducing exposure to extreme heat and air pollution through better environmental management.",
        "connection": "Shaded public spaces, heat adaptation and methane emission reduction."
    },
    {
        "number": 7,
        "name": "Affordable and Clean Energy",
        "description": "Supporting more efficient energy use and exploring energy recovery from captured landfill methane.",
        "connection": "Methane recovery and improved energy management."
    },
    {
        "number": 9,
        "name": "Industry, Innovation and Infrastructure",
        "description": "Encouraging environmental monitoring technology and more sustainable industrial infrastructure.",
        "connection": "Montrose VeriPlume and improved methane monitoring."
    },
    {
        "number": 11,
        "name": "Sustainable Cities and Communities",
        "description": "Developing urban environments that are more resilient to extreme temperatures.",
        "connection": "Urban greening, shaded walkways and reflective building materials."
    },
    {
        "number": 12,
        "name": "Responsible Consumption and Production",
        "description": "Improving waste management and reducing biodegradable waste sent to landfills.",
        "connection": "Organic waste separation and composting."
    },
    {
        "number": 13,
        "name": "Climate Action",
        "description": "Monitoring greenhouse gas trends and introducing measures to reduce emissions and adapt to climate change.",
        "connection": "Methane detection, methane recovery and urban cooling."
    },
    {
        "number": 15,
        "name": "Life on Land",
        "description": "Supporting terrestrial ecosystems and protecting vegetation through sustainable land management.",
        "connection": "Drought-resistant planting and water-efficient landscaping."
    }
]


# --------------------------------------------------
# FIND THE CORRECT SDG ICON
# --------------------------------------------------

def find_sdg_icon(number):

    possible_names = [
        f"sdg{number}",
        f"sdg{number:02d}",
        f"goal{number}",
        f"goal{number:02d}",
        f"e-web-goal-{number:02d}",
        f"e-web-goal-{number}"
    ]

    for file in ASSETS_DIR.rglob("*"):

        if file.is_file() and file.suffix.lower() in [
            ".png", ".jpg", ".jpeg", ".webp"
        ]:

            filename = file.stem.lower()

            normalized = (
                filename
                .replace("-", "")
                .replace("_", "")
                .replace(" ", "")
            )

            for name in possible_names:

                target = (
                    name
                    .replace("-", "")
                    .replace("_", "")
                    .replace(" ", "")
                )

                if normalized == target or normalized.startswith(target):
                    return file

    return None


# --------------------------------------------------
# DISPLAY THE SEVEN SDGs IN TWO COLUMNS
# --------------------------------------------------

for i in range(0, len(sdgs), 2):

    col1, col2 = st.columns(2)

    for col, sdg in zip([col1, col2], sdgs[i:i+2]):

        with col:

            with st.container(border=True):

                icon_path = find_sdg_icon(sdg["number"])

                if icon_path is not None:

                    st.image(
                        str(icon_path),
                        width=130
                    )

                else:

                    st.warning(
                        f"Icon for SDG {sdg['number']} not found. "
                        "Check the assets folder."
                    )

                st.subheader(
                    f"SDG {sdg['number']}: {sdg['name']}"
                )

                st.write(sdg["description"])

                st.caption(
                    f"Project connection: {sdg['connection']}"
                )


st.success("""
🌍 **Our overall contribution**

By combining NASA environmental data, methane monitoring
technology and practical solutions, our project aims to
support a cooler, greener and more sustainable Bahrain.
""")


# --------------------------------------------------
# DATA SOURCES AND ADDITIONAL RESEARCH RESOURCES
# --------------------------------------------------

st.divider()

st.header("📚 Data Sources")

st.subheader("Scientific and Environmental Datasets")

st.markdown("""
- [NASA MODIS MOD11A2](https://search.earthdata.nasa.gov/search?q=MOD11A2): Land Surface Temperature.
- [NASA MODIS MOD13Q1](https://search.earthdata.nasa.gov/search?q=MOD13Q1): Vegetation Indices.
- [NOAA Global Monitoring Laboratory](https://gml.noaa.gov/ccgg/trends/): Global Atmospheric Methane Measurements.
- [UNEP MARS](https://www.unep.org/explore-topics/energy/what-we-do/monitoring-methane-emissions): Methane plume monitoring.
""")

st.subheader("Additional Research Resources")

st.markdown("""
1. **Kingdom of Bahrain's First Biennial Transparency Report (BTR1) to the UNFCCC**

   Bahrain's official report containing information about national greenhouse gas emissions and climate action.

   [Access the official BTR1 report](https://unfccc.int/documents/655129)

2. **GHGSat – Methane Compared to CO₂**

   An educational resource explaining the differences between methane and carbon dioxide, including their contributions to global warming.

   [Read the GHGSat article](https://www.ghgsat.com/resources/methane-compared-to-co2/)

3. **NASA Earthdata Search**

   NASA's platform for discovering and accessing Earth observation datasets, including MODIS vegetation data.

   [Explore MOD13Q1 on NASA Earthdata](https://search.earthdata.nasa.gov/search?q=MOD13Q1&lat=14.462567921207366&long=32.57053608994562&zoom=1.9955620564830796)

4. **NASA Image and Video Library**

   A collection of NASA images and videos that can support the visual presentation of environmental and scientific topics.

   [Explore NASA Images](https://images.nasa.gov/)
""")

st.info(
    "NASA and NOAA datasets are presented separately because "
    "they measure different environmental variables. Additional "
    "research resources provide background information and "
    "support the interpretation of the investigation."
)
