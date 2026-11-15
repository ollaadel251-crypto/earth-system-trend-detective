
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

# --------------------------------------------------
# INTRODUCTION
# --------------------------------------------------

st.title("🌍 Earth System Trend Detective")

st.subheader("Understanding the relationship between land surface temperature and methane")

st.markdown("""
Welcome to the Earth System Trend Detective!

This application explores environmental changes using scientific
data from NASA and NOAA. It focuses on land surface temperature,
vegetation and atmospheric methane to help investigate changes
in Earth's environment over time.

By analysing satellite observations and atmospheric measurements,
users can explore environmental trends, identify possible
relationships and understand how human activities contribute
to environmental challenges.

The project focuses on Bahrain, examining the importance of
understanding rising temperatures and exploring possible
solutions for a more sustainable future.
""")

# --------------------------------------------------
# THE ENVIRONMENTAL PROBLEM
# --------------------------------------------------

st.divider()

st.header("⚠️ The Environmental Problem")

st.markdown("""
Bahrain faces environmental challenges associated with
high temperatures, urban development and greenhouse gas
emissions.

Urban areas can experience increased surface temperatures
because buildings and roads absorb and retain solar energy.
Limited vegetation can also reduce natural shade and cooling.

Methane is another important concern. It is a powerful
greenhouse gas released through activities such as waste
disposal and fossil fuel production. Its accumulation in
the atmosphere contributes to global warming.

Understanding these challenges requires examining
environmental data, identifying emission sources and
considering how human activities influence the environment.
""")

# --------------------------------------------------
# THE PROPOSED SOLUTION
# --------------------------------------------------

st.divider()

st.header("💡 The Proposed Solution")

st.markdown("""
This project uses environmental data and monitoring
technology to investigate environmental changes and
explore possible solutions for Bahrain.

The proposed approach combines:

**1. Environmental data analysis**
Using NASA satellite observations to investigate land
surface temperature and vegetation changes.

**2. Methane monitoring**
Using NOAA atmospheric measurements to examine global
methane trends and exploring technologies such as the
Montrose VeriPlume system to investigate individual
emission sources.

**3. Sustainable environmental strategies**
Proposing practical measures such as urban greening,
reflective roofs, improved waste management and methane
recovery.

Together, these approaches aim to support better
environmental understanding and encourage a cooler,
greener and more sustainable Bahrain.
""")

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
        # BAHRAIN METHANE PLUME INVESTIGATION
        # AWALI OIL & GAS
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

        st.caption(
            "Source: UNEP MARS figures as reported in the "
            "project research document. These are individual "
            "plume detections, not annual emissions."
        )

        
        # --------------------------------------------------
        # WASTE-SECTOR METHANE EMISSIONS
        # BAHRAIN
        # --------------------------------------------------

        st.divider()

        st.subheader("Methane Emissions from Bahrain's Waste Sector")

        st.write(
            "The project research reports an increase in annual "
            "waste-sector methane emissions between 2015 and 2022. "
            "The figures below are based on the annual estimates "
            "provided in the research document."
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

        st.write(
            "**Connection to Askar Landfill:** The research "
            "document separately reports that Askar Landfill "
            "methane emissions peaked at approximately "
            "14,844 tonnes per year around 2020. This is a "
            "separate figure and should not be confused with "
            "the waste-sector series in the graph."
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


# --------------------------------------------------
# 6. UNDERSTANDING COOLING STRATEGIES
# --------------------------------------------------

st.header("6. Understanding Cooling Strategies")

st.write(
    "Different strategies can help reduce urban heat, "
    "manage energy consumption and improve thermal comfort."
)

# GREEN INFRASTRUCTURE
with st.expander("🌳 Green Infrastructure"):

    st.write("""
    Trees, parks and vegetation provide shade and contribute
    to cooling through evapotranspiration. Expanding green
    spaces can help reduce temperatures in urban areas.
    """)

# COOL ROOFS
with st.expander("🏠 Cool Roofs"):

    st.write("""
    Reflective roofing materials reduce the amount of solar
    energy absorbed by buildings. This can lower indoor
    temperatures and reduce cooling demand.
    """)

# URBAN SHADING
with st.expander("🌿 Urban Shading"):

    st.write("""
    Trees, shaded walkways and shading structures reduce
    direct solar exposure in urban spaces, making outdoor
    environments more comfortable.
    """)

# DISTRICT COOLING
with st.expander("❄️ District Cooling"):

    st.write("""
    Centralised cooling systems distribute chilled water
    to multiple buildings through a network of pipes.
    This can improve cooling efficiency and help manage
    electricity demand compared with separate cooling
    systems in individual buildings.
    """)

# METHANE REDUCTION
st.subheader("Reducing Methane Emissions")

st.write("""
Reducing methane emissions is an important way to slow
global warming. Methane comes from sources such as livestock,
landfills, rice cultivation and fossil fuel production.

Strategies include detecting and repairing gas leaks,
capturing methane from landfills and improving waste
management practices.
""")

# URBAN HEAT ISLAND EFFECT
st.subheader("Understanding the Urban Heat Island Effect")

st.write("""
Cities can become hotter than surrounding areas because
buildings, roads and other surfaces absorb and retain heat.
This is known as the urban heat island effect.

Increasing vegetation, using reflective materials and
creating shaded public spaces can help reduce this effect.
""")

# ADAPTATION
st.subheader("Adapting to Rising Temperatures")

st.write("""
Communities can prepare for higher temperatures by
developing heat action plans, providing shaded public
spaces, improving access to drinking water and creating
cooling centres during periods of extreme heat.

These measures help reduce people's exposure to heat.
""")

# BAHRAIN
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

# MITIGATION VS ADAPTATION
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

with st.expander("🔬 Discover the Montrose VeriPlume Capture System", expanded=True):

    st.subheader("What is the VeriPlume device?")

    st.write("""
    The Montrose VeriPlume Capture System is a methane
    monitoring technology developed by Montrose Environmental
    Group.

    Introduced in January 2026, it is designed to capture
    and measure emissions from high-volume methane vents,
    particularly in the oil and gas industry.
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

        st.markdown(f"**Environmental challenge:** {solution['problem']}")

        st.markdown(f"**Proposed solution:** {solution['solution']}")

        st.markdown(f"**Expected environmental benefit:** {solution['impact']}")

        st.success(f"Related goals: {solution['sdgs']}")


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
        "number": "SDG 3",
        "name": "Good Health and Well-being",
        "description": "Reducing exposure to extreme heat and air pollution through better environmental management.",
        "connection": "Shaded public spaces, heat adaptation and methane emission reduction."
    },
    {
        "number": "SDG 7",
        "name": "Affordable and Clean Energy",
        "description": "Supporting more efficient energy use and exploring energy recovery from captured landfill methane.",
        "connection": "Methane recovery and improved energy management."
    },
    {
        "number": "SDG 9",
        "name": "Industry, Innovation and Infrastructure",
        "description": "Encouraging environmental monitoring technology and more sustainable industrial infrastructure.",
        "connection": "Montrose VeriPlume and improved methane monitoring."
    },
    {
        "number": "SDG 11",
        "name": "Sustainable Cities and Communities",
        "description": "Developing urban environments that are more resilient to extreme temperatures.",
        "connection": "Urban greening, shaded walkways and reflective building materials."
    },
    {
        "number": "SDG 12",
        "name": "Responsible Consumption and Production",
        "description": "Improving waste management and reducing biodegradable waste sent to landfills.",
        "connection": "Organic waste separation and composting."
    },
    {
        "number": "SDG 13",
        "name": "Climate Action",
        "description": "Monitoring greenhouse gas trends and introducing measures to reduce emissions and adapt to climate change.",
        "connection": "Methane detection, methane recovery and urban cooling."
    },
    {
        "number": "SDG 15",
        "name": "Life on Land",
        "description": "Supporting terrestrial ecosystems and protecting vegetation through sustainable land management.",
        "connection": "Drought-resistant planting and water-efficient landscaping."
    }
]

for i in range(0, len(sdgs), 2):

    col1, col2 = st.columns(2)

    for col, sdg in zip(
        [col1, col2],
        sdgs[i:i+2]
    ):

        with col:

            with st.container(border=True):

                st.subheader(sdg["number"])

                st.markdown(f"**{sdg['name']}**")

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
# DATA SOURCES
# --------------------------------------------------

st.divider()

st.header("📚 Data Sources")

st.markdown(
    """
    - NASA MODIS MOD11A2: Land Surface Temperature.
    - NASA MODIS MOD13Q1: Vegetation Indices.
    - NOAA Global Monitoring Laboratory: Atmospheric Methane.
    - Montrose Environmental Group: VeriPlume Capture System.
    
    NASA and NOAA datasets are presented separately because
    they measure different environmental variables.
    """
)
