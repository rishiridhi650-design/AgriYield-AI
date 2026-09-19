import streamlit as st
import pandas as pd
import joblib
import html


# ==========================================================
# AGRIYIELD AI - PREMIUM FRONTEND
# ==========================================================

st.set_page_config(
    page_title="AgriYield AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================================
# SESSION STATE
# ==========================================================

if "entered_app" not in st.session_state:
    st.session_state.entered_app = False


# ==========================================================
# PREMIUM CSS
# IMPORTANT: st.html() avoids the Markdown/HTML bug
# ==========================================================

st.html("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@300;400;500&family=Playfair+Display:ital,wght@0,600;0,700;1,600&family=Space+Grotesk:wght@300;400;500;600;700&display=swap');


/* ======================================================
   GLOBAL
   ====================================================== */

html, body {
    background: #020503;
}

.stApp {
    background:
        radial-gradient(
            circle at 15% 5%,
            rgba(32, 232, 120, 0.10),
            transparent 28%
        ),
        radial-gradient(
            circle at 90% 15%,
            rgba(20, 110, 58, 0.18),
            transparent 28%
        ),
        linear-gradient(
            180deg,
            #010302 0%,
            #061009 52%,
            #020503 100%
        );

    color: white;
}


/* page width */

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}


/* hide Streamlit decorations */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* ======================================================
   INTRO
   ====================================================== */

.landing {

    min-height: 80vh;

    position: relative;

    overflow: hidden;

    display: flex;

    align-items: center;

    justify-content: center;

    border-radius: 32px;

    border:
        1px solid rgba(74,255,150,0.14);

    background:

        radial-gradient(
            circle at 78% 27%,
            rgba(39, 232, 121, 0.26),
            transparent 23%
        ),

        radial-gradient(
            circle at 24% 78%,
            rgba(0, 120, 60, 0.16),
            transparent 28%
        ),

        linear-gradient(
            135deg,
            #000000,
            #041109 48%,
            #000000
        );

    box-shadow:
        0 45px 120px rgba(0,0,0,0.72);

}


/* grid background */

.landing-grid {

    position: absolute;

    inset: 0;

    background-image:

        linear-gradient(
            rgba(255,255,255,0.025) 1px,
            transparent 1px
        ),

        linear-gradient(
            90deg,
            rgba(255,255,255,0.025) 1px,
            transparent 1px
        );

    background-size:
        70px 70px;

}


/* glowing rings */

.ring-one {

    position: absolute;

    width: 720px;
    height: 720px;

    border-radius: 50%;

    border:
        1px solid rgba(53,255,139,0.11);

    right: -310px;
    top: -360px;

}


.ring-two {

    position: absolute;

    width: 500px;
    height: 500px;

    border-radius: 50%;

    border:
        1px solid rgba(53,255,139,0.07);

    right: -120px;
    top: -220px;

}


/* intro content */

.landing-content {

    position: relative;

    z-index: 10;

    width: min(1150px, 90%);

    text-align: center;

}


/* small heading */

.landing-kicker {

    font-family:
        'DM Mono',
        monospace;

    font-size:
        0.76rem;

    letter-spacing:
        0.58rem;

    color:
        #52ff9c;

    margin-bottom:
        1.6rem;

    animation:
        fadeUp 0.9s ease forwards;

}


/* gigantic title */

.landing-title {

    font-family:
        'Bebas Neue',
        Impact,
        sans-serif;

    font-size:
        clamp(7rem, 17vw, 14rem);

    letter-spacing:
        0.025em;

    line-height:
        0.76;

    margin: 0;

    color: white;

    animation:
        revealTitle 1.15s
        cubic-bezier(.16,.84,.44,1)
        forwards;

}


.landing-title span {

    color:
        #21e878;

    text-shadow:
        0 0 45px
        rgba(33,232,120,0.32);

}


/* tagline */

.landing-tagline {

    font-family:
        'Playfair Display',
        Georgia,
        serif;

    font-size:
        1.45rem;

    font-style:
        italic;

    margin-top:
        2rem;

    color:
        rgba(255,255,255,0.66);

    animation:
        fadeUp 1.4s ease forwards;

}


/* glowing line */

.landing-line {

    width:
        220px;

    height:
        2px;

    margin:
        2.5rem auto;

    background:
        linear-gradient(
            90deg,
            transparent,
            #2bf082,
            white,
            #2bf082,
            transparent
        );

    box-shadow:
        0 0 25px
        rgba(43,240,130,0.65);

    animation:
        pulseLine 2s
        ease-in-out infinite alternate;

}


/* technical subtitle */

.landing-tech {

    font-family:
        'DM Mono',
        monospace;

    font-size:
        0.72rem;

    letter-spacing:
        0.21rem;

    color:
        rgba(255,255,255,0.40);

}


/* ======================================================
   ANIMATIONS
   ====================================================== */

@keyframes fadeUp {

    from {
        opacity: 0;
        transform:
            translateY(28px);
    }

    to {
        opacity: 1;
        transform:
            translateY(0);
    }

}


@keyframes revealTitle {

    from {
        opacity: 0;

        transform:
            translateY(30px)
            scale(1.12);

        filter:
            blur(15px);
    }

    to {
        opacity: 1;

        transform:
            translateY(0)
            scale(1);

        filter:
            blur(0);
    }

}


@keyframes pulseLine {

    from {
        opacity: 0.45;

        transform:
            scaleX(0.7);
    }

    to {
        opacity: 1;

        transform:
            scaleX(1.08);
    }

}


/* ======================================================
   MAIN HERO
   ====================================================== */

.hero {

    min-height: 500px;

    position: relative;

    overflow: hidden;

    display: flex;

    flex-direction: column;

    justify-content: flex-end;

    padding: 4.5rem;

    margin-bottom: 2rem;

    border-radius: 30px;

    border:
        1px solid rgba(62,255,144,0.14);

    background:

        linear-gradient(
            90deg,
            rgba(0,0,0,0.96) 0%,
            rgba(4,33,17,0.88) 50%,
            rgba(0,0,0,0.60) 100%
        ),

        radial-gradient(
            circle at 82% 20%,
            #176438,
            #06130b 45%,
            #000000 100%
        );

    box-shadow:
        0 30px 100px
        rgba(0,0,0,0.58);

}


.hero-ring {

    position: absolute;

    width: 650px;

    height: 650px;

    border-radius: 50%;

    right: -230px;

    top: -300px;

    border:
        1px solid rgba(60,255,143,0.10);

}


.hero-kicker {

    position: relative;

    z-index: 5;

    font-family:
        'DM Mono',
        monospace;

    letter-spacing:
        0.32rem;

    font-size:
        0.73rem;

    color:
        #55ff9c;

    margin-bottom:
        1rem;

}


.hero-title {

    position: relative;

    z-index: 5;

    font-family:
        'Bebas Neue',
        Impact,
        sans-serif;

    font-size:
        clamp(5.5rem, 10vw, 8.5rem);

    line-height:
        0.82;

    margin: 0;

    color:
        white;

}


.hero-title span {

    color:
        #22e879;

}


.hero-description {

    position: relative;

    z-index: 5;

    max-width:
        720px;

    margin-top:
        1.8rem;

    font-family:
        'Space Grotesk',
        sans-serif;

    font-size:
        1.06rem;

    line-height:
        1.8;

    color:
        rgba(255,255,255,0.65);

}


/* ======================================================
   SECTION HEADERS
   ====================================================== */

.section-index {

    font-family:
        'DM Mono',
        monospace;

    font-size:
        0.70rem;

    letter-spacing:
        0.25rem;

    color:
        #48f591;

    margin-bottom:
        0.4rem;

}


.section-heading {

    font-family:
        'Playfair Display',
        Georgia,
        serif;

    font-size:
        2.6rem;

    font-weight:
        700;

    color:
        white;

    margin-bottom:
        1.5rem;

}


/* ======================================================
   METRIC CARDS
   ====================================================== */

div[data-testid="stMetric"] {

    padding:
        1.25rem;

    border-radius:
        16px;

    background:
        linear-gradient(
            145deg,
            rgba(255,255,255,0.072),
            rgba(255,255,255,0.022)
        );

    border:
        1px solid rgba(66,255,146,0.11);

    box-shadow:
        0 14px 38px rgba(0,0,0,0.25);

}


div[data-testid="stMetricLabel"] {

    font-family:
        'DM Mono',
        monospace;

}


/* ======================================================
   INPUTS
   ====================================================== */

div[data-baseweb="input"] > div {

    border-radius:
        9px !important;

}


div[data-baseweb="select"] > div {

    border-radius:
        9px !important;

}


/* ======================================================
   BUTTON
   ====================================================== */

div.stButton > button {

    width: 100%;

    min-height:
        56px;

    border-radius:
        0;

    border:
        1px solid #27ed80;

    background:
        #27ed80;

    color:
        #001508;

    font-family:
        'Space Grotesk',
        sans-serif;

    font-weight:
        800;

    font-size:
        0.86rem;

    letter-spacing:
        0.17rem;

    transition:
        all 0.25s ease;

}


div.stButton > button:hover {

    background:
        transparent;

    color:
        white;

    border:
        1px solid #27ed80;

    box-shadow:
        0 0 38px
        rgba(39,237,128,0.28);

    transform:
        translateY(-2px);

}


/* ======================================================
   TABS
   ====================================================== */

button[data-baseweb="tab"] {

    font-family:
        'DM Mono',
        monospace;

    font-size:
        0.77rem;

    letter-spacing:
        0.06rem;

}


/* ======================================================
   PREDICTION RESULT
   ====================================================== */

.result {

    position: relative;

    overflow: hidden;

    margin-top:
        2rem;

    padding:
        3.2rem;

    background:
        linear-gradient(
            135deg,
            #071b0f,
            #020604
        );

    border:
        1px solid #28ed81;

    box-shadow:
        0 30px 80px
        rgba(0,0,0,0.45);

}


.result-ring {

    position:
        absolute;

    width:
        370px;

    height:
        370px;

    border-radius:
        50%;

    right:
        -170px;

    top:
        -180px;

    border:
        1px solid rgba(46,243,134,0.14);

}


.result-tag {

    font-family:
        'DM Mono',
        monospace;

    letter-spacing:
        0.28rem;

    font-size:
        0.70rem;

    color:
        #4eff99;

}


.result-number {

    font-family:
        'Bebas Neue',
        Impact,
        sans-serif;

    font-size:
        7rem;

    line-height:
        1;

    color:
        white;

    margin-top:
        0.8rem;

}


.result-description {

    font-family:
        'Playfair Display',
        Georgia,
        serif;

    font-size:
        1.3rem;

    font-style:
        italic;

    color:
        rgba(255,255,255,0.62);

}


/* ======================================================
   SIDEBAR
   ====================================================== */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #010301,
            #061009
        );

    border-right:
        1px solid
        rgba(54,255,140,0.10);

}


/* ======================================================
   FOOTER
   ====================================================== */

.footer {

    margin-top:
        5rem;

    padding:
        2rem;

    text-align:
        center;

    border-top:
        1px solid
        rgba(255,255,255,0.07);

    font-family:
        'DM Mono',
        monospace;

    font-size:
        0.70rem;

    letter-spacing:
        0.14rem;

    color:
        rgba(255,255,255,0.30);

}

</style>
""")


# ==========================================================
# INTRO PAGE
# ==========================================================

if not st.session_state.entered_app:

    st.html("""
    <div class="landing">

        <div class="landing-grid"></div>

        <div class="ring-one"></div>

        <div class="ring-two"></div>

        <div class="landing-content">

            <div class="landing-kicker">
                INTELLIGENT AGRICULTURE
            </div>

            <div class="landing-title">
                AGRI<span>YIELD</span>
            </div>

            <div class="landing-tagline">
                Predicting tomorrow's harvest.
            </div>

            <div class="landing-line"></div>

            <div class="landing-tech">
                MACHINE LEARNING / AGRICULTURE / INDIA
            </div>

        </div>

    </div>
    """)

    st.write("")

    if st.button("ENTER AGRIYIELD AI"):

        st.session_state.entered_app = True

        st.rerun()

    st.stop()


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    return pd.read_csv(
        "data/processed_crop_yield.csv"
    )


@st.cache_resource
def load_metadata():

    return joblib.load(
        "models/multicrop/multicrop_metadata.pkl"
    )


@st.cache_resource
def load_model(model_path):

    return joblib.load(
        model_path
    )


df = load_data()

metadata = load_metadata()


# ==========================================================
# VALIDATED CROPS
# ==========================================================

validated_crops = sorted([
    crop
    for crop, info in metadata.items()
    if info["test_r2"] > 0
])


if not validated_crops:

    st.error(
        "No validated crop models are available."
    )

    st.stop()


# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.markdown("# 🌾 AGRIYIELD AI")

    st.caption(
        "Agricultural Intelligence Platform"
    )

    st.divider()

    crop = st.selectbox(
        "SELECT CROP",
        validated_crops
    )

    crop_info = metadata[crop]

    st.markdown("### MODEL SYSTEM")

    st.write(
        crop_info["best_algorithm"]
    )

    st.metric(
        "Temporal R²",
        f"{crop_info['test_r2']:.3f}"
    )

    st.metric(
        "Temporal MAE",
        f"{crop_info['test_mae']:.3f}"
    )

    st.divider()

    st.caption(
        f"Dataset period: "
        f"{crop_info['minimum_year']}–"
        f"{crop_info['maximum_year']}"
    )

    st.caption(
        f"Historical records: "
        f"{crop_info['records']}"
    )


# ==========================================================
# SELECT MODEL
# ==========================================================

crop_df = df[
    df["Crop"] == crop
].copy()


model = load_model(
    crop_info["model_path"]
)


safe_crop = html.escape(
    str(crop)
)


safe_model = html.escape(
    str(crop_info["best_algorithm"])
)


# ==========================================================
# MAIN HERO
# ==========================================================

st.html("""
<div class="hero">

    <div class="hero-ring"></div>

    <div class="hero-kicker">
        AGRICULTURAL INTELLIGENCE / AGRIYIELD AI
    </div>

    <div class="hero-title">
        CULTIVATE<br>
        <span>INTELLIGENCE.</span>
    </div>

    <div class="hero-description">

        Transform historical agricultural information
        into crop-specific machine-learning predictions.

        Explore yield behaviour, geographical patterns,
        historical trends and model performance through
        one intelligent agricultural platform.

    </div>

</div>
""")


# ==========================================================
# TOP METRICS
# ==========================================================

c1, c2, c3, c4 = st.columns(4)


with c1:

    st.metric(
        "CROP",
        crop
    )


with c2:

    st.metric(
        "MODEL",
        crop_info["best_algorithm"]
    )


with c3:

    st.metric(
        "TEMPORAL R²",
        f"{crop_info['test_r2']:.3f}"
    )


with c4:

    st.metric(
        "RECORDS",
        crop_info["records"]
    )


st.write("")


# ==========================================================
# TABS
# ==========================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "YIELD PREDICTION",
    "HISTORICAL INTELLIGENCE",
    "MODEL PERFORMANCE",
    "PROJECT"
])


# ==========================================================
# TAB 1 - PREDICTION
# ==========================================================

with tab1:

    st.html("""
    <div class="section-index">
        01 / PREDICTION ENGINE
    </div>

    <div class="section-heading">
        Agricultural Conditions
    </div>
    """)


    left, right = st.columns(2)


    with left:

        year = st.number_input(
            "Crop Year",
            min_value=int(
                crop_info["minimum_year"]
            ),
            max_value=int(
                crop_info["maximum_year"]
            ),
            value=int(
                crop_info["maximum_year"]
            ),
            step=1
        )


        season = st.selectbox(
            "Season",
            crop_info["seasons"]
        )


        state = st.selectbox(
            "State",
            crop_info["states"]
        )


        area = st.number_input(
            "Cultivated Area",
            min_value=0.01,
            value=float(
                crop_df["Area"].median()
            )
        )


    with right:

        rainfall = st.number_input(
            "Annual Rainfall",
            min_value=0.0,
            value=float(
                crop_df[
                    "Annual_Rainfall"
                ].median()
            )
        )


        fertilizer = st.number_input(
            "Fertilizer",
            min_value=0.0,
            value=float(
                crop_df[
                    "Fertilizer"
                ].median()
            )
        )


        pesticide = st.number_input(
            "Pesticide",
            min_value=0.0,
            value=float(
                crop_df[
                    "Pesticide"
                ].median()
            )
        )


    st.write("")


    if st.button(
        "RUN YIELD PREDICTION"
    ):

        input_data = pd.DataFrame([{
            "Crop_Year":
                year,

            "Season":
                season,

            "State":
                state,

            "Area":
                area,

            "Annual_Rainfall":
                rainfall,

            "Fertilizer":
                fertilizer,

            "Pesticide":
                pesticide
        }])


        prediction = model.predict(
            input_data
        )[0]


        st.html(
            f"""
            <div class="result">

                <div class="result-ring"></div>

                <div class="result-tag">
                    ML YIELD ESTIMATION
                </div>

                <div class="result-number">
                    {prediction:.4f}
                </div>

                <div class="result-description">
                    Predicted {safe_crop} Yield
                </div>

                <div style="
                    margin-top: 2rem;
                    font-family: 'DM Mono', monospace;
                    color: #4eff99;
                    letter-spacing: 0.12rem;
                    font-size: 0.72rem;
                ">
                    MODEL /
                    {safe_model.upper()}
                </div>

            </div>
            """
        )


        st.write("")


        st.html("""
        <div class="section-index">
            INPUT DATA
        </div>

        <div class="section-heading">
            Prediction Summary
        </div>
        """)


        display_data = input_data.copy()


        display_data.insert(
            0,
            "Crop",
            crop
        )


        st.dataframe(
            display_data,
            hide_index=True,
            width="stretch"
        )


# ==========================================================
# TAB 2 - HISTORICAL
# ==========================================================

with tab2:

    st.html(
        f"""
        <div class="section-index">
            02 / HISTORICAL INTELLIGENCE
        </div>

        <div class="section-heading">
            {safe_crop} Yield Behaviour
        </div>
        """
    )


    historical = (
        crop_df
        .groupby(
            "Crop_Year"
        )["Yield"]
        .mean()
    )


    st.line_chart(
        historical,
        height=430
    )


    h1, h2, h3 = st.columns(3)


    with h1:

        st.metric(
            "AVERAGE YIELD",
            f"{crop_df['Yield'].mean():.3f}"
        )


    with h2:

        st.metric(
            "MINIMUM YIELD",
            f"{crop_df['Yield'].min():.3f}"
        )


    with h3:

        st.metric(
            "MAXIMUM YIELD",
            f"{crop_df['Yield'].max():.3f}"
        )


    st.write("")


    st.html("""
    <div class="section-index">
        GEOGRAPHIC INTELLIGENCE
    </div>

    <div class="section-heading">
        Leading States
    </div>
    """)


    state_yield = (
        crop_df
        .groupby(
            "State"
        )["Yield"]
        .mean()
        .sort_values(
            ascending=False
        )
        .head(12)
    )


    st.bar_chart(
        state_yield,
        height=430
    )


# ==========================================================
# TAB 3 - PERFORMANCE
# ==========================================================

with tab3:

    st.html("""
    <div class="section-index">
        03 / MODEL PERFORMANCE
    </div>

    <div class="section-heading">
        Temporal Validation
    </div>
    """)


    p1, p2, p3 = st.columns(3)


    with p1:

        st.metric(
            "MAE",
            f"{crop_info['test_mae']:.4f}"
        )


    with p2:

        st.metric(
            "RMSE",
            f"{crop_info['test_rmse']:.4f}"
        )


    with p3:

        st.metric(
            "R²",
            f"{crop_info['test_r2']:.4f}"
        )


    st.info(
        "Training period: 1997–2017 | "
        "Temporal testing period: 2018–2020"
    )


    st.markdown(
        """
### Validation Methodology

Three regression algorithms were evaluated:

- **Linear Regression**
- **Random Forest Regression**
- **Gradient Boosting Regression**

Models were compared using:

- Mean Absolute Error
- Root Mean Squared Error
- R² score

Models with negative temporal R² were removed from
the user-facing system.
        """
    )


# ==========================================================
# TAB 4 - PROJECT
# ==========================================================

with tab4:

    st.html("""
    <div class="section-index">
        04 / PROJECT
    </div>

    <div class="section-heading">
        AgriYield AI
    </div>
    """)


    st.markdown(
        """
### Machine Learning for Agricultural Decision Support

AgriYield AI is a multi-crop yield-prediction prototype
developed using historical agricultural records.

### Inputs

- Crop year
- Season
- State
- Cultivated area
- Annual rainfall
- Fertilizer usage
- Pesticide usage

### Development Pipeline

**01 — Data Collection**

Historical crop records were collected and organized.

**02 — Data Cleaning**

Missing values, duplicate records and inconsistent
categories were investigated.

**03 — Exploratory Data Analysis**

Yield distributions, rainfall relationships and crop
characteristics were studied.

**04 — Machine Learning**

Linear Regression, Random Forest and Gradient Boosting
were evaluated.

**05 — Temporal Validation**

Historical years were used for training and later years
for testing.

**06 — Crop-Specific Architecture**

Separate models were created because yield behaviour
differs significantly between crops.

**07 — Deployment**

Validated models were integrated into a Streamlit web
application.

### Important Limitation

The historical dataset ends in 2020.

AgriYield AI is an internship and research prototype.
It should not replace official agricultural forecasts
or professional agronomic advice.
        """
    )


# ==========================================================
# FOOTER
# ==========================================================

st.html("""
<div class="footer">

    AGRIYIELD AI / AGRICULTURAL INTELLIGENCE

    <br><br>

    MACHINE LEARNING BASED MULTI-CROP
    YIELD PREDICTION SYSTEM

</div>
""")