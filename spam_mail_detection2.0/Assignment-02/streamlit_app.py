import streamlit as st
import requests
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "https://spam-mail-detector-2-0-3.onrender.com"

st.set_page_config(
    page_title="Spam Detector 2.0",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Hero */
.hero {
    padding: 30px;
    border-radius: 20px;
    background: linear-gradient(135deg, #111827, #374151);
    color: white;
    margin-bottom: 25px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 5px;
}

.hero p {
    font-size: 18px;
    opacity: 0.85;
}

/* Feature cards */
.feature-card {
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #444;
    min-height: 150px;
}

/* Footer */
.footer {
    text-align: center;
    padding: 25px;
    opacity: 0.7;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# SESSION STATE
# ============================================================

if "message" not in st.session_state:
    st.session_state.message = ""


# ============================================================
# EXAMPLE FUNCTIONS
# ============================================================

def set_ham_example():
    st.session_state.message = (
        "Hey, are we still meeting for lunch today?"
    )


def set_spam_example():
    st.session_state.message = (
        "Congratulations! You have won a free prize. "
        "Call now to claim your reward."
    )


def set_phishing_example():
    st.session_state.message = (
        "Your bank account has been suspended. "
        "Verify your OTP and password immediately."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown("""
<div class="hero">

<h1>🛡️ Spam Detector 2.0</h1>

<p>
AI-powered SMS & Email Classification System
</p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# FLASK API STATUS
# ============================================================

try:

    health_response = requests.get(
        f"{API_URL}/health",
        timeout=3
    )

    if health_response.status_code == 200:

        st.success("🟢 Flask API is online")

    else:

        st.error("🔴 Flask API is not healthy")

except requests.exceptions.RequestException:

    st.error(
        "🔴 Flask API is offline. "
        "Start Flask using: python app.py"
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("🛡️ Spam Detector 2.0")

    st.markdown("""
### Classification

🟢 **Ham**

Normal / legitimate message

🔴 **Spam**

Unwanted or promotional message

🟠 **Phishing**

Attempts to steal sensitive information
""")

    st.divider()

    st.markdown("""
### ⚙️ Technology

**Frontend:** Streamlit

**Backend:** Flask

**Algorithm:** Logistic Regression

**Features:** TF-IDF

**Classes:** 3
""")

    st.divider()

    st.info(
        "Enter a message or select an example "
        "to test the classifier."
    )


# ============================================================
# TRY AN EXAMPLE
# ============================================================

st.subheader("💬 Try an Example")

col1, col2, col3 = st.columns(3)

with col1:

    st.button(
        "🟢 Normal Message",
        use_container_width=True,
        on_click=set_ham_example
    )

with col2:

    st.button(
        "🔴 Spam Message",
        use_container_width=True,
        on_click=set_spam_example
    )

with col3:

    st.button(
        "🟠 Phishing Message",
        use_container_width=True,
        on_click=set_phishing_example
    )


# ============================================================
# MESSAGE INPUT
# ============================================================

st.subheader("🔍 Analyze a Message")

message = st.text_area(
    "Enter your SMS or email message",
    height=180,
    placeholder=(
        "Example: Your bank account has been suspended. "
        "Verify your OTP immediately."
    ),
    key="message"
)


# ============================================================
# ANALYZE BUTTON
# ============================================================

analyze_button = st.button(
    "🔎 Analyze Message",
    type="primary",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if analyze_button:

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    if not message.strip():

        st.warning(
            "⚠️ Please enter a message before analyzing."
        )

    else:

        try:

            # ------------------------------------------------
            # CALL FLASK API
            # ------------------------------------------------

            with st.spinner("🤖 Analyzing message..."):

                response = requests.post(
                    f"{API_URL}/predict",
                    json={
                        "message": message
                    },
                    timeout=10
                )

            # ------------------------------------------------
            # SUCCESS
            # ------------------------------------------------

            if response.status_code == 200:

                result = response.json()

                prediction = result.get(
                    "class",
                    "Unknown"
                )

                confidence = result.get(
                    "confidence",
                    0
                )

                explanation = result.get(
                    "explanation",
                    "No explanation available."
                )

                # Make sure confidence is numeric

                try:
                    confidence = float(confidence)
                except:
                    confidence = 0.0


                # ==================================================
                # RESULT
                # ==================================================

                st.divider()

                st.subheader("🎯 Prediction Result")

                prediction_lower = str(
                    prediction
                ).lower()


                # ------------------------------------------------
                # HAM
                # ------------------------------------------------

                if prediction_lower == "ham":

                    st.success(
                        f"🟢 HAM\n\n"
                        f"**Confidence:** {confidence:.2%}\n\n"
                        f"This message appears to be legitimate."
                    )


                # ------------------------------------------------
                # SPAM
                # ------------------------------------------------

                elif prediction_lower == "spam":

                    st.error(
                        f"🔴 SPAM\n\n"
                        f"**Confidence:** {confidence:.2%}\n\n"
                        f"This message appears to be unwanted "
                        f"or promotional."
                    )


                # ------------------------------------------------
                # PHISHING
                # ------------------------------------------------

                elif prediction_lower == "phishing":

                    st.warning(
                        f"🟠 PHISHING\n\n"
                        f"**Confidence:** {confidence:.2%}\n\n"
                        f"This message may be attempting to "
                        f"obtain sensitive information."
                    )


                # ------------------------------------------------
                # UNKNOWN
                # ------------------------------------------------

                else:

                    st.info(
                        f"⚪ {str(prediction).upper()}\n\n"
                        f"**Confidence:** {confidence:.2%}"
                    )


                # ==================================================
                # DETAILS
                # ==================================================

                detail_col1, detail_col2 = st.columns(2)


                # ------------------------------------------------
                # EXPLANATION
                # ------------------------------------------------

                with detail_col1:

                    st.subheader("🧠 Why this prediction?")

                    st.info(
                        explanation
                    )


                # ------------------------------------------------
                # MESSAGE
                # ------------------------------------------------

                with detail_col2:

                    st.subheader("📩 Analyzed Message")

                    st.code(
                        message
                    )


                # ==================================================
                # CONFIDENCE
                # ==================================================

                st.subheader("📊 Confidence")

                confidence_df = pd.DataFrame(
                    {
                        "Class": [
                            str(prediction)
                        ],
                        "Confidence": [
                            confidence
                        ]
                    }
                )

                st.bar_chart(
                    confidence_df.set_index("Class")
                )


                # ==================================================
                # API RESPONSE
                # ==================================================

                with st.expander(
                    "🔧 View Flask API Response"
                ):

                    st.json(result)


            # ------------------------------------------------
            # API ERROR
            # ------------------------------------------------

            else:

                st.error(
                    f"❌ Flask API returned "
                    f"status code {response.status_code}"
                )

                try:

                    st.json(
                        response.json()
                    )

                except:

                    st.code(
                        response.text
                    )


        # ----------------------------------------------------
        # CONNECTION ERROR
        # ----------------------------------------------------

        except requests.exceptions.ConnectionError:

            st.error(
                "❌ Cannot connect to Flask API."
            )

            st.info(
                "Make sure Flask is running with:\n\n"
                "python app.py"
            )


        # ----------------------------------------------------
        # TIMEOUT ERROR
        # ----------------------------------------------------

        except requests.exceptions.Timeout:

            st.error(
                "⏱️ Flask API took too long to respond."
            )


        # ----------------------------------------------------
        # OTHER ERROR
        # ----------------------------------------------------

        except Exception as e:

            st.error(
                f"❌ Unexpected error: {e}"
            )


# ============================================================
# HOW IT WORKS
# ============================================================

st.divider()

st.subheader("🚀 How It Works")

info1, info2, info3 = st.columns(3)


with info1:

    st.markdown("""
### 1️⃣ Enter Message

Enter an SMS or email message
or select one of the examples.
""")


with info2:

    st.markdown("""
### 2️⃣ Flask API

Streamlit sends the message
to the Flask `/predict`
endpoint.
""")


with info3:

    st.markdown("""
### 3️⃣ AI Prediction

The ML model returns the
class, confidence and explanation.
""")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown("""
<div class="footer">

<b>🛡️ Spam Detector 2.0</b>

<br><br>

Flask API • Streamlit • TF-IDF • Logistic Regression

</div>
""", unsafe_allow_html=True)
