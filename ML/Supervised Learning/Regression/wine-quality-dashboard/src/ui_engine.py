import streamlit as st


def setup_ui():

    # -------------------------
    # PAGE CONFIG (GLOBAL)
    # -------------------------
    st.set_page_config(
        page_title="Wine AI Dashboard",
        page_icon="🍷",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # -------------------------
    # LOAD CSS
    # -------------------------
    with open("assets/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

    # -------------------------
    # OPTIONAL GLOBAL HEADER STYLE
    # -------------------------
    st.markdown("""
        <style>
        /* remove Streamlit default padding */
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        /* hide Streamlit footer */
        footer {visibility: hidden;}

        /* hide hamburger menu */
        #MainMenu {visibility: hidden;}
        </style>
    """, unsafe_allow_html=True)