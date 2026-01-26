import streamlit as st


def apply_sidebar_style():
    """Apply custom sidebar styling to Streamlit app."""
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid #1e293b;
        }
        .stSidebar [data-testid="stMarkdownContainer"] p {
            color: #cbd5e1;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
