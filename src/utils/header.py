import streamlit as st


def page_header(title, subtitle="", emoji=""):
    """Render a page header with title, subtitle."""
    st.markdown(f"# {emoji} {title}")
    if subtitle:
        st.markdown(f"*{subtitle}*")
    st.markdown("---")
