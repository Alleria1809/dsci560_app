import streamlit as st

st.set_page_config(
    page_title="LA Restaurant Inspection Portal",
    page_icon=":shallow_pan_of_food:",
    layout="wide"
)

st.write("# Welcome to LA Restaurant Inspection Portal!")

st.sidebar.success("Select a page above.")

st.markdown(
    """
    ##### This website is for improving the LA Restaurant inspection.
    There are 3 main functions:
    - Prediction: Extract valid predictors of restaurant scores for predictions 
    on restaurant qualities and health risks.
    - Segmentation: visualize the segmentation of restaurants to 
    provide insights on the adjustment of inspection frequencies.
    - Recommendation: Recommend the restaurants based on the citizens' preferences.
"""
)

st.markdown(
    """
    ##### Team HAL9000
    Bella Chen, Xiaoyi Gu, Ying Wang, Zhenmin Hua
"""
)