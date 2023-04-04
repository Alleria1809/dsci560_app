import streamlit as st
from PIL import Image
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
    ##### There are 3 main functions:
    - **_Prediction_**: Extract valid predictors of restaurant scores for predictions on restaurant qualities and health risks.
    - **_Segmentation_**: Visualize the segmentation of restaurants to provide insights on the adjustment of inspection frequencies.
    - **_Recommendation_**: Recommend the restaurants based on the citizens' preferences.
"""
)

c1, c2, c3 = st.columns(3)
with c1:
    st.info('**Open Data Source: [LA Open Data](https://data.lacity.org/Community-Economic-Development/Restaurant-and-Market-Health-Violations/ckya-qgys)**')
with c2:
    st.info('**GitHub: [@Project APP](https://github.com/Alleria1809/dsci560_app)**')    


c4, c5 = st.columns(2)
with c4:
    st.markdown(
        """
        ###### Team HAL9000
        Bella Chen, Xiaoyi Gu, Ying Wang, Zhenmin Hua
    """
    )
with c5:
    image = Image.open('hal9000_logo.jpeg')
    st.image(image, caption='Team Logo', width=100)