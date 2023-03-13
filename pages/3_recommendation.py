import streamlit as st
import pandas as pd


st.set_page_config(page_title="Recommendation", page_icon=":sparkles:", layout="wide")

st.markdown("# LA Restaurant Recommendation")
st.sidebar.header("Restaurant Recommendation")

st.markdown(
    """
    #### This page is designed for the citizens.
    Please select the features of restaurants you prefer:
    """
)

options_cat = st.multiselect(
    'Please select the restaurant category:',
    ['Brunch', 'Japanese', 'American', 'Chinese'],)

options_price = st.multiselect(
    'Please select the restaurant price:',
    ['$', '$$', '$$$'],)

options_risk = st.multiselect(
    'Please select the restaurant risk:',
    ['Low', 'Medium', 'High'],)

options_location = st.multiselect(
    'Please select the restaurant location:',
    ['USC', 'UCLA', 'Hollywood'],)

restaurants = {'Plan Check Kitchen + Bar': 'https://www.yelp.com/biz/plan-check-kitchen-bar-los-angeles-9',
               "Justin Queso's Tex-Mex Restaurant & Bar": 'https://www.yelp.com/biz/justin-quesos-tex-mex-restaurant-and-bar-west-hollywood?osq=Restaurants',
               'Sunny Grill': 'https://www.yelp.com/biz/sunny-grill-los-angeles?osq=Restaurants',}

st.markdown(f'#### common features of these restaurants: ')
for feature, link in restaurants.items():
    st.write(f'[{feature}]({link})')