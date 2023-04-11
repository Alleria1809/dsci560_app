import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(page_title="Recommendation", page_icon=":sparkles:", layout="wide")

st.markdown("# LA Restaurant Recommendation")
st.sidebar.header("Restaurant Recommendation")

st.markdown(
    """
    #### This page is designed for the citizens.
    """
)

# choice = st.selectbox("What's in your mind?", ['I want to explore the features!', 'I already have some ideal restaurants in my mind!'])

# st.markdown("#### What\'s in your mind?")
choice = st.radio(
    "What\'s in your mind?",
    (['I want to explore the features!', 'I already have some ideal restaurants in my mind!']))


if choice == 'I want to explore the features!':
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
    
else:
    name =  st.text_input(
        "Please input your ideal restaurant name: ",
        "CAVA",
        key="restaurant_name",
    )

# restaurants = {'Plan Check Kitchen + Bar': 'https://www.yelp.com/biz/plan-check-kitchen-bar-los-angeles-9',
#                "Justin Queso's Tex-Mex Restaurant & Bar": 'https://www.yelp.com/biz/justin-quesos-tex-mex-restaurant-and-bar-west-hollywood?osq=Restaurants',
#                'Sunny Grill': 'https://www.yelp.com/biz/sunny-grill-los-angeles?osq=Restaurants',}

# st.markdown(f'#### common features of these restaurants: ')
# for feature, link in restaurants.items():
#     st.write(f'[{feature}]({link})')
    
# progress_bar = st.sidebar.progress(0)
# status_text = st.sidebar.empty()
# last_rows = np.random.randn(1, 1)

# for i in range(1, 101):
#     # new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
#     status_text.text("%i%% Complete" % i)
#     progress_bar.progress(i)
#     # last_rows = new_rows
#     time.sleep(0.05)

# progress_bar.empty()

# # Streamlit widgets automatically run the script from top to bottom. Since
# # this button is not connected to any other logic, it just causes a plain
# # rerun.
# st.button("Re-run")