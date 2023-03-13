import streamlit as st
import time
import numpy as np
import pandas as pd
import plotly.express as px
from io import StringIO
import random

st.set_page_config(page_title="Risk Prediction", page_icon="📈", layout="wide")

st.markdown("# LA Restaurant Health Violation Risk Prediction")
st.sidebar.header("Risk Prediction")

st.markdown(
    """
    #### This page is designed for the government.
    Please select or upload the features of the restaurants:
    """
)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # # To convert to a string based IO:
    # stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    # # st.write(stringio)
    #
    # # To read file as string:
    # string_data = stringio.read()
    # st.write(string_data)
    #
    # # Can be used wherever a "file-like" object is accepted:
    # dataframe = pd.read_csv(uploaded_file)
    # st.write(dataframe)

options_feature = st.multiselect(
    'Please select the inpute features:',
    ['Category', 'Price', 'Yelp Score', 'Risk', 'Location'],)


st.subheader("")
chart_data = pd.DataFrame(
    {'Restaurant': ['USC Pancake', 'USC Noodles', 'USC Fried Chicken'],
     'Risk': ['low', 'medium', 'high']})

fig = px.bar(
    chart_data,
    x='Restaurant',
    y='Risk',
    color=['green', 'blue', 'red']
)
st.plotly_chart(fig, theme="streamlit", use_container_width=True)


progress_bar = st.sidebar.progress(0)
status_text = st.sidebar.empty()
last_rows = np.random.randn(1, 1)

for i in range(1, 101):
    # new_rows = last_rows[-1, :] + np.random.randn(5, 1).cumsum(axis=0)
    status_text.text("%i%% Complete" % i)
    progress_bar.progress(i)
    # last_rows = new_rows
    time.sleep(0.05)

progress_bar.empty()

# Streamlit widgets automatically run the script from top to bottom. Since
# this button is not connected to any other logic, it just causes a plain
# rerun.
st.button("Re-run")