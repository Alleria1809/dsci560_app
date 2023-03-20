import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from PIL import Image
import plotly.express as px

st.set_page_config(page_title="Segmentation", page_icon=":chopsticks:", layout="wide")

st.markdown("# LA Restaurant Segmentation")
st.sidebar.header("Restaurant Segmentation")

st.markdown(
    """
    #### This page is designed for the government.
    Please select or upload the data of the restaurants:
    """
)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

# start_year_to_filter, end_year_to_filter = st.select_slider(
#         'Select a range of year',
#         options=[x for x in range(2017, 2023)],
#         value=(2018, 2020))
# st.write('You selected year between', start_year_to_filter, 'and', end_year_to_filter)

# options = st.multiselect(
#     'Please select the restaurant category:',
#     ['Brunch', 'Japanese', 'American', 'Chinese'],)

# options_risk = st.multiselect(
#     'Please select the risk level:',
#     ['Low', 'Medium', 'High'],)

# st.write('You selected:', options)


def confidence_ellipse(x, y, ax, n_std=3.0, facecolor="none", **kwargs):
    if x.size != y.size:
        raise ValueError("x and y must be the same size")
    cov = np.cov(x, y)
    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        **kwargs
    )
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)
    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mean_x, mean_y)
    )
    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

# @st.cache_data
def data():
    X = np.random.normal(0, 1, 1000).reshape(-1, 2)
    return X


X = data()

cluster_slider = st.slider(
    min_value=1, max_value=6, value=3, label="Number of clusters: "
)
kmeans = KMeans(n_clusters=cluster_slider, random_state=0).fit(X)
labels = kmeans.labels_

# selectbox = st.selectbox("Visualize confidence bounds", [False, True])
# stdbox = st.selectbox("Number of standard deviations: ", [1, 2, 3])
selectbox = False

clrs = ["red", "seagreen", "orange", "blue", "yellow", "purple"]

n_labels = len(set(labels))

individual = st.selectbox("Individual subplots?", [False, True])

if individual:
    fig, ax = plt.subplots(ncols=n_labels, figsize=(5, 3))
else:
    fig, ax = plt.subplots(figsize=(7, 4))

for i, yi in enumerate(set(labels)):
    if not individual:
        a = ax
    else:
        a = ax[i]

    xi = X[labels == yi]
    x_pts = xi[:, 0]
    y_pts = xi[:, 1]
    a.scatter(x_pts, y_pts, c=clrs[yi])

    if selectbox:
        confidence_ellipse(
            x=x_pts,
            y=y_pts,
            ax=a,
            edgecolor="black",
            facecolor=clrs[yi],
            alpha=0.2,
            # n_std=stdbox,
        )

plt.tight_layout()
plt.legend([f"cluster {i}" for i in range(cluster_slider)])
plt.figure(figsize=(5, 3))

col00, col01, col02 = st.columns([1,3,1])
with col01:
    st.write(fig)
#st.plotly_chart(fig)

# count_df = pd.read_csv('count_res.csv')
# clusters = list(count_df.index)
# counts = list(count_df.score.values)
  
# fig = plt.figure(figsize = (10, 5))
 
# # creating the bar plot
# plt.bar(clusters, counts)
col10, col11, col12 = st.columns([1,3,1])
with col11:
    st.markdown('#### Restaurant Count in each cluster')
    count_df = {'cluster':[0,1,2], 'count':[1420, 1093, 271]}
    fig = px.bar(
        count_df,
        x='cluster',
        y='count',
        color=['green', 'blue', 'red']
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# res_features = ['Noodles', 'Near USC', 'Low Price']
# st.markdown(f'#### common features of these restaurants: ')
# for idx, feature in enumerate(res_features):
#     st.write(f'cluster {idx} : {feature}')

col1, col2= st.columns(2)
with col1:
    st.markdown('#### Yelp Score Mean in each cluster')
    image1 = Image.open('yelp_score_mean.jpg')
    st.image(image1, caption='yelp_score_mean in clusters')

with col2:
    st.markdown('#### Yelp Review Counts Mean in each cluster')
    image2 = Image.open('yelp_review_count_mean.jpg')
    st.image(image2, caption='yelp_review_count_mean in clusters')

col3, col4= st.columns(2)
with col3:
    st.markdown('#### Yelp Open Hour Mean in each cluster')
    image3 = Image.open('yelp_open_hour_mean.jpg')
    st.image(image3, caption='yelp_open_hour_mean in clusters')

with col4:
    st.markdown('#### Yelp Price & Size Mean in each cluster')
    image4 = Image.open('price_size_mean.jpg')
    st.image(image4, caption='price_size_mean in clusters')

col50, col51, col52 = st.columns([1,2,1])
with col51:
#     st.markdown('#### Yelp Price & Size Mean in each cluster')
#     image5 = Image.open('res.jpg')
#     st.image(image5, caption='Results in clusters')
    st.markdown('#### Below is the result:')
    res_df = pd.read_csv('res.csv')
    st.write(res_df)
