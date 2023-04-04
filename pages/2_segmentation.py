import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from PIL import Image
import plotly.express as px
import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)


st.set_page_config(page_title="Segmentation", page_icon=":chopsticks:", layout="wide")

st.markdown("# LA Restaurant Segmentation")
st.sidebar.header("Restaurant Segmentation")

st.markdown(
    """
    #### This page is designed for the government.
    """
)

# uploaded_file = st.file_uploader("Choose a file")
# if uploaded_file is not None:
#     # To read file as bytes:
#     bytes_data = uploaded_file.getvalue()
#     st.write(bytes_data)


# @st.cache_data
# load raw features
def raw_data():
    model_df = pd.read_csv('features_for_segmentation.csv')
    return model_df


# @st.cache_data
# load preprocessed features
def load_data():
    model_df = pd.read_csv('features_0403.csv')
    return model_df

def pca(std_df):
    #Initiating PCA to reduce dimentions aka features to 3
    pca = PCA(n_components=3)
    pca.fit(std_df)
    PCA_ds = pd.DataFrame(pca.transform(std_df), columns=(["feature1","feature2", "feature3"]))
    # PCA_ds.describe().T
    return PCA_ds

def kmean(best_k):
    model = KMeans(n_clusters=best_k, random_state=2009)
    model.fit(PCA_ds)
    labels = model.labels_
    return labels
    
def find_k(PCA_ds):
    model = KMeans(random_state=2009)
    X = np.array(PCA_ds)
    visualizer = KElbowVisualizer(model, k=(1,11)).fit(X)
    best_k = visualizer.elbow_value_ # Get elbow value
    return best_k
    
def pca_plot(PCA_ds, best_k, X_df):
    labels = kmean(best_k)
    #Adding the Clusters feature to the orignal dataframe.
    PCA_ds['clusters'] = labels
    X_df['clusters'] = labels
    
    #A 3D Projection Of Data In The Reduced Dimension
    x =PCA_ds["feature1"]
    y =PCA_ds["feature2"]
    z =PCA_ds["feature3"]
    #To plot
    fig = plt.figure(figsize=(10,8))
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(x,y,z, c="maroon", marker="o" )
    # ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
    # plt.show()
    
    fig = px.scatter_3d(PCA_ds, x, y, z, color='clusters', title="A 3D Projection Of Data In The PCA Reduced Dimension(3 PCs)",
                        color_continuous_scale=px.colors.sequential.Inferno)
    # tight layout
    # fig.show()
    return fig, X_df

def count_plot(X_df):
    fig = px.histogram(X_df, x=X_df['clusters'], title="Distribution Of The Clusters",
                    color = 'clusters')
    fig.update_layout(bargap=0.2)
    return fig

def scatter_plot_2d(X_df, x, y):
    fig = px.scatter(X_df, x=x, y=y,color='clusters', color_continuous_scale=px.colors.sequential.Inferno,
                     title = f"2D plot of {x} and {y}")
    return fig

def scatter_plot_3d(X_df, x, y, z):
    fig = px.scatter_3d(X_df, x=x, y=y, z=z,color='clusters', 
                        color_continuous_scale=px.colors.sequential.Inferno,
                     title = f"3D plot of {x}, {y} and {z}")
    # fig = px.scatter(X_df, x=x, y=y,color='clusters', color_continuous_scale=px.colors.sequential.Inferno,
    #                  title = f"2D plot of {x} and {y}")
    return fig

def result_plot(res):
    res.rename({'binned_score':'score_level', 'score':'count'}, axis=1, inplace=True)
    fig = px.histogram(res, x=res['clusters'], y='count',title="Distribution of Score Levels among the Clusters",
                    color = 'score_level', nbins = len(res['clusters'].unique()))
    fig.update_layout(bargap=0.5)
    return fig
    
    
raw_df = raw_data()
with st.expander("See raw data"):
    st.write('Show the raw data.')
    # show the raw dataframe
    st.dataframe(raw_df.drop(columns=['name']), 1000, 300)


model_df = load_data()
X_df = model_df.copy(deep=True)
y_df = X_df.binned_score # true segments

# standardize the data
std_scaler = StandardScaler()
std_df = pd.DataFrame(std_scaler.fit_transform(X_df.drop(columns=['binned_score','binned_score_y'])), columns=X_df.drop(columns=['binned_score','binned_score_y']).columns)
# cluster_df_scaled = std_df.copy(deep=True)

PCA_ds = pca(std_df)
# visualizer.show()
best_k = find_k(PCA_ds)
st.markdown(f"##### The current best number of clusters is {best_k}")

# allow the user to select the number of clusters
cluster_slider = st.sidebar.slider(
    min_value=1, max_value=10, value=int(best_k), label="Please select the number of clusters: "
)
best_k = cluster_slider
fig, X_df = pca_plot(PCA_ds, best_k, X_df)
st.plotly_chart(fig)

st.markdown(f"##### Statistics with {best_k} cluster(s)")
# 1. count plot
count_fig = count_plot(X_df)
st.plotly_chart(count_fig)


# 2. result
res = X_df.groupby(['clusters','binned_score']).count()['score'].reset_index()
with st.expander("See resulted data"):
    st.write('Show the resulted data.')
    # show the raw dataframe
    st.dataframe(res, 1000, 300)
res_fig = result_plot(res)
st.plotly_chart(res_fig)
 
    
# 3. scatter plots
attributes = []
dimension = st.sidebar.selectbox("3D plot?", [False, True])

attributes = ['score', 'open_hours_week', 'coded_size',
       'coded_review_counts', 'coded_price', 'num_photos',
       'num_attributes', 'num_questions',
       'polarity', 'subjectivity']
if dimension:
    feature_x = st.sidebar.selectbox("Please select x axis:", attributes[5:]+attributes[:5])
    feature_y = st.sidebar.selectbox("Please select y axis:", attributes[3:]+attributes[:3])
    feature_z = st.sidebar.selectbox("Please select y axis:", attributes[4:]+attributes[:4])
    fig = scatter_plot_3d(X_df, feature_x, feature_y, feature_z)
    plt.figure(figsize=(5, 3))
    st.plotly_chart(fig)
else:
    feature_x = st.sidebar.selectbox("Please select x axis:", attributes[5:]+attributes[:5])
    feature_y = st.sidebar.selectbox("Please select y axis:", attributes[3:]+attributes[:3])
    scatter_fig2d = scatter_plot_2d(X_df,feature_x, feature_y)
    st.plotly_chart(scatter_fig2d)
    
# if individual:
#     fig, ax = plt.subplots(ncols=n_labels, figsize=(5, 3))
# else:
#     fig, ax = plt.subplots(figsize=(7, 4))

# for i, yi in enumerate(set(labels)):
#     if not individual:
#         a = ax
#     else:
#         a = ax[i]

#     xi = X[labels == yi]
#     x_pts = xi[:, 0]
#     y_pts = xi[:, 1]
#     a.scatter(x_pts, y_pts, c=clrs[yi])

#     if selectbox:
#         confidence_ellipse(
#             x=x_pts,
#             y=y_pts,
#             ax=a,
#             edgecolor="black",
#             facecolor=clrs[yi],
#             alpha=0.2,
#             # n_std=stdbox,
#         )

# plt.tight_layout()
# plt.legend([f"cluster {i}" for i in range(cluster_slider)])
# plt.figure(figsize=(5, 3))

# col00, col01, col02 = st.columns([1,3,1])
# with col01:
#     st.write(fig)
# #st.plotly_chart(fig)

# # count_df = pd.read_csv('count_res.csv')
# # clusters = list(count_df.index)
# # counts = list(count_df.score.values)
  
# # fig = plt.figure(figsize = (10, 5))
 
# # # creating the bar plot
# # plt.bar(clusters, counts)
# col10, col11, col12 = st.columns([1,3,1])
# with col11:
#     st.markdown('#### Restaurant Count in each cluster')
#     count_df = {'cluster':[0,1,2], 'count':[1420, 1093, 271]}
#     fig = px.bar(
#         count_df,
#         x='cluster',
#         y='count',
#         color=['green', 'blue', 'red']
#     )
#     st.plotly_chart(fig, theme="streamlit", use_container_width=True)

# # res_features = ['Noodles', 'Near USC', 'Low Price']
# # st.markdown(f'#### common features of these restaurants: ')
# # for idx, feature in enumerate(res_features):
# #     st.write(f'cluster {idx} : {feature}')

# col1, col2= st.columns(2)
# with col1:
#     st.markdown('#### Yelp Score Mean in each cluster')
#     image1 = Image.open('yelp_score_mean.jpg')
#     st.image(image1, caption='yelp_score_mean in clusters')

# with col2:
#     st.markdown('#### Yelp Review Counts Mean in each cluster')
#     image2 = Image.open('yelp_review_count_mean.jpg')
#     st.image(image2, caption='yelp_review_count_mean in clusters')

# col3, col4= st.columns(2)
# with col3:
#     st.markdown('#### Yelp Open Hour Mean in each cluster')
#     image3 = Image.open('yelp_open_hour_mean.jpg')
#     st.image(image3, caption='yelp_open_hour_mean in clusters')

# with col4:
#     st.markdown('#### Yelp Price & Size Mean in each cluster')
#     image4 = Image.open('price_size_mean.jpg')
#     st.image(image4, caption='price_size_mean in clusters')

# col50, col51, col52 = st.columns([1,2,1])
# with col51:
# #     st.markdown('#### Yelp Price & Size Mean in each cluster')
# #     image5 = Image.open('res.jpg')
# #     st.image(image5, caption='Results in clusters')
#     st.markdown('#### Below is the result:')
#     res_df = pd.read_csv('res.csv')
#     st.write(res_df)
