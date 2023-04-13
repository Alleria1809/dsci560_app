import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
# import matplotlib.transforms as transforms
from PIL import Image
import plotly.express as px
# import seaborn as sns
import warnings
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text 
import pickle
# import spacy


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

st.write("""
          Pipeline: Standardize data -> KMeans find the best number of clusters -> PCA visualization\
             + t-SNE visualization + Topic Modeling
         """)

# load raw features
# @st.cache_data
def raw_data():
    raw_df = pd.read_csv('features_for_segmentation_0403.csv')
    # raw_df = pd.read_csv('features_for_segmentation.csv')
    return raw_df

# load preprocessed features
# @st.cache_data
def load_data():
    model_df = pd.read_csv('features_0403.csv')
    # model_df = pd.read_csv('features.csv')  # up to date data
    return model_df

def pca(std_df):
    #Initiating PCA to reduce dimentions aka features to 3
    pca = PCA(n_components=3)
    pca.fit(std_df)
    PCA_ds = pd.DataFrame(pca.transform(std_df), columns=(["dim1","dim2", "dim3"]))
    # PCA_ds.describe().T
    return PCA_ds

def kmeans(df, best_k):
    model = KMeans(n_clusters=best_k, random_state=2009)
    model.fit(df)
    labels = model.labels_
    # save the labels generate with k-means(20)
    # pickle.dump(labels, open("labels.p", "wb" ))
    return labels
    
def find_k(df):
    model = KMeans(random_state=2009)
    X = np.array(df)
    visualizer = KElbowVisualizer(model, k=(1,11)).fit(X)
    best_k = visualizer.elbow_value_ # Get elbow value
    return best_k
    
def pca_plot(PCA_ds, best_k, X_df):
    labels = kmeans(PCA_ds, best_k)
    #Adding the Clusters feature to the orignal dataframe.
    PCA_ds['clusters'] = labels
    X_df['clusters'] = labels
    
    #A 3D Projection Of Data In The Reduced Dimension
    x =PCA_ds["dim1"]
    y =PCA_ds["dim2"]
    z =PCA_ds["dim3"]
    #To plot
    fig = plt.figure(figsize=(10,8))
    # ax = fig.add_subplot(111, projection="3d")
    # ax.scatter(x,y,z, c="maroon", marker="o" )
    # ax.set_title("A 3D Projection Of Data In The Reduced Dimension")
    # plt.show()
    
    fig = px.scatter_3d(PCA_ds, x, y, z, color='clusters', title="A 3D Projection Of Data In The PCA Reduced Dimension(3 PCs)",
                        color_continuous_scale=px.colors.sequential.Agsunset)
    # tight layout
    # fig.show()
    return fig, X_df

def count_plot(X_df):
    fig = px.histogram(X_df, x=X_df['clusters'], title="Distribution Of The Clusters",
                    color = 'clusters', color_discrete_sequence=px.colors.qualitative.Plotly)
    fig.update_layout(bargap=0.2)
    return fig

def scatter_plot_2d(X_df, x, y):
    fig = px.scatter(X_df, x=x, y=y,color='clusters', color_continuous_scale=px.colors.sequential.Agsunset,
                     title = f"2D plot of {x} and {y}")
    return fig

def scatter_plot_3d(X_df, x, y, z):
    fig = px.scatter_3d(X_df, x=x, y=y, z=z,color='clusters', 
                        color_continuous_scale=px.colors.sequential.Agsunset,
                     title = f"3D plot of {x}, {y} and {z}")
    # fig = px.scatter(X_df, x=x, y=y,color='clusters', color_continuous_scale=px.colors.sequential.Agsunset,
    #                  title = f"2D plot of {x} and {y}")
    return fig

def result_plot(res):
    res.rename({'binned_score':'score_level', 'score':'count'}, axis=1, inplace=True)
    fig = px.histogram(res, x=res['clusters'], y='count',title="Distribution of Score Levels among the Clusters",
                    color = 'score_level', 
                    color_discrete_map={'low':'#EF553B', 'medium':'#00CC96', 'high':'#636EFA'}, 
                    nbins = len(res['clusters'].unique()),
                    text_auto=True)
    fig.update_layout(bargap=0.5, width=450)
    return fig
    
def tsne(df, best_k, is_load):
    if not is_load:
        tsne = TSNE(verbose=1, perplexity=55, random_state=2009, n_iter=1000, learning_rate=200)  # Changed perplexity from 100 to 50 per FAQ
        X_embedded = tsne.fit_transform(df)
        # save the final t-SNE
        pickle.dump(X_embedded, open("X_embedded.p", "wb" ))
    else:
        file = open('X_embedded.p','rb')
        X_embedded = pickle.load(file)
        
    labels = kmeans(df, best_k)
    
    data_sne = pd.DataFrame({
    'axis-1': X_embedded[:,0],
    'axis-2': X_embedded[:,1],
    'cluster': labels
    })
    fig = px.scatter(data_sne, x='axis-1', y='axis-2', title='t-SNE with KMeans Labels(Based on PCA)',
                     color='cluster', color_continuous_scale=px.colors.sequential.Agsunset)
    fig.update_layout(width=450)
    return fig, X_embedded

# Functions for printing keywords for each topic
def selected_topics(model, vectorizer, top_n=3):
    current_words = []
    keywords = []
    
    for idx, topic in enumerate(model.components_):
        words = [(vectorizer.get_feature_names_out()[i], topic[i]) for i in topic.argsort()[:-top_n - 1:-1]]
        for word in words:
            if word[0] not in current_words:
                keywords.append(word)
                current_words.append(word[0])
                
    keywords.sort(key = lambda x: x[1])  
    keywords.reverse()
    return_values = []
    for ii in keywords:
        return_values.append(ii[0])
    # print('**',str(return_values))
    return return_values

# def spacy_tokenizer(sentence):
#     nlp = nlp = spacy.load("en_core_web_sm")
#     return [word.lemma_ for word in nlp(sentence) if not (word.like_num or word.is_stop or word.is_punct or word.is_space or len(word)==1)]

def LDA(best_k, X_df):
    # First, we will create 5 vectorizers, one for each of our cluster labels
    vectorizers = []
    
    # add stopwords
    my_additional_stop_words = {'nthe','just','blvd', 'comida','place', 'new', 'got', 'food', 'like', 'really'}
    stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)
    for ii in range(0, best_k):
        # Creating a vectorizer
        vectorizers.append(CountVectorizer(min_df=3, stop_words=stop_words, lowercase=True, token_pattern='[a-zA-Z\-][a-zA-Z\-]{2,}'))
    
    # Now we will vectorize the data from each of our clusters
    vectorized_data = []
    for current_cluster, cvec in enumerate(vectorizers):
        try:
            vectorized_data.append(cvec.fit_transform(X_df.loc[X_df['clusters'] == current_cluster, 'comments_list']))
        except Exception as e:
            # print("Not enough instances in cluster: " + str(current_cluster))
            vectorized_data.append(None)
            
    # number of topics per cluster
    NUM_TOPICS_PER_CLUSTER = best_k
    lda_models = []
    for ii in range(0, best_k):
        # Latent Dirichlet Allocation Model
        lda = LatentDirichletAllocation(n_components=NUM_TOPICS_PER_CLUSTER, max_iter=10, learning_method='online',verbose=False, random_state=2009)
        lda_models.append(lda)
    
    # For each cluster, we had created a corresponding LDA model in the previous step. We will now fit_transform all the LDA models on their respective cluster vectors
    clusters_lda_data = []
    for current_cluster, lda in enumerate(lda_models):
        # print("Current Cluster: " + str(current_cluster))
        
        if vectorized_data[current_cluster] != None:
            clusters_lda_data.append((lda.fit_transform(vectorized_data[current_cluster])))
    
    # Extracts the keywords from each cluster
    all_keywords = []
    for current_vectorizer, lda in enumerate(lda_models):
        # print("Current Cluster: " + str(current_vectorizer))

        if vectorized_data[current_vectorizer] != None:
            all_keywords.append(selected_topics(lda, vectorizers[current_vectorizer]))
    
    f=open('topics.txt','w')
    count = 0
    for ii in all_keywords:
        if vectorized_data[count] != None:
            f.write(', '.join(ii) + "\n")
        else:
            f.write("Not enough instances to be determined. \n")
            f.write(', '.join(ii) + "\n")
        count += 1
    f.close()
    
    return all_keywords
    
    
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
# print(std_df)

# reduce dimension to 3d at first
PCA_ds = pca(std_df)
best_k = find_k(PCA_ds)
# visualizer.show()
# run kmeans to find the best k
# best_k = find_k(PCA_ds)
st.markdown(f"##### The current best number of clusters is {best_k}")
# add the elbow plot?

# # allow the user to select the number of clusters, maybe not!
# cluster_slider = st.sidebar.slider(
#     min_value=1, max_value=10, value=int(best_k), label="Please select the number of clusters: "
# )
# best_k = cluster_slider
fig, X_df = pca_plot(PCA_ds, best_k, X_df)
st.plotly_chart(fig)

st.markdown(f"##### Statistics with {best_k} cluster(s)")
# 1. count plot
count_fig = count_plot(X_df)
st.plotly_chart(count_fig)
 
    
# 3. scatter plots
attributes = []
dimension = st.sidebar.selectbox("Would you like to look into 3 features?", [False, True])

# compile the raw features
# size_map = {'0-30':0, '31-60':1, '61-150':2, '151 + ':2}
attributes = ['score', 'open_hours_week', 'size',
       'coded_review_counts', 'price', 'num_photos',
       'num_attributes', 'num_questions',
       'polarity', 'subjectivity']
X_df['price'] = raw_df['price'].tolist()
X_df['size'] = raw_df['size'].tolist()


if dimension:
    feature_x = st.sidebar.selectbox("Please select x axis:", attributes[5:]+attributes[:5])
    feature_y = st.sidebar.selectbox("Please select y axis:", attributes[3:]+attributes[:3])
    feature_z = st.sidebar.selectbox("Please select z axis:", attributes[4:]+attributes[:4])
    fig = scatter_plot_3d(X_df, feature_x, feature_y, feature_z)
    plt.figure(figsize=(5, 3))
    st.plotly_chart(fig)
else:
    feature_x = st.sidebar.selectbox("Please select x axis:", attributes[5:]+attributes[:5])
    feature_y = st.sidebar.selectbox("Please select y axis:", attributes[3:]+attributes[:3])
    scatter_fig2d = scatter_plot_2d(X_df,feature_x, feature_y)
    st.plotly_chart(scatter_fig2d)
    
    
    
# 4. t-SNE plot
col1, col2 = st.columns(2)
with col1:
    # this tsne is transferred from pca to 2d
    tsne_fig, X_embedded = tsne(PCA_ds, best_k, is_load=True)  # need a saved tsne model, if have trained tsne, then is_load=True
    st.plotly_chart(tsne_fig, use_container_width=True)  


# 5. result shown with the keywords
res = X_df.groupby(['clusters','binned_score']).count()['score'].reset_index()
res.rename({'score':'count'}, inplace=True)
with st.expander("See resulted data"):
    st.write('Show the resulted data.')
    # show the raw dataframe
    st.dataframe(res, 1000, 300)
res_fig = result_plot(res)
with col2:
    st.plotly_chart(res_fig, use_container_width=True)

X_df['comments_list'] = raw_df['comments_list'].to_list()

# all_keywords = LDA(best_k, X_df)
# print(all_keywords)
# # st.write(all_keywords)

# load the stored keywords
all_keywords = []
stop_words_appended = ['really','don','good','great']
with open('topics.txt', 'r') as f:
    for line in f.readlines():
        words = line.split(',')
        # print(words)
        tmp = []
        for word in words:
            word = word.strip('\n')
            if word.strip(' ') not in stop_words_appended:
                tmp.append(word)
        all_keywords.append(','.join(tmp))

for idx in range(len(all_keywords)):
    st.write(f"cluster {idx} has the keywords: {str(all_keywords[idx])}")