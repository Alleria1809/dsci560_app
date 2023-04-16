import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from PIL import Image
import plotly.express as px
from IPython.display import HTML
import matplotlib.pyplot as plt
import joblib
import warnings
# preprocessing
from textblob import TextBlob

warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None)

st.set_page_config(page_title="Prediction", page_icon=":chopsticks:", layout="wide")

st.markdown("# LA Restaurant Health Risk Prediction")
st.sidebar.header("Restaurant Prediction")

st.markdown(
    """
    #### This page is designed for the government.
    """
)

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    with st.expander("See uploaded data"):
        dff = pd.read_csv(uploaded_file)
        st.write('Show the uploaded data.')
        # show the raw dataframe
        st.dataframe(dff.drop(columns=['name']), 1000, 300)
        # st.write(dff)

    # with st.expander("See raw data"):
    #     st.write('Show the raw data.')
    #     # show the raw dataframe
    #     st.dataframe(dff, 1000, 300)
        
    # load model
    model = joblib.load('rf_model.joblib') # ml prediction model
    model_state = st.text('Predicting...')
    
    dff_m = dff[['FACILITY_NAME', 'name', 'FACILITY_ZIP', 'review_counts', 'price', 'category', 'type', 
           'size', 'open_hours_week', 'SCORE', 'score', 'num_photos', 'is_bus_web', 'is_phone_number', 'is_message_bus',
           'num_attributes', 'num_questions', 'comments_list']]
    dff_m = dff_m[dff_m.type=='restaurant']  # type includes restaurant, food market, other
    dff_m.rename(columns={'score': 'rating'}, inplace=True)
    dff_m.columns = map(lambda x: str(x).upper(), dff_m.columns)
    #zip
    dff_m.FACILITY_ZIP = dff_m.FACILITY_ZIP.apply(lambda x: x[:5])
    #review_counts
    def func0(x):
        if 'k' in x:
            return float(x.replace('k', '')) * 1000
        else:
            return int(x)
    dff_m.REVIEW_COUNTS.fillna('0 reviews', inplace=True)
    dff_m.REVIEW_COUNTS = dff_m.REVIEW_COUNTS.apply(lambda x: x.strip('(').strip(')').split()[0])
    dff_m.REVIEW_COUNTS = dff_m.REVIEW_COUNTS.apply(func0)
    #price
    dff_m.PRICE = dff_m.PRICE.map({'$': '1', '$$': '2', '$$$': '3', '$$$$': '4'})
    dff_m.PRICE.fillna(dff_m.PRICE.mode()[0], inplace=True)
    #size
    dff_m.SIZE = dff_m.SIZE.map({'0-30': '1', '31-60': '2', '61-150': '3', '151 +': '4'})
    dff_m.SIZE.fillna(dff_m.SIZE.mode()[0], inplace=True)
    #open_hours_week
    dff_m.OPEN_HOURS_WEEK.fillna(dff_m.OPEN_HOURS_WEEK.mean(), inplace=True)
    #rating
    dff_m.RATING.fillna(dff_m.RATING.mode()[0], inplace=True)
    #num_photos
    def func(x):
        if pd.isnull(x)==False:
            num = x.split()[2]
            try:
                return int(num)
            except:
                if 'k' in num:
                    return float(num.replace('k', '')) * 1000
    dff_m.NUM_PHOTOS = dff_m.NUM_PHOTOS.apply(func)
    dff_m.NUM_PHOTOS.fillna(0, inplace=True)
    #is_bus_web
    dff_m.IS_BUS_WEB = dff_m.IS_BUS_WEB.map({True: 1, False: 0})
    #is_phone_number
    dff_m.IS_PHONE_NUMBER = dff_m.IS_PHONE_NUMBER.map({True: 1, False: 0})
    #is_message_bus
    dff_m.IS_MESSAGE_BUS = dff_m.IS_MESSAGE_BUS.map({True: 1, False: 0})
    #num_attributes
    dff_m.NUM_ATTRIBUTES.fillna(0, inplace=True) 
    #num_questions
    dff_m.NUM_QUESTIONS = dff_m.NUM_QUESTIONS.apply(lambda x: int(x.split()[2]) if pd.isnull(x)==False else 0) 
    #comments_list
    dff_m['SENTIMENT_POLARITY'] = dff_m.COMMENTS_LIST.apply(lambda x: TextBlob(x).sentiment.polarity)
    #label:risk_level
    dff_m['RISK_LEVEL'] = pd.cut(dff_m.SCORE, 3, labels=['high risk', 'medium risk', 'low risk']).astype(str)  #pd.cut/pd.qcut, retbins=True
    dff_m['RISK_LEVEL'] = dff_m['RISK_LEVEL'].map({'low risk': 0, 'medium risk': 1, 'high risk': 2})


    # prediction
    dff_rf = dff_m.copy()
    all_x = dff_rf[['FACILITY_ZIP', 'REVIEW_COUNTS', 'PRICE', 'SIZE', 'OPEN_HOURS_WEEK', 'RATING',
           'NUM_PHOTOS', 'IS_BUS_WEB', 'IS_PHONE_NUMBER', 'IS_MESSAGE_BUS', 
           'NUM_ATTRIBUTES', 'NUM_QUESTIONS', 'SENTIMENT_POLARITY']]
    all_y = dff_rf['RISK_LEVEL']
    
    
    y_pred = model.predict(all_x)
    model_state.text("Done!")
    
    dff_m['prediction_result'] = y_pred 
    dff_m['prediction_result'] = dff_m['prediction_result'].map({0: 'low risk', 1: 'medium risk', 2: 'high risk'})   

    # show prediction results in dataframe
    st.markdown("#### Prediction Results")
    with st.expander("See Prediction Results"):
        st.dataframe(dff_m.drop(['RISK_LEVEL'], axis=1), 1000, 300)   

    # show high/medium risk
    st.markdown("#### Show medium/high risk")
    st.dataframe(dff_m[(dff_m['prediction_result']=='high risk') | (dff_m['prediction_result']=='medium risk')].drop(['RISK_LEVEL'], axis=1), 1000, 300) 

    #visualize
    st.markdown("#### Prediction Distribution")
    res_df = pd.DataFrame({
        'prediction_result': dff_m['prediction_result'].value_counts().keys(),
        'count':  dff_m['prediction_result'].value_counts().values
    })
       
    # print(res_df)
    fig = px.histogram(res_df, x='prediction_result', y='count', color='prediction_result', title='Prediction Results',
                       color_discrete_map={'high risk':'#EF553B', 'medium risk':'#00CC96', 'low risk':'#636EFA'},
                       text_auto=True)
    #HTML(fig.to_html())
    st.plotly_chart(fig)
    
    # yelp ratings
    medium_low_df = dff_m[(dff_m['prediction_result']=='high risk') | (dff_m['prediction_result']=='medium risk')]
    fig = px.histogram(medium_low_df['RATING'], x='RATING', text_auto=True, 
                       title='Yelp Rating Distribution of the Medium and Low Risk Restaurants')
    st.plotly_chart(fig)
    

#    # zipcode and city
#     zipcode_df = pd.read_csv('zipcode.csv')
#     zipcode_map = dict(zip(zipcode_df.zipcode.tolist(), zipcode_df.city.tolist()))
#     medium_low_df = dff_m[(dff_m['prediction_result']=='high risk') | (dff_m['prediction_result']=='medium risk')]
#     city_df = medium_low_df['FACILITY_ZIP'].astype(int).map(zipcode_map)
#     print(city_df)
#     st.write(city_df)
   

    