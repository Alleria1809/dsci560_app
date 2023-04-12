import streamlit as st
import pandas as pd
import numpy as np
import time
import random
import functools
import plotly.graph_objects as go

st.set_page_config(page_title="Recommendation", page_icon=":sparkles:", layout="wide")

st.markdown("# LA Restaurant Recommendation")
st.sidebar.header("Restaurant Recommendation")

st.markdown(
    """
    #### This page is designed for the citizens.
    """
)

# split amenities
def check_delivery(row):
  if type(row["amenities"])!=type(np.nan):
      if 'Offers Delivery' in row['amenities']:
        return 1
      elif 'No Delivery' in row['amenities']:
        return 0
      else:
        return np.nan

def check_takeout(row):
  if type(row["amenities"])!=type(np.nan):
      if 'Offers Takeout' in row['amenities']:
        return 1
      elif 'No Takeout' in row['amenities']:
        return 0
      else:
        return np.nan

def check_reservation(row):
    if type(row["amenities"])!=type(np.nan):
      if 'Takes Reservations' in row['amenities']:
        return 1
      elif 'No Reservations' in row['amenities']:
        return 0
      else:
        return np.nan

def check_vegetarian(row):
    if type(row["amenities"])!=type(np.nan):
      if 'Limited Vegetarian Options' in row['amenities'] or 'Many Vegetarian Options' in row['amenities']:
        return 1
      elif 'No Vegetarian Options' in row['amenities']:
        return 0
      else:
        return np.nan

def check_groups(row):
    if type(row["amenities"])!=type(np.nan):
      if 'Not Good For Groups' in row['amenities']:
        return 0
      elif 'Good for Groups' in row['amenities']:
        return 1
      else:
        return np.nan

def check_wheelchair(row):
    if type(row["amenities"])!=type(np.nan):
      if 'Wheelchair Accessible' in row['amenities']:
        return 1
      else:
        return 0

# Add a column for tags
def combine_tags(row):
  tags = []
  tags.append(str(row['size']))
  tags.append(str(row['risk_level']))
  tags.append(str(row['price']))
  for category in row['category'].split(', '):
    tags.append(str(category))
  tags.append('offers delivery' if row['delivery'] ==  1 else 'no delivery')
  tags.append('offers takeout' if row['takeout'] ==  1 else 'no takeout')
  tags.append('good for groups' if row['groups'] ==  1 else 'not good for groups')
  tags.append('offers vegetarian options' if row['vegetarian'] ==  1 else 'no vegetarian options')
  tags.append('offers reservation' if row['reservation'] ==  1 else 'no reservation')
  tags.append('wheelchair accessible' if row['reservation'] ==  1 else 'wheelchair not accessible')
  tags.append(str(row['zipcode']))
  return tags

# Demo #1 - recommendation based on features

# Find restaurants that contain all input features
def exact_match(df, tags):
  candidate_restaurant = []
  candidate_pairs = []
  for i in range(len(df)):
    restaurant = df.iloc[i]
    if tags.issubset(set(restaurant['tags'])):
      candidate_restaurant.append(restaurant['name'])
      candidate_pairs.append((restaurant['name'], restaurant['url'], 'exact_match'))
  return candidate_restaurant, candidate_pairs


# Remove restaurants that are already in candidate_pairs
def remaining_df(df, candidate_restaurant):
  final_df = df[df.name.isin(candidate_restaurant) == False]
  return final_df


# Calculate jaccard similarity
def jaccard_similarity(row, tags):
  intersection = set(row['tags']).intersection(tags)
  union = set(row['tags']).union(tags)
  res = round(len(intersection)/len(union), 4)
  return res


# If len(candidate_pairs) < 10, choose remaining candidates by calculating jaccard similarities
def jaccard_match(df, tags, candidate_pairs):

  # Calculate jaccard similarity between input tags and each restaurant tags
  df['jaccard'] = df.apply(lambda row: jaccard_similarity(row, tags), axis = 1)
  final_df = df.sort_values(by = ['jaccard'], ascending = False).reset_index(drop = True)

  # Fill the remaining spots using the top 10-len(candidates) jaccard similarities
  i = 0
  curr_jaccard_candidates = []
  curr_jaccard = final_df.iloc[0]['jaccard']

  while i < len(final_df) and len(candidate_pairs) < 10:
    # If there are ties, record a list of restaurants with the current jaccard similarity
    if final_df.iloc[i]['jaccard'] == curr_jaccard:
      curr_jaccard_candidates.append((final_df.iloc[i]['name'], final_df.iloc[i]['url'], final_df.iloc[i]['jaccard']))
      i += 1
    # As soon as a different, lower jaccard similarity appears
    else:
      # If the recorded list can fit into the remaining spots of candidate_pairs, add all pairs in
      if len(curr_jaccard_candidates) <= 10 - len(candidate_pairs):
        for pair in curr_jaccard_candidates:
          if pair not in candidate_pairs:
            candidate_pairs.append(pair)
      # Else choose a random sample that can fit into the remaining spots
      else:
        new_candidates = random.sample(curr_jaccard_candidates, 10 - len(candidate_pairs))
        for pair in new_candidates:
          if pair not in candidate_pairs:
            candidate_pairs.append(pair)

      # Reset that different, lower jaccard similariy as the current value
      curr_jaccard = final_df.iloc[i]['jaccard']
      curr_jaccard_candidates = []
    
  return candidate_pairs


# Recommendation
def recommendation_from_features(df, input):
  tags = set()
  for value in input.values():
    if type(value) == type('str'):
      tags.add(value)
    else:
      for subvalue in value:
        tags.add(subvalue)
  
  candidate_restaurant, candidate_pairs = exact_match(df, tags)
  # If candidate size from exact match is more than 10, choose a random sample
  if len(candidate_pairs) >= 10:
    final_candidates = random.sample(candidate_pairs, 10)
  # Else use jaccard match to fill the spots
  else:
    df = remaining_df(df, candidate_restaurant)
    final_candidates = jaccard_match(df, tags, candidate_pairs)

  return final_candidates

# get unique categories
def get_category(data):
    categories = set()
    for x in list(data.category):
        # 'Korean, Barbeque'
        clists = x.split(',')
        # print(clists)
        for c in clists:
            c = c.replace(' ','')
            categories.add(c)
    return list(categories)

# Demo #2 - recommendation based on current resuaurant

# Choose candidates by calculating jaccard similarities
def jaccard_top10(df, tags):

  # Calculate jaccard similarity between input tags and each restaurant tags
  df['jaccard'] = df.apply(lambda row: jaccard_similarity(row, tags), axis = 1)
  final_df = df.sort_values(by = ['jaccard'], ascending = False).reset_index(drop = True)

  # Choose the top 10-len(candidates) jaccard similarities
  candidate_pairs = []
  i = 0
  curr_jaccard_candidates = []
  curr_jaccard = final_df.iloc[0]['jaccard']

  while i < len(final_df) and len(candidate_pairs) < 11:
    # If there are ties, record a list of restaurants with the current jaccard similarity
    if final_df.iloc[i]['jaccard'] == curr_jaccard:
      curr_jaccard_candidates.append((final_df.iloc[i]['name'], final_df.iloc[i]['url'], final_df.iloc[i]['jaccard']))
      i += 1
    # As soon as a different, lower jaccard similarity appears
    else:
      # If the recorded list can fit into the remaining spots of candidate_pairs, add all pairs in
      if len(curr_jaccard_candidates) <= 11 - len(candidate_pairs):
        for pair in curr_jaccard_candidates:
          if pair not in candidate_pairs:
            candidate_pairs.append(pair)
      # Else choose a random sample that can fit into the remaining spots
      else:
        new_candidates = random.sample(curr_jaccard_candidates, 11 - len(candidate_pairs))
        for pair in new_candidates:
          if pair not in candidate_pairs:
            candidate_pairs.append(pair)

      # Reset that different, lower jaccard similariy as the current value
      curr_jaccard = final_df.iloc[i]['jaccard']
      curr_jaccard_candidates = []
    
  return candidate_pairs

# Recommendation
def recommendation_from_restaurant(df, input):
  tags = set(df[df['name'] == input]['tags'].tolist()[0])
  final_candidates = jaccard_top10(df, tags)
  return final_candidates

def show_restaurants(recommendations):
  matched_rest = []
  rest_score = []
  for i in range(len(recommendations)):
      matched_rest.append(recommendations[i][0])
      rest_score.append(recommendations[i][-1])
      
  # chart = functools.partial(st.plotly_chart, use_container_width=True)
  labels = matched_rest
  values = rest_score
  
  # Use `hole` to create a donut-like pie chart
  fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
  fig.update_layout(
  title={'text': "Recommended 10 Restaurants for you"})
  return fig
  

# @st.cache_data
def load_data():
    data = pd.read_csv('combined_rest_health_0331.csv')
    data['delivery'] = data.apply(lambda row: check_delivery(row), axis = 1)
    data['takeout'] = data.apply(lambda row: check_takeout(row), axis = 1)
    data['groups'] = data.apply(lambda row: check_groups(row), axis = 1)
    data['vegetarian'] = data.apply(lambda row: check_vegetarian(row), axis = 1)
    data['reservation'] = data.apply(lambda row: check_reservation(row), axis = 1)
    data['wheelchair'] = data.apply(lambda row: check_wheelchair(row), axis = 1)
    # Extract key features
    data_key = data[['name', 'SCORE', 'size', 'risk_level', 'url', 'score', 'review_counts', 'price', 'category',
                    'delivery', 'takeout', 'groups', 'vegetarian', 'reservation','wheelchair',"address","zipcode"]]
    data_key['tags'] = data_key.apply(lambda row: combine_tags(row), axis = 1)
    
    return data_key

data_key = load_data()
# choice = st.selectbox("What's in your mind?", ['I want to explore the features!', 'I already have some ideal restaurants in my mind!'])

# st.markdown("#### What\'s in your mind?")
choice = st.radio(
    "What\'s in your mind?",
    (['I want to explore the features!', 'I already have some ideal restaurants in my mind!']))


if choice == 'I want to explore the features!':
    risk_level = st.selectbox(
        'Please select the risk level:',
        ['LOW', 'MEDIUM', 'HIGH'],)

    price = st.selectbox(
        'Please select the restaurant price:',
        ['$', '$$', '$$$'],)
    
    categories = get_category(data_key)
    category = st.multiselect(
        'Please select the restaurant category:',
        categories,)
    
    delivery = st.selectbox(
        'Please select the risk level:',
        ['offers delivery', 'no Delivery'],)
    
    takeout = st.selectbox(
        'Please select the risk level:',
        ['offers takeout', 'no takeout'],)
    
    vegetarian = st.selectbox(
        'Please select the risk level:',
        ['offers vegetarian options', 'no vegetarian options'],)
    
    reservation = st.selectbox(
        'Please select the risk level:',
        ['offers reservation', 'no reservation'],)
    
    input1 = {'risk_level': risk_level,
         'price': price,
         'category': category,
         'delivery': delivery,
         'takeout': takeout,
         'vegetarian': vegetarian,
         'reservation': reservation}
    with st.form(key="Form :", clear_on_submit = True):
      Submit = st.form_submit_button(label='Search')
    if Submit:
      recommendations = recommendation_from_features(data_key, input1) 
      
      col1, col2 = st.columns(2)
      with col1:
        chart = functools.partial(st.plotly_chart, use_container_width=True)
        fig = show_restaurants(recommendations)
        chart(fig)
                    
      with col2:  
        st.write(f'##### Recommended Restaurants list:')
        for  name, url, score in recommendations:
            # [@Project APP](https://github.com/Alleria1809/dsci560_app)
            st.write(f"restaurant: [{name}]({url})")
    
else:
    name =  st.text_input(
        "Please input your ideal restaurant name: ",
        "Moon BBQ 2",
        key="restaurant_name",
    )
    # input2 = 'Moon BBQ 2'
    with st.form(key="Form :", clear_on_submit = True):
      Submit = st.form_submit_button(label='Search')
      
    if Submit:
      input2 = name
      recommendations = recommendation_from_restaurant(data_key, input2)
      
      col3, col4 = st.columns(2)
      with col3:
        chart = functools.partial(st.plotly_chart, use_container_width=True)
        fig = show_restaurants(recommendations)
        chart(fig)
                    
      with col4:  
        st.write(f'##### Recommended Restaurants list:')
        for  name, url, score in recommendations:
            # [@Project APP](https://github.com/Alleria1809/dsci560_app)
            st.write(f"restaurant: [{name}]({url})")

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