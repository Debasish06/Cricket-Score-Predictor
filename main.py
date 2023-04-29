import streamlit as st
import pickle
import pandas as pd
import numpy as np
import sklearn
import xgboost

pipe = pickle.load(open('pipe.pkl','rb'))

teams = ['New Zealand', 'Bangladesh', 'West Indies', 'South Africa',
       'India', 'England', 'Pakistan', 'Australia', 'Zimbabwe',
       'Sri Lanka', 'Ireland', 'Afghanistan']

cities = ['Barbados', 'Mirpur', 'Lauderhill', 'Johannesburg', 'Trinidad',
       'Melbourne', 'Durban', 'Manchester', 'Wellington', 'Auckland',
       'Abu Dhabi', 'Sydney', 'Dhaka', 'Cape Town', 'Adelaide', 'Colombo',
       'Southampton', 'Pallekele', 'Hamilton', 'Lahore', 'London',
       'Nottingham', 'Chittagong', 'St Kitts', 'Mount Maunganui',
       'Mumbai', 'Harare', 'Dubai', 'Sharjah', 'Delhi', 'Chandigarh',
       'Guyana', 'St Lucia', 'Centurion', 'Nagpur', 'Bangalore',
       'Cardiff', 'Kolkata', 'Hambantota', 'Greater Noida',
       'Christchurch']

st.title(":blue[T20 International Cricket Score Predictor]")

col1,col2 = st.columns(2)
with col1:
    batting_team=st.selectbox(":red[Batting Team]",sorted(teams))
with col2:
    bowling_team=st.selectbox(":red[Bowling Team]",sorted(teams))

city=st.selectbox(":red[City]",sorted(cities))

col3,col4 = st.columns(2)
with col3:
    current_score = st.number_input(":red[Current Score]",min_value=0)
with col4:
    overs_done = st.number_input(":red[Overs done (should be greater than 5)]",min_value=5,max_value=20)

wickets=st.slider(":red[Wicket]",0,10,1)

last_five = int(st.number_input(":red[Last 5 overs runs]",min_value=0))

if st.button("Predict Score",type="primary"):
    balls_left = 120 - overs_done*6
    wickets_left = 10 - wickets
    crr = current_score / overs_done

    input_df = pd.DataFrame({'batting_team' : [batting_team], 'bowling_team'   : [bowling_team], 'city' : city,
                'current_score':[current_score], 'balls_left':[balls_left], 'wickets_left':[wickets_left],
                'crr':[crr],'last_five':[last_five]})

    result = pipe.predict(input_df)
    st.header("Predicted Score - " +str(int(result[0])))

