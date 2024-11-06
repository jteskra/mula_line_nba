import pandas as pd
from bs4 import BeautifulSoup, Comment
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import time
import requests

# st.write(pd.__version__)
# import os
# st.write(os.listdir('.'))



# model = load_model('nba_third_try.keras')
# st.write(model.summary())
# st.title('NBA Moneyline Model')
# st.video('https://www.youtube.com/watch?v=21y14JbQo8A')
# st.sidebar.write('**Please fill out the criteria to get a result**')
# st.info('**Machine learning models must use numbers to make predictions. So we must convert teams into numbers and keep them  consistant.**')

# Add a thicker divider using HTML
# st.sidebar.markdown('<hr style="border:2px solid black">', unsafe_allow_html=True)
# #decimal odds deltas
# team1_decimal_odds = st.sidebar.slider('team 1 decimal odds', .1, 12.1, 6.2)
# team2_decimal_odds = st.sidebar.slider('team 2 decimal odds', .1, 12.1, 6.2)
# # Dynamically calculate decimal_delta as the absolute difference
# decimal_delta = abs(team1_decimal_odds - team2_decimal_odds)
# st.sidebar.write(f'**Odds Delta** (Absolute Difference)**:** **{decimal_delta:.2f}**')
# # Add a thicker divider using HTML
# st.sidebar.markdown('<hr style="border:2px solid black">', unsafe_allow_html=True)


# #points deltas
# favorite_team_points = st.sidebar.slider('Favorite Team avg pts scored per game', 95, 130, 100)
# underdog_team_points = st.sidebar.slider('Underdog Team avg pts scored per game', 95, 130, 100)
# points_delta = (favorite_team_points-underdog_team_points)
# st.sidebar.write(f'**Points Delta:** **{points_delta:.2f}**')
# # Add a thicker divider using HTML
# st.sidebar.markdown('<hr style="border:2px solid black">', unsafe_allow_html=True)

# #Assist Deltas
# favorite_team_assists = st.sidebar.slider('Favorite Team Assists per game', 15.0, 35.0, step=0.1)
# underdog_team_assists = st.sidebar.slider('Underdog Team Assists per game',  15.0, 35.0, step=0.1)
# delta_assists_per_game = (favorite_team_assists-underdog_team_assists)
# st.sidebar.write(f'**Assists Delta:** **{delta_assists_per_game:.2f}**')
# # Add a thicker divider using HTML
# st.sidebar.markdown('<hr style="border:2px solid black">', unsafe_allow_html=True)

# #free throw percentage delta
# favorite_team_ft=st.sidebar.slider('Favorite team Free Throw Percentage', 60, 100, 1)
# underdog_team_ft=st.sidebar.slider('Underdog team Free Throw Percentage', 60, 100, 1)
# delta_free_throw_percentage = (favorite_team_ft-underdog_team_ft)
# st.sidebar.write(f'**Free Throw% Delta: {delta_free_throw_percentage:.2f}**')
# # Add a thicker divider using HTML
# st.sidebar.markdown('<hr style="border:2px solid black">', unsafe_allow_html=True)

# team1_encoded = st.sidebar.slider('team 1 encoded', 0, 30, 1)

# favorite_encoded = st.sidebar.slider('favorite encoded', 0, 30, 1)

# visitor_encoded = st.sidebar.slider('visitor team encoded', 0, 30, 1)
# # Add a thicker divider using HTML
# st.sidebar.markdown('<hr style="border:2px solid black">', unsafe_allow_html=True)

# with st.expander('**Why am I entering these random numbers?**'):
#     st.write('We found that these exact parameters yeilded the highest accuracy on unseen data AKA: val_accuracy. It may seem odd and counterproductive to enter some of these values, but all of these columns have significant coeffiences for correlation.')
#     st.write('Taking the difference between two teams, rather than using each teams general statistic yeilded 5 percent higher val_accuracy and the log_loss dropped around .3')
#     st.write('Achieving ~70 percent accuracy is remarkably difficult when trying to predict something with extremely high randomness, we apologize for the inconvience. Some things we just cannot get around :(')

# with st.expander('**Data for Deltas**'):
#     st.markdown("[Click for NBA Odds](https://www.espn.com/nba/odds)")
#     st.write('**Using 2024 Stats for the beginning part of the 2025 NBA season could yeild better accuracy; just make sure the starters are the same from 2024 to 2025**')
#     st.markdown("[Click for 2024 NBA Stats](https://www.basketball-reference.com/leagues/NBA_2024.html)")
#     st.markdown("[Click for 2025 NBA Stats](https://www.basketball-reference.com/leagues/NBA_2025.html)")


# Use st.expander to group the content
# with st.expander('**Data for Deltas**'):
#     st.markdown("[Click for NBA Odds](https://www.espn.com/nba/odds)")
#     st.write('**Using 2024 Stats for the beginning part of the 2025 NBA season could yeild better accuracy; just make sure the starters are the same from 2024 to 2025. Just change the 2025 in the URL to 2024!**')

    
#     # Dictionary mapping team names to URLs (you'll need to replace the URLs with the actual ones)
#     teams_urls = {
#         'Atlanta Hawks': 'https://www.basketball-reference.com/teams/ATL/2025.html',
#         "Boston Celtics": 'https://www.basketball-reference.com/teams/BOS/2025.html', 
#         "Brooklyn Nets": 'https://www.basketball-reference.com/teams/BRK/2025.html', 
#         "Charlotte Hornets": 'https://www.basketball-reference.com/teams/CHA/2025.html', 
#         "Chicago Bulls": 'https://www.basketball-reference.com/teams/CHI/2025.html',
#         'Cleveland Cavaliers': 'https://www.basketball-reference.com/teams/CLE/2025.html',
#         'Dallas Mavericks': 'https://www.basketball-reference.com/teams/DAL/2025.html',
#         'Denver Nuggets': 'https://www.basketball-reference.com/teams/DEN/2025.html',
#         'Detroit Pistons': 'https://www.basketball-reference.com/teams/DET/2025.html',
#         'Golden State Warriors': 'https://www.basketball-reference.com/teams/GSW/2025.html',
#         'Houston Rockets': 'https://www.basketball-reference.com/teams/HOU/2025.html',
#         'Indiana Pacers': 'https://www.basketball-reference.com/teams/IND/2025.html',
#         'Los Angeles Clippers': 'https://www.basketball-reference.com/teams/LAC/2025.html',
#         'Los Angeles Lakers': 'https://www.basketball-reference.com/teams/LAL/2025.html',
#         'Memphis Grizzlies': 'https://www.basketball-reference.com/teams/MEM/2025.html',
#         'Miami Heat': 'https://www.basketball-reference.com/teams/MIA/2025.html',
#         'Milwaukee Bucks': 'https://www.basketball-reference.com/teams/MIL/2025.html',
#         'Minnesota Timberwolves': 'https://www.basketball-reference.com/teams/MIN/2025.html',
#         'New Orleans Pelicans': 'https://www.basketball-reference.com/teams/NOP/2025.html',
#         'New York Knicks': 'https://www.basketball-reference.com/teams/NYK/2025.html',
#         'OKC Thunder': 'https://www.basketball-reference.com/teams/OKC/2025.html',
#         'Orlando Magic': 'https://www.basketball-reference.com/teams/ORL/2025.html',
#         'Philadelphia 76ers': 'https://www.basketball-reference.com/teams/PHI/2025.html',
#         'Phoenix Suns': 'https://www.basketball-reference.com/teams/PHO/2025.html',
#         'Portland Trail Blazers': 'https://www.basketball-reference.com/teams/POR/2025.html',
#         'Sacramento Kings': 'https://www.basketball-reference.com/teams/SAC/2025.html',
#         'San Antonio Spurs': 'https://www.basketball-reference.com/teams/SAS/2025.html',
#         'Toronto Raptors': 'https://www.basketball-reference.com/teams/TOR/2025.html',
#         'Utah Jazz': 'https://www.basketball-reference.com/teams/UTA/2025.html',
#         'Washington Wizards': 'https://www.basketball-reference.com/teams/WAS/2025.html',
#     }
    
#     # Loop through team names and their corresponding URLs
#     for team, url in teams_urls.items():
#         col1, col2 = st.columns([3, 3])  # Two equal width columns
#         col1.write(team)  # Display team name on the left
#         col2.markdown(f"[2025 Team Data]({url})", unsafe_allow_html=True)  # Create clickable link on the right


# with st.expander('**Teams Encoded**'):
#         # Loop through team names and numbers to create a two-column layout
#     teams_encoded = {
#         'Atlanta Hawks': 0,
#         "Boston Celtics": 1, 
#         "Brooklyn Nets": 2, 
#         "Charlotte Hornets": 3, 
#         "Chicago Bulls": 4,
#         'Cleveland Cavaliers': 5,
#         'Dallas Mavericks': 6,
#         'Denver Nuggets': 7,
#         'Detroit Pistons': 8,
#         'Golden State Warriors': 9,
#         'Houston Rockets': 10,
#         'Indiana Pacers': 11,
#         'Los Angeles Clippers': 12,
#         'Los Angeles Lakers': 13,
#         'Memphis Grizzlies': 14,
#         'Miami Heat': 15,
#         'Milwaukee Bucks': 16,
#         'Minnesota Timberwolves': 17,
#         'New Orleans Pelicans': 18,
#         'New_York Knicks':19,
#         'OKC Thunder': 20,
#         'Orlando Magic':21,
#         'Philadelphia 76ers':22,
#         'Phoenix Suns':23,
#         'Portland Trail Blazers': 24,
#         'Sacramento Kings': 25,
#         'San Antonio Spurs': 26,
#         'Toronto Raptors': 28,
#         'Utah Jazz': 29,
#         'Washington Wizards': 30,


#         # Add more teams as needed
#     }
    
#     for team, code in teams_encoded.items():
#         col1, col2 = st.columns([3, 1])  # Create two columns, left wider than right
#         col1.write(team)  # Display team name on the left
#         col2.write(code)  # Display encoded number on the right
#     # st.write('Boston = 1, Brooklyn = 2, Charlotte = 3, Chicago = 4,')


# # Create an expander section
# with st.expander('**American Odds to Decimal Odds Converter**'):
    
#     # Function to convert American odds to Decimal odds
#     def american_to_decimal(american_odds):
#         if american_odds > 0:
#             # For positive American odds
#             decimal_odds = (american_odds / 100) + 1
#         else:
#             # For negative American odds
#             decimal_odds = (100 / abs(american_odds)) + 1
#         return decimal_odds

#     # Create an input box for the user to enter American odds
#     american_odds = st.number_input("Enter American odds:", value=100)  # Default value set to 100

#     # Button to perform conversion
#     if st.button("Convert to Decimal Odds"):
#         # Perform the conversion
#         decimal_odds = american_to_decimal(american_odds)
        
#         # Display the result
#         st.write(f"American odds: {american_odds} -> Decimal odds: {decimal_odds}")




#columns
#'team1_odds', 'odds_delta', 'delta_points_per_game', 'delta_assists_per_game', 
#'delta_free_throw_percentage', 'team1_encoded', 'favorite_encoded', 'visitor_encoded'
# submit = st.sidebar.button('**Submit**')  # ,

# if submit:



#     # print(petal_width, sepal_length)
#     data_dict = {'team1_decimal_odds':team1_decimal_odds, 'team2_decimal_odds':team2_decimal_odds, 'decimal_delta':decimal_delta, 'delta_points_per_game': points_delta, 
#     'delta_assists_per_game':delta_assists_per_game,'delta_free_throw_percentage':delta_free_throw_percentage, 
#     'team1_encoded':team1_encoded, 'favorite_encoded': favorite_encoded,'visitor_encoded': visitor_encoded}


#     # Convert to DataFrame
#     sample_df = pd.DataFrame(data_dict, index=[0])

#     # Convert DataFrame to NumPy array to feed into the model
#     input_array = sample_df.to_numpy()

#     #Display 
#     st.write(sample_df)

#     print(sample_df)

#     #perform the predictions
#     prediction =model.predict(input_array)

#     #Display the prediction
#     # st.write(prediction)

#     # Assuming prediction is a probability between 0 and 1, convert it to percentage
#     prediction_percent = prediction[0][0] * 100  # Multiply by 100 to convert to percentage

#     # Display the custom message with the prediction
#     st.write(f"The home team has a {prediction_percent:.2f}% chance of winning the game")
#     print(prediction)


#################################################################################################################################################################################

# Add a thicker divider using HTML
st.sidebar.markdown('<hr style="border:2px solid black">', unsafe_allow_html=True)

#########################    LOAD MODEL    ################################

model = load_model('nba_third_try.keras')


#########################    AMERICAN TO DECIMAL ODDS CONVERTER    #########################

# Create an expander section for the odds converter
with st.expander('**American Odds to Decimal Odds Converter**'):
    
    # Function to convert American odds to Decimal odds
    def american_to_decimal(american_odds):
        return (american_odds / 100) + 1 if american_odds > 0 else (100 / abs(american_odds)) + 1

    # Input box for the user to enter American odds
    american_odds = st.number_input("Enter American odds:", value=100)

    # Button to perform conversion
    if st.button("Convert to Decimal Odds"):
        decimal_odds = american_to_decimal(american_odds)
        st.write(f"American odds: {american_odds} -> Decimal odds: {decimal_odds}")


#########################    PREPARING LISTS AND VARIABLES    #########################

# List of NBA team abbreviations
teams = [
    'BOS', 'NYK', 'MIL', 'CLE', 'ORL', 'IND', 'PHI', 'MIA', 'CHI', 'ATL',
    'BRK', 'TOR', 'CHO', 'WAS', 'DET', 'OKC', 'DEN', 'MIN', 'LAC', 'DAL',
    'PHO', 'NOP', 'LAL', 'SAC', 'GSW', 'HOU', 'UTA', 'MEM', 'SAS', 'POR'
]

# Set custom headers for web scraping
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

# Create dropdowns to select two teams and input year range
visitor_team = st.selectbox('Visitor Team:', teams)
home_team = st.selectbox('Home Team:', teams)
year_range = st.selectbox('Select Year Range:', [2023, 2024])


#########################    WEB SCRAPING FUNCTION    #########################

# Submit button to trigger scraping
submit = st.sidebar.button('**Submit**')

if submit:
    def scrape_team_data(team, year):
        # Initialize DataFrame for Team and Opponent stats combined
        df = pd.DataFrame(columns=[
            'G', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 
            '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 
            'STL', 'BLK', 'TOV', 'PF', 'PTS', 'def_G', 'def_MP', 'def_FG', 'def_FGA', 'def_FG%', 'def_3P', 'def_3PA', 'def_3P%', 
            'def_2P', 'def_2PA', 'def_2P%', 'def_FT', 'def_FTA', 'def_FT%', 'def_ORB', 'def_DRB', 'def_TRB', 'def_AST', 
            'def_STL', 'def_BLK', 'def_TOV', 'def_PF', 'def_PTS'
        ])

        url = f"https://www.basketball-reference.com/teams/{team}/{year}.html"
        st.write(f"Scraping {team} in {year} at URL: {url}")

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract relevant table inside the comment block
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        team_data, opponent_data = [], []

        # Iterate through the comments to find "team_and_opponent" table
        for comment in comments:
            if 'team_and_opponent' in comment:
                comment_soup = BeautifulSoup(comment, 'html.parser')
                table = comment_soup.find('table', {'id': 'team_and_opponent'})
                if table:
                    rows = table.find_all('tr')
                    for row in rows:
                        cols = [ele.text.strip() for ele in row.find_all(['td', 'th']) if ele.text.strip()]
                        if cols:
                            if cols[0] == 'Team/G':
                                cols.pop(0)  # Remove label
                                team_data = cols
                            elif cols[0] == 'Opponent/G':
                                cols.pop(0)  # Remove label
                                opponent_data = cols

        # Combine Team/G and Opponent/G rows into one row
        if team_data and opponent_data:
            combined_data = team_data + opponent_data
            if len(combined_data) == len(df.columns):  # Ensure column alignment
                df.loc[len(df)] = combined_data
            else:
                st.write(f"Skipping {team} due to mismatched columns")
        else:
            st.write(f"Skipping {team} due to missing data")

        return df

#########################    ENCODE & DISPLAY DATA    #########################

# Scrape data for both teams
df_team = scrape_team_data(visitor_team, year_range)  # Visitor team
df_opponent = scrape_team_data(home_team, year_range)  # Home team
st.write(df_team)

# Prefix columns for the second team to avoid duplication
df_opponent.columns = ['home_' + col if col not in ['Visitor Team', 'Home Team', 'Year'] else col for col in df_opponent.columns]

# Combine both teams' data horizontally into one DataFrame
final_combined_df = pd.concat([df_team, df_opponent], axis=1).head(1)  # Keep only one row

# Add visitor and home team information
final_combined_df['Visitor Team'] = visitor_team
final_combined_df['Home Team'] = home_team
final_combined_df['Year'] = year_range

# Display the final DataFrame
st.write("Final Combined DataFrame:", final_combined_df)



#########################    ENCODE    ################################

    # df['Visitor Team'] = visitor_team
    # df['Home Team'] = home_team

    # # Model input columns
    # feature_columns = ['FG', 'FGA', '3P', '3PA', 'PTS'] 
    # X=df[feature_columns]
    # st.write(f"Selected feature columns: {feature_columns}")



#########################    RUN THE MODEL    ################################

    # Load your machine learning model
    # model = load_model()

    # Select the features required for your model (replace with actual features)
    # Assume the model needs columns like ['FG', 'FGA', '3P', '3PA', 'PTS']

    # X = df[feature_columns]

    # # Run the model prediction
    # predictions = model.predict(X)

    # # Display the outcome of the prediction
    # st.write(f"Predictions for {team1} and {team2}:", predictions)





# 'Home_Decimal',  
# 'Decimal_Visitor',    
# 'Offense_FG%_Visitor', 
# 'Defense_FG%_Home',    
# 'Defense_TRB_Home',     
# 'Offense_2P%_Visitor',  
# 'Defense_DRB_Home',      
# 'Offense_FG%_Home',     
# 'Defense_2P%_Home',       
# 'Offense_PTS_Visitor',   
# 'Defense_AST_Home',       
# 'Offense_FG_Visitor',     
# 'Offense_3P%_Visitor',    
# 'Defense_FG%_Visitor',    
# 'Defense_BLK_Home',      
# 'Defense_FG_Home',
#         'Home',
#         'Visitor',
# 'Offense_3P%_Home',        
# 'Defense_PTS_Home'

#         'Team', 'Year', 'G', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 
#         '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 
#         'STL', 'BLK', 'TOV', 'PF', 'PTS'