import pandas as pd
from bs4 import BeautifulSoup, Comment
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import time
import requests
from sklearn.preprocessing import StandardScaler
import joblib





########################    CSS    #########################
# Add a thicker divider using HTML
st.sidebar.markdown('<hr style="border:2px solid black">', unsafe_allow_html=True)

#########################    LOAD MODEL    ################################

file_path = "not_scaled_nba.keras"
# scaler = joblib.load('scaler_nba.joblib')




########################    PREPARING LISTS AND VARIABLES    #########################


# List of team abbreviations
teams = [
    'BOS', 'NYK', 'MIL', 'CLE', 'ORL', 'IND', 'PHI', 'MIA', 'CHI', 'ATL',
    'BRK', 'TOR', 'CHO', 'WAS', 'DET', 'OKC', 'DEN', 'MIN', 'LAC', 'DAL',
    'PHO', 'NOP', 'LAL', 'SAC', 'GSW', 'HOU', 'UTA', 'MEM', 'SAS', 'POR'
]

# Set custom headers for requests
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}

# Streamlit app layout
# st.title('Machine Learning NBA Moneyline Model')
st.markdown("<h1 style='color: blue;'>Machine Learning NBA Moneyline Model</h1>", unsafe_allow_html=True)




#########################    TEAM SELECTION    #########################
st.markdown('<hr style="border:3px solid black">', unsafe_allow_html=True)

# Add space
st.markdown("<br><br>", unsafe_allow_html=True)

team_away = st.selectbox('**Select the away team**', teams)
away_team_odds = st.slider('**Enter away team decimal odds**', min_value=0.1, max_value=10.0, value=1.0, step=0.05)


# Add space
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown('<hr style="border:3px solid black">', unsafe_allow_html=True)

# Add space
st.markdown("<br><br>", unsafe_allow_html=True)
team_home = st.selectbox('**Select the home team**', teams)
home_team_odds = st.slider('**Enter home team decimal odds**', min_value=0.1, max_value=10.0, value=1.0, step=0.05)

# Add space
st.markdown("<br><br>", unsafe_allow_html=True)


st.markdown('<hr style="border:3px solid black">', unsafe_allow_html=True)


year = st.selectbox('**Select a year**', options=[2023, 2024])

st.write('**American Odds to Decimal Odds Converter**')


#########################    AMERICAN TO DECIMAL ODDS CONVERTER    #########################

# Create an expander section for the odds converter
with st.expander('**American Odds to Decimal Odds Converter:**'):
    
    # Function to convert American odds to Decimal odds
    def american_to_decimal(american_odds):
        return (american_odds / 100) + 1 if american_odds > 0 else (100 / abs(american_odds)) + 1

    # Input box for the user to enter American odds
    american_odds = st.number_input("Enter American odds:", value=-100)

    # Button to perform conversion
    if st.button("Convert to Decimal Odds"):
        decimal_odds = american_to_decimal(american_odds)
        st.write(f"American odds: {american_odds} ---> Decimal odds: {decimal_odds}")
        # st.write(f"Error details: {e}")



st.markdown('<hr style="border:3px solid black">', unsafe_allow_html=True)


submit = st.button('**Submit**')

# Add space
st.markdown("<br><br>", unsafe_allow_html=True)



# Create a list to store the scraped data
all_data = []

########################    Scraping Bot    #########################
# Declare df_combined as a global variable
global df_combined
df_combined = pd.DataFrame()

# Function to scrape data and populate the DataFrame
def scrape_data(team, year):
    url = f"https://www.basketball-reference.com/teams/{team}/{year}.html"
    # st.write(f"Scraping {team} in {year} from URL: {url}")
    
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Find all comment blocks (which include the table we want)
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))

    # Iterate through the comments to find the relevant table
    for comment in comments:
        if 'team_and_opponent' in comment:
            # Parse the comment as HTML to find the table
            comment_soup = BeautifulSoup(comment, 'html.parser')
            table = comment_soup.find('table', {'id': 'team_and_opponent'})
            
            # If the table is found, extract rows
            if table:
                rows = table.find_all('tr')
                
                for row in rows:
                    # Extract the text from each column (cell)
                    cols = row.find_all(['td', 'th'])
                    cols = [ele.text.strip() for ele in cols]  # Clean up the text
                    
                    if len(cols) > 1:  # Make sure to skip rows without data
                        # Add team and year info
                        cols.insert(0, year)
                        cols.insert(0, team)
                        all_data.append(cols)


########################    Submit Button    #########################

# Trigger scraping if the submit button is clicked
if submit:
    # Scrape data for home team
    scrape_data(team_home, year)
    
    # Scrape data for away team
    scrape_data(team_away, year)

    # Convert the data into a DataFrame
    if all_data:  # Check if data was scraped successfully
        headers = ['Team', 'invisible_column','Year', 'G', 'MP', 'FG', 'FGA', 'FG%', '3P', '3PA', '3P%', 
                   '2P', '2PA', '2P%', 'FT', 'FTA', 'FT%', 'ORB', 'DRB', 'TRB', 'AST', 
                   'STL', 'BLK', 'TOV', 'PF', 'PTS']
        df_combined = pd.DataFrame(all_data, columns=headers)

        # Display the combined scraped data
        st.subheader('Combined Team Data')
        st.dataframe(df_combined)
    else:
        st.write("No data was scraped. The table might not be available for the selected teams or year.")

try:
    # List of indices to drop
    indices_to_drop = [0, 1, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 16, 17]

    # Drop the rows using the index list
    df_dropped = df_combined.drop(indices_to_drop, axis=0)

    # Drop unnecessary columns
    df_dropped = df_dropped.drop(columns=['Year', 'invisible_column', 'G', 'MP', 'FGA', '3PA', '2P', '2PA', 'FT', 'FT%', 'FTA', 
                                          'ORB', 'STL', 'TOV', 'PF'])

    # Split the data into parts and assign prefixes for each section
    visitor_offense = df_dropped.iloc[0].add_prefix('Visitor_offense_')
    visitor_defense = df_dropped.iloc[1].add_prefix('Visitor_defense_')
    home_offense = df_dropped.iloc[2].add_prefix('Home_offense_')
    home_defense = df_dropped.iloc[3].add_prefix('Home_defense_')

    # Concatenate all rows horizontally into a single row
    combined_row = pd.concat([visitor_offense, visitor_defense, home_offense, home_defense])

    # Add the away_team_odds and home_team_odds as new Series and concatenate them
    odds_series = pd.Series({
        'away_team_odds': away_team_odds,
        'home_team_odds': home_team_odds
    })

    combined_row = pd.concat([combined_row, odds_series])

    # Convert to DataFrame for display
    df_combined_2 = pd.DataFrame([combined_row])

    # Encoding teams
    team_mapping = {
        'BOS': 0, 'NYK': 1, 'MIL': 2, 'CLE': 3, 'ORL': 4, 'IND': 5, 'PHI': 6, 'MIA': 7,
        'CHI': 8, 'ATL': 9, 'BRK': 10, 'TOR': 11, 'CHO': 12, 'WAS': 13, 'DET': 14,
        'OKC': 15, 'DEN': 16, 'MIN': 17, 'LAC': 18, 'DAL': 19, 'PHO': 20, 'NOP': 21,
        'LAL': 22, 'SAC': 23, 'GSW': 24, 'HOU': 25, 'UTA': 26, 'MEM': 27, 'SAS': 28, 'POR': 29
    }
    df_combined_2['Visitor_offense_Team'] = df_combined_2['Visitor_offense_Team'].replace(team_mapping)
    df_combined_2['Visitor_defense_Team'] = df_combined_2['Visitor_defense_Team'].replace(team_mapping)
    df_combined_2['Home_offense_Team'] = df_combined_2['Home_offense_Team'].replace(team_mapping)
    df_combined_2['Home_defense_Team'] = df_combined_2['Home_defense_Team'].replace(team_mapping)

    # Load your machine learning model
    model = load_model(file_path)

    # Select the features required for your model
    X = df_combined_2[[
        'Visitor_offense_FG%', 'Visitor_offense_2P%', 'Visitor_offense_PTS',
        'Visitor_offense_FG', 'Visitor_offense_3P%', 'Visitor_defense_FG%',
        'Visitor_offense_Team', 'Home_defense_FG%', 'Home_defense_TRB',
        'Home_defense_2P%', 'Home_defense_AST', 'Home_defense_BLK',
        'Home_offense_Team', 'Home_offense_FG%', 'Home_offense_3P%',
        'Home_defense_FG', 'Home_defense_PTS', 'away_team_odds', 'home_team_odds'
    ]]

    # Run the model prediction
    predictions = model.predict(X)
    st.write("The home team has a", predictions * 100, "chance of winning the game!")

except (KeyError, NameError) as e:
    # Display a warning message if any of the data is missing or undefined
    st.warning("")




# ########################    Dropping Unwanted Rows    #########################

# # # List of indices to drop
# indices_to_drop = [0, 1, 3, 4, 5, 7, 8, 9, 10, 12, 13, 14, 16, 17]

# # Drop the rows using the index list
# df_dropped = df_combined.drop(indices_to_drop, axis=0)
# # List of indices to drop



# # Display the resulting DataFrame after dropping rows
# # st.dataframe(df_dropped)

# ########################    Fixing Input Shape for ML Model    #########################
# df_dropped=df_dropped.drop(columns=['Year','invisible_column', 'G', 'MP', 'FGA', '3PA', '2P', '2PA', 'FT', 'FT%', 'FTA', 
#                                     'ORB', 'STL', 'TOV', 'PF'])
# # st.dataframe(df_dropped)

# # st.write('yes')
# # st.write(df_dropped)
# # Split the data into parts and assign the prefixes for each section
# visitor_offense = df_dropped.iloc[0].add_prefix('Visitor_offense_')
# visitor_defense = df_dropped.iloc[1].add_prefix('Visitor_defense_')
# home_offense = df_dropped.iloc[2].add_prefix('Home_offense_')
# home_defense = df_dropped.iloc[3].add_prefix('Home_defense_')

# # Concatenate all rows horizontally into a single row
# combined_row = pd.concat([visitor_offense, visitor_defense, home_offense, home_defense])

# # Add the away_team_odds and home_team_odds as new Series and concatenate them
# odds_series = pd.Series({
#     'away_team_odds': away_team_odds,
#     'home_team_odds': home_team_odds
# })

# combined_row = pd.concat([combined_row, odds_series])

# # Convert to DataFrame for display
# df_combined_2 = pd.DataFrame([combined_row])

# # Display the combined row in Streamlit
# # st.dataframe(df_combined_2.columns)

# ########################    Encoding    #########################

# #encoding teams
# team_mapping = {
#     'BOS': 0,
#     'NYK': 1,
#     'MIL': 2,
#     'CLE': 3,
#     'ORL': 4,
#     'IND': 5,
#     'PHI': 6,
#     'MIA': 7,
#     'CHI': 8,
#     'ATL': 9,
#     'BRK': 10,
#     'TOR': 11,
#     'CHO': 12,
#     'WAS': 13,
#     'DET': 14,
#     'OKC': 15,
#     'DEN': 16,
#     'MIN': 17,
#     'LAC': 18,
#     'DAL': 19,
#     'PHO': 20,
#     'NOP': 21,
#     'LAL': 22,
#     'SAC': 23,
#     'GSW': 24,
#     'HOU': 25,
#     'UTA': 26,
#     'MEM': 27,
#     'SAS': 28,
#     'POR': 29
# }
# df_combined_2['Visitor_offense_Team'] = df_combined_2['Visitor_offense_Team'].replace(team_mapping)
# df_combined_2['Visitor_defense_Team'] = df_combined_2['Visitor_defense_Team'].replace(team_mapping)
# df_combined_2['Home_offense_Team'] = df_combined_2['Home_offense_Team'].replace(team_mapping)
# df_combined_2['Home_defense_Team'] = df_combined_2['Home_defense_Team'].replace(team_mapping)

# # st.dataframe(df_combined_2)
# # st.dataframe(df_combined_2)

# # Define the new column order
# # new_column_order = [
# #     'home_team_odds', 'away_team_odds', 'Visitor_offense_FG%', 'Home_defense_FG%',
# #     'Home_defense_TRB', 'Visitor_offense_2P%', 'Home_defense_DRB', 'Home_offense_FG%', 'Home_defense_2P%',
# # 'Visitor_offense_PTS', 'Home_defense_AST', 'Visitor_offense_FG', 'Visitor_offense_3P%', 'Visitor_defense_FG%',
# # 'Home_defense_BLK', 'Home_defense_FG', 'Home_offense_Team', 'Visitor_offense_Team', 'Home_offense_3P%', 'Home_defense_PTS'
# # ]





# # df_reordered = df_combined_2[new_column_order]
# # st.write('hello')
# # st.write(df_combined_2.columns)






# #########################    RUN THE MODEL    ################################

# # Load your machine learning model
# model = load_model(file_path)

# #Select the features required for your model (replace with actual features)
# #Assume the model needs columns like ['FG', 'FGA', '3P', '3PA', 'PTS']

# X = df_combined_2[[
#     'Visitor_offense_FG%',
#     'Visitor_offense_2P%',
#     'Visitor_offense_PTS',
#     'Visitor_offense_FG',
#     'Visitor_offense_3P%',
#     'Visitor_defense_FG%',
#     'Visitor_offense_Team',
#     'Home_defense_FG%',
#     'Home_defense_TRB',
#     'Home_defense_2P%',
#     'Home_defense_AST',
#     'Home_defense_BLK',
#     'Home_offense_Team',
#     'Home_offense_FG%',
#     'Home_offense_3P%',
#     'Home_defense_FG',
#     'Home_defense_PTS',
#     'away_team_odds',
#     'home_team_odds'
# ]]

# # Run the model prediction
# predictions = model.predict(X)
# st.write("The home team has a", predictions * 100, "chance of winning the game!")





















# df_combined_2.rename({  'home_team_odds': 'Home_Decimal', 'away_team_odds':'Decimal_Visitor',  'Visitor_offense_FG%':'Offense_FG%_Visitor', 
#                              'Home_defense_FG%':'Defense_FG%_Home','Home_defense_TRB':'Defense_TRB_Home', 'Visitor_offense_2P%':'Offense_2P%_Visitor', 
#                              'Home_defense_DRB':'Defense_DRB_Home', 'Home_offense_FG%':'Offense_FG%_Home', 'Home_defense_2P%':'Defense_2P%_Home',
#                              'Visitor_offense_PTS': 'Offense_PTS_Visitor', 'Home_defense_AST':'Defense_AST_Home', 'Visitor_offense_FG': 'Offense_FG_Visitor', 
#                              'Visitor_offense_3P%':'Offense_3P%_Visitor', 'Visitor_defense_FG%':'Defense_FG%_Visitor','Home_defense_BLK':'Defense_BLK_Home', 
#                              'Home_defense_FG':'Defense_FG_Home', 'Home_offense_Team':'Home', 'Visitor_offense_Team':'Visitor', 'Home_offense_3P%':'Offense_3P%_Home', 
#                              'Home_defense_PTS':'Defense_PTS_Home'}, inplace=True, axis='columns')

# st.write(df_combined_2.columns)
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
# df_reordered = df_reordered.apply(lambda x: pd.to_numeric(x, errors='raise'))
# st.dataframe(df_reordered)



#######################################    SCALING    ###########################################


# # Display the current data types
# # st.write(df_combined_2.dtypes)

# #scaling data
# # Columns that should not be scaled
# columns_to_exclude = ['Visitor_offense_Team', 'away_team_odds', 'Home_offense_Team', 'home_team_odds']

# # Separate the data into the columns to scale and columns to exclude
# columns_to_scale = df_combined_2.drop(columns=columns_to_exclude).columns  # All other columns

# # Create a scaler
# # scaler = StandardScaler()

# # Scale the columns that need scaling
# df_scaled = pd.DataFrame(scaler.fit_transform(df_combined_2[columns_to_scale]), columns=columns_to_scale)
# # joblib.dump(scaler, 'scaler_nba.joblib')
# # Concatenate the scaled columns back with the excluded columns
# df_final = pd.concat([df_combined_2[columns_to_exclude].reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)

# # Display the resulting DataFrame
# st.write(df_final)



# # Identify columns that are of type 'object' (likely containing strings or mixed data)
# object_columns = df_combined_2.select_dtypes(include=['object']).columns

# st.write("Columns with object data type (potential strings):")
# st.write(object_columns)

#scaling data
# Columns that should not be scaled
# columns_to_exclude = ['Visitor_offense_Team', 'away_team_odds', 'Home_offense_Team', 'home_team_odds']
# # Check for zero variance columns before scaling
# st.subheader("Variance of Columns Before Scaling")
# variance_before_scaling = df_reordered.drop(columns=columns_to_exclude).var()
# st.dataframe(variance_before_scaling)


# # Separate the data into the columns to scale and columns to exclude
# columns_to_scale = df_reordered.drop(columns=columns_to_exclude).columns  # All other columns

# # Create a scaler
# scaler = StandardScaler()

# # Scale the columns that need scaling
# # Use df_reordered[columns_to_scale] to pass the actual data, not just column names
# df_scaled = pd.DataFrame(scaler.fit_transform(df_reordered[columns_to_scale]), columns=columns_to_scale)


# # Concatenate the scaled columns back with the excluded columns
# df_final = pd.concat([df_reordered[columns_to_exclude].reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)

# # Display the resulting DataFrame
# st.dataframe(df_final)

# 'Home_Decimal',  

# 'Decimal_Visitor',   
#  
# 'Offense_FG%_Visitor', 

# 'Defense_FG%_Home',    

# 'Defense_TRB_Home',     

# 'Offense_2P%_Visitor',  

# 'Defense_DRB_Home',  
#     
# 'Offense_FG%_Home',     

# 'Defense_2P%_Home',       

# 'Offense_PTS_Visitor',   

# 'Defense_AST_Home',    
#    
# 'Offense_FG_Visitor', 
#     
# 'Offense_3P%_Visitor',    

# 'Defense_FG%_Visitor',    

# 'Defense_BLK_Home',   
#    
# 'Defense_FG_Home',
#         'Home',
#         'Visitor',
# 'Offense_3P%_Home',        
# 'Defense_PTS_Home'

#Visitor stats
# 'Decimal_Visitor',    
# 'Offense_FG%_Visitor', 
# 'Offense_2P%_Visitor',  
# 'Offense_PTS_Visitor', 
# 'Offense_3P%_Visitor',    
# 'Defense_FG%_Visitor', 
# 'Offense_FG_Visitor', 
#visitor

#home stats# 
#home_decimal
# 'Defense_FG%_Home',    
# 'Defense_TRB_Home',   
# 'Defense_DRB_Home',      
# 'Offense_FG%_Home',     
# 'Defense_2P%_Home', 
# 'Defense_AST_Home', 
# 'Defense_BLK_Home',      
# 'Defense_FG_Home',
#home
# 'Offense_3P%_Home',        
# 'Defense_PTS_Home'