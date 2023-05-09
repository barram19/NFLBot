import requests
import numpy as np
import xgboost as xgb
import pandas as pd

# Set API endpoint URL and API key
url = 'https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds-history/?apiKey=00ab27442da4a2b1f8460c5c70d0b3d8&regions=us&markets=h2h,spreads,totals&oddsFormat=american&date=2021-10-18T12:00:00Z&bookmakers=fanduel'
#^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^Will need to update URL above for future odds vs historical ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#########https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds/?apiKey=00ab27442da4a2b1f8460c5c70d0b3d8&regions=us&markets=h2h,spreads,totals&oddsFormat=american
api_key = '00ab27442da4a2b1f8460c5c70d0b3d8'

# Set headers for the request
headers = {
    'Content-Type': 'application/json',
    'x-api-key': api_key,
}

# Make the request and get the response data
response = requests.get(url, headers=headers)

if response.status_code == 200:
    response_data = response.json()
    # Process the odds data

    # Extract the necessary information from the odds data
game_data = []
for game in odds_data:
    home_team = game['home_team']
    away_team = game['away_team']
    commence_time = game['commence_time'][:10] # Extract the date from the commence_time field
    bookmakers = game['bookmakers']
    for bookmaker in bookmakers:
        markets = bookmaker['markets']
        for market in markets:
            if market['key'] == 'h2h':
                outcomes = market['outcomes']
                home_odds = outcomes[0]['price']
                away_odds = outcomes[1]['price']
            elif market['key'] == 'totals':
                outcomes = market['outcomes']
                point = outcomes[0]['point']
                over_odds = outcomes[0]['price']
                under_odds = outcomes[1]['price']
                if point is not None:
                    game_data.append([home_team, away_team, commence_time, home_odds, away_odds, point, over_odds, under_odds])

# Convert the game data to a numpy array and split it into features and labels
game_data = np.array(game_data)
X = game_data[:, 3:6].astype(float)
y_win = np.where(game_data[:, 0] == game_data[:, 1], 2, np.where(game_data[:, 3] > game_data[:, 4], 1, 0))
y_total = np.where(game_data[:, 3].astype(float) > 8.5, 1, 0) #may need to update to y_total = np.where(game_data[:, 5].astype(float) > 8.5, 1, 0) for furture odds vs historical


# Train the XGB model on the data for game winner
model_win = xgb.XGBClassifier()
model_win.fit(X, y_win)

# Train the XGB model on the data for over/under
model_total = xgb.XGBClassifier()
model_total.fit(X, y_total)

# Make predictions on new data using the trained model
new_data = X
y_pred_win = model_win.predict(new_data)
y_pred_total = model_total.predict(new_data)

# Create lists to store the table data
dates = []
matchups = []
win_predictions = []
total_predictions = []

# Output the predicted labels along with the matchup and date
for i, game in enumerate(game_data):
    home_team = game[0]
    away_team = game[1]
    date = game[2]
    win_prediction = y_pred_win[i]
    total_prediction = y_pred_total[i]
    if win_prediction == 0:
        win_pred = away_team + ' ML'
    elif win_prediction == 1:
        win_pred = home_team + ' ML'
    else:
        win_pred = 'Draw'
    if total_prediction == 0:
        total_pred = 'u' + str(game[5])
    else:
        total_pred = 'o' + str(game[5])
    
    # Add the data to the lists
    dates.append(date)
    matchups.append(away_team + ' vs ' + home_team)
    win_predictions.append(win_pred)
    total_predictions.append(total_pred)

# Create the table using pandas
df = pd.DataFrame({
    'Date': dates,
    'Matchup': matchups,
    'Predicted Winner': win_predictions,
    'Predicted Total': total_predictions
})
# Set max column width to unlimited
pd.set_option('display.max_colwidth', None)

# Display the table in-line
display(df)
