import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Output, Input
from twilio.rest import Client
import requests
import pandas as pd
from dash_app.line_scraper_functions import extract_games_from_df, process_lines, process_games

from dash_app.nba_scraper import get_games

account_sid = 'AC500cdffd41d5ae4f275e2eecc7c8677d'
auth_token = '772f732acf5ff166e06e2b076c2f220b'
client = Client(account_sid, auth_token)
messages_sent = set()
id_to_team = dict()
team_to_id = dict()
id_to_book = {55: '5Dimes', 24: 'Bet365', 42: 'BetMania', 34: 'Bookmaker', 3: 'Pinnacle', 1: 'BetOnline',
              16: 'Heritage', 8: 'MyBookie.ag', 21: 'Bovada', 4: 'Intertops', 9: 'Youwager', 28: 'SportsBetting',
              2: 'JustBet', 37: 'SportBet', 33: 'The Greek', 64: 'Nitrogen', 50: 'GTBets'}
book_to_id = dict((b, i) for i, b in id_to_book.items())

app = dash.Dash(__name__)
server = app.server
app.layout = html.Div(
    html.Div([
        html.H4('Bet Dashboard'),
        html.Div(id='live-update-table'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000*60*2, # 1s * 1000ms * 60s/min * 30min,
            n_intervals=0
        )
    ])
)

def calc_concensus(lines):
    lines = lines.sort_values('line_datetime')
    lines['concensus_ml_fav'] = np.nan
    lines['concensus_ml_dog'] = np.nan
    lines['ml_dog_std'] = np.nan
    lines['ml_fav_std'] = np.nan

    for key, grp in lines.groupby('game_uuid'):  # For each game
        # Keep each bookie's most recent line for the game
        grp = grp[grp.line_datetime < grp.game_datetime].drop_duplicates('bookie', keep='last')
        teams = set(grp.dog.unique()) | set(grp.fav.unique())
        probs = dict()
        for team in teams:
            probs[team] = []
        for idx, label in enumerate(grp.index):  # For each line for the current game
            row = grp.iloc[idx, :]
            for team in teams:
                if row.fav == team:
                    probs[team].append(row.ml_fav_prob)
                if row.dog == team:
                    probs[team].append(row.ml_dog_prob)

        means = dict()
        stds = dict()
        for team in teams:
            means[team] = np.mean(probs[team])
            stds[team] = np.std(probs[team])

        for idx, label in enumerate(grp.index):  # Now we'll record the mean/std
            row = grp.iloc[idx, :]

            for team in teams:
                if row.fav == team:
                    lines.loc[label, 'concensus_ml_fav'] = means[team]
                    lines.loc[label, 'ml_fav_std'] = stds[team]
                if row.dog == team:
                    lines.loc[label, 'concensus_ml_dog'] = means[team]
                    lines.loc[label, 'ml_dog_std'] = stds[team]
            # lines.loc[label, 'concensus_ml_fav'] = _grp['ml_fav_prob'].mean()
            # lines.loc[label, 'concensus_ml_dog'] = _grp['ml_dog_prob'].mean()
            # lines.loc[label, 'ml_dog_std'] = _grp['ml_dog_prob'].std()
            # lines.loc[label, 'ml_fav_std'] = _grp['ml_fav_prob'].std()
    lines = lines.dropna()
    lines['ml_fav_z'] = (lines['ml_fav_prob'] - lines['concensus_ml_fav']) / lines['ml_fav_std']
    lines['ml_dog_z'] = (lines['ml_dog_prob'] - lines['concensus_ml_dog']) / lines['ml_dog_std']
    return lines


def gen_table(df):
    cols = ['Game Date', 'Book', 'Line Time', 'Away', 'Away Ml', 'Away Z', 'Away Prob', 'Away Consensus',
         'Home', 'Home Ml', 'Home Z', 'Home Prob', 'Home Consensus', 'Game ID']
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in cols])] +

        # Body
        [html.Tr([
            html.Td(df.iloc[i][col]) for col in df.columns
        ]) for i in range(min(len(df), len(df)))]
    )

def gen_text(row, z):
    print(row)
    if z:
        if row.home_ml_z <= -3:
            return 'Bet on ' + row.home_team + ' on ' + str(row.game_date) + ' at ' + str(row.home_ml) + ' odds at ' + str(row.book)
        if row.away_ml_z <= -3:
            return 'Bet on ' + row.away_team + ' on ' + str(row.game_date) + ' at ' + str(row.away_ml) + ' odds at ' + str(row.book)
    else:
        print(row)
        if row.home_ml_prob <= (row.home_ml_consensus - .05):
            return 'Bet on ' + row.home_team + ' on ' + str(row.game_date) + ' at ' + str(row.home_ml) + ' odds at ' + str(row.book)
        if row.away_ml_prob <= (row.away_ml_consensus - .05):
            return 'Bet on ' + row.away_team + ' on ' + str(row.game_date) + ' at ' + str(row.away_ml) + ' odds at ' + str(row.book)


def line_to_prob(line):
    if line < 0:
        return -line / (-line + 100.)
        # Do some stuff
    elif line > 0:
        return 100. / (line + 100.)
        # Do some other stuff


@app.callback(Output('live-update-table', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_table_v2(n_intervals):
    divs = []
    url = 'https://api-prod.sprtactn.co/web/v1/scoreboard/nba?bookIds='
    for i in range(100):
        if i != 45:
            url = url + str(i) + ','
    url = url[:-1]
    r = requests.get(url)
    json = r.json()
    games = json['games']
    df = dict(home_ml=[], away_ml=[], book_id=[], line_dt=[], home_id=[], away_id=[], game_id=[], game_date=[])
    for game in games:
        teams = game['teams']
        for team in teams:
            id_to_team[team['id']] = team['display_name']
            team_to_id[team['display_name']] = team['id']

        home_team = id_to_team[game['home_team_id']]
        away_team = id_to_team[game['away_team_id']]
        odds = game['odds']
        for odd in odds:
            if odd['type'] == 'game':
                df['home_ml'].append(odd['ml_home'])
                df['away_ml'].append(odd['ml_away'])
                df['book_id'].append(odd['book_id'])
                df['line_dt'].append(odd['inserted'])
                df['home_id'].append(game['home_team_id'])
                df['away_id'].append(game['away_team_id'])
                df['game_id'].append(game['id'])
                df['game_date'].append(game['start_time'])

    df = pd.DataFrame(df)
    df = df.dropna()
    df = df[df['book_id'].isin(id_to_book.keys())]
    # Process lines. Convert American odds to probabilities
    df['home_ml_prob'] = df['home_ml'].apply(lambda x: line_to_prob(x))
    df['away_ml_prob'] = df['away_ml'].apply(lambda x: line_to_prob(x))
    df['home_team'] = df['home_id'].apply(lambda x: id_to_team[x])
    df['away_team'] = df['away_id'].apply(lambda x: id_to_team[x])
    df['home_ml_consensus'] = np.nan
    df['away_ml_consensus'] = np.nan
    df['home_ml_std'] = np.nan
    df['away_ml_std'] = np.nan
    df['line_dt'] = pd.to_datetime(df['line_dt'])
    df['game_date'] = pd.to_datetime(df['game_date'])
    df['Pretty_Line_Time'] = df['line_dt'].apply(lambda x: (x-datetime.timedelta(seconds=5*60*60)).strftime('%d-%m-%y %I:%M %p'))
    df['Pretty_Game_Time'] = df['game_date'].apply(lambda x: x.strftime('%d-%m-%y'))
    df['book'] = df['book_id'].apply(lambda x: id_to_book[x])

    # Calc consensus
    df = df.sort_values('line_dt')
    for key, grp in df.groupby('game_id'):
        grp = grp.drop_duplicates('book_id', keep='last')
        df.loc[grp.index, 'home_ml_consensus'] = grp['home_ml_prob'].mean()
        df.loc[grp.index, 'away_ml_consensus'] = grp['away_ml_prob'].mean()

        df.loc[grp.index, 'home_ml_std'] = grp['home_ml_prob'].std()
        df.loc[grp.index, 'away_ml_std'] = grp['away_ml_prob'].std()

    df = df.dropna()

    df['home_ml_prob'] = df['home_ml_prob'].round(4)
    df['away_ml_prob'] = df['away_ml_prob'].round(4)

    df['home_ml_consensus'] = df['home_ml_consensus'].round(4)
    df['away_ml_consensus'] = df['away_ml_consensus'].round(4)

    df['home_ml_std'] = df['home_ml_std'].round(4)
    df['away_ml_std'] = df['away_ml_std'].round(4)

    df['home_ml_z'] = ((df['home_ml_prob'] - df['home_ml_consensus']) / df['home_ml_std']).round(4)
    df['away_ml_z'] = ((df['away_ml_prob'] - df['away_ml_consensus']) / df['away_ml_std']).round(4)

    # Check for betting opportunities
    for key, grp in df.groupby('game_id'):
        z_grp = grp.loc[(grp.home_ml_z <= -2.75) | (grp.away_ml_z <= -2.75), :]
        p_grp = grp.loc[(grp.home_ml_prob <= (grp.home_ml_consensus - .05)) |
                        (grp.away_ml_prob <= (grp.away_ml_consensus - .05)), :]

        if len(z_grp) > 0 and len(grp) >= 10:
            for i in range(len(z_grp)):
                row = z_grp.iloc[i, :]
                # if not gen_text(row, True) in messages_sent:
                #     message = client.messages.create(to='+16179356853',
                #                                      from_='+17814606736',
                #                                      body=gen_text(row, True))
                #     messages_sent.add(gen_text(row, True))
        if len(p_grp) > 0:
            for i in range(len(p_grp)):
                row = p_grp.iloc[i, :]
                # if not gen_text(row, False) in messages_sent:
                #     message = client.messages.create(to='+16179356853',
                #                                      from_='+17814606736',
                #                                      body=gen_text(row, False))
                #     messages_sent.add(gen_text(row, False))
        c = ['Pretty_Game_Time', 'book', 'Pretty_Line_Time',
             'away_team', 'away_ml', 'away_ml_z', 'away_ml_prob', 'away_ml_consensus',
             'home_team', 'home_ml', 'home_ml_z', 'home_ml_prob', 'home_ml_consensus', 'game_id']
        div = html.Div([
            html.H1('{} @ {}'.format(away_team, home_team)),
            html.H4('All Current Lines'),
            gen_table(grp.loc[:, c]),
            html.H4('Lines 3 z-scores off'),
            gen_table(grp.loc[(grp.home_ml_z <= -3) | (grp.away_ml_z <= -3), c]),
            html.H4('Lines 5% probability off'),
            gen_table(grp.loc[(grp.home_ml_prob <= (grp.home_ml_consensus - .05)) |
                              (grp.away_ml_prob <= (grp.away_ml_consensus - .05)), c])

        ])
        divs.append(div)
    return html.Div(divs)

# @app.callback(Output('live-update-table', 'children'),
#               [Input('interval-component', 'n_intervals')])
# def update_table(n_intervals):
#
#     today = datetime.datetime.today()
#     year = today.year
#     month = today.strftime("%B").lower()
#     nba_games = get_games([year], [month])
#     nba_games = process_games(nba_games)
#     nba_games = nba_games.replace('blazers', 'trail-blazers')
#     nba_games = nba_games.loc[(nba_games.date >= today) & (nba_games.date - today <= datetime.timedelta(days=2)), :]
#     lines = extract_games_from_df(nba_games, 'nba')
#
#     lines = process_lines(lines)
#     lines = calc_concensus(lines)
#
#     # requests.post('https://api.twilio.com/2010-04-01/Accounts/AC500cdffd41d5ae4f275e2eecc7c8677d/Messages.json',
#     #               data={'To': '+16179356853', 'From': '+17814606736', 'Body': str(datetime.datetime.now())})
#
#     # message = client.messages.create(to='+16179356853',
#     #                                  from_='+17814606736',
#     #                                  body='Successfully refreshed data at {}'.format(str(datetime.datetime.now())))
#
#     c = ['game_datetime', 'bookie', 'line_datetime', 'dog', 'ml_dog_line', 'fav', 'ml_fav_line', 'ml_fav_z', 'ml_dog_z',
#          'ml_fav_prob', 'ml_dog_prob', 'concensus_ml_fav', 'concensus_ml_dog', 'game_uuid']
#     lines = lines.loc[:, c]
#     divs = []
#     for key, grp in lines.groupby('game_uuid'):
#         grp = grp.drop_duplicates('bookie', keep='last')
#         home_team = nba_games.loc[nba_games.uuid == key, 'home_team'].values[0]
#         away_team = nba_games.loc[nba_games.uuid == key, 'away_team'].values[0]
#         print('Home team: {}, Away team: {}'.format(home_team, away_team))
#         z_grp = grp.loc[(grp.ml_fav_z <= -3) | (grp.ml_dog_z <= -3), c[:-1]]
#         p_grp = grp.loc[(grp.ml_fav_prob <= (grp.concensus_ml_fav - .05)) |
#                               (grp.ml_dog_prob <= (grp.concensus_ml_dog - .05)), c[:-1]]
#
#         if len(z_grp) > 0:
#             for i in range(len(z_grp)):
#                 row = z_grp.iloc[i, :]
#                 print(messages_sent)
#                 print(gen_text(row, True) in messages_sent)
#                 if not gen_text(row, True) in messages_sent:
#                     message = client.messages.create(to='+16179356853',
#                                                      from_='+17814606736',
#                                                      body=gen_text(row, True))
#                     messages_sent.add(gen_text(row, True))
#                 # if row.ml_fav_z <= -3:
#                 #     body = 'Bet on ' + row.fav + ' on ' + str(row.game_datetime) + ' at ' + str(row.ml_fav_line) + ' odds at ' + row.bookie
#                 # if row.ml_dog_z <= -3:
#                 #     body = 'Bet on ' + row.dog + ' on ' + str(row.game_datetime) + ' at ' + str(row.ml_dog_line) + ' odds at ' + row.bookie
#                 print(messages_sent)
#                 # if body not in messages_sent:
#                 #     message = client.messages.create(to='+16179356853',
#                 #                                      from_='+17814606736',
#                 #                                      body=body)
#                 #     messages_sent |=
#                 # requests.post(
#                 #     'https://api.twilio.com/2010-04-01/Accounts/AC500cdffd41d5ae4f275e2eecc7c8677d/Messages.json',
#                 #     data={'To': '+16179356853', 'From': '+17814606736', 'Body': body})
#         if len(p_grp) > 0:
#             for i in range(len(p_grp)):
#                 row = p_grp.iloc[i, :]
#                 if not gen_text(row, False) in messages_sent:
#                     message = client.messages.create(to='+16179356853',
#                                                      from_='+17814606736',
#                                                      body=gen_text(row, False))
#                     messages_sent.add(gen_text(row, False))
#                 #
#                 # if body not in messages_sent:
#                 #     message = client.messages.create(to='+16179356853',
#                 #                                      from_='+17814606736',
#                 #                                      body=body)
#                 #     messages_sent |= set([body])
#                 # requests.post(
#                 #     'https://api.twilio.com/2010-04-01/Accounts/AC500cdffd41d5ae4f275e2eecc7c8677d/Messages.json',
#                 #     data={'To': '+16179356853', 'From': '+17814606736', 'Body': body})
#         div = html.Div([
#             html.H1('{} @ {}'.format(away_team, home_team)),
#             html.H4('All Current Lines'),
#             gen_table(grp.loc[:, c[:-5]]),
#             html.H4('Lines 3 z-scores off'),
#             gen_table(grp.loc[(grp.ml_fav_z <= -3) | (grp.ml_dog_z <= -3), c[:-1]]),
#             html.H4('Lines 5% probability off'),
#             gen_table(grp.loc[(grp.ml_fav_prob <= (grp.concensus_ml_fav - .05)) |
#                               (grp.ml_dog_prob <= (grp.concensus_ml_dog - .05)), c[:-1]])
#
#         ])
#         divs.append(div)
#     # z_lines = current_lines.loc[(current_lines.ml_fav_z <= -3) | (current_lines.ml_dog_z <= -3)]
#     # p_lines = current_lines.loc[(current_lines.ml_fav_prob <= (current_lines.concensus_ml_fav - .05)) |
#     #                             (current_lines.ml_dog_prob <= (current_lines.concensus_ml_dog - .05))]
#     return html.Div(divs)


if __name__ == '__main__':
    app.run_server(debug=True, port=8050, host='192.168.1.194')