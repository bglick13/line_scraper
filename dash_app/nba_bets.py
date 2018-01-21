import dash
from dash.dependencies import Output, Input
import dash_core_components as dcc
import dash_html_components as html
import requests
from util.nba_scraper import get_games
from util.line_scraper_functions import extract_games_from_df, process_lines, process_games
import datetime
import numpy as np
from twilio.rest import Client

account_sid = 'AC500cdffd41d5ae4f275e2eecc7c8677d'
auth_token = '772f732acf5ff166e06e2b076c2f220b'
client = Client(account_sid, auth_token)
messages_sent = set()

app = dash.Dash(__name__)
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

    for key, grp in lines.groupby('game_uuid'):
        grp = grp[grp.line_datetime < grp.game_datetime]
        for idx, label in enumerate(grp.index):
            _grp = grp.drop_duplicates('bookie', keep='last')
            lines.loc[label, 'concensus_ml_fav'] = _grp['ml_fav_prob'].mean()
            lines.loc[label, 'concensus_ml_dog'] = _grp['ml_dog_prob'].mean()
            lines.loc[label, 'ml_dog_std'] = _grp['ml_dog_prob'].std()
            lines.loc[label, 'ml_fav_std'] = _grp['ml_fav_prob'].std()
    lines = lines.dropna()
    lines['ml_fav_z'] = (lines['ml_fav_prob'] - lines['concensus_ml_fav']) / lines['ml_fav_std']
    lines['ml_dog_z'] = (lines['ml_dog_prob'] - lines['concensus_ml_dog']) / lines['ml_dog_std']
    return lines


def gen_table(df):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in df.columns])] +

        # Body
        [html.Tr([
            html.Td(df.iloc[i][col]) for col in df.columns
        ]) for i in range(min(len(df), len(df)))]
    )

def gen_text(row, z):
    if z:
        if row.ml_fav_z <= -3:
            return 'Bet on ' + row.fav + ' on ' + str(row.game_datetime) + ' at ' + str(row.ml_fav_line) + ' odds at ' + row.bookie + ' (Z of ' + str(row.ml_fav_z) + ')'
        if row.ml_dog_z <= -3:
            return 'Bet on ' + row.dog + ' on ' + str(row.game_datetime) + ' at ' + str(row.ml_dog_line) + ' odds at ' + row.bookie + ' (Z of ' + str(row.ml_dog_z) + ')'
    else:
        if row.ml_fav_prob <= (row.concensus_ml_fav - .05):
            return 'Bet on ' + row.fav + ' on ' + str(row.game_datetime) + ' at ' + str(row.ml_fav_line) + ' odds at ' + row.bookie + ' (P of ' + str(row.ml_fav_prob) + ')'
        if row.ml_dog_prob <= (row.concensus_ml_dog - .05):
            return 'Bet on ' + row.dog + ' on ' + str(row.game_datetime) + ' at ' + str(row.ml_dog_line) + ' odds at ' + row.bookie + ' (P of ' + str(row.ml_dog_prob) + ')'


@app.callback(Output('live-update-table', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_table(n_intervals):

    today = datetime.datetime.today()
    year = today.year
    month = today.strftime("%B").lower()
    nba_games = get_games([year], [month])
    nba_games = process_games(nba_games)
    nba_games = nba_games.replace('blazers', 'trail-blazers')
    nba_games = nba_games.loc[(nba_games.date >= today) & (nba_games.date - today <= datetime.timedelta(days=2)), :]
    lines = extract_games_from_df(nba_games, 'nba')
    lines = process_lines(lines)
    lines = calc_concensus(lines)

    # requests.post('https://api.twilio.com/2010-04-01/Accounts/AC500cdffd41d5ae4f275e2eecc7c8677d/Messages.json',
    #               data={'To': '+16179356853', 'From': '+17814606736', 'Body': str(datetime.datetime.now())})

    # message = client.messages.create(to='+16179356853',
    #                                  from_='+17814606736',
    #                                  body='Successfully refreshed data at {}'.format(str(datetime.datetime.now())))

    c = ['game_datetime', 'bookie', 'line_datetime', 'dog', 'ml_dog_line', 'fav', 'ml_fav_line', 'ml_fav_z', 'ml_dog_z',
         'game_uuid', 'ml_fav_prob', 'ml_dog_prob', 'concensus_ml_fav', 'concensus_ml_dog']
    lines = lines.loc[:, c]
    divs = []
    for key, grp in lines.groupby('game_uuid'):
        grp = grp.drop_duplicates('bookie', keep='last')
        home_team = nba_games.loc[nba_games.uuid == key, 'home_team']
        away_team = nba_games.loc[nba_games.uuid == key, 'away_team']
        z_grp = grp.loc[(grp.ml_fav_z <= -3) | (grp.ml_dog_z <= -3), c[:-5]]
        p_grp = grp.loc[(grp.ml_fav_prob <= (grp.concensus_ml_fav - .05)) |
                              (grp.ml_dog_prob <= (grp.concensus_ml_dog - .05)), c[:-5]]

        if len(z_grp) > 0:
            for i in range(len(z_grp)):
                row = z_grp.iloc[i, :]
                print(messages_sent)
                print(gen_text(row, True) in messages_sent)
                if not gen_text(row, True) in messages_sent:
                    message = client.messages.create(to='+16179356853',
                                                     from_='+17814606736',
                                                     body=gen_text(row, True))
                    messages_sent.add(gen_text(row, True))
                # if row.ml_fav_z <= -3:
                #     body = 'Bet on ' + row.fav + ' on ' + str(row.game_datetime) + ' at ' + str(row.ml_fav_line) + ' odds at ' + row.bookie
                # if row.ml_dog_z <= -3:
                #     body = 'Bet on ' + row.dog + ' on ' + str(row.game_datetime) + ' at ' + str(row.ml_dog_line) + ' odds at ' + row.bookie
                print(messages_sent)
                # if body not in messages_sent:
                #     message = client.messages.create(to='+16179356853',
                #                                      from_='+17814606736',
                #                                      body=body)
                #     messages_sent |=
                # requests.post(
                #     'https://api.twilio.com/2010-04-01/Accounts/AC500cdffd41d5ae4f275e2eecc7c8677d/Messages.json',
                #     data={'To': '+16179356853', 'From': '+17814606736', 'Body': body})
        if len(p_grp) > 0:
            for i in range(len(p_grp)):
                row = p_grp.iloc[i, :]
                if not gen_text(row, False) in messages_sent:
                    message = client.messages.create(to='+16179356853',
                                                     from_='+17814606736',
                                                     body=gen_text(row, False))
                    messages_sent.add(gen_text(row, False))
                #
                # if body not in messages_sent:
                #     message = client.messages.create(to='+16179356853',
                #                                      from_='+17814606736',
                #                                      body=body)
                #     messages_sent |= set([body])
                # requests.post(
                #     'https://api.twilio.com/2010-04-01/Accounts/AC500cdffd41d5ae4f275e2eecc7c8677d/Messages.json',
                #     data={'To': '+16179356853', 'From': '+17814606736', 'Body': body})
        div = html.Div([
            html.H1('{} @ {}'.format(away_team.values[0], home_team.values[0])),
            html.H4('All Current Lines'),
            gen_table(grp.loc[:, c[:-5]]),
            html.H4('Lines 3 z-scores off'),
            gen_table(grp.loc[(grp.ml_fav_z <= -3) | (grp.ml_dog_z <= -3), c[:-5]]),
            html.H4('Lines 5% probability off'),
            gen_table(grp.loc[(grp.ml_fav_prob <= (grp.concensus_ml_fav - .05)) |
                              (grp.ml_dog_prob <= (grp.concensus_ml_dog - .05)), c[:-5]])
            
        ])
        divs.append(div)
    # z_lines = current_lines.loc[(current_lines.ml_fav_z <= -3) | (current_lines.ml_dog_z <= -3)]
    # p_lines = current_lines.loc[(current_lines.ml_fav_prob <= (current_lines.concensus_ml_fav - .05)) |
    #                             (current_lines.ml_dog_prob <= (current_lines.concensus_ml_dog - .05))]
    return html.Div(divs)


if __name__ == '__main__':
    app.run_server(debug=True)