import datetime

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
from dash.dependencies import Output, Input
from twilio.rest import Client
import requests
import pandas as pd
import MySQLdb
from MySQLdb.constants import FIELD_TYPE
import datetime
# from dash_app.db_config import user, password
from dash_app.line_scraper_functions import extract_games_from_df, process_lines, process_games
from dash_app.selenium_member_test import SeleniumSpider
from dash_app.nba_scraper import get_games

db = MySQLdb.connect(host="127.0.0.1", port=3306, user="root",passwd="",db="betting")
spider = SeleniumSpider()
account_sid = 'AC500cdffd41d5ae4f275e2eecc7c8677d'
auth_token = '772f732acf5ff166e06e2b076c2f220b'
client = Client(account_sid, auth_token)
messages_sent = set()
id_to_team = dict(nba=dict(), nhl=dict(), ncaab=dict())
team_to_id = dict(nba=dict(), nhl=dict(), ncaab=dict())
id_to_book = {55: '5Dimes', 24: 'Bet365', 42: 'BetMania', 34: 'Bookmaker', 3: 'Pinnacle', 1: 'BetOnline',
              16: 'Heritage',
              8: 'MyBookie.ag', 21: 'Bovada', 4: 'Intertops', 9: 'Youwager', 28: 'SportsBetting',
              13: 'Casears', 14: 'Westgate', 19: 'MGM Mirage', 32: 'Wynn',
              2: 'JustBet', 37: 'SportBet', 33: 'The Greek', 64: 'Nitrogen', 50: 'GTBets', 36: 'Jazz Sports',
              23: 'ABCislands', 52: 'LooseLines'}
book_to_id = dict((b, i) for i, b in id_to_book.items())

white_list = set(['Bovada', 'Bookmaker', 'Sportbet', 'Nitrogen', 'BetOnline', 'YouWager',
                  'JustBet', 'Heritage', 'MyBookie.ag'])


Z_CUTOFF = -2.75
P_CUTOFF = .026

# con = MySQLdb.connect(host='127.0.0.1', user=user, password=password, db='betting')
app = dash.Dash(__name__)
server = app.server
app.layout = html.Div(
    html.Div([
        html.H4('Bet Dashboard'),
        html.Div(id='live-update-table'),
        dcc.Interval(
            id='interval-component',
            interval=1*1000*60*1.5, # 1s * 1000ms * 60s/min * 30min,
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


def gen_table(df, draws=False):
    if draws:
        cols = ['Game Date', 'Book', 'Best Prob. Diff', 'Away', 'Away Ml', 'Away Prob. Diff.',
                'Home', 'Home Ml', 'Home Prob. Diff.', 'Draw Ml', 'Draw Prob. Diff']
    else:
        cols = ['Game Date', 'Book', 'Best Prob. Diff', 'Away', 'Away Ml', 'Away Prob. Diff.',
         'Home', 'Home Ml', 'Home Prob. Diff.']
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in cols])] +

        # Body
        [html.Tr([
            html.Td(df.iloc[i][col]) for col in df.columns
        ]) for i in range(min(len(df), len(df)))]
    )

def gen_text(row, z, draws=False):
    if z:
        if row.home_ml_z <= Z_CUTOFF:
            return 'Bet on ' + row.home_team + ' on ' + str(row.game_date) + ' at ' + str(row.home_ml) + ' odds at ' + str(row.book)
        if row.away_ml_z <= Z_CUTOFF:
            return 'Bet on ' + row.away_team + ' on ' + str(row.game_date) + ' at ' + str(row.away_ml) + ' odds at ' + str(row.book)
    else:
        if row.home_ml_prob <= (row.home_ml_consensus - P_CUTOFF):
            return 'Bet on ' + row.home_team + ' on ' + str(row.game_date) + ' at ' + str(row.home_ml) + ' odds at ' + str(row.book)
        if row.away_ml_prob <= (row.away_ml_consensus - P_CUTOFF):
            return 'Bet on ' + row.away_team + ' on ' + str(row.game_date) + ' at ' + str(row.away_ml) + ' odds at ' + str(row.book)
        if draws:
            if row.draw_ml_prob <= (row.draw_ml_consensus - P_CUTOFF):
                return 'Bet on ' + row.away_team + ' & ' + row.home_team + ' to draw on ' + str(row.game_date) + ' at ' + str(
                    row.draw_ml) + ' odds at ' + str(row.book)

def line_to_prob(line):
    if line < 0:
        return -line / (-line + 100.)
        # Do some stuff
    elif line > 0:
        return 100. / (line + 100.)
        # Do some other stuff


def prob_to_line(prob):
    if prob > .5:
        return prob / (1 - prob) * (-100)
    else:
        return ((1 - prob) / prob) * 100


def get_data_action_network(league):
    url = 'https://api-prod.sprtactn.co/web/v1/scoreboard/{}?bookIds='.format(league)
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
            id_to_team[league][team['id']] = team['display_name']
            team_to_id[league][team['display_name']] = team['id']

        home_team = id_to_team[league][game['home_team_id']]
        away_team = id_to_team[league][game['away_team_id']]
        try:
            odds = game['odds']
        except KeyError:
            continue
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
    df['home_team'] = df['home_id'].apply(lambda x: id_to_team[league][x])
    df['away_team'] = df['away_id'].apply(lambda x: id_to_team[league][x])
    df['home_ml_consensus'] = np.nan
    df['away_ml_consensus'] = np.nan
    df['home_ml_std'] = np.nan
    df['away_ml_std'] = np.nan
    df['line_dt'] = pd.to_datetime(df['line_dt'])
    df['game_date'] = pd.to_datetime(df['game_date']) - datetime.timedelta(seconds=5 * 60 * 60)
    df['Pretty_Line_Time'] = df['line_dt'].apply(
        lambda x: (x - datetime.timedelta(seconds=5 * 60 * 60)).strftime('%d-%m-%y %I:%M %p'))
    df['Pretty_Game_Time'] = df['game_date'].apply(lambda x: x.strftime('%d-%m-%y %I:%M %p'))
    df = df[df['game_date'] >= datetime.datetime.now()]
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

    # df['home_ml_prob'] = df['home_ml_prob']
    # df['away_ml_prob'] = df['away_ml_prob']
    #
    # df['home_ml_consensus'] = df['home_ml_consensus']
    # df['away_ml_consensus'] = df['away_ml_consensus']
    #
    # df['home_ml_std'] = df['home_ml_std']
    # df['away_ml_std'] = df['away_ml_std']

    df['home_ml_z'] = ((df['home_ml_prob'] - df['home_ml_consensus']) / df['home_ml_std']).round(4)
    df['away_ml_z'] = ((df['away_ml_prob'] - df['away_ml_consensus']) / df['away_ml_std']).round(4)

    df['home_p_diff'] = (df['home_ml_prob'] - df['home_ml_consensus']).round(4)
    df['away_p_diff'] = (df['away_ml_prob'] - df['away_ml_consensus']).round(4)

    df['best_p_diff'] = df[['home_p_diff', 'away_p_diff']].min(axis=1)
    df = df.sort_values('best_p_diff')
    return df


def insert_game(home_team, away_team, dt, league):
    c = db.cursor()
    try:
        c.execute("""INSERT INTO betting.games(home_team, away_team, game_datetime, league) VALUES (%s, %s, %s, %s)""", (home_team, away_team, dt, league))
        c.commit()
    except MySQLdb.IntegrityError as e:
        print("Game {} @ {}: {} already exists in db".format(home_team, away_team, dt))
    q = """SELECT game_id FROM betting.games WHERE home_team = %s and away_team = %s and game_datetime = %s""" % (home_team, away_team, dt)
    c.execute(q)
    game_id = c.fetchone()
    print(game_id)
    return game_id

def insert_line(values):
    c = db.cursor()
    try:
        q = """INSERT INTO betting.lines(bookie, home_prob, away_prob, line_datetime, game_id) VALUES (%s, %f, %f, %s, %d)"""
        c.executemany(q, values)
        c.commit()
    except MySQLdb.IntegrityError:
        print("Line aleady in db")


def prune_lines(df):
    idx = []
    c = db.cursor()
    q = """SELECT (home_prob, away_prob) FROM betting.lines where bookie = %s and game_id = %d order by line_datetime desc limit 1"""
    for i in range(len(df)):
        row = df.iloc[i, :]
        c.execute(q, (row.book, row.game_id))
        result = c.fetchone()
        if row.home_ml_prob == result[0] and row.away_ml_prob == result[1]:
            continue
        idx.append(i)
    return idx

@app.callback(Output('live-update-table', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_table_selenium(n_intervals):
    leagues = ['nba', 'nhl', 'ncaab']#, 'prem', 'la liga']

    divs = []
    for league in leagues:
        if league == 'ncaab':
            n_rows = 75
        else:
            n_rows = 15
        divs.append(html.H1(league))
        for row in range(n_rows):  # Each row represents one game
            try:
                away_team, home_team, df, game_date = spider.get_table_row(row, league)
                if away_team is None:
                    continue
            except IndexError:  # Row without game information
                continue
            game_id = insert_game(home_team, away_team, game_date, league)
            df = df.dropna()
            df['game_id'] = game_id
            df['line_datetime'] = datetime.datetime.now()
            values = df.loc[:, ['book', 'home_ml_prob', 'away_ml_prob', 'line_datetime', 'game_id']]
            idx = prune_lines(values)
            values = values.iloc[idx, :].values
            insert_line(values)

            if league == 'nhl':
                df = df[df.book != 'Heritage']
            if len(df) == 0:
                continue
            home_z_cutoff = df.home_ml_consensus.values[0] - P_CUTOFF
            away_z_cutoff = df.away_ml_consensus.values[0] - P_CUTOFF
            target_home_line = np.round(prob_to_line(home_z_cutoff), 0)
            target_away_line = np.round(prob_to_line(away_z_cutoff), 0)
            home_team = df.home_team.values[0]
            away_team = df.away_team.values[0]
            p_grp = df.loc[(df.home_ml_prob <= (df.home_ml_consensus - P_CUTOFF)) |
                            (df.away_ml_prob <= (df.away_ml_consensus - P_CUTOFF)), :]
            print(df)

            # if len(p_grp) > 0:
            #     for i in range(len(p_grp)):
            #         row = p_grp.iloc[i, :]
            #         if not gen_text(row, False) in messages_sent and row.book in white_list:
            #             try:
            #                 message = client.messages.create(to='+16179356853',
            #                                                  from_='+17814606736',
            #                                                  body=gen_text(row, False))
            #                 messages_sent.add(gen_text(row, False))
            #             except:
            #                 pass
            if league == 'prem' or league == 'la liga':
                c = ['game_date', 'book', 'best_p_diff',
                     'away_team', 'away_ml', 'away_p_diff',
                     'home_team', 'home_ml', 'home_p_diff',
                     'draw_ml', 'draw_p_diff']
            else:
                c = ['game_date', 'book', 'best_p_diff',
                     'away_team', 'away_ml', 'away_p_diff',
                     'home_team', 'home_ml', 'home_p_diff']

            arb = False

            if league == 'prem' or league == 'la liga':
                if (df.home_ml_prob.min() + df.away_ml_prob.min() + df.draw_ml_prob.min()) < 1.:
                    arb = True
                draws = True
            else:
                draws = False
                if (df.home_ml_prob.min() + df.away_ml_prob.min()) < 1.:
                    arb = True

            if arb:
                _d = [html.H2('{} (Target Line = {}) @ {} (Target Line = {})'.format(away_team,
                                                                               target_away_line,
                                                                               home_team,
                                                                               target_home_line)),
                html.H4('All Current Lines ({} available)'.format(len(df))),
                gen_table(df.loc[:, c].head(10), draws=draws),
                    html.H4("ARB. OPPORTUNITY"), gen_table(df.sort_values('home_ml_prob').head()[c], draws=draws),
                      gen_table(df.sort_values('away_ml_prob').head()[c], draws=draws)]
            else:
                _d = [html.H2('{} (Target Line = {}) @ {} (Target Line = {})'.format(away_team,
                                                                               target_away_line,
                                                                               home_team,
                                                                               target_home_line)),
                html.H4('All Current Lines ({} available)'.format(len(df))),
                gen_table(df.loc[:, c].head(10), draws=draws)]
            div = html.Div(_d)
            divs.append(div)
    return html.Div(divs)



if __name__ == '__main__':
    app.run_server(debug=True, port=8050, host='192.168.1.194')