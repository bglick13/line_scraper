import MySQLdb
import numpy as np
import pandas as pd
import datetime

bookies = ['5Dimes',
 'Bet365',
 'BetDSI',
 'BetMania',
 'BetOnline',
 'Bookmaker',
 'Bovada',
 'Caesars',
 'Consensus',
 'GTBets',
 'Greek',
 'Heritage',
 'Intertops',
 'JustBet',
 'LooseLines',
 # 'MGM',
 'Nitrogen',
 'Pinnacle',
 'Sportbet',
 'SportsBetting',
 'Sportsbk',
 'Westgate',
 'YouWager']

def get_data():
    """

    :return: DF of every line with an outcome
    """
    db = MySQLdb.connect(host="127.0.0.1", port=3306, user="root", passwd="", db="betting")
    c = db.cursor()
    c.execute('select * from games where outcome is not null')
    games = c.fetchall()
    games = pd.DataFrame(columns=['home_team', 'away_team', 'game_datetime', 'league', 'game_id', 'outcome'],
                         data=np.array(games))
    output = pd.DataFrame(columns=['bookie', 'home_prob', 'away_prob', 'line_datetime', 'game_id'])
    for game_id, game_dt, outcome in games.loc[:, ['game_id', 'game_datetime', 'outcome']].values:
        q = """select * from betting.lines where game_id_fk = %s and bookie != 'MGM' order by line_datetime""" % (game_id,)
        c.execute(q)
        lines = c.fetchall()
        lines = pd.DataFrame(columns=['bookie', 'home_prob', 'away_prob', 'line_datetime', 'game_id'],
                             data=np.array(lines))
        lines['time_to_game'] = game_dt - lines['line_datetime']
        lines['time_to_game'] = lines['time_to_game'].apply(lambda x: x.total_seconds() / 60.)
        lines['outcome'] = outcome
        output = output.append(lines, ignore_index=True)
    db.close()
    output['home_prob'] = pd.to_numeric(output['home_prob'])
    output['away_prob'] = pd.to_numeric(output['away_prob'])
    output['game_id'] = pd.to_numeric(output['game_id'])
    return output


def get_live_input(game_id, sequence_length=10):
    db = MySQLdb.connect(host="127.0.0.1", port=3306, user="root", passwd="", db="betting")
    c = db.cursor()
    q = """select * from betting.lines where game_id_fk = %s and bookie != 'MGM' order by line_datetime""" % (game_id, )
    c.execute(q)
    lines = list(c.fetchall())
    # q = """select * from betting.lines where game_id_fk = %s order by line_datetime limit 1""" % (game_id, )
    # c.execute(q)
    # _open = c.fetchone()
    # lines.append(_open)
    lines = pd.DataFrame(columns=['bookie', 'home_prob', 'away_prob', 'line_datetime', 'game_id'],
                         data=np.array(lines)).sort_values('line_datetime')
    lines['home_prob'] = pd.to_numeric(lines['home_prob'])
    lines['away_prob'] = pd.to_numeric(lines['away_prob'])
    pivot = pd.pivot_table(lines, 'home_prob', 'line_datetime', 'bookie')
    pivot = pivot.fillna(method='ffill').dropna().values
    # pivot = pivot.apply(lambda row: row.fillna(row.mean()), axis=1).values
    pivot = np.vstack((pivot[0], pivot[-sequence_length:]))
    away_pivot = pd.pivot_table(lines, 'away_prob', 'line_datetime', 'bookie')
    away_pivot = away_pivot.fillna(method='ffill').dropna().values
    # away_pivot = away_pivot.apply(lambda row: row.fillna(row.mean()), axis=1).values
    away_pivot = np.vstack((away_pivot[0], away_pivot[-sequence_length:]))

    db.close()
    return np.array([pivot]), np.array([away_pivot])

def build_flat_dataset(data=None):
    if data is None:
        data = get_data()
    X = None
    y = None
    for key, grp in data.groupby('game_id'):
        pivot = pd.pivot_table(grp, 'home_prob', 'line_datetime', 'bookie')
        pivot = pivot.fillna(method='ffill').dropna()
        if len(pivot.columns) != len(bookies):
            continue
        if X is None:
            X = pivot
            y = [[grp['outcome'].values[0]] * len(pivot)]
        else:
            X = np.vstack((X, pivot.values))
            y = np.append(y, [grp['outcome'].values[0]] * len(pivot))
    return X, y


def build_sequence_dataset(data=None, sequence_length=10, window=2, home=True):
    if home:
        odds = 'home_prob'
    else:
        odds = 'away_prob'
    if data is None:
        data = get_data()
    X = []
    y = []
    games = []
    for key, grp in data.groupby('game_id'):
        pivot = pd.pivot_table(grp, odds, 'line_datetime', 'bookie')
        pivot = pivot.fillna(method='ffill').dropna().values  # Fill N/A since some bookies (MGM) post lines very late
        # pivot = pivot.apply(lambda row: row.fillna(row.mean()), axis=1).values
        outcome = grp['outcome'].values[0]
        for i in range(sequence_length, len(pivot), 1):
            _arr = np.vstack((pivot[0], pivot[i-sequence_length:i]))
            if _arr.shape != (sequence_length+1, len(bookies)):
                continue
            X.append(_arr)
            y.append(outcome)
            games.append(key)
    return np.array(X), np.array(y), np.array(games)

