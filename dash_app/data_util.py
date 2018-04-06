import MySQLdb
import numpy as np
import pandas as pd
import datetime


def get_data():
    db = MySQLdb.connect(host="127.0.0.1", port=3306, user="root", passwd="", db="betting")
    c = db.cursor()
    c.execute('select * from games where outcome is not null')
    games = c.fetchall()
    games = pd.DataFrame(columns=['home_team', 'away_team', 'game_datetime', 'league', 'game_id', 'outcome'],
                         data=np.array(games))
    output = pd.DataFrame(columns=['bookie', 'home_prob', 'away_prob', 'line_datetime', 'game_id'])
    for game_id, game_dt, outcome in games.loc[:, ['game_id', 'game_datetime', 'outcome']].values:
        q = """select * from betting.lines where game_id_fk = %s""" % (game_id,)
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


def build_flat_dataset(data=None):
    if data is None:
        data = get_data()
    X = None
    y = None
    for key, grp in data.groupby('game_id'):
        pivot = pd.pivot_table(grp, 'home_prob', 'line_datetime', 'bookie')
        pivot = pivot.fillna(method='ffill').dropna()
        if len(pivot.columns) != 23:
            continue
        if X is None:
            X = pivot
            y = [[grp['outcome'].values[0]] * len(pivot)]
        else:
            X = np.vstack((X, pivot.values))
            y = np.append(y, [grp['outcome'].values[0]] * len(pivot))
    return X, y


def build_sequence_dataset(data=None, sequence_length=10, window=2):
    if data is None:
        data = get_data()
    X = []
    y = []
    games = []
    for key, grp in data.groupby('game_id'):
        pivot = pd.pivot_table(grp, 'home_prob', 'line_datetime', 'bookie')
        pivot = pivot.fillna(method='ffill').dropna().values
        outcome = grp['outcome'].values[0]
        for i in range(sequence_length, len(pivot), 1):
            _arr = np.vstack((pivot[0], pivot[i-sequence_length:i]))
            if _arr.shape != (sequence_length+1, 23):
                continue
            X.append(_arr)
            y.append(outcome)
            games.append(key)
    return np.array(X), np.array(y), np.array(games)

