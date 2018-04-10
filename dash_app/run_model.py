from dash_app.data_util import build_flat_dataset, build_sequence_dataset
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import numpy as np
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error as mse
from keras.optimizers import RMSprop, Adam
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm, tqdm_notebook
import itertools

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
white_list = set(['BetOnline, Bookmaker', 'Bovada', 'Heritage', 'JustBet', 'Nitrogen', 'Sportbet',
                  'SportsBetting', 'YouWager'])


def calc_payout(amount, line):
    if line > 0:
        return amount * (line / 100.)
    elif line < 0:
        return amount / (np.abs(line) / 100.)

def prob_to_ml(p):
    p *= 100
    if p >= 50:
        return -(p/(100.-p))*100.
    else:
        return ((100-p)/p)*100


def dict_product(dicts):
    """
    >>> list(dict_product(dict(number=[1,2], character='ab')))
    [{'character': 'a', 'number': 1},
     {'character': 'a', 'number': 2},
     {'character': 'b', 'number': 1},
     {'character': 'b', 'number': 2}]
    """
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


class Backtest:

    def __init__(self, sequence_length=10, n_hidden_layers=1, lstm_units=128, tol=.03, lr=.001, epochs=25, **kwargs):
        self.sequence_length = sequence_length
        self.n_hidden_layers = n_hidden_layers
        self.lstm_units = lstm_units
        self.lr = lr
        self.tol = tol
        self.input_shape = (sequence_length+1, len(bookies))
        self.model = build_model(input_shape=self.input_shape, n_hidden_layers=self.n_hidden_layers,
                                 lstm_units=self.lstm_units, lr=self.lr)
        self.home_X, self.home_y, self.games = build_sequence_dataset(sequence_length=self.sequence_length)
        self.away_X, self.away_y, _ = build_sequence_dataset(sequence_length=self.sequence_length, home=False)
        self.games_bet = []
        self.balances = []
        self.probs = []
        self.scores = []
        self.epochs = epochs

    def fit(self, X, y, forget=False):
        if forget:
            self.model = build_model(input_shape=self.input_shape, n_hidden_layers=self.n_hidden_layers,
                                 lstm_units=self.lstm_units, lr=self.lr)
        _fit = self.model.fit(X, y, batch_size=128, epochs=self.epochs, verbose=False)

    def predict(self, X):
        return self.model.predict_proba(X, batch_size=128)

    def score(self, home_X, away_X, y, games):
        bankroll = 1000.
        balance = [1000]
        probs = []
        games_bet = set([])
        pred = self.predict(home_X)
        for p, home_x, away_x, y, g in zip(pred, home_X, away_X, y, games):
            if g not in games_bet:
                home_p_diffs = p - home_x[-1]
                away_p_diffs = (1 - p) - away_x[-1]
                if np.max(home_p_diffs) >= self.tol:
                    bookie = bookies[np.argmax(home_p_diffs)]
                    if bookie not in white_list:
                        continue

                    odds = home_x[-1][np.argmax(home_p_diffs)]
                    ml = prob_to_ml(odds)
                    if y == 1:
                        bankroll += calc_payout(50, ml)
                    else:
                        bankroll -= 50

                    # print("Betting on home team in game {} at odds {}({}) - result: {}".format(g, odds, ml, y))
                    games_bet |= set([g])
                    balance.append(bankroll)
                    probs.append(odds)
                elif np.max(away_p_diffs) >= self.tol:
                    bookie = bookies[np.argmax(away_p_diffs)]
                    if bookie not in white_list:
                        continue

                    odds = away_x[-1][np.argmax(away_p_diffs)]
                    ml = prob_to_ml(odds)
                    if y == 0:
                        bankroll += calc_payout(50, ml)
                    else:
                        bankroll -= 50

                    # print("Betting on away team in game {} at odds {}({}) - result: {}".format(g, odds, ml, y))
                    games_bet |= set([g])
                    balance.append(bankroll)
                    probs.append(odds)
        self.games_bet.append(games_bet)
        self.probs.append(probs)
        self.balances.append(balance)
        self.scores.append(balance[-1])
        return balance[-1]


def build_model(input_shape=(10,len(bookies)), n_hidden_layers=1, lstm_units=128, lr=.001):
    optimizer = Adam(lr=lr)
    model = Sequential()
    model.add(LSTM(units=lstm_units, activation='sigmoid', recurrent_activation='hard_sigmoid',
                   input_shape=input_shape, return_sequences=True))
    # model.add(Dropout(0.5))
    for l in range(n_hidden_layers):
        if l == n_hidden_layers-1:
            rs = False
        else:
            rs = True
        model.add(LSTM(units=lstm_units, activation='sigmoid', recurrent_activation='hard_sigmoid', return_sequences=rs))
        # model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer)
    return model


def run_gridsearch(**params):
    list_of_params = dict_product(params)
    results = dict(bt=[], min=[], max=[], mean=[], sd=[], params=[])
    for l in tqdm(list_of_params):
        bt = Backtest(**l)
        kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        for train_idx, test_idx in tqdm(kf.split(bt.home_X, bt.home_y)):
            train_X, train_y = bt.home_X[train_idx], bt.home_y[train_idx]
            test_home_X, test_y, test_away_X, games = bt.home_X[test_idx], bt.home_y[test_idx], bt.away_X[test_idx], bt.games[test_idx]
            bt.fit(train_X, train_y, forget=True)
            score = bt.score(test_home_X, test_away_X, test_y, games)
        results['bt'].append(bt)
        results['min'].append(np.min(bt.scores))
        results['max'].append(np.max(bt.scores))
        results['mean'].append(np.mean(bt.scores))
        results['sd'].append(np.std(bt.scores))
        results['params'].append(l)
    return results


if __name__ == "__main__":
    params = {
        'tol': [.025, .03, .07],
        'n_hidden_layers': [1, 2]
    }
    # for p in dict_product(params):
    #     print(p)
    results = run_gridsearch(**params)
    print(results)
    # input_shape = (20, 23)
    # model = build_model(input_shape)
    # X, y = build_sequence_dataset(sequence_length=input_shape[0], window=20)
    # base_x, base_y = build_flat_dataset()
    # print(X.shape)
    # model.fit(X[:50], y[:50], batch_size=128, epochs=20)
    # pred = model.predict(X[50:])
    # print(pred)
    # true = y[50:]
    # print(true)
    #
    # print("MSE: {}".format(mse(true, pred)))
    # print("Baseline MSE: {}".format(mse(base_y, base_x.mean(axis=1))))

