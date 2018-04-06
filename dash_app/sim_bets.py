import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc3 as pm

from dash_app.line_scraper_functions import abbrev_to_name, nba_mapping, calc_concensus, process_lines
from dash_app.line_scraper_functions import process_games


def calc_payout(amount, line):
    if line > 0:
        return amount * (line / 100.)
    elif line < 0:
        return amount / (np.abs(line) / 100.)


def simulate_games(concensus_lines, games, team_mapping, min_lines, tol=.05, bankroll=10000., risk=.01, use_z=True):
    outcomes = []
    open_bankroll = bankroll
    open_outcomes = []
    implied_probabilities = []
    open_implied_probabilities = []
    bankroll_over_time = [bankroll]
    open_bankroll_over_time = [bankroll]
    for key, grp in concensus_lines.groupby('game_uuid'):
        grp = grp[grp.n_lines_available >= min_lines]
        bet_size = min(bankroll * risk, 1000)
        # bet_size=100.
        if use_z:
            line_taken = grp.loc[(grp.ml_fav_z <= tol) | (grp.ml_dog_z <= tol)]
        else:
            open_line_taken = grp.loc[(grp.ml_fav_prob <= (grp.fav_open - tol)) |
                                 (grp.ml_dog_prob <= (grp.dog_open - tol))]
            line_taken = grp.loc[(grp.ml_fav_prob <= (grp.ml_fav_prob - tol)) | (grp.ml_dog_prob <= (grp.ml_dog_prob - tol))]
        if len(line_taken) == 0 and len(open_line_taken) == 0:
            continue
        if len(line_taken) > 0:
            line_taken = line_taken.iloc[0, :]
            if (use_z and line_taken.ml_dog_z <= tol) or (
                        not use_z and line_taken.ml_dog_prob <= (line_taken.ml_dog_prob - tol)):
                team = abbrev_to_name(line_taken.dog, team_mapping)
                line = line_taken.ml_dog_line
                prob = line_taken.ml_dog_prob
            elif (use_z and line_taken.ml_fav_z <= tol) or (
                        not use_z and line_taken.ml_fav_prob <= (line_taken.ml_fav_prob - tol)):
                team = abbrev_to_name(line_taken.fav, team_mapping)
                line = line_taken.ml_fav_line
                prob = line_taken.ml_fav_prob
            game = games.loc[games.uuid == key, :]
            won = ((game.home_team.values[0] == team and game.home_team_win.values[0] == 1) or
                   (game.away_team.values[0] == team and game.home_team_win.values[0] == 0))
            implied_probabilities.append(prob)

            if won:
                winnings = calc_payout(bet_size, line)
                outcomes.append(winnings)
                bankroll += winnings
                bankroll_over_time.append(bankroll)
            else:
                outcomes.append(-bet_size)
                bankroll -= bet_size
                bankroll_over_time.append(bankroll)
        if len(open_line_taken) > 0:
            open_line_taken = open_line_taken.iloc[0, :]
            if (use_z and line_taken.ml_dog_z <= tol) or (
                        not use_z and open_line_taken.ml_dog_prob <= (open_line_taken.dog_open - tol)):
                team = abbrev_to_name(open_line_taken.dog, team_mapping)
                line = open_line_taken.ml_dog_line
                prob = open_line_taken.ml_dog_prob
            elif (use_z and open_line_taken.ml_fav_z <= tol) or (
                        not use_z and open_line_taken.ml_fav_prob <= (open_line_taken.fav_open - tol)):
                team = abbrev_to_name(open_line_taken.fav, team_mapping)
                line = open_line_taken.ml_fav_line
                prob = open_line_taken.ml_fav_prob
            game = games.loc[games.uuid == key, :]
            won = ((game.home_team.values[0] == team and game.home_team_win.values[0] == 1) or
                   (game.away_team.values[0] == team and game.home_team_win.values[0] == 0))

            open_implied_probabilities.append(prob)
            if won:
                winnings = calc_payout(bet_size, line)
                open_outcomes.append(winnings)
                open_bankroll += winnings
                open_bankroll_over_time.append(bankroll)
            else:
                open_outcomes.append(-bet_size)
                open_bankroll -= bet_size
                open_bankroll_over_time.append(bankroll)
    return outcomes, implied_probabilities, bankroll_over_time, open_outcomes, open_implied_probabilities, open_bankroll_over_time

if __name__ == '__main__':
    nba_games = process_games(pd.read_pickle('../nba_games.pkl'))
    nba_games = nba_games.replace('blazers', 'trail-blazers')
    nba_games['home_team_win'] = np.where(nba_games.home_points.astype(float) > nba_games.away_points.astype(float), 1, 0)
    lines = pd.read_pickle('new_nba_c_lines.pkl')
    lines['dog_open'] = np.nan
    lines['fav_open'] = np.nan
    for key, grp in lines.groupby('game_uuid'):
        try:
            dog_open = grp.loc[grp.bookie == 'BOOKMAKER LINE MOVEMENTS', 'ml_dog_prob'].values[0]
        except IndexError:
            dog_open = np.nan
        try:
            fav_open = grp.loc[grp.bookie == 'BOOKMAKER LINE MOVEMENTS', 'ml_fav_prob'].values[0]
        except IndexError:
            fav_open = np.nan
        # dog_open = grp.drop_duplicates('bookie', keep='first')['ml_dog_prob'].mean()
        # fav_open = grp.drop_duplicates('bookie', keep='first')['ml_fav_prob'].mean()
        lines.loc[lines.game_uuid == key, 'dog_open'] = dog_open
        lines.loc[lines.game_uuid == key, 'fav_open'] = fav_open

    lines = lines.dropna()

    print(lines.head())
    # lines = process_lines(lines)
    # lines = calc_concensus(lines)
    # pd.to_pickle(lines, 'new_nba_c_lines.pkl')

    # z_tols = [-3.5, -3, -2.5, -2, -.5]
    z_tols = []
    p_tols = [.0225, .0255, .026, .0275]
    min_lines = [12]

    for ml in min_lines:
        print("Sim with min lines = {}".format(ml))
        for z in z_tols:
            outcomes, probs, br, open_out, open_prob, open_br = simulate_games(lines, nba_games, nba_mapping, min_lines=ml, tol=z, risk=.05, use_z=True)
            fig, ax = plt.subplots(figsize=(17, 8))
            ax.plot(range(len(br)), br)
            ax.plot(range(len(open_br)), open_br)
            plt.legend(['Normal', 'Open'])
            plt.savefig('z{}_{}lines_returns_vs_open.png'.format(z, ml))
            avg_return = np.mean(outcomes)
            avg_open_return = np.mean(open_out)
            percent_bet = float(len(outcomes)) / float(len(lines.game_uuid.unique()))
            open_percent_bet = float(len(open_out)) / float(len(lines.game_uuid.unique()))
            accuracy = np.mean(np.array(outcomes) > 0)
            open_accuracy = np.mean(np.array(open_out) > 0)
            # with pm.Model() as model:
            #     mu = pm.Uniform('mu', -1000, 1000)
            #     sd = pm.HalfCauchy('sd', 3)
            #     returns = pm.Normal('returns', mu=mu, sd=sd, observed=np.array(outcomes))
            #     trace = pm.sample(10000, tune=5000, init='advi')
            # prob_of_profit = np.mean(trace['mu'] > 0)
            print("Sim with Z = {}".format(z))
            print("Percent of games bet: {}".format(percent_bet))
            print("Percent of games bet w/ open: {}".format(open_percent_bet))
            print("Average Return: {}".format(avg_return))
            print("Average Return w/ open: {}".format(avg_open_return))
            print('Accuracy w/ open: {}'.format(open_accuracy))
            # print("Probability of Profitability: {}".format(prob_of_profit))

        for p in p_tols:
            outcomes, probs, br, open_out, open_prob, open_br = simulate_games(lines, nba_games, nba_mapping,
                                                                               min_lines=ml, tol=p, risk=.05,
                                                                               use_z=False)
            fig, ax = plt.subplots(figsize=(17, 8))
            ax.plot(range(len(br)), br)
            ax.plot(range(len(open_br)), open_br)
            plt.legend(['Normal', 'Open'])
            plt.savefig('p{}_{}lines_returns_vs_open.png'.format(p, ml))
            avg_return = np.mean(outcomes)
            avg_open_return = np.mean(open_out)
            percent_bet = float(len(outcomes)) / float(len(lines.game_uuid.unique()))
            open_percent_bet = float(len(open_out)) / float(len(lines.game_uuid.unique()))
            accuracy = np.mean(np.array(outcomes) > 0)
            open_accuracy = np.mean(np.array(open_out) > 0)
            # with pm.Model() as model:
            #     mu = pm.Uniform('mu', -1000, 1000)
            #     sd = pm.HalfCauchy('sd', 3)
            #     returns = pm.Normal('returns', mu=mu, sd=sd, observed=np.array(outcomes))
            #     trace = pm.sample(10000, tune=5000, init='advi')
            # prob_of_profit = np.mean(trace['mu'] > 0)
            print("Sim with P = {}".format(p))
            print("Percent of games bet: {}".format(percent_bet))
            print("Percent of games bet w/ open: {}".format(open_percent_bet))
            print("Average Return: {}".format(avg_return))
            print("Average Return w/ open: {}".format(avg_open_return))
            print('Accuracy w/ open: {}'.format(open_accuracy))

        print('\n')


