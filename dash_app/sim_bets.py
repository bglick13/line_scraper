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
    implied_probabilities = []
    bankroll_over_time = [bankroll]
    for key, grp in concensus_lines.groupby('game_uuid'):
        grp = grp[grp.n_lines_available >= min_lines]
        bet_size = min(bankroll * risk, 1000)
        # bet_size=100.
        if use_z:
            line_taken = grp.loc[(grp.ml_fav_z <= tol) | (grp.ml_dog_z <= tol)]
        else:
            line_taken = grp.loc[(grp.ml_fav_prob <= (grp.concensus_ml_fav - tol)) |
                                 (grp.ml_dog_prob <= (grp.concensus_ml_dog - tol))]
        if len(line_taken) == 0:
            continue
        else:
            line_taken = line_taken.iloc[0, :]

        if (use_z and line_taken.ml_dog_z <= tol) or (
            not use_z and line_taken.ml_dog_prob <= (line_taken.concensus_ml_dog - tol)):
            team = abbrev_to_name(line_taken.dog, team_mapping)
            line = line_taken.ml_dog_line
            prob = line_taken.ml_dog_prob
        elif (use_z and line_taken.ml_fav_z <= tol) or (
            not use_z and line_taken.ml_fav_prob <= (line_taken.concensus_ml_fav - tol)):
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
    return outcomes, implied_probabilities, bankroll_over_time

if __name__ == '__main__':
    nba_games = process_games(pd.read_pickle('../nba_games.pkl'))
    nba_games = nba_games.replace('blazers', 'trail-blazers')
    nba_games['home_team_win'] = np.where(nba_games.home_points.astype(float) > nba_games.away_points.astype(float), 1, 0)
    lines = pd.read_pickle('new_nba_c_lines.pkl')
    # lines = process_lines(lines)
    # lines = calc_concensus(lines)
    # pd.to_pickle(lines, 'new_nba_c_lines.pkl')

    # z_tols = [-3.5, -3, -2.5, -2, -.5]
    z_tols = []
    p_tols = [.0255, .026, .0275]
    min_lines = [12]

    for ml in min_lines:
        print("Sim with min lines = {}".format(ml))
        for z in z_tols:
            outcomes, probs, br = simulate_games(lines, nba_games, nba_mapping, min_lines=ml, tol=z, risk=.05, use_z=True)
            fig, ax = plt.subplots(figsize=(17, 8))
            ax.plot(range(len(br)), br)
            plt.savefig('z{}_{}lines_returns.png'.format(z, ml))
            avg_return = np.mean(outcomes)
            percent_bet = float(len(outcomes)) / float(len(lines.game_uuid.unique()))
            accuracy = np.mean(np.array(outcomes) > 0)
            with pm.Model() as model:
                mu = pm.Uniform('mu', -1000, 1000)
                sd = pm.HalfCauchy('sd', 3)
                returns = pm.Normal('returns', mu=mu, sd=sd, observed=np.array(outcomes))
                trace = pm.sample(10000, tune=5000, init='advi')
            prob_of_profit = np.mean(trace['mu'] > 0)
            print("Sim with Z = {}".format(z))
            print("Percent of games bet: {}".format(percent_bet))
            print("Average Return: {}".format(avg_return))
            print('Accuracy: {}'.format(accuracy))
            print("Probability of Profitability: {}".format(prob_of_profit))

        for p in p_tols:
            outcomes, probs, br = simulate_games(lines, nba_games, nba_mapping, min_lines=ml, tol=p, risk=.05, use_z=False)
            fig, ax = plt.subplots(figsize=(17, 8))
            ax.plot(range(len(br)), br)
            plt.savefig('p{}_{}lines_returns.png'.format(p, ml))
            avg_return = np.mean(outcomes)
            percent_bet = float(len(outcomes)) / float(len(lines.game_uuid.unique()))
            accuracy = np.mean(np.array(outcomes) > 0)
            with pm.Model() as model:
                mu = pm.Uniform('mu', -1000, 1000)
                sd = pm.HalfCauchy('sd', 3)
                returns = pm.Normal('returns', mu=mu, sd=sd, observed=np.array(outcomes))
                trace = pm.sample(10000, tune=5000, init='advi')
            prob_of_profit = np.mean(trace['mu'] > 0)
            print("Sim with P tolerance = {}".format(p))
            print("Percent of games bet: {}".format(percent_bet))
            print("Average Return: {}".format(avg_return))
            print('Accuracy: {}'.format(accuracy))
            print('Average implied Prob: {}'.format(np.mean(probs)))
            print("Probability of Profitability: {}".format(prob_of_profit))

        print('\n')


