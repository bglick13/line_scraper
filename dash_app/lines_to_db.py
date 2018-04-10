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
import time
from tqdm import tqdm
# from dash_app.db_config import user, password
from dash_app.line_scraper_functions import extract_games_from_df, process_lines, process_games
from dash_app.selenium_member_test import SeleniumSpider
from dash_app.nba_scraper import get_games
from bs4 import BeautifulSoup
from multiprocessing import Process
from dash_app.data_util import get_live_input
from keras.models import load_model

db = MySQLdb.connect(host="127.0.0.1", port=3306, user="root", passwd="", db="betting")
account_sid = 'AC500cdffd41d5ae4f275e2eecc7c8677d'
auth_token = '772f732acf5ff166e06e2b076c2f220b'
client = Client(account_sid, auth_token)
games_bet = set([])
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


team_maps = dict(
    mlb = dict(
        TB = 'TBR',
        SF = 'SFN',
        SD = 'SDN',
        NYM = 'NYN',
        LAA = 'ANA',
        LAD = 'LAN',
        NYY = 'NYA'

    ),
    nhl = {'TB': 'TBL',
           'LA': 'LAK',
           'SJ': 'SJS',
           },
    nba = {
        'Toronto Raptors': 'TOR',
        'Cleveland Cavaliers': 'CLE',
        'Atlanta Hawks': 'ATL',
        'Brooklyn Nets': 'BRK',
        'Minnesota Timberwolves': 'MIN',
        'Denver Nuggets': 'DEN',
        'Golden State Warriors': 'GSW',
        'Portland Trail Blazers': 'POR',
        'Los Angeles Lakers': 'LAL',
        'Philadelphia 76ers': 'PHI',
        'Miami Heat': 'MIA',
        'New York Knicks': 'NYK',
        'Orlando Magic': 'ORL',
        'Milwaukee Bucks': 'MIL',
        'Oklahoma City Thunder': 'OKC',
        'Boston Celtics': 'BOS',
        'Chicago Bulls': 'CHI',
        'Dallas Mavericks': 'DAL',
        'Los Angeles Clippers': 'LAC',
        'San Antonio Spurs': 'SAS',
        'Indiana Pacers': 'IND',
        'Houston Rockets': 'HOU',
        'Phoenix Suns': 'PHO',
        'Washington Wizards': 'WAS',
        'Charlotte Hornets': 'CHO',
        'Utah Jazz': 'UTA',
        'Sacramento Kings': 'SAC',
        'New Orleans Pelicans': 'NOP',
        'Memphis Grizzlies': 'MEM',
        'Detroit Pistons': 'DET'

    },
    ncaab = {'Villanova (n)': 'Villanova'}

)

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


def insert_game(home_team, away_team, dt, league):
    c = db.cursor()
    try:
        c.execute("""INSERT INTO betting.games(home_team, away_team, game_datetime, league) VALUES (%s, %s, %s, %s)""", (home_team, away_team, dt, league))
        db.commit()
    except MySQLdb.IntegrityError as e:
       pass
    q = """SELECT game_id FROM betting.games WHERE home_team = "%s" and away_team = "%s" and game_datetime = '%s'""" % (home_team, away_team, dt)
    c.execute(q)
    game_id = c.fetchone()[0]
    return game_id

def insert_line(values):
    c = db.cursor()
    values = [list(v) for v in values]
    try:
        q = """INSERT INTO betting.lines(bookie, home_prob, away_prob, line_datetime, game_id_fk) VALUES (%s, %s, %s, %s, %s)"""
        c.executemany(q, values)
        db.commit()
    except MySQLdb.IntegrityError:
       pass


def prune_lines(df):
    idx = []
    c = db.cursor()

    for i in range(len(df)):
        row = df.iloc[i, :]
        q = """SELECT home_prob, away_prob FROM betting.lines where bookie = "%s" and game_id_fk = %d order by line_datetime desc limit 1""" % (row.book, int(row.game_id))
        c.execute(q)
        result = c.fetchone()
        if result is not None:
            if abs(row.home_ml_prob - float(result[0])) > .0001 and abs(row.away_ml_prob - float(result[1])) > .0001:
                idx.append(i)
        else:
            idx.append(i)

    return idx


def _get_result(home_team, away_team, game_datetime, league):
    league = league.lower()
    date_fortmat = game_datetime.strftime("%Y%m%d")
    try:
        if league == 'nba':
            url = 'https://www.basketball-reference.com/boxscores/{}0{}.html'.format(date_fortmat, home_team)
            soup = BeautifulSoup(requests.get(url).text)
            scores = soup.find_all('div', class_="score")
            away = scores[0].get_text()
            home = scores[1].get_text()
        elif league == 'nfl':
            pass
        elif league == 'nhl':
            url = 'https://www.hockey-reference.com/boxscores/{}0{}.html'.format(date_fortmat, home_team)
            soup = BeautifulSoup(requests.get(url).text)
            scores = soup.find_all('div', class_="score")
            if scores is None:
                url = 'https://www.hockey-reference.com/boxscores/{}{}0.html'.format(date_fortmat, home_team)
                soup = BeautifulSoup(requests.get(url).text)
                scores = soup.find_all('div', class_="score")
            away = scores[0].get_text()
            home = scores[1].get_text()
        elif league == 'ncaab':
            pass
        elif league == 'mlb':

            url = 'https://www.baseball-reference.com/boxes/{}/{}{}0.shtml'.format(home_team, home_team, date_fortmat)
            soup = BeautifulSoup(requests.get(url).text)
            scores = soup.find_all('div', class_="score")
            _append = 1
            while scores is None:
                url = 'https://www.baseball-reference.com/boxes/{}/{}{}{}.shtml'.format(home_team, home_team,
                                                                                       date_fortmat, _append)
                soup = BeautifulSoup(requests.get(url).text)
                scores = soup.find_all('div', class_="score")
                _append += 1
            away = scores[0].get_text()
            home = scores[1].get_text()
        if home > away:
            return 1
        else:
            return 0
    except:
        raise Exception(url)



def get_game_results():
    c = db.cursor()
    today = datetime.datetime.today()
    q = """select * from betting.games where outcome is null and game_datetime < '%s' and league != 'ncaab'""" % (today, )
    print(q)
    c.execute(q)
    games = c.fetchall()
    # Home team, Away team, datetime, league, game_id
    success = 0
    failures = 0
    failed_games = []
    for game in tqdm(games):
        try:
            home_team, away_team, league = game[0], game[1], game[3]
            if home_team in team_maps[league].keys():
                home_team = team_maps[league][home_team]
            if away_team in team_maps[league].keys():
                away_team = team_maps[league][away_team]
            result = _get_result(home_team, away_team, game[2], game[3])
            q = """update betting.games set outcome = %s where game_id = %s""" % (result, game[4])
            c.execute(q)
            db.commit()
            success += 1
        except Exception as e:
            failures += 1
            failed_games.append(e)
            print(e)
    print("Successfully updated {} of {} games".format(success, success+failures))
    print("Failed on the following home teams...")
    print(failed_games)


def predict_live(X, away_lines, tol=.025):
    # try:
    #     model = load_model('trained_model.h5')
    # except:
    #     print("Warning: No live model trained yet")
    #     return
    # X, away_lines = get_live_input(game_id, sequence_length)
    # if X.shape != (1, sequence_length + 1, len(bookies)):
    #     print("Invalid data for {} @ {}. Expected (1, {}, {}), got {}".format(away_team, home_team, sequence_length+1, len(bookies), X.shape))
    #     return

    prob = model.predict_proba(X)
    home_p_diffs = prob - X[-1][-1]
    away_p_diffs = (1-prob) - away_lines[-1][-1]
    team = None
    if np.max(home_p_diffs) > tol:
        idx = np.argmax(home_p_diffs)
        p = X[-1][-1][idx]
        true_p = prob - tol #p + home_p_diffs[idx]
        bookie = bookies[idx]
        if bookie in white_list:
            team = home_team
    elif np.max(away_p_diffs) > tol:
        idx = np.argmax(away_p_diffs)
        p = away_lines[-1][-1][idx]
        true_p = (1-prob) - tol #p + away_p_diffs[idx]
        bookie = bookies[idx]
        if bookie in white_list:
            team = away_team
    if team is not None:
        ml = prob_to_ml(p)
        fair_ml = prob_to_ml(true_p)
        s = "Bet on {} ({} @ {}) at odds {} or better at {}".format(team, away_team, home_team, fair_ml, bookie)
        print(s)
        try:
            message = client.messages.create(to='+16179356853',
                                             from_='+17814606736',
                                             body=s)
            games_bet.add(game_id)
        except:
            pass
    else:
        best_home_prob = X[-1][-1][np.argmin(X[-1][-1])]
        best_away_prob = away_lines[-1][-1][np.argmin(away_lines[-1][-1])]
        print("Predicted true prob: {}, best home odds: {}, best away odds: {}".format(prob, best_home_prob, best_away_prob))
        # print("Bet on {} ({} @ {}) at odds {} (implied: {})".format(team, away_team, home_team, ml, fair_ml))


def collect_lines(spider, model, tol=.05):
    leagues = ['nba', 'nhl', 'mlb']
    cutoff = datetime.time(22, 30)
    while datetime.datetime.now().time() < cutoff:
        for league in leagues:
            n_rows = 30
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
                _len = len(values)
                idx = prune_lines(values)
                values = values.iloc[idx, :].values
                if len(values) > 0:
                    insert_line(values)
                    X, away_lines = get_live_input(game_id, 10)
                    if X.shape == (1, 10 + 1, len(bookies)):
                        prob = model.predict_proba(X)[0][0]
                        home_p_diffs = prob - X[-1][-1]
                        print("Home p diffs: {}".format(home_p_diffs))
                        away_p_diffs = (1 - prob) - away_lines[-1][-1]
                        print("Away p diffs: {}".format(away_p_diffs))

                        team = None
                        if np.max(home_p_diffs) > tol:
                            idx = np.argmax(home_p_diffs)
                            p = X[-1][-1][idx]
                            true_p = prob - tol  # p + home_p_diffs[idx]
                            bookie = bookies[idx]
                            if bookie in white_list:
                                team = home_team
                        elif np.max(away_p_diffs) > tol:
                            idx = np.argmax(away_p_diffs)
                            p = away_lines[-1][-1][idx]
                            true_p = (1 - prob) - tol  # p + away_p_diffs[idx]
                            bookie = bookies[idx]
                            if bookie in white_list:
                                team = away_team
                        if team is not None:
                            ml = prob_to_ml(p)
                            fair_ml = prob_to_ml(true_p)
                            s = "Bet on {} ({} @ {}) at odds {} or better at {}".format(team, away_team, home_team,
                                                                                        fair_ml, bookie)
                            print(s)
                            if game_id not in games_bet:
                                try:
                                    message = client.messages.create(to='+16179356853',
                                                                     from_='+17814606736',
                                                                     body=s)
                                    games_bet.add(game_id)
                                except:
                                    pass
                        else:
                            best_home_prob = X[-1][-1][np.argmin(X[-1][-1])]
                            best_away_prob = away_lines[-1][-1][np.argmin(away_lines[-1][-1])]
                            print("Predicted true prob: {}, best home odds: {}, best away odds: {}".format(prob,
                                                                                                           best_home_prob,
                                                                                                           best_away_prob))
                            # print("Bet on {} ({} @ {}) at odds {} (implied: {})".format(team, away_team, home_team, ml, fair_ml))
                    else:
                        print("Invalid data for {} @ {}. Expected (1, {}, {}), got {}".format(away_team, home_team,
                                                                                              10 + 1,
                                                                                              len(bookies), X.shape))
                    # p = Process(target=predict_live, args=(game_id, home_team, away_team))
                    # p.start()
                    # p.join()

                    print("Successfully inserted {} out of {} potential new lines".format(len(values), _len))
        # time.sleep(120)


if __name__ == '__main__':
    spider = SeleniumSpider()
    model = load_model('trained_model.h5')
    collect_lines(spider, model, .025)
    get_game_results()
    spider.close_driver()
    db.close()