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

db = MySQLdb.connect(host="127.0.0.1", port=3306, user="root", passwd="", db="betting")


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


def collect_lines(spider):
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
                insert_line(values)
                print("Successfully inserted {} out of {} potential new lines".format(len(values), _len))
        time.sleep(120)

if __name__ == '__main__':
    spider = SeleniumSpider()
    collect_lines(spider)
    spider.close_driver()
    db.close()