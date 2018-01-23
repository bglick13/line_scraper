import pandas as pd
import requests
from bs4 import BeautifulSoup
import uuid
import datetime


def load_month(year, month):
    """
    Loads the html page for the overview of all games in a week
    :param year:
    :param month:
    :return:
    """
    url = 'https://www.basketball-reference.com/leagues/NBA_{}_games-{}.html'.format(year, month)
    page = requests.get(url).text
    return page


def extract_game_info(game, scores=True):
    """
    Get the relevant information for a single game from the html elements
    :param game:
    :param scores:
    :return:
    """
    cells = game.find_all('td')
    date = game.find('th').get_text()

    time = cells[0].get_text()
    dt = datetime.datetime.strptime(date + ' ' + time, '%a, %b %d, %Y %I:%M %p')
    away_team = cells[1].get_text()
    away_points = cells[2].get_text()
    home_team = cells[3].get_text()
    home_points = cells[4].get_text()

    return dt, away_team, away_points, home_team, home_points


def get_games(years: list = None, months: list = None):
    """
    Wrapper function to get data on all games in a set of years and weeks
    :param years:
    :param weeks:
    :return:
    """
    output = dict(date=[], away_team=[], away_points=[], home_team=[], home_points=[], season=[], uuid=[])
    if years is None:
        years = range(2010, 2018)
    if months is None:
        months = ['october', 'november', 'december', 'january', 'february', 'march', 'april', 'may']

    for year in years:
        for month in months:
            page = load_month(year, month)
            soup = BeautifulSoup(page)
            try:
                games = soup.find('table', id='schedule').find('tbody').find_all('tr')
                for game in games:
                    try:
                        _uuid = uuid.uuid1()
                        date, away_team, away_points, home_team, home_points = extract_game_info(game)
                        output['date'].append(date)
                        output['away_team'].append(away_team)
                        output['away_points'].append(away_points)
                        output['home_team'].append(home_team)
                        output['home_points'].append(home_points)
                        output['season'].append(year)
                        output['uuid'].append(_uuid)
                    except IndexError:
                        pass
            except AttributeError:
                pass

    df = pd.DataFrame.from_dict(output, orient='columns')
    return df