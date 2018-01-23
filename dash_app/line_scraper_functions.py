import pandas as pd
import requests
from bs4 import BeautifulSoup
import numpy as np
import datetime


header_class = 'component_head'
table_class  = 'rt_railbox_border2'
div_class = 'SLTables1'

columns = ['date', 'dog', 'dog_spread_line', 'fav', 'fav_spread_line', 'ml_dog_line', 'ml_fav_line',
           'over', 'spread', 'time', 'total', 'under', 'bookie', 'game_uuid']


nfl_mapping = {
        'PHI': 'eagles',
        'JAC': 'jaguars',
        'NWE': 'patriots',
        'WAS': 'redskins',
        'ATL': 'falcons',
        'DAL': 'cowboys',
        'TEN': 'titans',
        'NYJ': 'jets',
        'NYG': 'giants',
        'CAR': 'panthers',
        'HOU': 'texans',
        'DEN': 'broncos',
        'IND': 'colts',
        'DET': 'lions',
        'KAN': 'chiefs',
        'LAC': 'chargers',
        'SFO': '49ers',
        'DAL': 'cowboys',
        'NOR': 'saints',
        'SEA': 'seahawks',
        'STL': 'rams',
        'BAL': 'ravens',
        'MIN': 'vikings',
        'BUF': 'bills',
        'ARI': 'cardinals',
        'CHI': 'bears',
        'PIT': 'steelers',
        'GNB': 'packers',
        'TAM': 'buccaneers',
        'CIN': 'bengals',
        'OAK': 'raiders',
        'MIA': 'dolphins',
        'CLE': 'browns'
    }
nba_mapping = {
    'WAS': 'wizards',
    'ATL': 'hawks',
    'CHI': 'bulls',
    'CHA': 'bobcats',
    'DAL': 'mavericks',
    'UTH': 'jazz',
    'GSW': 'warriors',
    'IND': 'pacers',
    'MIA': 'heat',
    'CLE': 'cavaliers',
    'HOU': 'rockets',
    'TOR': 'raptors',
    'SAS': 'spurs',
    'SAC': 'kings',
    'MIN': 'timberwolves',
    'BOS': 'celtics',
    'POR': 'trail-blazers',
    'MEM': 'grizzlies',
    'MIL': 'bucks',
    'PHO': 'suns',
    'LAL': 'lakers',
    'DET': 'pistons',
    'LAC': 'clippers',
    'OKL': 'thunder',
    'DEN': 'nuggets',
    'NYK': 'knicks',
    'NOR': 'pelicans',
    'ORL': 'magic',
    'NJN': 'nets',
    'PHI': '76ers'
}


def extract_games_from_df(df, league):
    games = pd.DataFrame(columns=columns)
    successes = 0
    failures = 1
    for i in range(len(df)):
        row = df.iloc[i, :]
        try:
            game = extract_tables_from_game(row.home_team, row.away_team, row.date_, row.uuid, league)
            games = games.append(game, ignore_index=True)
            successes += 1
        except ConnectionError:
            # print('Failed on the following game')
            # print(row)
            failures += 1
        except ValueError:
            # print('Failed on the following game')
            # print(row)
            failures += 1
        except requests.exceptions.ConnectionError:
            failures += 1
        if i % 10 == 0:
            print("Succeeded on {} of {} games so far".format(successes, successes+failures))
    print("Succeeded on {} of {} games".format(successes, successes+failures))
    return games


def extract_tables_from_game(home_team, away_team, date, game_uuid, league):
    url = 'http://www.vegasinsider.com/{}/odds/offshore/line-movement/{}-@-{}.cfm/date/{}'.format(league, away_team, home_team, date)
    print(url)
    page = requests.get(url).text
    soup = BeautifulSoup(page)
    try:
        game_info = soup.find('div', class_=div_class).find_all('table')[1].find_all('tr')
        game_date = game_info[0].find('td').get_text().replace(u'\xa0', u'').split(':')[1]
        game_time = game_info[1].find('td').get_text().replace(u'\xa0', u'').split('e:')[1]
    except:
        print(url)
    game_datetime = datetime.datetime.strptime(game_date + ' ' + game_time, '%A, %B %d, %Y %I:%M %p ')
    tables = soup.find_all('table', class_=table_class)
    headers = soup.find_all('tr', class_=header_class)
    game = pd.DataFrame(columns=columns)
    for t, h in zip(tables, headers):
        table = extract_data_from_table(t, h, game_datetime.year)
        game = game.append(table, ignore_index=True)
    game['game_uuid'] = game_uuid
    game['game_datetime'] = game_datetime
    return game


def extract_data_from_table(t, h, year):
    bookie = h.find('td').get_text().strip()
    table = pd.DataFrame(columns=columns)
    rows = t.find_all('tr')
    for r in rows[2:]:
        try:
            data = r.find_all('td')
            ret  = extract_data_from_cell(data, year)
            table = table.append(ret, ignore_index=True)
        except IndexError:
            pass
    table['bookie'] = bookie
    return table


def extract_data_from_cell(d, year):
    """
    Extract a single line data point from a table of lines from a single bookie for a single game
    """
    date = d[0].get_text().strip()
    time = d[1].get_text().strip()
    dt_str = date + ' ' + str(year) + ' ' + time
    dt = datetime.datetime.strptime(dt_str, '%m/%d %Y %I:%M%p')

    # Getting the moneyline info is easy
    try:
        fav, ml_fav_line = d[2].get_text().strip().split('-')
        ml_fav_line = int(ml_fav_line) * -1
        dog, ml_dog_line = d[3].get_text().strip().split('+')
    except ValueError:  # The bookie may not have ML posted yet
        fav, ml_fav_line = (np.nan, np.nan)
        dog, ml_dog_line = (np.nan, np.nan)

    try:
        # The spread line for the favorite in the game. This allows us to always split on '-' to get the number
        spread, fav_spread_line = d[4].get_text().strip().split(' ')
        _, spread = spread.split('-')
        # Since we already know the spread and the fav/dog teams,
        # all we need is the odds line for betting the dog on the spread
        _, dog_spread_line = d[5].get_text().strip().split(' ')
    except ValueError:
        spread, fav_spread_line, dog_spread_line = (np.nan, np.nan, np.nan)

    # Get point total data here
    try:
        total, over = d[6].get_text().strip().split(' ')
        _, under = d[7].get_text().strip().split(' ')
    except ValueError:
        total, over, under = (np.nan, np.nan, np.nan)

    if ml_dog_line == 'XX':
        ml_dog_line = np.nan
    if ml_fav_line == 'XX':
        ml_fav_line = np.nan
    if spread == 'XX':
        spread = np.nan
    if fav_spread_line == 'XX':
        fav_spread_line = np.nan
    if dog_spread_line == 'XX':
        dog_spread_line = np.nan
    if total == 'XX':
        total = np.nan
    if over == 'XX':
        over = np.nan
    if under == 'XX':
        under = np.nan

    return pd.Series(dict(date=date, time=time, fav=fav, dog=dog, ml_fav_line=ml_fav_line,
                          ml_dog_line=float(ml_dog_line), spread=float(spread), fav_spread_line=float(fav_spread_line),
                          dog_spread_line=float(dog_spread_line), total=float(total), over=float(over),
                          under=float(under), line_datetime=dt))


def line_to_prob(line):
    if line < 0:
        return -line / (-line + 100.)
        # Do some stuff
    elif line > 0:
        return 100. / (line + 100.)
        # Do some other stuff


def calc_concensus(lines):
    lines = lines.sort_values('line_datetime')
    lines['concensus_ml_fav'] = np.nan
    lines['concensus_ml_dog'] = np.nan
    lines['ml_dog_std'] = np.nan
    lines['ml_fav_std'] = np.nan
    lines['n_lines_available'] = np.nan

    for key, grp in lines.groupby('game_uuid'):
        grp = grp[grp.line_datetime < grp.game_datetime]
        for idx, label in enumerate(grp.index):
            _grp = grp.iloc[:idx+1, :].drop_duplicates('bookie', keep='last')
            lines.loc[label, 'concensus_ml_fav'] = _grp['ml_fav_prob'].mean()
            lines.loc[label, 'concensus_ml_dog'] = _grp['ml_dog_prob'].mean()
            lines.loc[label, 'ml_dog_std'] = _grp['ml_dog_prob'].std()
            lines.loc[label, 'ml_fav_std'] = _grp['ml_fav_prob'].std()
            lines.loc[label, 'n_lines_available'] = idx + 1
    lines = lines.dropna()
    lines['ml_fav_z'] = (lines['ml_fav_prob'] - lines['concensus_ml_fav']) / lines['ml_fav_std']
    lines['ml_dog_z'] = (lines['ml_dog_prob'] - lines['concensus_ml_dog']) / lines['ml_dog_std']
    return lines


def abbrev_to_name(abbrev, mapping):
    return mapping[abbrev]


def process_games(games):
    games['away_team'] = games.away_team.apply(lambda x: x.split(' ')[-1].lower())
    games['home_team'] = games.home_team.apply(lambda x: x.split(' ')[-1].lower())
    games['date_'] = games.date.apply(lambda x: datetime.datetime.strftime(pd.to_datetime(str(x)), '%m-%d-%y'))
    return games


def process_lines(lines):
    lines = lines.dropna().reset_index()
    lines['fav_spread_prob'] = lines['fav_spread_line'].apply(line_to_prob)
    lines['dog_spread_prob'] = lines['dog_spread_line'].apply(line_to_prob)
    lines['ml_dog_prob'] = lines['ml_dog_line'].apply(line_to_prob)
    lines['ml_fav_prob'] = lines['ml_fav_line'].apply(line_to_prob)
    lines['over_prob'] = lines['over'].apply(line_to_prob)
    lines['under_prob'] = lines['under'].apply(line_to_prob)
    return lines
