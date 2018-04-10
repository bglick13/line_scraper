from selenium.webdriver.support.ui import WebDriverWait

from selenium.webdriver.support.ui import Select
from pyvirtualdisplay import Display

import numpy as np
from time import sleep
from random import randint
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import selenium.common.exceptions as exc
import pandas as pd
import datetime

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

NBA_TEAMS = frozenset(
    [
        'Hawks', 'Celtics', 'Nets', 'Hornets', 'Bulls', 'Cavaliers', 'Mavericks', 'Nuggets', 'Pistons', 'Warriors',
        'Rockets', 'Pacers', 'Clippers', 'Lakers', 'Grizzlies', 'Heat', 'Bucks', 'Timberwolves', 'Pelicans', 'Knicks',
        'Thunder', 'Magic', '76ers', 'Suns', 'Trail Blazers', 'Kings', 'Spurs', 'Raptors', 'Jazz'
    ]
)

PREM_TEAMS = frozenset(
    [
        'Arsenal', 'Bournemouth', 'Brighton', 'Burnley', 'Chelsea', 'Crystal Palace', 'Everton', 'Huddersfield',
        'Leicester', 'Liverpool', 'Man City', 'Man Utd', 'Newcastle', 'Southampton', 'Stoke City', 'Swansea',
        'Tottenham', 'Watford', 'West Brom', 'West Ham'

    ]
)

NHL_TEAMS = frozenset(
    [
        'CAR', 'CBJ', 'NJD', 'NYI', 'NYR', 'PHI', 'PIT', 'WAS', 'BOS',
        'BUF', 'DET', 'FLA', 'MTL', 'OTT', 'TB', 'TOR', 'CHI', 'COL',
        'DAL', 'MIN', 'NSH', 'STL', 'WIN', 'ANA', 'ARI', 'CGY', 'EDM', 'LA', 'SJ',
        'VAN', 'VGK'
    ]
)

NCAAB_TEAMS = frozenset(
    [
        'Northwestern', 'Michigan', 'Notre Dame', 'Duke', 'Illinois Chicago', 'Wisc Milwaukee', 'Kansas', 'Kansas State',
        'Nebraska', 'Wisconsin', 'E Tennessee St', 'Citadel', 'Western Carolina', 'Wofford', 'NC Greensboro',
        'Tennessee Chat', 'Monmouth', 'Rider', "Saint Peter's", 'Siena', 'Iona', 'Fairfield', 'VMI', 'Samford', 'Lehigh',
        'Holy Cross', 'Rhode Island', 'Massachusetts', 'Ohio', 'Bowling Green', 'North Carolina', 'Clemson', 'Buffalo',
        'Lent', 'Northern Illinois', 'Central Michigan', 'Eastern Michigan', 'Akron', 'Miami Ohio', 'Ball State',
        'Toledo', 'TCU', 'Oklahoma State', 'Indiana', 'Ohio State', 'Rutgers', 'Illinois', 'Florida', 'Georgia',
        'Illinois State', 'Missouri State', 'So Illinois', 'Drake', 'Xavier', "St John's", 'Baylor', 'Oklahoma',
        'Minnesota', 'Iowa', 'Auburn', 'Mississippi', 'Arkansas', 'Texas A&M', 'Vanderbilt', 'Kentucky', 'Marist',
        'Manhattan', 'Omaha', 'South Dakota State', 'Tulane', 'East Carolina', 'Penn State', 'Michigan State', 'LSU',
        'Tennessee', 'Houston', 'Cincinnati', 'Saint Louis', "St Joseph's", 'Louisville', 'Virginia', 'St Bonaventure',
        'George Mason', 'Duquesne', 'George Washington', 'La Salle', 'Davidson', 'Pittsburgh', 'Miami Florida',
        'West Virginia', 'Iowa State', 'Providence', 'Seton Hall', 'Syracuse', 'Georgia Tech', 'Florida State',
        'Wake Forest', 'Northern Iowa', 'Valparaiso', 'Loyola Chicago', 'Bradley', 'Mississippi St', 'South Carolina',
        'Memphis', 'South Florida', 'Maryland', 'Purdue', 'Virginia Tech', 'Boston College', 'Texas', 'Texas Tech',
        'Missouri', 'Alabama', 'Butler', 'Marquette', 'Arizona'
    ]
)

TEAMS = dict(nba=NBA_TEAMS, prem=PREM_TEAMS, nhl=NHL_TEAMS, ncaab=NCAAB_TEAMS)

BOOKIES = frozenset(
       [
           'Bookmaker', 'Pinnacle', '5Dimes', 'BetOnline', 'Bovada', 'GTBets', 'Sportsbk', 'Westgate', 'MGM',
           'Will Hill', 'Heritage', 'Greek', 'BetMania', 'Bet365', 'BetDSI', 'Caesars', 'Intertops', 'JustBet',
           'LooseLines', 'Nitrogen', 'Sportbet', 'SportsBetting', 'YouWager', 'Wynn', 'Consensus'
       ]
)

CONSENSUS = frozenset(['Consensus'])

class SeleniumSpider:

    def __init__(self):
        self.selected_sport = None
        self.sport_tabs = None
        self.table_header = None
        self.bookies = None
        self.select = None
        self.display = None
        self.driver = None
        self.url = 'http://account.sportsinsights.com/live-odds/#'
        self.start_driver()
        self.get_sport_tabs()

    def start_driver(self):
        # chrome_options = Options()
        # chrome_options.add_argument("--headless")
        # chrome_options.add_argument("--window-size=3600x1080")

        self.display = Display(visible=False, size=(3200, 1200))
        self.display.start()
        self.driver = webdriver.Chrome()
        self.driver.set_window_size(3600, 1700)
        self.driver.wait = WebDriverWait(self.driver, 30)
        self.driver.get(self.url)
        self.login()
        self.driver.get(self.url)
        sleep(5)

    # Close chromedriver
    def close_driver(self):
        print('closing driver...')
        self.display.stop()
        self.driver.quit()
        print('closed!')

    def login(self):
        print('getting pass the gate page...')
        try:
            form = self.driver.find_element_by_id('login-form')
            form.find_element_by_name('email').send_keys('benglickenhaus@gmail.com')
            form.find_element_by_name('password').send_keys('C0gZnaj5u119')
            form.find_element_by_class_name('btn-primary').click()
            sleep(randint(3, 5))
        except Exception:
            pass

    def get_sport_tabs(self):
        self.sport_tabs = self.driver.find_element_by_id('sport-tabs-list').find_elements_by_class_name('btn-sport')

    def select_sport(self, sport):
        # print("selecting {} tab".format(sport))
        if sport.lower() == 'nba':
            # print(self.sport_tabs[1].text)
            self.sport_tabs[1].click()
        if sport.lower() == 'mlb':
            self.sport_tabs[2].click()
        if sport.lower() == 'nhl':
            self.sport_tabs[3].click()
        if sport.lower() == 'ncaab':
            self.sport_tabs[5].click()
        if sport.lower() == 'prem':
            self.sport_tabs[6].click()
        if sport.lower() == 'la liga':
            self.sport_tabs[8].click()

    def select_moneylines(self):
        # print("selecting moneylines")
        self.select = Select(self.driver.find_element_by_id('view-options').find_element_by_class_name('smart-form'))
        self.select.select_by_visible_text('Moneyline')

    def get_table_headers(self):
        self.table_header = self.driver.find_elements_by_id('agText')
        headers = [h.text for h in self.table_header]
        self.bookies = [x for x in headers if x in BOOKIES]

    def _get_generic_row(self, row, sport):
        sport = sport.lower()
        if sport == 'prem' or sport == 'la liga':
            draws = True
            out = dict(game_date=[], away_ml=[], away_ml_consensus=[], home_ml=[], home_ml_consensus=[],
                       draw_ml=[], draw_ml_consensus=[], book=[], home_team=[], away_team=[])
        elif sport == 'nba' or sport == 'nhl' or sport == 'ncaab' or sport == 'mlb' or sport == 'nfl':
            draws = False
            out = dict(game_date=[], away_ml=[], away_ml_consensus=[], home_ml=[], home_ml_consensus=[],
                       book=[], home_team=[], away_team=[])

        success = False
        while not success:
            try:
                date_time_team = self.driver.find_elements_by_xpath("//div[@row='{}']".format(row))[0].text.split('\n')
                date_time = ' '.join(date_time_team[:2])

                success = True
            except exc.StaleElementReferenceException:
                pass
        if len(date_time_team) > 1:
            try:
                now = datetime.datetime.now()
                game_date = date_time + ' {}'.format(now.year)
                if int(game_date.split('/')[0]) < 10:
                    game_date = '0' + game_date
                game_date = datetime.datetime.strptime(game_date, '%m/%d %I:%M %p %Y')
                if game_date < now:
                    return None, None, None, None
            except ValueError:
                return None, None, None, None

        if sport == 'nhl' or sport == 'mlb':
            away_team = date_time_team[4].split('-')[0]
            home_team = date_time_team[5].split('-')[0]

        elif draws:
            away_team = date_time_team[-4]
            home_team = date_time_team[-3]
        else:
            away_team = date_time_team[-2]
            home_team = date_time_team[-1]



        success = False
        while not success:
            try:
                lines = self.driver.find_elements_by_xpath("//div[@row='{}']".format(row))[2]
                lines = lines.find_elements_by_class_name('ag-cell')[8:]
                lines = [l.text for l in lines]
                success = True
            except exc.StaleElementReferenceException:
                pass

        consensus = lines[-1].replace('\n', '').strip().split(' ')
        away_ml_consensus = consensus[0]
        home_ml_consensus = consensus[1]
        if draws:
            draw_ml_consensus = consensus[2]

        for i, b in enumerate(self.bookies):
            line = lines[i]
            if line:
                line = line.replace('\n', '').strip().split(' ')
                away_ml_line = line[0]
                home_ml_line = line[1]
                if draws:
                    draw_ml_line = line[2]
            else:
                away_ml_line = np.nan
                home_ml_line = np.nan
                if draws:
                    draw_ml_line = np.nan

            out['game_date'].append(date_time)
            out['away_ml'].append(away_ml_line)
            out['away_ml_consensus'].append(away_ml_consensus)
            out['home_ml'].append(home_ml_line)
            out['home_ml_consensus'].append(home_ml_consensus)
            out['home_team'].append(home_team)
            out['away_team'].append(away_team)
            if draws:
                out['draw_ml'].append(draw_ml_line)
                out['draw_ml_consensus'].append(draw_ml_consensus)
            out['book'].append(b)
        df = pd.DataFrame(out)

        df['home_ml'] = pd.to_numeric(df['home_ml'])
        df['home_ml_consensus'] = pd.to_numeric(df['home_ml_consensus'])
        df['away_ml'] = pd.to_numeric(df['away_ml'])
        df['away_ml_consensus'] = pd.to_numeric(df['away_ml_consensus'])
        if draws:
            df['draw_ml'] = pd.to_numeric(df['draw_ml'])
            df['draw_ml_consensus'] = pd.to_numeric(df['draw_ml_consensus'])

        df['away_ml_consensus'] = df['away_ml_consensus'].apply(lambda x: line_to_prob(x))
        df['home_ml_consensus'] = df['home_ml_consensus'].apply(lambda x: line_to_prob(x))
        df['home_ml_prob'] = df['home_ml'].apply(lambda x: line_to_prob(x))
        df['away_ml_prob'] = df['away_ml'].apply(lambda x: line_to_prob(x))
        if draws:
            df['draw_ml_consensus'] = df['draw_ml_consensus'].apply(lambda x: line_to_prob(x))
            df['draw_ml_prob'] = df['draw_ml'].apply(lambda x: line_to_prob(x))
            df['draw_p_diff'] = (df['draw_ml_prob'] - df['draw_ml_consensus']).round(4)
        try:
            df['home_p_diff'] = (df['home_ml_prob'] - df['home_ml_consensus']).round(4)
            df['away_p_diff'] = (df['away_ml_prob'] - df['away_ml_consensus']).round(4)
        except AttributeError:
            return None, None, None, None
        df['best_p_diff'] = df[['home_p_diff', 'away_p_diff']].min(axis=1)
        df = df.sort_values('best_p_diff')
        return away_team, home_team, df, game_date

    def _get_nba_row(self, row):
        out = dict(date=[], away_ml_line=[], away_ml_consensus=[], home_ml_line=[], home_ml_consensus=[], book=[])
        date_time_team = self.driver.find_elements_by_xpath("//div[@row='{}']".format(row))[0].text.replace('\n', ' ').split(' ')
        date_time = ' '.join(date_time_team[:3])
        teams = [x for x in date_time_team if x in NBA_TEAMS]
        away_team = teams[0]
        home_team = teams[1]
        lines = self.driver.find_elements_by_xpath("//div[@row='{}']".format(row))[2]
        lines = lines.find_elements_by_class_name('ag-cell')[8:]
        lines = [l.text for l in lines]

        consensus = lines[-1].replace('\n', '').strip().split(' ')
        away_ml_consensus = consensus[0]
        home_ml_consensus = consensus[1]
        print("# of lines: {}".format(len(lines)))
        print("# of bookies: {}".format(len(self.bookies)))
        for i, b in enumerate(self.bookies):
            line = lines[i]
            if line:
                line = line.replace('\n', '').strip().split(' ')
                away_ml_line = line[0]
                home_ml_line = line[1]
            else:
                away_ml_line = np.nan
                home_ml_line = np.nan

            out['date'].append(date_time)
            out['away_ml_line'].append(away_ml_line)
            out['away_ml_consensus'].append(away_ml_consensus)
            out['home_ml_line'].append(home_ml_line)
            out['home_ml_consensus'].append(home_ml_consensus)
            out['book'].append(b)
        return away_team, home_team, pd.DataFrame(out)

    def _get_prem_row(self, row):
        out = dict(date=[], away_ml_line=[], away_ml_consensus=[], home_ml_line=[], home_ml_consensus=[],
                   draw_ml_line=[], draw_ml_consensus=[], book=[])
        date_time_team = self.driver.find_elements_by_xpath("//div[@row='{}']".format(row))[0].text.replace('\n', ' ').split(' ')
        date_time = ' '.join(date_time_team[:3])
        teams = [x for x in date_time_team if x in PREM_TEAMS]
        away_team = teams[0]
        home_team = teams[1]
        lines = self.driver.find_elements_by_xpath("//div[@row='{}']".format(row))[2]
        lines = lines.find_elements_by_class_name('ag-cell')[8:]
        lines = [l.text for l in lines]

        consensus = lines[-1].replace('\n', '').strip().split(' ')
        away_ml_consensus = consensus[0]
        home_ml_consensus = consensus[1]
        draw_ml_consensus = consensus[2]
        print("# of lines: {}".format(len(lines)))
        print("# of bookies: {}".format(len(self.bookies)))
        for i, b in enumerate(self.bookies):
            line = lines[i]
            if line:
                line = line.replace('\n', '').strip().split(' ')
                away_ml_line = line[0]
                home_ml_line = line[1]
                draw_ml_line = line[2]
            else:
                away_ml_line = np.nan
                home_ml_line = np.nan
                draw_ml_line = np.nan
                # away_ml_line = lines[i*3]
                # home_ml_line = lines[i*3+1]
                # draw_ml_line = lines[i*3+2]
            out['date'].append(date_time)
            out['away_ml_line'].append(away_ml_line)
            out['away_ml_consensus'].append(away_ml_consensus)
            out['home_ml_line'].append(home_ml_line)
            out['home_ml_consensus'].append(home_ml_consensus)
            out['draw_ml_line'].append(draw_ml_line)
            out['draw_ml_consensus'].append(draw_ml_consensus)
            out['book'].append(b)
        df = pd.DataFrame(out)
        df['']
        return away_team, home_team, df

    def get_table_row(self, row, sport):
        sport = sport.lower()
        if self.selected_sport != sport:
            self.select_sport(sport)
            self.select_moneylines()
            self.get_table_headers()
            self.selected_sport = sport
        return self._get_generic_row(row, sport)

if __name__ == '__main__':
    spider = SeleniumSpider()

    away, home, df = spider.get_table_row(5, 'nba')
    print(df.head(20))
    sleep(60)
    away, home, df = spider.get_table_row(5, 'nba')
    print(df.head(20))

    # for r in range(30):
    #     try:
    #         away, home, df = spider.get_table_row(r, 'nba')
    #         df = df.dropna()
    #         print("{} @ {}".format(away, home))
    #         print(df.dropna().head(20))
    #         print()
    #     except IndexError:
    #         pass
    #
    # for r in range(30):
    #     try:
    #         away, home, df = spider.get_table_row(r, 'prem')
    #         print("{} @ {}".format(away, home))
    #         print(df.dropna().head(20))
    #         print()
    #     except IndexError:
    #         pass
    #
    # for r in range(30):
    #     try:
    #         away, home, df = spider.get_table_row(r, 'nhl')
    #         print("{} @ {}".format(away, home))
    #         print(df.dropna().head(20))
    #         print()
    #     except IndexError:
    #         pass
    #
    # for r in range(30):
    #     try:
    #         away, home, df = spider.get_table_row(r, 'ncaab')
    #         print("{} @ {}".format(away, home))
    #         print(df.dropna().head(20))
    #         print()
    #     except IndexError:
    #         pass

    spider.close_driver()
