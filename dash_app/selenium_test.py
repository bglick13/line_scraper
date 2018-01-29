from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium import webdriver
from bs4 import BeautifulSoup
from time import sleep
from random import randint
from pyvirtualdisplay import Display
from selenium import webdriver
import pandas as pd


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


class SeleniumNBA:

    def __init__(self):
        self.url = 'https://www.sportsinsights.com/nba/'
        self.start_driver()

    def start_driver(self):
        self.display = Display(visible=0, size=(800, 600))
        self.display.start()
        self.driver = webdriver.Chrome()
        self.driver.wait = WebDriverWait(self.driver, 30)
        self.driver.get(self.url)
        frame = self.driver.find_element_by_id('sportsinsights-iframe')
        self.driver.switch_to.frame(frame)
        # print(self.driver.page_source)
        self.driver.wait.until(
            EC.visibility_of_element_located((By.CLASS_NAME, "tableOdds")))
        select = Select(self.driver.find_element_by_id('view'))
        select.select_by_value('Moneyline')
        sleep(5)

    def get_table(self):
        html_page = self.driver.page_source
        soup = BeautifulSoup(html_page)
        tables = soup.find_all('table', class_='tableOdds')
        header = tables[0]
        table = tables[1]
        return header, table

    def _extract_date_time_team(self, td):
        divs = td.find_all('div')
        date_time = divs[2].find_all('div')
        date = date_time[0].find_all('div')[1].text.strip()
        time = date_time[3].find_all('div')[1].text.strip()
        away_team = divs[-2].text.strip()
        home_team = divs[-1].text.strip()
        return date, time, away_team, home_team

    def process_header(self, header):
        bookies = []
        data = header.find_all('td', class_='sportsbook')[1:]
        for d in data:
            bookies.append(d.find('div').text)
        return bookies

    def extract_table(self, header, table, _league):
        out = dict(game_date=[], game_time=[], away_team=[], away_ml=[], away_ml_consensus=[],
                   home_team=[], home_ml=[], home_ml_consensus=[], book=[], league=[])
        bookies = self.process_header(header)
        rows = table.find_all('tr', {'class': ['row-odd', 'row-group']})
        # league = _league
        for row in rows:
            _class = set(row.get('class'))
            if 'row-group' in _class:
                league = row.find('td', class_='team').text.strip().split('-')[0]
                continue
            data = row.find_all('td')

            # if _league not in league:
            #     continue


            date_time_team = data[1]
            date, time, away_team, home_team = self._extract_date_time_team(date_time_team)
            lines = row.find_all('td', class_='sportsbook')

            consensus = lines[-1]
            spans = consensus.find_all('span')
            try:
                away_consensus = spans[0].text.strip()
                home_consensus = spans[1].text.strip()
            except IndexError:
                continue

            try:
                for i, line in enumerate(lines[1:-1]):
                    spans = line.find_all('span')
                    away_line = spans[0].text.strip()
                    home_line = spans[1].text.strip()
                    out['game_date'].append(date)
                    out['game_time'].append(time)
                    out['away_team'].append(away_team)
                    out['away_ml'].append(away_line)
                    out['away_ml_consensus'].append(away_consensus)
                    out['home_team'].append(home_team)
                    out['home_ml'].append(home_line)
                    out['home_ml_consensus'].append(home_consensus)
                    out['book'].append(bookies[i])
                    out['league'].append(league)
            except IndexError:
                continue
        df = pd.DataFrame(out)
        return df

    def team_name_helper(self, row):
        s = row.split(' ')
        if len(s) == 2:
            return s[1]
        else:
            return ' '.join(s[1:])

    def process_table(self, df):
        df['home_ml'] = pd.to_numeric(df['home_ml'].str.replace(',', ''))
        df['away_ml'] = pd.to_numeric(df['away_ml'].str.replace(',', ''))
        df['home_ml_consensus'] = pd.to_numeric(df['home_ml_consensus'].str.replace(',', ''))
        df['away_ml_consensus'] = pd.to_numeric(df['away_ml_consensus'].str.replace(',', ''))
        df['home_team'] = df['home_team'].apply(self.team_name_helper)
        df['away_team'] = df['away_team'].apply(self.team_name_helper)
        df['home_ml_prob'] = df['home_ml'].apply(lambda x: line_to_prob(x))
        df['home_ml_consensus'] = df['home_ml_consensus'].apply(lambda x: line_to_prob(x))
        df['away_ml_prob'] = df['away_ml'].apply(lambda x: line_to_prob(x))
        df['away_ml_consensus'] = df['away_ml_consensus'].apply(lambda x: line_to_prob(x))
        df['home_p_diff'] = (df['home_ml_prob'] - df['home_ml_consensus']).round(4)
        df['away_p_diff'] = (df['away_ml_prob'] - df['away_ml_consensus']).round(4)
        df['game_id'] = df['home_team']+df['away_team']
        df['best_p_diff'] = df[['home_p_diff', 'away_p_diff']].min(axis=1)

        df['Pretty_Game_Time'] = df.apply(lambda x: x.game_date + ' ' + x.game_time, axis=1)
        df = df.sort_values('best_p_diff')
        return df

    def parse(self, league):
        header, table = self.get_table()
        df = self.extract_table(header, table, league.upper())
        try:
            df = self.process_table(df)
        except ValueError as e:
            print(e)
            return None
        print(df.groupby('league').count())
        return df

if __name__ == '__main__':
    spider = SeleniumNBA()
    spider.parse('bla')