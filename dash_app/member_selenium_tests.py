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


class SeleniumNBA:

    def __init__(self):
        self.url = 'http://account.sportsinsights.com/live-odds/#'
        self.start_driver()

    def start_driver(self):
        # self.display = Display(visible=0, size=(800, 600))
        # self.display.start()
        self.driver = webdriver.Chrome()
        self.driver.wait = WebDriverWait(self.driver, 30)
        self.driver.get(self.url)
        self.login()
        self.driver.get(self.url)
        # frame = self.driver.find_element_by_id('sportsinsights-iframe')
        # self.driver.switch_to.frame(frame)
        # # print(self.driver.page_source)
        # self.driver.wait.until(
        #     EC.visibility_of_element_located((By.CLASS_NAME, "tableOdds")))
        # select = Select(self.driver.find_element_by_id('view'))
        # select.select_by_value('Moneyline')
        sleep(5)

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

if __name__ == '__main__':
    spider = SeleniumNBA()
