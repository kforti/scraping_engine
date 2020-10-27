import json
import time

import scrapy
from bs4 import BeautifulSoup



class AmazonSpider(scrapy.Spider):
    name = "amazon_food_browse"
    urls = {}
    def start_requests(self):
        base_url = 'https://www.amazon.com/gp/browse.html?node=16310101&ref_=nav_em__gro_0_2_18_2'
        url = base_url
        while True:
            yield scrapy.Request(url=url, callback=self.parse_result)
            time.sleep(1)

    def parse_result(self, response):
        soup = BeautifulSoup(response.text, 'html.parser')
        data = soup.find_all('a')
        for d in data:
            print(d.get('href'))


    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'quotes-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)

        self.log('Saved file %s' % filename)