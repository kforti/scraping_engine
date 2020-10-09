import json
import time

import scrapy
from bs4 import BeautifulSoup




class WalmartFoodBrowseSpider(scrapy.Spider):
    name = "walmart_food_browse"
    products = []
    def start_requests(self):

        base_url = 'https://www.walmart.com/browse/food/976759?page={page}'
        page = 1
        while True:
            url = base_url.format(page=page)
            yield scrapy.Request(url=url, callback=self.parse_result)
            page += 1
            time.sleep(1)
            if page == 26:
                break
        with open("../../../data/products.json", "w") as f:
            json.dump(self.products, f)

    def parse_result(self, response):
        soup = BeautifulSoup(response.text, 'html.parser')
        data = soup.find_all('script', attrs={'id':'searchContent'})

        for d in data:
            contents = d.contents
            if len(contents) == 0:
                continue
            results = json.loads(contents[0])
            items = results["searchContent"]["preso"]["items"]
            self.products.extend(items)



    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'quotes-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)

        self.log('Saved file %s' % filename)