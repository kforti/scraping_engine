
from scrapy.crawler import CrawlerProcess

from product_page_scrapers.product_page_scrapers.spiders.walmart_product_spider import WalmartProductSpider
from product_page_scrapers.product_page_scrapers.spiders.walmart_food_browse_spider import WalmartFoodBrowseSpider
from product_page_scrapers.product_page_scrapers.spiders.walmart_spider import WalmartSpider



def test_product_spider():
    spider = WalmartProductSpider
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })
    process.crawl(spider, urls=[]
                  )
    process.start()

def test_food_bowse_spider():
    spider = WalmartFoodBrowseSpider
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })
    process.crawl(spider)
    process.start()

def test_walmart_spider():
    spider = WalmartSpider
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1)'
    })
    process.crawl(spider)
    process.start()



if __name__ == '__main__':
    #test_product_spider()
    #test_food_bowse_spider()
    test_walmart_spider()
