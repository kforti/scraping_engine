import scrapy
from scrapy_selenium import SeleniumRequest



class WalmartProductSpider(scrapy.Spider):
    name = "walmart_product"

    def __init__(self, urls):
        self.urls = urls

    def start_requests(self):
        urls = [
            'https://www.walmart.com/ip/Kellogg-s-Raisin-Bran-Breakfast-Cereal-Original-Family-Size-24-Oz/844236833?wpa_bd=&wpa_pg_seller_id=F55CDC31AB754BB68FE0B39041159D63&wpa_ref_id=wpaqs:oY8Uq2-8EsahZtHhSEzEeVwHB6KFQ-2wPkD5VyawtMxIYPgQAsp6jmUmkwzZFwGZ9YYJoTUwhNTE3zkULQJUAwZ6-I_svxLJlxdcfY2bU_opWocBK2pCFMjzs5mU6gFz3aFZCMVRxXNGYPCn89XFvZXbUWzN_Fo0dZx_hmGeXnbjl0cZ9Vv_6nDckrTxwmaplNjY_3WoYkVJIUxhnI3EXQ&wpa_tag=&wpa_aux_info=&wpa_pos=1&wpa_plmt=1145x1145_T-C-IG_TI_1-2_HL-INGRID-GRID-NY&wpa_aduid=690c098b-8b3c-417e-9f77-433c1632d561&wpa_pg=browse&wpa_pg_id=976759&wpa_st=__searchterms__&wpa_tax=976759&wpa_bucket=__bkt__',

        ]
        for url in urls:
            yield SeleniumRequest(url=url, callback=self.parse_result, screenshot=True)

    def parse_result(self, response):
        print(response.request.meta['driver'].title)
        with open('image.png', 'wb') as image_file:
            image_file.write(response.meta['screenshot'])

    def parse(self, response):
        page = response.url.split("/")[-2]
        filename = 'quotes-%s.html' % page
        with open(filename, 'wb') as f:
            f.write(response.body)

        self.log('Saved file %s' % filename)