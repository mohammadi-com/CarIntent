import scrapy
import pandas as pd


class AllMakesModelsSpider(scrapy.Spider):
    name = 'all_makes_models'

    start_urls = [
        'https://classics.autotrader.com/',
        'https://www.autotrader.com/'
    ]

    def parse(self, response):
        print(len(response.xpath('//optgroup//option/text()')))
