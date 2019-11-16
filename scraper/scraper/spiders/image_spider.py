"""
This file defines the image scrapers. There is one spider for the stock image
website, and one for pinterest.
"""

import scrapy

class IStockSpider(scrapy.Spider):
    name = 'iStock'
    start_urls = [
        'https://www.istockphoto.com/fr/photos/naked-man?page=2&phrase=naked%20man&sort=mostpopular',
        ]

    def __init__(self):
        self.count = 0

    def parse(self, response):
        self.count += 1
        # get links of images
        for link in response.css('img.gallery-asset__thumb::attr(src)'):
            yield {'image_urls': link.get()}

        # follow pagination links
        for href in response.css(
            'a.search-pagination__button--next::attr(href)'):
            yield response.follow(href, self.parse)