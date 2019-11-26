"""
This file defines the image scrapers. There is one spider for the stock image
website, and one for pinterest.
"""

import scrapy

# from myitems import ImageItem

class IStockSpider(scrapy.Spider):
    name = 'iStock'
    start_urls = [
        'https://www.istockphoto.com/fr/photos/naked-woman?istockcollection=&mediatype=photography&phrase=naked%20woman&sort=mostpopular',
        ]

    def __init__(self):
        self.count = 0

    def parse(self, response):
        self.count += 1
        items = []
        # get links of images
        for link in response.css('img.gallery-asset__thumb::attr(src)'):
            yield {'image_urls': [link.get()]}
            # item = ImageItem()
            # item['image_urls'] = link.get()
            # items.append(item)

        # follow pagination links
        for href in response.css(
            'a.search-pagination__button--next::attr(href)'):
            yield response.follow(href, self.parse)