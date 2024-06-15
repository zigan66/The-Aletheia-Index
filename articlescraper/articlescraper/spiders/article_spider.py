# command for running the spider "scrapy crawl article" on terminal

import scrapy
from articlescraper.items import ArticleItem
import urllib.parse
import tldextract
import json


class ArticleSpider(scrapy.Spider):
    name = "article"
    article_count = 0 

    def start_requests(self):
        urls = read_json_data()
        for url in urls:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        self.article_count += 1
        item = ArticleItem()
        item['id'] = self.article_count
        item['title'] = response.css('h1::text').get(default='No title')
        item['platform'] = self.extract_platform_from_url(response.url)
        item['article_text'] = ' '.join(response.css('article p::text').extract())
        
        yield item
    
    
    def extract_platform_from_url(self, url):
        extracted = tldextract.extract(url)
        # The registered domain is a combination of the second-level domain (SLD) and the top-level domain (TLD)
        # For example, 'bbc.co.uk' -> (SLD='bbc', TLD='co.uk')
        platform = extracted.domain
        return platform
    


# Function to read JSON data
def read_json_data():
  try:
    with open('../urls.json', "r") as f:
      return json.load(f)
  except FileNotFoundError:
    print(f"Error: File '{'../urls.json'}' not found.")
    return None