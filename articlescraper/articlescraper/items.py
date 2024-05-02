import scrapy

class ArticleItem(scrapy.Item):
    id = scrapy.Field()
    title = scrapy.Field()
    platform = scrapy.Field()
    article_text = scrapy.Field()
