import scrapy
from scrapy.spider import BaseSpider
from scrapy.selector import HtmlXPathSelector
from craigslist_sample.items import CraigslistSampleItem

class MySpider(scrapy.Spider):
    name = "craig"
    #allowed_domains = ["craigslist.org"]
    def start_requests(self):
        ## Create huge map of all of the lat / longs from our image set 
        ## Use that to create a list of URLs with the pattern below
        ## Change this to a for URL in URLs, yield ... [calls parse at the end] 
        url = "http://data.fcc.gov/api/block/find?format=json&latitude=37.714439&longitude=-122.214456&showall=true"
        yield scrapy.Request(url, self.parse)

    def parse(self, response):
        for quote in response.css('div.quote'):
            print quote 
        # This is initialized with an "items" object 
        
        # hxs = HtmlXPathSelector(response)
        # titles = hxs.xpath("//span[@class='pl']")
        # print titles 
        # items = []
        # for titles in titles:
        #     item = CraigslistSampleItem()
        #     item["title"] = titles.select("a/text()").extract()
        #     item["link"] = titles.select("a/@href").extract()
        #     items.append(item)
        # return items


    # https://doc.scrapy.org/en/latest/topics/debug.html 
    # using the type of URL given above 
