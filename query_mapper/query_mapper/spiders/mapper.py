import scrapy
import pandas as pd
import time


class MapperSpider(scrapy.Spider):
    name = 'mapper'
    csv_file_address = './query_mapper/spiders/autotrader.com-organic.Positions-uk-20200507-2020-05-08T15_39_18Z.csv'
    df = None
    output_df = pd.DataFrame(columns=['Keyword', 'Make', 'Model'])
    makes = ['amc', 'acura', 'alfa romeo', 'aston martin', 'audi', 'bmw', 'bentley', 'bugatti', 'buick', 'cadillac', 'chevrolet', 'chrysler', 'daewoo', 'datsun', 'delorean', 'dodge', 'eagle', 'fiat', 'ferrari', 'fisker', 'ford', 'freightliner', 'gmc', 'genesis', 'geo', 'hummer', 'honda', 'hyundai', 'infiniti', 'isuzu', 'jaguar', 'jeep', 'karma', 'kia', 'lamborghini', 'land rover', 'lexus', 'lincoln', 'lotus', 'lucid', 'mazda', 'mini', 'maserati', 'maybach', 'mclaren', 'mercedes-benz', 'mercury', 'mitsubishi', 'nissan', 'oldsmobile', 'plymouth', 'polestar', 'pontiac', 'porsche', 'ram', 'rivian', 'rolls-royce', 'srt', 'saab', 'saturn', 'scion', 'subaru', 'suzuki', 'tesla', 'toyota', 'volkswagen', 'volvo', 'yugo', 'smart']

    def read_csv_file(self):
        data = pd.read_csv(self.csv_file_address)
        self.df = pd.DataFrame(data, columns=['Keyword', 'URL'])
        return self.df

    def start_requests(self):
        self.read_csv_file()
        for index, row in self.df.iterrows():
            time.sleep(1.5)
            yield scrapy.Request(url=row['URL'], callback=self.parse, meta={'Keyword': row['Keyword']})

    def parse(self, response):
        if response.url.__contains__('classics.autotrader.com'):
            make = response.xpath("//div[@class='filter-tag'][contains(text(),'Make')]/text()")
            model = response.xpath("//div[@class='filter-tag'][contains(text(),'Model')]/text()")
            # print('This is the meta: ', response.meta['Keyword'])
            if len(make) == 1 and len(model) == 1:  # We are checking to be on ly match
                make = make.get()[6:].lower()
                model = model.get()[7:].lower()
                # print('This is going to be saved: ', response.meta['Keyword'], make, model)
                self.output_df.loc[len(self.output_df)] = [response.meta['Keyword'], make, model]  # save to df
                self.output_df.to_csv('mapper.csv')
            return
        else:
            url_parts = response.url.split('/')
            for i, url_part in enumerate(url_parts):
                if (i+1) < len(url_parts) and url_part.lower().replace('+', ' ') in self.makes:
                    make = url_parts[i].lower().replace('+', ' ').lower()
                    model = url_parts[i+1].lower().replace('+', ' ').lower()
                    # print('This is going to be saved: ', response.meta['Keyword'], make, model)
                    self.output_df.loc[len(self.output_df)] = [response.meta['Keyword'], make, model]
                    self.output_df.to_csv('mapper.csv')
                    return
        make = response.xpath('//div[@data-cmp="filterCheckboxes" and descendant::span[contains(text(),"Make")]]//input[@type="checkbox" and @checked]//following-sibling::*//text()')  # //div[@data-cmp="filterCheckboxes"][12]//label[input[@type="checkbox" and @checked]]//text()
        model = response.xpath('//div[@data-cmp="filterCheckboxes" and descendant::span[contains(text(),"Model")]]//input[@type="checkbox" and @checked]//following-sibling::*//text()')
        if len(make) == 1 and len(model) == 1:  # We are checking to be on ly match
            make = make.get()
            model = model.get()
            # print('This is going to be saved: ', response.meta['Keyword'], make, model)
            self.output_df.loc[len(self.output_df)] = [response.meta['Keyword'], make, model]
            self.output_df.to_csv('mapper.csv')
