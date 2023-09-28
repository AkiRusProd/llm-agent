import requests
from bs4 import BeautifulSoup
from dotenv import dotenv_values
env = dotenv_values(".env")
# from googlesearch import search



API_KEY = env['GOOGLE_CLOUD_API_KEY']
SEARCH_ENGINE_ID = env['GOOGLE_SEARCH_ENGINE_ID']




class SearchEngine():
    def __init__(self) -> None:
        # self.url = lambda query: f'https://www.googleapis.com/customsearch/v1?key={API_KEY}&cx={SEARCH_ENGINE_ID}&q={query}'
        self.url = 'https://www.googleapis.com/customsearch/v1'

    def payload(self, query, start = 1, n_results = 3, date_restrict='m1', **params):
        payload = {
            'key': API_KEY,
            'q': query,  # Query string
            'start': start,  # Index of the first search result
            'cx': SEARCH_ENGINE_ID,
            'num': n_results,  # Number of search results to return
            # 'dateRestrict': date_restrict,  # Date restriction for search results
        
        }
        payload.update(params)

        return payload

    def scrape(self, url):
        try:
            response = requests.get(url)
        except:
            return None
        
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        
        for script in soup(['script', 'style']):
            script.extract()
    
        text = soup.get_text()
    
        clean_text = ' '.join(text.split())
        return clean_text

    def search(self, query:str, n_results: int = 3):
        # response = requests.get(self.url(query))
        response = requests.get(self.url, params = self.payload(query, n_results = n_results))

        request = []
       
        if response.status_code == 200:
            data = response.json()
            
            items = data.get('items', [])
            for item in items:
                
                title = item.get('title')
                link = item.get('link')
                content = self.scrape(link)

                request.append({'title': title, 'link': link, 'content': content}) if content is not None else None
        else:
            print('Error occurred during the search.')

        return request



# search_engine = SearchEngine()
# response = search_engine.search('python')

