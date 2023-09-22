import requests
from bs4 import BeautifulSoup
#TODO: add Search engine logic
def clean_html(url): # test
    response = requests.get(url)
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')
    
    for script in soup(['script', 'style']):
        script.extract()
   
    text = soup.get_text()
  
    clean_text = ' '.join(text.split())
    return clean_text


url = ""
clean_text = clean_html(url)
print(clean_text)