import datetime
import requests
import csv
import pandas as pd 

from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from urllib.request import Request, urlopen

WORKERS = 4
SKIPDAY = 1
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
PORTALS = [
        'eugene-sheffer',
        'los-angeles-times-daily',
        'los-angeles-times-mini',
        'newsday-com',
        'ny-times',
        'ny-times-mini',
        'two-speed',
        'telegraph-quick',
        'thomas-joseph',
        'universal',
        'usa-today',
        'wsj',
        'family-time',
        'mirror',
        'mirror-quick',
        'mirror-quiz',
        'mirror-teatime',
        'premier-sunday',
        'puzzler',
        'puzzler-backwords',
        'commuter',
        'tca-tv',
        'guardian-quick',
        'guardian-speedy',
        'guardian-weekend',
        'times-concise',
        'times-specialist-sunday',
        'telegraph-toughie'
    ]

CURRENT_PORTAL = 0

crossword_data = []

# starting and ending dates
start_date = datetime.date(2023, 12, 28)
end_date = datetime.date(2023, 12, 28)

# base_outline_url
base_url = "https://crossword-solver.io/crossword-answers/" + PORTALS[CURRENT_PORTAL] + '/'


def download_crossword_data(date):
    # Format the date in the required format
    formatted_date = date.strftime("%Y-%m-%d")

    # Build the complete URL
    url = base_url + formatted_date + "-answers/"

    print(url)

    try:
        # Send a GET request to the URL with SSL verification
        response = requests.get(url, verify = True, headers = HEADERS, timeout = 15)

        # Check if the response is successful (status code 200)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            wrapper_div = soup.find('ul', class_ = 'list-group')
            list_elements = wrapper_div.find_all('li')

            for clue_box in list_elements:
                clue_txt = clue_box.find('a').text.strip()
                clue_txt = clue_txt.replace(' Crossword Clue', '')

                sub_link = clue_box.find('a')['href']
                sub_link = sub_link.replace(r'%e2%80%9c', '')
                sub_link = sub_link.replace(r'%e2%80%9d', '')

                link  = "https://crossword-solver.io" + sub_link
                req = Request(link, headers = HEADERS)

                with urlopen(req) as response:
                    sub_link_status_code = response.getcode()
                    answer_page_content = response.read()

                if sub_link_status_code == 200:
                    answer_soup = BeautifulSoup(answer_page_content, 'html.parser')
                    ul_tag = answer_soup.find('ul', class_ = 'inline-list')
                    li_tags = ul_tag.find_all('li')
                    answer = li_tags[0].text.replace(',', '').lower()
    
                    crossword_data.append((clue_txt, answer.lower(), formatted_date))
                    print(f"Count: {len(crossword_data)} ||| Clue: {clue_txt} ||| Answer: {answer} ||| Date: {formatted_date}")

    except requests.exceptions.RequestException as e:
        print(f"Error scrapping crossword data for: {formatted_date}: {e}")

# thread pool executor
executor = ThreadPoolExecutor(max_workers = WORKERS)

current_date = start_date
while current_date <= end_date:
    executor.submit(download_crossword_data, current_date)
    current_date += datetime.timedelta(days = SKIPDAY)

executor.shutdown()

output_path = f"./{PORTALS[CURRENT_PORTAL]}-crossword-data.csv"
crossword_df = pd.DataFrame(crossword_data, columns = ['Clue', 'Answer', 'Date'])

crossword_df.to_csv(output_path)