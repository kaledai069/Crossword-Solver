import datetime
import requests
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
import csv

WORKERS = 10
SKIPDAY = 1
# foolin' the server
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
}

crossword_data = []

# Define the start and end dates for the year
start_date = datetime.date(2023, 6, 26)
end_date = datetime.date(2023, 6, 26)


# Define the base URL
base_url = "https://crossword-solver.io/crossword-answers/eugene-sheffer/"

# Function to download e-paper for a given date
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
                text = clue_box.find('a').text.strip()
                sub_link = clue_box.find('a')['href']
            
                link  = "https://crossword-solver.io/" + sub_link

                response2 = requests.get(link, verify = True, headers  = HEADERS, timeout = 15)
                answer_soup = BeautifulSoup(response2.content, 'html.parser')
                h2_tags = answer_soup.find_all('h2', class_ = 'h4')
                answer = h2_tags[-1].find('b').text.replace(',', '').lower()
   
                crossword_data.append((text, answer.lower(), formatted_date))
                print((text, answer.lower(), formatted_date))
                print(len(crossword_data))

    except requests.exceptions.RequestException as e:
        print(f"Error occurred while downloading e-paper for {formatted_date}: {e}")

# Create a thread pool executor
executor = ThreadPoolExecutor(max_workers = WORKERS)

# Iterate over each date in the range and submit the download task to the executor
current_date = start_date
while current_date <= end_date:
    executor.submit(download_crossword_data, current_date)
    current_date += datetime.timedelta(days = SKIPDAY)

# Wait for all tasks to complete
executor.shutdown()

# Save crossword_data to csv
csv_file = "wsj.csv"
with open(csv_file, "a", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerows(crossword_data)