import requests
import json

def fetch_nyt_crossword(dateStr):
    '''
        Fetch NYT puzzle from a specific date.
    '''

    headers = {
        'Referer': 'https://www.xwordinfo.com/JSON/'
    }
    # mm/dd/yyyy

    url = 'https://www.xwordinfo.com/JSON/Data.ashx?date=' + dateStr

    response = requests.get(url, headers=headers)

    context = {}
    grid_data = {}
    if response.status_code == 200:
        bytevalue = response.content
        jsonText = bytevalue.decode('utf-8').replace("'", '"')
        puzzle_data = json.loads(jsonText)
        with open("./today_nyt_7-27-20203.json", 'w') as file:
            json.dump(puzzle_data, file)
        return puzzle_data

    else:
        print(f"Request failed with status code {response.status_code}.")

fetch_nyt_crossword("2023/07/27")