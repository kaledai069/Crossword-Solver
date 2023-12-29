import urllib.request
import urllib.parse
import urllib.error
from tqdm import tqdm
import csv
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor


# URL of the main page 
base_url = "https://crossword-answers.com"

news_outlet = "spyscape" # CHANGE THE SITE HERE

main_puzzle_url = f"{base_url}/{news_outlet}-crossword-answers"
pagination_value = 0 # you get an error in 285
pagination_end = 270

def process_puzzle(puzzle_link):
    crossword_data = []
    
    # do not scrape for 2020
    if puzzle_link['href'][-10:-6] == '//an':
        return crossword_data
    
    clue_url = puzzle_link["href"]
    clue_url = base_url + clue_url

    try: 
        # Send a GET request to the puzzle clues URL
        clues_html = urllib.request.urlopen(clue_url).read()
        clue_soup = BeautifulSoup(clues_html, "html.parser")
    except:
        return crossword_data
    else:
        # Find all the clue rows
        clue_rows = clue_soup.select(".cluerow a")

        # Process each clue row
        for clue_row in clue_rows:
            clue = clue_row.text.strip()  # extracting the clue

            answer_url = clue_row["href"]
            answer_url = main_puzzle_url + 'puzzle' + answer_url[1:] #removing the "." in the beginning
            try:
                # Send a GET request to the answer URL
                answer_html = urllib.request.urlopen(answer_url).read()
                answer_soup = BeautifulSoup(answer_html, "html.parser")
                # Find the answer link
                answer_link = answer_soup.select_one(".text-success a")
                answer = answer_link.text.strip()
            except:
                print(clue_url, answer_url)
            else:
                # Append the clue and answer to the crossword data list
                crossword_data.append((clue, answer.lower(), clue_url[-10:]))
        
        return crossword_data


def process_page(pagination_value):
    crossword_data = []
    
    main_url = main_puzzle_url + f'page/{pagination_value}'
    print(main_url)

    # Send a GET request to the main URL
    html = urllib.request.urlopen(main_url).read()
    # Create a BeautifulSoup object to parse the HTML content
    soup = BeautifulSoup(html, "html.parser")
    # Find all the links to the puzzle pages
    puzzle_links = soup.select(".h20 a")

    # Process each puzzle link
    with tqdm(total=len(puzzle_links)) as pbar:
        with ThreadPoolExecutor() as executor:
            futures = []
            for puzzle_link in puzzle_links:
                future = executor.submit(process_puzzle, puzzle_link)
                futures.append(future)

            for future in futures:
                crossword_data.extend(future.result())
                pbar.update(1)

    # Save the crossword data to a CSV file
    csv_file = f"{news_outlet}_crossword.csv"
    with open(csv_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(crossword_data)

while pagination_value <= pagination_end:
    process_page(pagination_value)
    pagination_value += 15