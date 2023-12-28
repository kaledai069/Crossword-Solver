from urllib.request import Request, urlopen

url = "https://crossword-solver.io/clue/actress-falco/"
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}

req = Request(url, headers=headers)

with urlopen(req) as response:
    print(response.read())