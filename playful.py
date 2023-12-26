import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

base_url = 'https://camo.githubusercontent.com/9306488cfa658cb30bf60139e4020c624a7fb14d11b1422578df0ee2c097a0b9/68747470733a2f2f6b6f6d617265762e636f6d2f67687076632f3f757365726e616d653d6b616c65646169303639266c6162656c3d50726f66696c65253230766965777326636f6c6f723d306537356236267374796c653d666c6174'


def request_(url):
    try:
        response = requests.get(url)

    except requests.exceptions.RequestException as e:
        print(f"Error occurred")

# Create a thread pool executor
executor = ThreadPoolExecutor(max_workers = 16)

# Iterate over each date in the range and submit the download task to the executor
for i in tqdm(range(1000), ncols = 120):
    executor.submit(request_, base_url)

# Wait for all tasks to complete
executor.shutdown()