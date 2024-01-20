import requests
import json

def getGrid(dateStr):

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
        grid_data = json.loads(jsonText) 
        save_file = open("crossword_3_today.json", "w")  
        json.dump(grid_data, save_file, indent = 6)  
        save_file.close()  
        context['data'] = grid_data
        print("Saved into crossword.json")
    else:
        print(f"Request failed with status code {response.status_code}.")

if __name__ == "__main__":
    
    #mm//dd//yy
    getGrid("1/14/2024")