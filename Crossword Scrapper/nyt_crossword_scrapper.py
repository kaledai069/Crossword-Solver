import json
import requests
import datetime
import os

def json_CA_json_converter(json_file_path, is_path):
    try:
        if is_path:
            with open(json_file_path, "r") as file:
                data = json.load(file)
        else:
            data = json_file_path

        json_conversion_dict = {}

        rows = data["size"]["rows"]
        cols = data["size"]["cols"]
        date = data["date"]

        clues = data["clues"]
        answers = data["answers"]

        json_conversion_dict["metadata"] = {"date": date, "rows": rows, "cols": cols}

        across_clue_answer = {}
        down_clue_answer = {}

        for clue, ans in zip(clues["across"], answers["across"]):
            split_clue = clue.split(" ")
            clue_num = split_clue[0][:-1]
            clue_ = " ".join(split_clue[1:])
            clue_ = clue_.replace("[", "").replace("]", "")
            across_clue_answer[clue_num] = [clue_, ans]

        for clue, ans in zip(clues["down"], answers["down"]):
            split_clue = clue.split(" ")
            clue_num = split_clue[0][:-1]
            clue_ = " ".join(split_clue[1:])
            clue_ = clue_.replace("[", "").replace("]", "")
            down_clue_answer[clue_num] = [clue_, ans]

        json_conversion_dict["clues"] = {
            "across": across_clue_answer,
            "down": down_clue_answer,
        }

        grid_info = data["grid"]
        grid_num = data["gridnums"]

        grid_info_list = []
        for i in range(rows):
            row_list = []
            for j in range(cols):
                if grid_info[i * rows + j] == ".":
                    row_list.append("BLACK")
                else:
                    if grid_num[i * rows + j] == 0:
                        row_list.append(["", grid_info[i * rows + j]])
                    else:
                        row_list.append(
                            [str(grid_num[i * rows + j]), grid_info[i * rows + j]]
                        )
            grid_info_list.append(row_list)

        json_conversion_dict["grid"] = grid_info_list

        return json_conversion_dict
    
    except:
        print("ERROR has occured.")

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
        grid_data = json.loads(jsonText)
        puzzle_data = json_CA_json_converter(grid_data, False)
        for dim in ['across', 'down']:
            for grid_num in puzzle_data['clues'][dim].keys():
                clue_answer_list = puzzle_data['clues'][dim][grid_num]
                clue_section = clue_answer_list[0]
                ans_section = clue_answer_list[1]
                clue_section = clue_section.replace("&quot;", "'").replace("&#39;", "'")
                puzzle_data['clues'][dim][grid_num] = [clue_section, ans_section]
        return puzzle_data

    else:
        print(f"Request failed with status code {response.status_code}.")

publication_keyword = "nyt"
publication = 'new-york-times'
start_date = datetime.date(2024, 1, 8)
end_date= datetime.date(2024, 1, 8)
puz_output_dir = "./puz"
json_output_dir = "./json"

if not os.path.exists(os.path.join(json_output_dir, publication)):
    os.mkdir(os.path.join(json_output_dir, publication))

current_date = start_date

while current_date <= end_date:
    formatted_date_str = current_date.strftime("%Y/%m/%d")
    output_date_str = current_date.strftime("%m/%d/%Y")
    try:
        puzzle_data = fetch_nyt_crossword(formatted_date_str)
        output_path = os.path.join(json_output_dir, publication, f"crossword_{output_date_str.replace('/', '-')}.json")
        print("Successfully crossword fetched for: ", formatted_date_str)
        with open(output_path, 'w') as f:
            json.dump(puzzle_data, f)
    except:
        print("An error as occured for ", formatted_date_str)
    current_date += datetime.timedelta(days = 1)