import datetime
import subprocess
import puz
import glob
import os
import json

def puz_to_json(fname):
    """ Converts a puzzle in .puz format to .json format
    """
    p = puz.read(fname)
    numbering = p.clue_numbering()

    grid = [[None for _ in range(p.width)] for _ in range(p.height)]
    for row_idx in range(p.height):
        cell = row_idx * p.width
        row_solution = p.solution[cell:cell + p.width]
        for col_index, item in enumerate(row_solution):
            if p.solution[cell + col_index:cell + col_index + 1] == '.':
                grid[row_idx][col_index] = 'BLACK'
            else:
                grid[row_idx][col_index] = ["", row_solution[col_index: col_index + 1]]

    across_clues = {}
    for clue in numbering.across:
        answer = ''.join(p.solution[clue['cell'] + i] for i in range(clue['len']))
        across_clues[str(clue['num'])] = [clue['clue'] + ' ', ' ' + answer]
        grid[int(clue['cell'] / p.width)][clue['cell'] % p.width][0] = str(clue['num'])

    down_clues = {}
    for clue in numbering.down:
        answer = ''.join(p.solution[clue['cell'] + i * numbering.width] for i in range(clue['len']))
        down_clues[str(clue['num'])] = [clue['clue'] + ' ', ' ' + answer]
        grid[int(clue['cell'] / p.width)][clue['cell'] % p.width][0] = str(clue['num'])


    mydict = {'metadata': {'date': None, 'rows': p.height, 'cols': p.width}, 'clues': {'across': across_clues, 'down': down_clues}, 'grid': grid}
    return mydict

def download_crossword(publication, publication_keyword, start_date, end_date, output_dir):
    if not os.path.exists(os.path.join(output_dir, publication)):
        os.mkdir(os.path.join(output_dir, publication))

    current_date = start_date

    while current_date <= end_date:
        formatted_date_str = current_date.strftime("%m/%d/%y")
        output_date_str = current_date.strftime("%m-%d-%Y")

        try:
            output_file_path = f"{output_dir}/{publication}/crossword_{output_date_str}.puz"
            command = f'xword-dl {publication_keyword} --date {formatted_date_str} -o {output_file_path}'
            subprocess.run(command, shell = True, check = True, stdout = subprocess.PIPE, text = True)
            print(f"Successfully downloaded for {publication}: Filename - crossword_{output_date_str}.puz")
        except:
            print(f"Error in downloading crossword for: {output_date_str}")

        current_date += datetime.timedelta(days = 1)

def convert_to_json(publication, puz_output_dir, json_output_dir):

    all_puz_files = glob.glob(os.path.join(puz_output_dir, publication, '*.puz'))

    if not os.path.exists(os.path.join(json_output_dir, publication)):
        os.mkdir(os.path.join(json_output_dir, publication))

    for puz_file_path in all_puz_files:
        json_output_path = puz_file_path.replace('.puz', '.json').replace('puz', 'json')
        json_data = puz_to_json(puz_file_path)
        with open(json_output_path, 'w') as f:
            json.dump(json_data, f)


publication_keyword = "lat"
publication = 'the-LA-times'
start_date = datetime.date(2023, 8, 1)
end_date= datetime.date(2023, 12, 31)
puz_output_dir = "./puz"
json_output_dir = "./json"

download_crossword(publication, publication_keyword, start_date, end_date, puz_output_dir)
convert_to_json(publication, puz_output_dir, json_output_dir)