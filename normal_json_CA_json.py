import json
from pprint import pprint
import re

with open("/content/savedata.json", "r") as file:
  data = json.load(file)

json_conversion_dict = {}

rows = data['size']['rows']
cols = data['size']['cols']
date = data['date']

clues = data['clues']
answers = data['answers']

json_conversion_dict['metadata'] = {'date': date, 'rows': rows, 'cols': cols}

across_clue_answer = {}
down_clue_answer = {}

for clue, ans in zip(clues['across'], answers['across']):
  split_clue = clue.split(' ')
  clue_num = split_clue[0][:-1]
  clue_ = " ".join(split_clue[1:])
  clue_ = clue_.replace("[", '').replace("]", '')
  across_clue_answer[clue_num] = [clue_, ans]

for clue, ans in zip(clues['down'], answers['down']):
  split_clue = clue.split(' ')
  clue_num = split_clue[0][:-1]
  clue_ = " ".join(split_clue[1:])
  clue_ = clue_.replace("[", '').replace("]", '')
  down_clue_answer[clue_num] = [clue_, ans]

json_conversion_dict['clues'] = {'across' : across_clue_answer, 'down' : down_clue_answer}

grid_info = data['grid']
grid_num = data['gridnums']

grid_info_list = []
for i in range(rows):
  row_list = []
  for j in range(cols):
    if grid_info[i * rows + j] == '.':
      row_list.append('BLACK')
    else:
      row_list.append([str(grid_num[i * rows + j]), grid_info[i * rows + j]])
  grid_info_list.append(row_list)
  
json_conversion_dict['grid'] = grid_info_list

pprint(json_conversion_dict)