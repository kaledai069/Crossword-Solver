import numpy as np
import cv2

def draw_grid(data, grid_size, overlay_truth_matrix, grid_num_matrix, accu_list, all_clue_info, wrong_clues, puzzle_date = None):
    rows, cols = grid_size
    cell_size = 38
    padding_w = 10

    BOX_OFFSET = 15

    wrong_A_num, wrong_D_num = wrong_clues

    padding_h = 60
    width = cols * cell_size + 2 * padding_w
    height = rows * cell_size + 2 * padding_h

    image = np.ones((height, width, 3), dtype=np.uint8) * 255
    font_scale = 0.65
    font_thickness = 1
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(rows):
        for j in range(cols):
            cell_value = data[i][j]
            cell_x = j * cell_size + padding_w
            cell_y = i * cell_size + padding_h - BOX_OFFSET

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.65
            font_thickness = 1
            
            if cell_value == 0:
                cv2.rectangle(image, (cell_x, cell_y), (cell_x + cell_size, cell_y + cell_size), (0, 0, 0), -1)  # Fill the cell with black
            elif overlay_truth_matrix[i][j] == 1:
                text_size = cv2.getTextSize(cell_value, font, font_scale, font_thickness)[0]
                text_x = cell_x + (cell_size - text_size[0]) // 2
                text_y = cell_y + (cell_size + text_size[1]) // 2
                cv2.rectangle(image, (cell_x, cell_y), (cell_x + cell_size, cell_y + cell_size), (63, 27, 196), -1)
                cv2.putText(image, cell_value, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
            else:
                text_size = cv2.getTextSize(cell_value, font, font_scale, font_thickness)[0]
                text_x = cell_x + (cell_size - text_size[0]) // 2
                text_y = cell_y + (cell_size + text_size[1]) // 2
                cv2.putText(image, cell_value, (text_x, text_y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
            
            if grid_num_matrix[i][j] != '-':
                grid_num = grid_num_matrix[i][j]
                
                grid_num_x = cell_x + 2
                grid_num_y = cell_y + 10
                cv2.putText(image, grid_num, (grid_num_x, grid_num_y), font, 0.28, (0, 0, 0, 96), 1, cv2.LINE_AA)
    
    letter_accuracy_text = f"Letter Accuracy: {accu_list[0]:.2f} %"
    text_size = cv2.getTextSize(letter_accuracy_text, font, font_scale, font_thickness)[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    t_x = padding_w
    t_y = 30
    cv2.putText(image, letter_accuracy_text, (t_x, t_y), font, 0.65, (0, 0, 0), font_thickness, cv2.LINE_AA)

    word_accuracy_text = f"Word Accuracy: {accu_list[1]:.2f} %"
    word_text_size = cv2.getTextSize(word_accuracy_text, font, font_scale, font_thickness)[0]
    font = cv2.FONT_HERSHEY_DUPLEX
    t_x = width // 2 + (rows * cell_size) // 2 - word_text_size[0]
    t_y = 30
    cv2.putText(image, word_accuracy_text, (t_x, t_y), font, 0.65, (0, 0, 0), font_thickness, cv2.LINE_AA)

    text_limit = 500

    y_start_ind = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(rows + 1):
        y = i * cell_size + padding_h - BOX_OFFSET
        cv2.line(image, (padding_w, y), (width - padding_w, y), (0, 0, 0), 1)

    for j in range(cols + 1):
        x = j * cell_size + padding_w 
        cv2.line(image, (x, padding_h - BOX_OFFSET), (x, height - padding_h - BOX_OFFSET), (0, 0, 0), 1)
        
    # Draw a border around the grid
    border_thickness = 2  # You can adjust this as needed
    cv2.rectangle(image, (padding_w, padding_h - BOX_OFFSET), (width - padding_w - 1, height - padding_h - 1 - BOX_OFFSET), (0, 0, 0), border_thickness)

    # Display the grid with characters, padding, and inner grid lines
#     cv2.imshow('Solved Crossword', image)
    #     month, day, year = puzzle_date.split('/')
#     output_file_path = f"./solved_crosswords/crossword_{month}-{day}-{year}.jpg"
#     print(output_file_path)
#     cv2.imwrite(output_file_path, image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
    return image

def get_grid(grid_solution, puzzle, accuracy_list):
    
    # preparing grid
    for i in range(len(grid_solution)):
        for j in range(len(grid_solution[0])):
            if grid_solution[i][j] == '':
                grid_solution[i][j] = 0

    num_rows = puzzle['metadata']['rows']
    num_cols = puzzle['metadata']['cols']

    overlay_truth_matrix = [[0] * num_cols for _ in range(num_rows)]
    grid_num_matrix = [["-"] * num_cols for _ in range(num_rows)]
    gold_grid_info = puzzle['grid']

    wrong_clues_list = []

    for i in range(num_rows):
        for j in range(num_cols):
            cell_info = gold_grid_info[i][j]
            cell_char = cell_info[1]
            cell_num = cell_info[0]
            
            if cell_num != 'BLACK':
                grid_num_matrix[i][j] = cell_num
                
            if grid_solution[i][j] != cell_char and cell_info != 'BLACK':
                start_i = i
                start_j = j
                if cell_num == '':
                    while start_j > -1:
                        start_j -= 1
                        if gold_grid_info[i][start_j][0] != '':
                            wrong_clues_list.append(gold_grid_info[i][start_j][0] + " A")
                            break
                    while start_i > - 1:
                        start_i -= 1
                        if gold_grid_info[start_i][j][0] != '':
                            wrong_clues_list.append(gold_grid_info[start_i][j][0]+ " D")
                            break
                else:
                    contained_grid_num = gold_grid_info[i][j][0]
                    if contained_grid_num in puzzle['clues']['across'].keys():
                        wrong_clues_list.append(contained_grid_num +" A")

                    if contained_grid_num in puzzle['clues']['down'].keys():
                        wrong_clues_list.append(contained_grid_num + " D")
                overlay_truth_matrix[i][j] = 1

    wrong_A_num = [x.split(' ')[0] for x in list(set(wrong_clues_list)) if x.split(' ')[1] == 'A']
    wrong_D_num = [x.split(' ')[0] for x in list(set(wrong_clues_list)) if x.split(' ')[1] == 'D']

    wrong_clues = [wrong_A_num, wrong_D_num]

    all_clues = puzzle['clues']
    across_clue_data = []
    down_clue_data = []

    for dim in ['across', 'down']:
        for key in all_clues[dim].keys():
            clue = all_clues[dim][key][0]
            if dim == 'across':
                across_clue_data.append([key, clue])
            else:
                down_clue_data.append([key, clue])

    all_clue_info = [across_clue_data, down_clue_data]

    grid_img = draw_grid(grid_solution, [num_rows, num_cols], overlay_truth_matrix, grid_num_matrix, accuracy_list, all_clue_info, wrong_clues)
    return grid_img