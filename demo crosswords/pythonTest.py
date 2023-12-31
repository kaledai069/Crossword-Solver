def returnJSON(grid_formatted,rows,cols):
    grid = []
    grid_nums = []
    across_clue_num = []
    down_clue_num = []
    # if (x,y) is present in these array the cell (x,y) is already accounted as a part of answer of across or down
    in_horizontal = []
    in_vertical = []

    
    num = 0

    for x in range(0, cols ):
        for y in range(0, rows):

            # if the cell is black there's no need to number
            if grid_formatted[x][y] == '.':
                grid_nums.append(0)
                continue

            # if the cell is part of both horizontal and vertical cell then there's no need to number
            horizontal_presence = (x, y) in in_horizontal
            vertical_presence = (x, y) in in_vertical

            # present in both 1 1 
            if horizontal_presence and vertical_presence:
                grid_nums.append(0)
                continue

            # present in one i.e 1 0
            if not horizontal_presence and vertical_presence:
                horizontal_length = 0
                temp_horizontal_arr = []
                # iterate in x direction until the end of the grid or until a black box is found
                while x + horizontal_length < rows and grid_formatted[x + horizontal_length][y] != '.':
                    temp_horizontal_arr.append((x + horizontal_length, y))
                    horizontal_length += 1
                # if horizontal length is greater than 1, then append the temp_horizontal_arr to in_horizontal array
                if horizontal_length > 1:
                    in_horizontal.extend(temp_horizontal_arr)
                    num += 1
                    across_clue_num.append(num)
                    grid_nums.append(num)
                    continue
                grid_nums.append(0)
            # present in one 1 0        
            if not vertical_presence and horizontal_presence:
                # do the same for vertical
                vertical_length = 0
                temp_vertical_arr = []
                # iterate in y direction until the end of the grid or until a black box is found
                while y + vertical_length < cols  and grid_formatted[x][y+vertical_length] != '.':
                    temp_vertical_arr.append((x, y+vertical_length))
                    vertical_length += 1
                # if vertical length is greater than 1, then append the temp_vertical_arr to in_vertical array
                if vertical_length > 1:
                    in_vertical.extend(temp_vertical_arr)
                    num += 1
                    down_clue_num.append(num)
                    grid_nums.append(num)
                    continue
                grid_nums.append(0)
            
            if(not horizontal_presence and not vertical_presence):

                horizontal_length = 0
                temp_horizontal_arr = []
                # iterate in x direction until the end of the grid or until a black box is found
                while x + horizontal_length < rows  and grid_formatted[x + horizontal_length][y] != '.':
                    temp_horizontal_arr.append((x + horizontal_length, y))
                    horizontal_length += 1
                # if horizontal length is greater than 1, then append the temp_horizontal_arr to in_horizontal array
                                    
                # do the same for vertical
                vertical_length = 0
                temp_vertical_arr = []
                # iterate in y direction until the end of the grid or until a black box is found
                while y + vertical_length < cols  and grid_formatted[x][y+vertical_length] != '.':
                    temp_vertical_arr.append((x, y+vertical_length))
                    vertical_length += 1
                # if vertical length is greater than 1, then append the temp_vertical_arr to in_vertical array
                
                if horizontal_length > 1 and horizontal_length > 1:
                    in_horizontal.extend(temp_horizontal_arr)
                    in_vertical.extend(temp_vertical_arr)
                    num += 1
                    across_clue_num.append(num)
                    down_clue_num.append(num)
                    grid_nums.append(num)
                elif vertical_length > 1:
                    in_vertical.extend(temp_vertical_arr)
                    num += 1
                    down_clue_num.append(num)
                    grid_nums.append(num)
                elif horizontal_length > 1:
                    in_horizontal.extend(temp_horizontal_arr)
                    num += 1
                    across_clue_num.append(num)
                    grid_nums.append(num)
                else:
                    grid_nums.append(0)


    size = { 'rows' : rows,
            'cols' : cols,
            }
    
    
    dict = {
        'size' : size,
        'grid' : grid,
        'gridnums': grid_nums,
        'across_nums': down_clue_num,
        'down_nums' : across_clue_num,
    }
    
    return dict


if __name__ == "__main__":
    grid =  [[' ', ' ', ' ', '.', '.'],
            [' ', ' ', ' ', ' ', '.'],
            [' ', ' ', ' ', ' ', ' '],
            ['.', ' ', ' ', ' ', ' '],
            ['.', '.', ' ', ' ', ' ']]
    rows = 5
    cols = 5
    print(returnJSON(grid,rows,cols))