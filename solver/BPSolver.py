"""
A solver based on loopy belief propagation. We more or less follow the procedure in https://www.aaai.org/Papers/AAAI/1999/AAAI99-023.pdf
where we make a variable for each fill (BPVar) and for each letter (BPCell). 
BPVars connect to BPCells and vice versa in a connected graph, and we propagate alternating between the two, 
being careful to preserve the directionality of the updates (e.g., the update from A to B should take into account A's neighbors other than B.)
The paper's procedure of infilling / finding words not in the original provided candidate list is not implemented.
We instead using iterative search after the fact by replacing characters one-by-one in our solution.
"""
import math
import string
import re
import time

from collections import defaultdict
from copy import deepcopy

import numpy as np
from scipy.special import log_softmax, softmax
from tqdm import trange, tqdm

from solver.Utils import print_grid, get_word_flips
from solver.Solver import Solver
from models import setup_t5_reranker, t5_reranker_score_with_clue

# # our answer set
# answer_set = set()
# with open(r'./checkpoints/all_answer_list.tsv', 'r') as rf: 
#     for line in rf:
#         w = ''.join([c.upper() for c in (line.split('\t')[-1]).upper() if c in string.ascii_uppercase])
#         answer_set.add(w)

# the probability of each alphabetical character in the crossword
UNIGRAM_PROBS = [('A', 0.0897379968935765), ('B', 0.02121248877769636), ('C', 0.03482206634145926), ('D', 0.03700942543460491), ('E', 0.1159773210750429), ('F', 0.017257461694024614), ('G', 0.025429024796296124), ('H', 0.033122967601502), ('I', 0.06800036223479956), ('J', 0.00294611331754349), ('K', 0.013860682888259786), ('L', 0.05130800574373874), ('M', 0.027962776827660175), ('N', 0.06631994270448001), ('O', 0.07374646543246745), ('P', 0.026750756212433214), ('Q', 0.001507814175439393), ('R', 0.07080460813737305), ('S', 0.07410988246048224), ('T', 0.07242993582154593), ('U', 0.0289272388037645), ('V', 0.009153522059555467), ('W', 0.01434705167591524), ('X', 0.003096729223103298), ('Y', 0.01749958208224007), ('Z', 0.002659777584995724)]

# the LETTER_SMOOTHING_FACTOR controls how much we interpolate with the unigram LM. TODO this should be tuned. 
# Right now it is set according to the probability that the answer is not in the answer set
LETTER_SMOOTHING_FACTOR = [0.0, 0.0, 0.04395604395604396, 0.0001372495196266813, 0.0005752186417796561, 0.0019841824329989103, 0.0048042463338563764, 0.013325257419745608, 0.027154447774285505, 0.06513517299341645, 0.12527790128946198, 0.22003002358996354, 0.23172376584839494, 0.254873006497342, 0.3985086992543496, 0.2764976958525346, 0.672645739910314, 0.6818181818181818, 0.8571428571428571, 0.8245614035087719, 0.8, 0.71900826446281, 0.0]

T5_COUNTER = 0

class BPVar:
    def __init__(self, name, variable, candidates, cells): 
        '''
            name - name of the filling position along with orientation like 1A, 2A, 1D and such
            variable - crossword.variables.value field which contains as below
            {'clue': 'Lettuce variety', 'gold': 'BIBB', 'cells': [(0, 0), (0, 1), (0, 2), (0, 3)], 'crossing': ['1D', '2D', '3D', '4D']}
            candidates - {'words', 'bit_array', 'weights'} --> bit_array is used to create some sort of distribution with the answers generated for that particular clue
            cells - [BPCell(1A), BPCell(2A), BPCell(3A)]
        '''

        self.name = name # name of the filling, which goes by 1A, 2A, 1D and such
        cells_by_position = {}
        for cell in cells:
            cells_by_position[cell.position] = cell # this line is for (0, 0) -> 1A (BPCell)
            cell._connect(self) # this line is for connecting which BPCell is connected to which BPVariable

        self.length = len(cells) # length of the answer for that particular filling or variable
        self.ordered_cells = [cells_by_position[pos] for pos in variable['cells']] #[(0, 0), (0, 1), (0, 2)] -> [BPCell(1A), BPCell(2A), BPCell(3A)]{Ordered Cells}
        self.candidates = candidates # [words, bit_array, weights] for each filling produced by the First Pass Model
        self.words = self.candidates['words'] # all the possible answers that could fit in that particular across or down clue (less than the maximum candidates )
        
        self.word_indices = np.array([[string.ascii_uppercase.index(l) for l in fill] for fill in self.words]) # words x length of letter indices
        '''
            example: words = ['SUNDAY', 'MONDAY']
                     word_indices -> array([[18, 20, 13,  3,  0, 24],
                                           [12, 14, 13,  3,  0, 24]])
        '''
        self.scores = -np.array([self.candidates['weights'][fill] for fill in self.candidates['words']]) # the incoming 'weights' are costs
        self.prior_log_probs = log_softmax(self.scores)
        self.log_probs = log_softmax(self.scores)
        self.directional_scores = [np.zeros(len(self.log_probs)) for _ in range(len(self.ordered_cells))]
        '''
        len(self.ordered_cells) is the length of field of required answer
        directional_scores zeros "words x length of letter to be filled"
        '''
    
    def _propagate_to_var(self, other, belief_state):
        assert other in self.ordered_cells
        other_idx = self.ordered_cells.index(other)
        letter_scores = belief_state
        self.directional_scores[other_idx] = letter_scores[self.word_indices[:, other_idx]]
    
    def _postprocess(self, all_letter_probs):
        # unigram smoothing
        unigram_probs = np.array([x[1] for x in UNIGRAM_PROBS])
        for i in range(len(all_letter_probs)):
            all_letter_probs[i] = (1 - LETTER_SMOOTHING_FACTOR[self.length]) * all_letter_probs[i] + LETTER_SMOOTHING_FACTOR[self.length] * unigram_probs
        return all_letter_probs
    
    def sync_state(self):
        self.log_probs = log_softmax(sum(self.directional_scores) + self.prior_log_probs)
    
    def propagate(self): # very first task in the main loop - applied across all the variables for that particular crossword
        all_letter_probs = []
        for i in range(len(self.ordered_cells)):
            word_scores = self.log_probs - self.directional_scores[i]
            word_probs = softmax(word_scores)
            letter_probs = (self.candidates['bit_array'][:, i] * np.expand_dims(word_probs, axis=0)).sum(axis=1) + 1e-8
            all_letter_probs.append(letter_probs)
        all_letter_probs = self._postprocess(all_letter_probs) # unigram postprocessing
        for i, cell in enumerate(self.ordered_cells):
            cell._propagate_to_cell(self, np.log(all_letter_probs[i]))


class BPCell:
    def __init__(self, position, clue_pair):
        self.crossing_clues = clue_pair
        self.position = tuple(position)
        self.letters = list(string.ascii_uppercase)
        self.log_probs = np.log(np.array([1./len(self.letters) for _ in range(len(self.letters))]))
        self.crossing_vars = []
        self.directional_scores = []
        self.prediction = {}
    
    def _connect(self, other):
        self.crossing_vars.append(other)
        self.directional_scores.append(None)
        assert len(self.crossing_vars) <= 2 # this particular line is connected to the rules of American Crosswords, as a single cell atmost be part of a across and down clue

    def _propagate_to_cell(self, other, belief_state):
        assert other in self.crossing_vars
        other_idx = self.crossing_vars.index(other)
        self.directional_scores[other_idx] = belief_state
    
    def sync_state(self):
        self.log_probs = log_softmax(sum(self.directional_scores))

    def propagate(self):
        assert len(self.crossing_vars) == 2
        # try:
        for i, v in enumerate(self.crossing_vars):
            v._propagate_to_var(self, self.directional_scores[1-i])
        # except IndexError:
            # pass


class BPSolver(Solver):
    def __init__(self, 
                 crossword, 
                 model_path,
                 ans_tsv_path,
                 dense_embd_path,
                 reranker_path,
                 reranker_model_type = 't5-small',
                 max_candidates = 40000,
                 score_improvement_threshold = 0.5,
                 process_id=0,
                 **kwargs):
        super().__init__(crossword,
                         model_path,
                         ans_tsv_path,
                         dense_embd_path, 
                         max_candidates = max_candidates,
                         process_id=process_id,
                         **kwargs)
        self.crossword = crossword
        self.reranker_path = reranker_path
        self.reranker_model_type = reranker_model_type
        self.score_improvement_threshold = score_improvement_threshold

         # our answer set
        self.answer_set = set()
        with open(ans_tsv_path, 'r') as rf: 
            for line in rf:
                w = ''.join([c.upper() for c in (line.split('\t')[-1]).upper() if c in string.ascii_uppercase])
                self.answer_set.add(w)

        self.reset()
    
    # the first function after solving the crossword using first pass model, begins with the reset local function 
    def reset(self):
        self.bp_cells = []
        self.bp_cells_by_clue = defaultdict(lambda: [])

        # defining every cells (one with letter in it) in the crossword
        # crossword.grid_cells.items() -> (0, 0): ['1A', '1D'], key -> Position, value -> Clue Positional Info

        for position, clue_pair in self.crossword.grid_cells.items():
            cell = BPCell(position, clue_pair) # need to seriously check this 
            self.bp_cells.append(cell)
            for clue in clue_pair:
                self.bp_cells_by_clue[clue].append(cell)

        self.bp_vars = []
        for key, value in self.crossword.variables.items():
            var = BPVar(key, value, self.candidates[key], self.bp_cells_by_clue[key])
            self.bp_vars.append(var)

    def extract_float(self, input_string):
        pattern = r"\d+\.\d+"
        matches = re.findall(pattern, input_string)
        float_numbers = [float(match) for match in matches]
        return float_numbers
    
    def solve(self, num_iters = 10, iterative_improvement_steps = 5, return_greedy_states = False, return_ii_states = False):
        global T5_COUNTER

        output_results = {}
        # run solving for num_iters iterations
        print('\nBeginning Belief Propagation iteration steps')
        for _ in tqdm(range(num_iters), ncols = 100):
            for var in self.bp_vars:
                var.propagate()
            for cell in self.bp_cells:
                cell.sync_state()
            for cell in self.bp_cells:
                cell.propagate()
            for var in self.bp_vars:
                var.sync_state()
        print('Belief Propagation iteration complete\n')
       
        # Get the current based grid based on greedy selection from the marginals
        if return_greedy_states:
            grid, all_grids = self.greedy_sequential_word_solution(return_grids = True)
        else:
            grid = self.greedy_sequential_word_solution()
            all_grids = []

        output_results['first pass model'] = {}
        output_results['first pass model']['grid'] = grid
        
        _, accu_log = self.evaluate(grid, False)
        [ori_letter_accu, ori_word_accu] = self.extract_float(accu_log)
        output_results['first pass model']['letter accuracy'] = ori_letter_accu
        output_results['first pass model']['word accuracy'] = ori_word_accu

        original_grid_solution = deepcopy(grid)

        print(f"Before Iterative Improvement with t5-small: {accu_log}")

        if iterative_improvement_steps < 1 or ori_letter_accu == 100.0 or ori_word_accu < 85.0:
            # if the letter accuracy reaches maximum leave this here without further second pass model 
            if return_greedy_states or return_ii_states:
                return output_results, all_grids
            else:
                return output_results
        
        #loading the ByT5 reranker model
        print(self.reranker_model_type)
        self.reranker, self.tokenizer = setup_t5_reranker(self.reranker_path, self.reranker_model_type)
        
        output_results['second pass model'] = {}
        output_results['second pass model']['all grids'] = []
        output_results['second pass model']['all letter accuracy'] = []
        output_results['second pass model']['all word accuracy'] = []
        intermediate_II_results = []

        second_pass_start_time = time.time()
        print('-'*100)
        print("Starting Iterative Improvement with T5-small")
        for i in range(iterative_improvement_steps):
            grid, did_iterative_improvement_make_edit = self.iterative_improvement(grid)
            _, accu_log = self.evaluate(grid, False)
            [temp_letter_accu, temp_word_accu] = self.extract_float(accu_log)

            # track the iterative ongoing grid accuracies
            intermediate_II_results.append([temp_letter_accu, temp_word_accu])
            print(f"{i+1}th iteration: {accu_log}")

            # saving output results
            output_results['second pass model']['all grids'].append(grid)
            output_results['second pass model']['all letter accuracy'].append(temp_letter_accu)
            output_results['second pass model']['all word accuracy'].append(temp_word_accu)
            
            # get the hell out of the II, if the consecutive improvement doesn't shows much result
            if len(intermediate_II_results) > 1:
                former_accuracies = sum(intermediate_II_results[-2])
                later_accuracies = sum(intermediate_II_results[-1])

                if later_accuracies <= former_accuracies:
                    break

            if not did_iterative_improvement_make_edit or temp_letter_accu == 100.0:
                break
            if return_ii_states:
                all_grids.append(deepcopy(grid))

        # track the time for the second pass model only
        second_pass_end_time = time.time()
        _, accu_log = self.evaluate(grid, False)
        print(f"\nAfter Iterative Improvement with t5-small: {accu_log}")

        if temp_letter_accu < ori_letter_accu or temp_word_accu < ori_word_accu:
            print("Reverting the changes due to worse output from second pass iterative handle. ")
            grid = deepcopy(original_grid_solution)
            _, accu_log = self.evaluate(grid, False)
            print(f"\nFinal Accuracy Stat: {accu_log}")

        print(f"Total time taken for t5-small: {second_pass_end_time - second_pass_start_time} seconds")
        
        temp_lett_accu_list = output_results['second pass model']['all letter accuracy'].copy()
        ii_max_index = temp_lett_accu_list.index(max(temp_lett_accu_list))

        output_results['second pass model']['final grid'] = output_results['second pass model']['all grids'][ii_max_index]
        output_results['second pass model']['final letter'] = output_results['second pass model']['all letter accuracy'][ii_max_index]
        output_results['second pass model']['final word'] = output_results['second pass model']['all word accuracy'][ii_max_index]
        
        print("-"*100)
        print("\nStarting last refinement step: ")

        first_pass_grid = deepcopy(original_grid_solution)
        second_pass_grid = output_results['second pass model']['final grid']
        did_some_improvement = False

        possible_wrong_cell_list = []
        for i in range(self.crossword.size[0]):
            for j in range(self.crossword.size[1]):
                if first_pass_grid[i][j] != second_pass_grid[i][j]:
                    possible_wrong_cell_list.append([(i, j), first_pass_grid[i][j], second_pass_grid[i][j]])    

        improvement_cells = {}
        for wrong_cell in possible_wrong_cell_list:
            cell_position = wrong_cell[0]

            improvement_cells[cell_position] = {}
            improvement_cells[cell_position]['fillings'] = []
            improvement_cells[cell_position]['cells'] = []
            improvement_cells[cell_position]['clues'] = []
            
            improvement_cells[cell_position]['f_pass value'] = wrong_cell[1]
            improvement_cells[cell_position]['s_pass value'] = wrong_cell[2]
            for key, var in self.crossword.variables.items():
                if cell_position in var['cells']:
                    improvement_cells[cell_position]['fillings'].append(key)
                    improvement_cells[cell_position]['cells'].append(var['cells'])
                    improvement_cells[cell_position]['clues'].append(var['clue'])
            
        improvement_captures = {}

        temp_grid = deepcopy(second_pass_grid)
        for pos, data in improvement_cells.items():
            improvement_captures[pos] = []
            before_improvement_grid = deepcopy(temp_grid)

            for i, answer_pos in enumerate(data['cells']):
                previous_ans = self.get_grid_ans(answer_pos, before_improvement_grid)

                temp_grid[pos[0]][pos[1]] = data['f_pass value']
                new_ans = self.get_grid_ans(answer_pos, temp_grid)

                improvement_captures[pos].append([data['clues'][i], (previous_ans, new_ans)])
            
            if self.do_improve(improvement_captures[pos], self.reranker, self.tokenizer):
                did_some_improvement = True
            else:
                temp_grid = deepcopy(before_improvement_grid)
    
        if did_some_improvement:
            output_results['second pass model']['last grid'] = temp_grid
            _, accu_log = self.evaluate(temp_grid, False)
            [temp_letter_accu, temp_word_accu] = self.extract_float(accu_log)
            output_results['second pass model']['last letter accuracy'] = temp_letter_accu
            output_results['second pass model']['last word accuracy'] = temp_word_accu
            print(f"\nAfter Refinement: {accu_log}")

        if return_greedy_states or return_ii_states:
            return output_results, all_grids
        else:
            print(f"Total times the second pass model is called: {T5_COUNTER}")
            output_results['second pass model']['call count'] = T5_COUNTER
            return output_results
        
    
    def do_improve(self, refinement_list, model, tokenizer):
        improvement_count = 0
        device = model.device
        for data in refinement_list:
            clue = data[0]
            ans_pair = data[1]
            input = tokenizer(["Q: " + clue], return_tensors = 'pt')['input_ids'].to(device)
            label_b = tokenizer([ans_pair[0].lower()], return_tensors = 'pt')['input_ids'].to(device)
            label_a = tokenizer([ans_pair[1].lower()], return_tensors = 'pt')['input_ids'].to(device)

            output_b = model(input, labels = label_b)
            loss_b = -output_b.loss.item() * len(ans_pair[0])

            output_a = model(input, labels = label_a)
            loss_a = -output_a.loss.item() * len(ans_pair[0])

            if loss_a > loss_b:
                improvement_count += 1

        return improvement_count == 2
    
    def get_grid_ans(self, ans_positions, grid):
        word = ''
        for pos in ans_positions:
            word += grid[pos[0]][pos[1]]
        return word

    def get_candidate_replacements(self, uncertain_answers, grid):
        # find alternate answers for all the uncertain words
        candidate_replacements = []
        replacement_id_set = set()

        # check against dictionaries
        for clue in uncertain_answers.keys():
            initial_word = uncertain_answers[clue]
            # print("INITIAL WORD: ", initial_word)
            clue_flips = get_word_flips(initial_word, 10) # flip then segment
            # print(clue_flips)
            clue_positions = [key for key, value in self.crossword.variables.items() if value['clue'] == clue]
            # print(clue_positions)
            for clue_position in clue_positions:
                cells = sorted([cell for cell in self.bp_cells if clue_position in cell.crossing_clues], key=lambda c: c.position)
                if len(cells) == len(initial_word):
                    break
            for flip in clue_flips:
                if len(flip) != len(cells):
                    import pdb; pdb.set_trace()
                assert len(flip) == len(cells)
                for i in range(len(flip)):
                    if flip[i] != initial_word[i]:
                        candidate_replacements.append([(cells[i], flip[i])])
                        break

        # also add candidates based on uncertainties in the letters, e.g., if we said P but G also had some probability, try G too
        for cell_id, cell in enumerate(self.bp_cells): 
            probs = np.exp(cell.log_probs)
            above_threshold = list(probs > 0.01)
            new_characters = ['ABCDEFGHIJKLMNOPQRSTUVWXYZ'[i] for i in range(26) if above_threshold[i]]
            # used = set()
            # new_characters = [x for x in new_characters if x not in used and (used.add(x) or True)] # unique the set
            new_characters = [x for x in new_characters if x != grid[cell.position[0]][cell.position[1]]] # ignore if its the same as the original solution
            if len(new_characters) > 0: 
                for new_character in new_characters:
                    id = '_'.join([str(cell.position), new_character])
                    if id not in replacement_id_set:
                        candidate_replacements.append([(cell, new_character)])
                    replacement_id_set.add(id)

        # create composite flips based on things in the same row/column
        composite_replacements = []
        for i in range(len(candidate_replacements)):
            for j in range(i+1, len(candidate_replacements)):
                flip1, flip2 = candidate_replacements[i], candidate_replacements[j]
                if flip1[0][0] != flip2[0][0]:
                    if len(set(flip1[0][0].crossing_clues + flip2[0][0].crossing_clues)) < 4: # shared clue
                        composite_replacements.append(flip1 + flip2)

        candidate_replacements += composite_replacements

        #print('\ncandidate replacements')
        for cr in candidate_replacements:
            modified_grid = deepcopy(grid)
            for cell, letter in cr:
                modified_grid[cell.position[0]][cell.position[1]] = letter
            variables = set(sum([cell.crossing_vars for cell, _ in cr], []))
            for var in variables:
                original_fill = ''.join([grid[cell.position[0]][cell.position[1]] for cell in var.ordered_cells])
                modified_fill = ''.join([modified_grid[cell.position[0]][cell.position[1]] for cell in var.ordered_cells])
                #print('original:', original_fill, 'modified:', modified_fill)
        
        return candidate_replacements

    def get_uncertain_answers(self, grid):
        original_qa_pairs = {} # the original puzzle preds that we will try to improve
        # first save what the argmax word-level prediction was for each grid cell just to make life easier
        for var in self.crossword.variables:
            # read the current word off the grid  
            cells = self.crossword.variables[var]["cells"]
            word = []
            for cell in cells:
                word.append(grid[cell[0]][cell[1]])
            word = ''.join(word)
            for cell in self.bp_cells: # loop through all cells
                if cell.position in cells: # if this cell is in the word we are currently handling
                    # save {clue, answer} pair into this cell
                    cell.prediction[self.crossword.variables[var]['clue']] = word
                    original_qa_pairs[self.crossword.variables[var]['clue']] = word

        uncertain_answers = {}

        # find uncertain answers
        # right now the heuristic we use is any answer that is not in the answer set
        for clue in original_qa_pairs.keys():
            if original_qa_pairs[clue] not in self.answer_set:
                uncertain_answers[clue] = original_qa_pairs[clue]

        return uncertain_answers
    
    def score_grid(self, grid):
        global T5_COUNTER
        global NUM_CLUE_ANSWER
        clues = []
        answers = []
        for clue, cells in self.bp_cells_by_clue.items():
            letters = ''.join([grid[cell.position[0]][cell.position[1]] for cell in sorted(list(cells), key=lambda c: c.position)])
            clues.append(self.crossword.variables[clue]['clue'])
            answers.append(letters)
        
        scores, t5_run_count = t5_reranker_score_with_clue(self.reranker, self.tokenizer, self.reranker_model_type, clues, answers)
        T5_COUNTER += t5_run_count
        return sum(scores)
    
    def greedy_sequential_word_solution(self, return_grids = False):
        all_grids = []
        # after we've run BP, we run a simple greedy search to get the final. 
        # We repeatedly pick the highest-log-prob candidate across all clues which fits the grid, and fill it. 
        # at the end, if you have any cells left (due to missing gold candidates) just fill it with the argmax on that letter.
        cache = [(deepcopy(var.words), deepcopy(var.log_probs)) for var in self.bp_vars]

        grid = [["" for _ in row] for row in self.crossword.letter_grid]
        unfilled_cells = set([cell.position for cell in self.bp_cells])
        for var in self.bp_vars:
            # postprocess log probs to estimate probability that you don't have the right candidate
            var.log_probs = var.log_probs + math.log(1 - LETTER_SMOOTHING_FACTOR[var.length])
        best_per_var = [var.log_probs.max() for var in self.bp_vars]
        while not all([x is None for x in best_per_var]):
            all_grids.append(deepcopy(grid))
            best_index = best_per_var.index(max([x for x in best_per_var if x is not None]))
            best_var = self.bp_vars[best_index]
            best_word = best_var.words[best_var.log_probs.argmax()]
            # print('greedy filling in', best_word)
            for i, cell in enumerate(best_var.ordered_cells):
                letter = best_word[i]
                grid[cell.position[0]][cell.position[1]] = letter
                if cell.position in unfilled_cells:
                    unfilled_cells.remove(cell.position)
                for var in cell.crossing_vars:
                    if var != best_var:
                        cell_index = var.ordered_cells.index(cell)
                        keep_indices = [j for j in range(len(var.words)) if var.words[j][cell_index] == letter]
                        var.words = [var.words[j] for j in keep_indices]
                        var.log_probs = var.log_probs[keep_indices]
                        var_index = self.bp_vars.index(var)
                        if len(keep_indices) > 0:
                            best_per_var[var_index] = var.log_probs.max()
                        else:
                            best_per_var[var_index] = None
            best_var.words = []
            best_var.log_probs = best_var.log_probs[[]]
            best_per_var[best_index] = None

        for cell in self.bp_cells: 
            if cell.position in unfilled_cells:
                grid[cell.position[0]][cell.position[1]] = string.ascii_uppercase[cell.log_probs.argmax()]
                '''
                some alternative improvement can be done at this point, i think so
                instead of filling the cell with highest valued letter, what if we find the associated variables or filling with the cell, and insert letter such that the joint probability of occuring the answers, or letters in all the associated filling is maximum
                '''

        for var, (words, log_probs) in zip(self.bp_vars, cache): # restore state
            var.words = words
            var.log_probs = log_probs
        if return_grids:
            return grid, all_grids
        else:
            return grid

    def iterative_improvement(self, grid):
        # check the grid for uncertain areas and save those words to be analyzed in local search, aka looking for alternate candidates
        uncertain_answers = self.get_uncertain_answers(grid) 
        # print(uncertain_answers)
        self.candidate_replacements = self.get_candidate_replacements(uncertain_answers, grid)
        # print(len(self.candidate_replacements))
        # print(self.candidate_replacements[:10])

        # print('\nstarting iterative improvement')
        original_grid_score = self.score_grid(grid)
        possible_edits = []
        for replacements in self.candidate_replacements:
            modified_grid = deepcopy(grid)
            for cell, letter in replacements:
                modified_grid[cell.position[0]][cell.position[1]] = letter
            modified_grid_score = self.score_grid(modified_grid)
            # print('candidate edit')
            variables = set(sum([cell.crossing_vars for cell, _ in replacements], []))

            # just to who the original answer, score and modified scores
            for var in variables:
                original_fill = ''.join([grid[cell.position[0]][cell.position[1]] for cell in var.ordered_cells])
                modified_fill = ''.join([modified_grid[cell.position[0]][cell.position[1]] for cell in var.ordered_cells])
                clue_index = list(set(var.ordered_cells[0].crossing_clues).intersection(*[set(cell.crossing_clues) for cell in var.ordered_cells]))[0]
                # print('original:', original_fill, 'modified:', modified_fill)
                # print('gold answer', self.crossword.variables[clue_index]['gold'])
                # print('clue', self.crossword.variables[clue_index]['clue'])
            # print('original score:', original_grid_score, 'modified score:', modified_grid_score)
                
            if modified_grid_score - original_grid_score > self.score_improvement_threshold:
                # print('found a possible edit')
                possible_edits.append((modified_grid, modified_grid_score, replacements))
            # print()
        
        if len(possible_edits) > 0:
            variables_modified = set()
            possible_edits = sorted(possible_edits, key=lambda x: x[1], reverse=True)
            selected_edits = []
            for edit in possible_edits:
                replacements = edit[2]
                variables = set(sum([cell.crossing_vars for cell, _ in replacements], []))
                if len(variables_modified.intersection(variables)) == 0: # we can do multiple updates at once if they don't share clues
                    variables_modified.update(variables)
                    selected_edits.append(edit)

            new_grid = deepcopy(grid)
            for edit in selected_edits:
                # print('\nactually applying edit')
                replacements = edit[2]
                for cell, letter in replacements:
                    new_grid[cell.position[0]][cell.position[1]] = letter
                variables = set(sum([cell.crossing_vars for cell, _ in replacements], []))
                for var in variables:
                    original_fill = ''.join([grid[cell.position[0]][cell.position[1]] for cell in var.ordered_cells])
                    modified_fill = ''.join([new_grid[cell.position[0]][cell.position[1]] for cell in var.ordered_cells])
                    # print('original:', original_fill, 'modified:', modified_fill)
            return new_grid, True
        else:
            return grid, False