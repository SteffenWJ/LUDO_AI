import numpy as np
import ann_model as ann
import helper_functions as hf
from numpy import random
import glob

import os
import datetime
rng = np.random.default_rng()

#These are the functions that helps to create the genetic alghorithm
#1. Create a population of random networks -> Done by ANN in play game
#2. Create a fitness function -> Done by genetic alghorithm
#3. Create a selection function -> Done by genetic alghorithm
#4. Create a crossover function -> Done by genetic alghorithm
#5. Create a mutation function -> Done by genetic alghorithm
    

class selection_of_pop:
    #This class takes all the populations and spits out the new weights to be used in the next generation
    def get_population(self):
        self.populations = np.sort(self.populations, order='fitness_value')[::-1]
        return self.populations
    
    def print_fitness_values(self):
        print(self.populations['fitness_value'])
        
        
    def __init__(self) -> None:
        self.populations = np.array([], dtype=[('Population', object), ('fitness_value', float)])
        self.lowest_fitnes = 0
        self.full = False
    
    def __call__(self, pop):
        if self.full == False:
            self.populations = np.append(self.populations, np.array([(pop, pop.get_fitness_value_raw())], dtype=[('Population', object), ('fitness_value', float)]))
            if (pop.get_fitness_value_raw() > self.lowest_fitnes):
                self.lowest_fitnes = np.min(self.populations['fitness_value'])
            if self.populations.shape[0] == 6:
                self.full = True
        else:
            print(f"Checking if {pop.get_fitness_value_raw()} is higher than {self.lowest_fitnes}")
            if pop.get_fitness_value_raw() > self.lowest_fitnes:
                print(f"Replacing index: {np.argmin(self.populations['fitness_value'])} with {self.populations[np.argmin(self.populations['fitness_value'])]}")
                print(f"replaced with {pop.get_fitness_value_raw()}") 
                self.populations    = np.delete(self.populations, np.argmin(self.populations['fitness_value']))
                self.populations    = np.append(self.populations, np.array([(pop, pop.get_fitness_value_raw())], dtype=[('Population', object), ('fitness_value', float)]))
                self.lowest_fitnes  = np.min(self.populations['fitness_value'])
                # Error #self.lowest_fitnes  = np.argmin(self.populations['fitness_value'])
                print(f"Lowest Fitness is now {self.lowest_fitnes}")

class Population_object:
    def kill_model(self):
        self.ann_model.clear_the_session()
    
    
    def get_generation(self):
        return self.gen_num
    def get_parent_A(self):
        return self.parent_A
    def get_parent_B(self):
        return self.parent_B
    def calculate_distance(self, max_distance = 57, value = 10):
        A = (self.distance[0]/ max_distance) * value
        B = (self.distance[1]/ max_distance) * value
        C = (self.distance[2]/ max_distance) * value
        D = (self.distance[3]/ max_distance) * value
        return A, B , C , D
    
    def set_kills(self,num):
        self.kills = num
    
    def update_win(self):
        self.win = True
    def update_kill(self):
        self.kills += 1
    def update_distance(self, pawn, distance):
        self.distance[pawn] = distance
    def update_safe_spot(self):
        self.safe_spot += 1
    def update_star_spot(self):
        self.star_spot += 1
    def set_kills(self,num):
        self.kills = num
    def update_globe_spot(self):
        self.globe += 1
    def update_self_kill(self):
        self.self_kills += 1
    def update_into_the_goal_area(self):
        self.get_to_goal += 1
    def update_get_in_goal(self):
        self.get_in_goal += 1
    def update_pawn_in_danger(self):
        self.pawn_in_danger += 1
    def get_piece_to_move(self):
        return self.piece_to_move
    def get_get_to_goal(self):
        return self.get_to_goal
    def get_get_in_goal(self):
        return self.get_in_goal
    def get_pawn_in_danger(self):
        self.pawn_in_danger
    def get_globe(self):
        return self.globe
    def get_kills(self):
        return self.kills
    def get_safe_spots(self):
        return self.safe_spot
    def get_self_kills(self):
        return self.self_kills
    def get_distance(self):
        return self.calculate_distance()
    def get_star_spots(self):
        return self.star_spot
    def get_fitness_value(self):
        dis_A, dis_B, dis_C, dis_D = self.calculate_distance()
        self.fitness = self.win*20 + dis_A+ dis_B + dis_C + dis_D + self.kills*1.5 + self.safe_spot*1.5 + self.star_spot*1.2 + self.globe*1.5 - self.self_kills + self.get_to_goal + self.get_in_goal - self.pawn_in_danger*0.1
        return self.fitness
    def get_fitness_value_raw(self):
        return self.fitness
    def get_if_won(self):
        return self.win
    
    def get_the_weights(self):
        A,B,C,D = self.ann_model.get_the_weights()
        return A,B,C,D
    
    def set_fitnes_value(self,num):
        #For debuging
        self.fitness = num
        
    def save_weight(self, path, generation, number = 0):
        the_weights = self.ann_model.get_all_the_weights()
        the_path = path+"/weights/generation_"+str(generation)+"/"
        os.makedirs(the_path,exist_ok=True)
        the_path = the_path+ str(number) + ".npy"
        np.save(the_path, the_weights)
        
        
    def reset_fitness(self):
        #This is used as part of validation so the fitness can be reset      
        self.kills       = 0
        self.safe_spot   = 0
        self.star_spot   = 0
        self.self_kills  = 0
        self.globe       = 0
        self.get_to_goal = 0
        self.get_in_goal = 0
        self.pawn_in_danger = 0
        self.piece_to_move = None
        self.distance    = np.zeros(4, dtype=int)
        self.win         = False
        self.fitness     = 0
        
    def print_the_values(self):
        #This is a debug print to see how the fitness was achived
        print(f"The AI had the following statistics - update me")
        print(f"AI had a fitness of {self.get_fitness_value()}")
        print(f"The AI kills :   {self.kills}")
        print(f"The AI stars :   {self.star_spot}")
        print(f"The AI globes:  {self.globe}")
        print(f"The AI Safe:  {self.safe_spot}")
        print(f"The AI Self Kill:  {self.self_kills}")
        print(f"The AI Get to goal:  {self.get_to_goal}")
        print(f"The AI Get in goal:  {self.get_in_goal}")
        print(f"The AI pawn in danger:  {self.pawn_in_danger}")
        print(f"The AI Dist  :    {self.get_distance()}")
        print(f"The AI Won  :    {self.win}")
    
    def update_the_score(self,pawn,dice,enemy_pieces_converted, freinds):
        #Calculate the score
        if hf.is_it_a_glob(pawn+dice):
                if hf.is_it_a_kill(pawn+dice, enemy_pieces_converted) == 0:
                    #Update the globe_score
                    self.update_globe_spot()
                if hf.is_it_a_kill(pawn+dice, enemy_pieces_converted) > 0:
                     #Update the killed your self score
                    self.update_self_kill()
        else:
            
            if hf.is_it_a_kill(pawn+dice, enemy_pieces_converted) == 2:
                #Update Killed by an enemy
                    self.update_self_kill()
            elif hf.is_it_a_kill(pawn+dice, enemy_pieces_converted) == 1:
                #Update you killed enemy Check start
                self.update_kill()
                if hf.is_it_a_star(pawn+dice):
                    #Update star score
                    self.update_star_spot()
            else:
                if hf.is_there_a_freindly(pawn+dice, freinds):
                    #Update the safe_score #Pawns on top of each ohters works as globes
                    self.update_safe_spot()
                if hf.is_it_a_star(pawn+dice):
                    #Update star score
                    self.update_star_spot()
                if hf.is_it_goal_area(pawn,dice):
                    #Update get to goal score
                    self.update_into_the_goal_area()
                if hf.is_it_the_goal(pawn+dice):
                    #Update get in goal score
                    self.update_get_in_goal()                
        for i in range(1,7): #6 sided diceDice
            if hf.is_pawn_dang(pawn+dice, enemy_pieces_converted+i):
            #update score
                self.update_pawn_in_danger()
                break #It dose not get more punishment for more
    
    
    def get_input(self,dice, move_pieces, player_pieces, enemy_pieces):
        #Function needs to return a flatten array of the inputs
        #print(move_pieces)
        _,_,_,enemy_pieces_converted = hf.get_enemy_list_convert(enemy_pieces)
        enemy_pieces_converted = enemy_pieces
        #print(enemy_pieces_converted)
        if np.size(move_pieces) == 0:
            _move_pieces = np.array([0, 0, 0, 0])
        elif np.size(move_pieces) > 0:
            _move_pieces = np.array([0, 0, 0, 0])
            for num in move_pieces: #The move piece from the game is [0,1,2,3] so if there is a number it can run.
                _move_pieces[num] = 1
        _glob_list = np.array([0, 0, 0, 0])
        _star_list = np.array([0, 0, 0, 0])
        _kill_list = np.array([0, 0, 0, 0])
        _pawn_dang = np.array([0, 0, 0, 0])
        _pawn_dies = np.array([0, 0, 0, 0])
        _pawn_safe = np.array([0, 0, 0, 0])
        _get_to_goal = np.array([0, 0, 0, 0])
        _get_in_goal = np.array([0, 0, 0, 0])
        for idx, pawn in enumerate(player_pieces):
            #I hate these giant IF statments, but it works.
            if hf.is_it_a_glob(pawn+dice) and _move_pieces[idx] == 1:
                if hf.is_it_a_kill(pawn+dice, enemy_pieces_converted) == 0:
                    _glob_list[idx] = 1
                else :
                    _pawn_dies[idx] = 1 #If there is a enemy on the globe it will kill the pawn.
            if hf.is_it_a_star(pawn+dice)  and _move_pieces[idx] == 1:
                _star_list[idx] = 1
            if hf.is_it_a_kill(pawn+dice, enemy_pieces_converted) == 2 and _move_pieces[idx]:
                _pawn_dies[idx] = 1
            elif hf.is_it_a_kill(pawn+dice, enemy_pieces_converted) == 1 and _move_pieces[idx]:
                _kill_list[idx] = 1
            for i in range(1,7): #6 sided diceDice
                if hf.is_pawn_dang(pawn, enemy_pieces_converted+i) and _move_pieces[idx] == 1:
                    _pawn_dang[idx] = 1
                    break
            if hf.is_it_a_glob(pawn) and _move_pieces[idx]:
                _pawn_safe[idx] = 1
            if hf.is_it_goal_area(pawn,dice) and _move_pieces[idx]:
                _get_to_goal[idx] = 1
            if hf.is_it_the_goal(pawn+dice) and _move_pieces[idx]:
                _get_in_goal[idx] = 1
        pawns_on_same = hf.are_we_on_the_same(player_pieces)
        #print(f"Pawns on the same {pawns_on_same}")
        _move_pieces = _move_pieces.flatten()
        _glob_list   = _glob_list.flatten()
        _star_list   = _star_list.flatten()
        _kill_list   = _kill_list.flatten()
        _pawn_dang   = _pawn_dang.flatten()
        _pawn_dies   = _pawn_dies.flatten()
        _pawn_safe   = _pawn_safe.flatten()
        _get_to_goal = _get_to_goal.flatten()
        _get_in_goal = _get_in_goal.flatten()
        #_dice = np.array([dice])
        _input = np.concatenate([_move_pieces, _glob_list,_star_list,_kill_list,_pawn_dang,_pawn_dies,_pawn_safe,_get_to_goal,_get_in_goal])
        #_input[_input == -1] = 0 #Converts the minus -1 to 0
        #print(f"The Input {_input}")
        return _input

    def __init__(self, generation_number, parent_A = None, parent_B = None, weights = None):
        self.gen_num     = generation_number
        self.kills       = 0
        self.safe_spot   = 0
        self.star_spot   = 0
        self.self_kills  = 0
        self.globe       = 0
        self.get_to_goal = 0
        self.get_in_goal = 0
        self.pawn_in_danger = 0
        self.piece_to_move = None
        self.distance    = np.zeros(4, dtype=int)
        self.win         = False
        self.fitness     = 0
        self.ann_model   = ann.ANN_network(weights=weights)
    
    def __call__(self, dice, move_pieces, player_pieces, enemy_pieces):
        #print(f"enemy_pieces {enemy_pieces}")
        _,_,_,enemy_pieces_converted = hf.get_enemy_list_convert(enemy_pieces) #The ohter parts are mroe for debugging
        #print(f"enemy_pieces_converted {enemy_pieces_converted}")
        the_input = self.get_input(dice, move_pieces, player_pieces, enemy_pieces_converted)
        output_net = self.ann_model.use_model(the_input)
        #sort_selection = np.argsort(output_net)[0][::-1]
        #print(f"Sorted output was {output_net}")
        piece_to_move = None
        
        #This part picks from the output with the highest score to the lowest
        for piece in output_net:                  #sort_selection:
            if piece in move_pieces:
                piece_to_move = piece
                break
        #print(f"piece_to_move is {piece_to_move} and it is type {type(piece_to_move)}")
        postion = player_pieces[piece_to_move]
        
        self.update_the_score(postion,dice,enemy_pieces_converted,player_pieces)
        
        self.update_distance(piece_to_move, postion+dice)
        #print(f"Done processing moving {piece_to_move} from {old_pos} to {new_pos}")
        self.piece_to_move = piece_to_move
            
def create_first_generation(ammount = 50): #Was 30, but limitation made it smaller
    populations = np.empty(ammount, dtype=[('Population', object)])
    for i in range(ammount):
        the_random_population = Population_object(1)
        #the_random_population.set_kills(i) # This is for debuging leaving it for some testing
        populations[i] = (the_random_population) # Initialize with dummy fitness value
    return populations

def load_weights(path,generation_number = 0, ammount = 6):
    #This function loads a set of weights from a path and returns a population object
    files_in_path = glob.glob(path+"*.npy")
    #files_in_path = sorted(files_in_path) #This is to make sure the files are loaded in the correct order
    populations = np.empty(ammount, dtype=[('Population', object)])
    print(f"files_in_path: {files_in_path}")
    for count, np_array in enumerate(files_in_path):
        temp_weights = np.load(np_array, allow_pickle=True)
        temp_Population = Population_object(generation_number, weights=temp_weights)
        temp_Population.set_kills(count) # This is for debuging leaving it for some testing
        populations[count] = temp_Population
    return populations    

def mutate_weights(weights):
    _temp_shape     = np.shape(weights)
    _flat_wheigt    = np.concatenate(weights)
    _how_many_mute  = np.random.randint(1,np.size(weights)) #op to half can be mutated
    _muate_select   = rng.choice(np.size(weights), size=_how_many_mute, replace=False)
    for i in _muate_select:
        #print(f"i: {i}")
        #print(f"weights[i] was: {_flat_wheigt[i]}")
        changed = np.random.uniform(-1,1)
        #print(f"changed: {changed}")
        #if _flat_wheigt[i]+changed < -1 or _flat_wheigt[i]+changed > 1:
        #    #print(f"Was to high or low with {_flat_wheigt[i]+changed}")
        #    _flat_wheigt[i] = _flat_wheigt[i]-changed
        #else:
        #    _flat_wheigt[i] = _flat_wheigt[i]+changed
        _flat_wheigt[i] = _flat_wheigt[i]+changed
        #print(f"weights[i] now: {_flat_wheigt[i]}")
    
    _reshape_array = np.reshape(_flat_wheigt, _temp_shape)
    return _reshape_array

def crossover_weights(weights_A, wieghts_b):
    _temp_shape     = np.shape(weights_A)
    _temp_A         = np.concatenate(weights_A)
    _temp_B         = np.concatenate(wieghts_b)
    _split          = random.random_integers(np.shape(_temp_A)[0])-1
    _half           = int(np.shape(_temp_A)[0]/2)
    for _ in range(_half):
        _temp_container = _temp_A[_split]
        _temp_A[_split] = _temp_B[_split]
        _temp_B[_split] = _temp_container
        if _split+1 >= np.shape(_temp_A)[0]:
            _split = 0
        else:
            _split += 1
    _temp_child_A = np.reshape(_temp_A, _temp_shape)
    _temp_child_B = np.reshape(_temp_B, _temp_shape)
    return _temp_child_A, _temp_child_B

def create_population(the_best_pop, generation = 0,first_run = False):
    #This function creates 50 new populations from a list of the best 6
    _the_new_populations = np.array([], dtype=[('Population', object)])
    _selection  = rng.choice(the_best_pop.shape[0], size=the_best_pop.shape[0], replace=False)
    #I split the selection in half so i can randomly mix them togheter to try and remove any bias this will give is 20 populations
    _left_half   = _selection[:len(_selection)//2]
    _right_half  = _selection[len(_selection)//2:]
    _object_list = the_best_pop['Population']
    #print(_object_list)
    #print(len(_right_half))
    for i in range(0,len(_right_half)):
        AA,AB,AC,AD = _object_list[_left_half][0].get_the_weights()
        BA,BB,BC,BD = _object_list[_right_half][0].get_the_weights()
        AA , BA = crossover_weights(AA,BA)
        AB , BB = crossover_weights(AB,BB)
        AC , BC = crossover_weights(AC,BC)
        #Not the pretiests way to do it, but easier for me to see what is going on
        BD[0] = AA
        BD[1] = AB
        BD[2] = AC
        AD[0] = BA
        AD[1] = BB
        AD[2] = BC
        temp_pop_A = Population_object(generation, weights = AD)
        temp_pop_B = Population_object(generation, weights = BD)
        _the_new_populations = np.append(_the_new_populations, temp_pop_A)
        _the_new_populations = np.append(_the_new_populations, temp_pop_B)
    for i in range(0,len(_object_list)):
        #print(_object_list[i])
        AA,AB,AC,AD = _object_list[i].get_the_weights()
        AA = mutate_weights(AA)
        AB = mutate_weights(AB)
        AC = mutate_weights(AC)
        AD[0] = AA
        AD[1] = AB
        AD[2] = AC
        temp_pop_A = Population_object(generation, weights = AD)
        _the_new_populations = np.append(_the_new_populations, temp_pop_A)
    
    #print(np.argsort(the_best_pop['fitness_value']))
    elite_offset = 3
    if not first_run:
        elite_3 = the_best_pop[:3]
        for i in elite_3:
            _the_new_populations = np.append(_the_new_populations, the_best_pop[i][0])  
    #Two new random pops
    #print(len(_the_new_populations))
    if len(_the_new_populations) % 2 == 0:
        for i in range(0,2+elite_offset):
            _the_new_populations = np.append(_the_new_populations, Population_object(generation))
    else:
        for i in range(0,1+elite_offset):
            _the_new_populations = np.append(_the_new_populations, Population_object(generation))
    return _the_new_populations


def create_population_doubel(the_best_pop, generation = 0):
    _the_new_populations = np.array([], dtype=[('Population', object)])
    #I split the selection in half so i can randomly mix them togheter to try and remove any bias this will give is 20 populations
    _left_half   = []
    _right_half  = []
    _object_list = the_best_pop['Population']
    for index, element in enumerate(_object_list):
        if index % 2 == 0:
            _left_half.append(element)
        else:
            _right_half.append(element)
    for i in range(0,len(_right_half)):
        #AA,AB,AC,AD = _object_list[_left_half][0].get_the_weights()
        #BA,BB,BC,BD = _object_list[_right_half][0].get_the_weights()
        AA,AB,AC,AD = _object_list[i].get_the_weights()
        BA,BB,BC,BD = _object_list[i].get_the_weights()
        
        AA , BA = crossover_weights(AA,BA)
        AB , BB = crossover_weights(AB,BB)
        AC , BC = crossover_weights(AC,BC)
        #Not the pretiests way to do it, but easier for me to see what is going on
        BD[0] = AA
        BD[1] = AB
        BD[2] = AC
        AD[0] = BA
        AD[1] = BB
        AD[2] = BC
        temp_pop_A = Population_object(generation, weights = AD)
        temp_pop_B = Population_object(generation, weights = BD)
        _the_new_populations = np.append(_the_new_populations, temp_pop_A)
        _the_new_populations = np.append(_the_new_populations, temp_pop_B)
    for m in range(0,len(_object_list)):
        #print(_object_list[i])
        AA,AB,AC,AD = _object_list[m].get_the_weights()
        AA = mutate_weights(AA)
        AB = mutate_weights(AB)
        AC = mutate_weights(AC)
        AD[0] = AA
        AD[1] = AB
        AD[2] = AC
        temp_pop_A = Population_object(generation, weights = AD)
        _the_new_populations = np.append(_the_new_populations, temp_pop_A)
    
    
    #Randomly Mixed
    _selection  = rng.choice(the_best_pop.shape[0], size=the_best_pop.shape[0], replace=False)
    #I split the selection in half so i can randomly mix them togheter to try and remove any bias this will give is 20 populations
    _left_half   = _selection[:len(_selection)//2]
    _right_half  = _selection[len(_selection)//2:]
    _object_list = the_best_pop['Population']
    #print(_object_list)
    #print(len(_right_half))
    for j in range(0,len(_right_half)):
        AA,AB,AC,AD = _object_list[_left_half][0].get_the_weights()
        BA,BB,BC,BD = _object_list[_right_half][0].get_the_weights()
        AA , BA = crossover_weights(AA,BA)
        AB , BB = crossover_weights(AB,BB)
        AC , BC = crossover_weights(AC,BC)
        #Not the pretiests way to do it, but easier for me to see what is going on
        BD[0] = AA
        BD[1] = AB
        BD[2] = AC
        AD[0] = BA
        AD[1] = BB
        AD[2] = BC
        temp_pop_A = Population_object(generation, weights = AD)
        temp_pop_B = Population_object(generation, weights = BD)
        _the_new_populations = np.append(_the_new_populations, temp_pop_A)
        _the_new_populations = np.append(_the_new_populations, temp_pop_B)
    for k in range(0,len(_object_list)):
        #print(_object_list[i])
        AA,AB,AC,AD = _object_list[k].get_the_weights()
        AA = mutate_weights(AA)
        AB = mutate_weights(AB)
        AC = mutate_weights(AC)
        AD[0] = AA
        AD[1] = AB
        AD[2] = AC
        temp_pop_A = Population_object(generation, weights = AD)
        _the_new_populations = np.append(_the_new_populations, temp_pop_A)
        _selection  = rng.choice(the_best_pop.shape[0], size=the_best_pop.shape[0], replace=False)
    #I split the selection in half so i can randomly mix them togheter to try and remove any bias this will give is 20 populations
    _left_half   = _selection[:len(_selection)//2]
    _right_half  = _selection[len(_selection)//2:]
    _object_list = the_best_pop['Population']
    #print(_object_list)
    #print(len(_right_half))
    for _ in range(0,len(_right_half)):
        AA,AB,AC,AD = _object_list[_left_half][0].get_the_weights()
        BA,BB,BC,BD = _object_list[_right_half][0].get_the_weights()
        AA , BA = crossover_weights(AA,BA)
        AB , BB = crossover_weights(AB,BB)
        AC , BC = crossover_weights(AC,BC)
        #Not the pretiests way to do it, but easier for me to see what is going on
        BD[0] = AA
        BD[1] = AB
        BD[2] = AC
        AD[0] = BA
        AD[1] = BB
        AD[2] = BC
        temp_pop_A = Population_object(generation, weights = AD)
        temp_pop_B = Population_object(generation, weights = BD)
        _the_new_populations = np.append(_the_new_populations, temp_pop_A)
        _the_new_populations = np.append(_the_new_populations, temp_pop_B)
    for o in range(0,len(_object_list)):
        #print(_object_list[i])
        AA,AB,AC,AD = _object_list[o].get_the_weights()
        AA = mutate_weights(AA)
        AB = mutate_weights(AB)
        AC = mutate_weights(AC)
        AD[0] = AA
        AD[1] = AB
        AD[2] = AC
        temp_pop_A = Population_object(generation, weights = AD)
        _the_new_populations = np.append(_the_new_populations, temp_pop_A)
    
    for l in range(0,len(_object_list)):
        #print(_object_list[i])
        AA,AB,AC,AD = _object_list[o].get_the_weights()
        AA = mutate_weights(AA)
        AB = mutate_weights(AB)
        AC = mutate_weights(AC)
        AD[0] = AA
        AD[1] = AB
        AD[2] = AC
        temp_pop_A = Population_object(generation, weights = AD)
        _the_new_populations = np.append(_the_new_populations, temp_pop_A)
    
    elite_3 = the_best_pop[:3]
    for elite in elite_3:
        _the_new_populations = np.append(_the_new_populations, elite[0])  
    
    if len(_the_new_populations) % 2 == 0:
        for i in range(0,6):
            _the_new_populations = np.append(_the_new_populations, Population_object(generation))
    else:
        for i in range(0,5):
            _the_new_populations = np.append(_the_new_populations, Population_object(generation))
    return _the_new_populations