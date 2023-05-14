from typing import Any
import numpy as np
import ann_model as ann
import helper_functions as hf
import names



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
        return self.populations
    
    def print_fitness_values(self):
        print(self.populations['fitness_value'])
        
        
    def __init__(self) -> None:
        self.populations = np.array([], dtype=[('object', object), ('fitness_value', float)])
        self.lowest_fitnes = 0
        self.full = False
    
    def __call__(self, pop):
        if self.full == False:
            self.populations = np.append(self.populations, np.array([(pop, pop.get_fitness_value())], dtype=[('object', object), ('fitness_value', float)]))
            if (pop.get_fitness_value() > self.lowest_fitnes):
                self.lowest_fitnes = pop.get_fitness_value()
            if self.populations.shape[0] == 10:
                self.full = True
        else:
            if pop.get_fitness_value() > self.lowest_fitnes:
                self.populations = np.delete(self.populations, np.argmin(self.populations['fitness_value']))
                self.populations = np.append(self.populations, np.array([(pop, pop.get_fitness_value())], dtype=[('object', object), ('fitness_value', float)]))
                self.lowest_fitnes = np.min(self.populations['fitness_value'])
                self.lowest_fitnes = np.argmin(self.populations['fitness_value'])

class Population_object:
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
    
    def get_fitness_value(self):
        dis_A, dis_B, dis_C, dis_D = self.calculate_distance()
        self.fitness = self.win*10 + self.kills + dis_A+ dis_B + dis_C + dis_D + self.safe_spot + self.star_spot - self.killed_num
        return self.fitness
    def update_win(self):
        self.win = True
    def update_kill(self, ammount = 1):
        self.kills += ammount #Should only be 1, but in rare cases it can be more if there are more enemy pawns my home spawn. 
    def update_distance(self, pawn, distance):
        self.distance[pawn] = distance
    def update_safe_spot(self):
        self.safe_spot += 1
    def update_star_spot(self):
        self.star_spot += 1
    def how_many_dead(self): #Not using it right now but would use to to try and keep it from getting killed
        return self.kills_num
    def get_piece_to_move(self):
        return self.piece_to_move
    def get_kills(self):
        return self.kills
    def get_safe_spots(self):
        return self.safe_spot
    def get_distance(self):
        return self.calculate_distance()
    def get_star_spots(self):
        return self.star_spot
    
    def print_the_values(self):
        #This is a debug print to see how the fitness was achived
        print(f"The AI had the following statistics")
        print(f"AI had a fitness of {self.get_fitness_value()}")
        print(f"The AI kills :   {self.get_kills()}")
        print(f"The AI stars :   {self.get_star_spots()}")
        print(f"The AI globes:  {self.get_safe_spots()}")
        print(f"The AI Dist  :    {self.get_distance()}")
    
    def get_input(self,dice, move_pieces, player_pieces, enemy_pieces):
        #Function needs to return a flatten array of the inputs
        #print(move_pieces)
        if np.size(move_pieces) == 0:
            _move_pieces = np.array([0, 0, 0, 0])
        elif np.size(move_pieces) > 0:
            _move_pieces = np.array([0, 0, 0, 0])
            for num in move_pieces:
                _move_pieces[num] = 1
        _move_pieces = _move_pieces.flatten()
        #print(f"Size: {np.size(_move_pieces)} and values where:\n {_move_pieces}")
        _player_pieces = player_pieces.flatten()
        #print(f"Size: {np.size(_player_pieces)} and values where:\n {_player_pieces}")
        _enemy_pieces = enemy_pieces.flatten()
        #print(f"Size: {np.size(_enemy_pieces)} and values where:\n {_enemy_pieces}")
        _dice = np.array([dice])
        _input = np.concatenate([_dice, _move_pieces, _player_pieces, _enemy_pieces])
        _input[_input == -1] = 0 #Converts the minus -1 to 0
        #print(_input)
        #_input = np.concatenate([_move_pieces, _player_pieces, _enemy_pieces])
        #_input = tf.convert_to_tensor(_input, dtype=tf.float32)
        return _input
    

    def __init__(self, generation_number, player_number, parent_A = None, parent_B = None, weights = None):
        self.gen_num    = generation_number
        self.name       = names.get_first_name()+"_"+names.get_last_name()+"_"+str(generation_number)
        self.parent_A   = parent_A
        self.parent_B   = parent_B
        self.player_num = player_number
        self.fitness    = 0
        self.kills      = 0
        self.safe_spot  = 0
        self.star_spot  = 0
        self.killed_num = 0
        self.piece_to_move = None
        self.distance   = np.zeros(4, dtype=int)
        self.win        = False
        self.ann_model  = ann.ANN_network(21,weights=weights)
    
    def __call__(self, dice, move_pieces, player_pieces, enemy_pieces):
        #print(f"enemy_pieces {enemy_pieces}")
        _,_,_,enemy_pieces_converted = hf.get_enemy_list_convert(enemy_pieces) #The ohter parts are mroe for debugging
        #print(f"enemy_pieces_converted {enemy_pieces_converted}")
        the_input = self.get_input(dice, move_pieces, player_pieces, enemy_pieces_converted)
        output_net = self.ann_model.use_model(the_input)
        sort_selection = np.argsort(output_net)[0][::-1]
        piece_to_move = None
        
        #This part picks from the output with the highest score to the lowest
        for piece in sort_selection:
            if piece in move_pieces:
                piece_to_move = piece
                break
            
        old_pos = player_pieces[piece_to_move]
        new_pos = old_pos + dice
        if hf.is_it_a_star(new_pos):
            self.update_star_spot()
        elif hf.is_it_a_kill(new_pos, enemy_pieces_converted):
            if hf.is_it_a_glob(new_pos):
                pass #Might make this a negative value to teach it not to knock it self home.
            else:
                self.update_kill()
        elif hf.is_it_a_glob(new_pos):
            self.update_safe_spot()
        
        self.update_distance(piece_to_move, new_pos)
        #print(f"Done processing moving {piece_to_move} from {old_pos} to {new_pos}")
        self.piece_to_move = piece_to_move
            
        
    
    
def mutate_weights(weights):
    _temp_shape     = np.shape(weights)
    _flat_wheigt    = np.concatenate(weights)
    _how_many_mute  = np.random.randint(1,np.size(weights)) #op to half can be mutated
    _muate_select   = rng.choice(np.size(weights), size=_how_many_mute, replace=False)
    
    #print(f"_temp_shape: {_temp_shape}")
    #print(f"_flat_wheigt: {_flat_wheigt}")
    #print(f"_how_many_mute: {_how_many_mute}")
    print(f"_muate_select: {_muate_select}")
    
    for i in _muate_select:
        print(f"i: {i}")
        print(f"weights[i] was: {_flat_wheigt[i]}")
        changed = np.random.uniform(-1,1)
        print(f"changed: {changed}")
        if _flat_wheigt[i]+changed < -1 or _flat_wheigt[i]+changed > 1:
            print(f"Was to high or low with {_flat_wheigt[i]+changed}")
            _flat_wheigt[i] = _flat_wheigt[i]-changed
        else:
            _flat_wheigt[i] = _flat_wheigt[i]+changed
        print(f"weights[i] now: {_flat_wheigt[i]}")
    
    _reshape_array = np.reshape(_flat_wheigt, _temp_shape)
    return _reshape_array

def crossover_weights(weights_A, wieghts_b):
    _temp_shape     = np.shape(weights_A)
    _temp_A         = np.concatenate(weights_A)
    _temp_B         = np.concatenate(wieghts_b)
    _selecton_select   = rng.choice(np.size(_temp_A), size=int(np.size(weights_A)/2), replace=False)
    _temp_child_A = _temp_A
    _temp_child_B = _temp_B
    for i in _selecton_select:
        _temp_child_A[i] = _temp_B[i]
        _temp_child_B[i] = _temp_A[i]
    _temp_child_A = np.reshape(_temp_A, _temp_shape)
    _temp_child_B = np.reshape(_temp_B, _temp_shape)
    return _temp_child_A, _temp_child_B