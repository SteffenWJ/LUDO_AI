import numpy as np
import ann_model as ann
import helper_functions as hf
import genetic_alghorithm as ga

rng = np.random.default_rng()



def create_population(the_best_pop, generation = 0):
    #This function creates 50 new populations from a list of the best 10
    _the_new_populations = np.array([], dtype=[('Population', object)])
    _selection  = rng.choice(the_best_pop.shape[0], size=the_best_pop.shape[0], replace=False)
    #I split the selection in half so i can randomly mix them togheter to try and remove any bias this will give is 20 populations
    _left_half   = _selection[:len(_selection)//2]
    _right_half  = _selection[len(_selection)//2:]
    _object_list = the_best_pop['Population']
    #print(_object_list)
    print(len(_right_half))
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
        temp_pop_A = ga.Population_object(generation, weights = AD)
        temp_pop_B = ga.Population_object(generation, weights = BD)
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
        temp_pop_A = ga.Population_object(generation, weights = AD)
        _the_new_populations = np.append(_the_new_populations, temp_pop_A)
    
    #print(np.argsort(the_best_pop['fitness_value']))
    elite_3 = np.argsort(the_best_pop['fitness_value'])[-3:]
    #print(the_best_pop[elite_3])
    for i in elite_3:
        _the_new_populations = np.append(_the_new_populations, the_best_pop[i])
    #Two new random pops
    #print(len(_the_new_populations))
    if len(_the_new_populations) % 2 == 0:
        for i in range(0,8):
            _the_new_populations = np.append(_the_new_populations, ga.Population_object(generation))
    else:
        for i in range(0,7):
            _the_new_populations = np.append(_the_new_populations, ga.Population_object(generation))
    return _the_new_populations
        
        
def mutate_weights(weights):
    _temp_shape     = np.shape(weights)
    _flat_wheigt    = np.concatenate(weights)
    _how_many_mute  = np.random.randint(1,np.size(weights)) #op to half can be mutated
    _muate_select   = rng.choice(np.size(weights), size=_how_many_mute, replace=False)
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


test_pop_1 = ga.Population_object(1)
test_pop_2 = ga.Population_object(1)
test_pop_3 = ga.Population_object(1)
test_pop_4 = ga.Population_object(1)
test_pop_5 = ga.Population_object(1)
test_pop_6 = ga.Population_object(1)
test_pop_7 = ga.Population_object(1)
test_pop_8 = ga.Population_object(1)
test_pop_9 = ga.Population_object(1)
test_pop_10 = ga.Population_object(1)

test_pop_1.set_kills(1.4)
test_pop_2.set_kills(3)
test_pop_3.set_kills(30)
test_pop_4.set_kills(20)
test_pop_5.set_kills(1)
test_pop_6.set_kills(21)
test_pop_7.set_kills(4)
test_pop_8.set_kills(53)
test_pop_9.set_kills(6)
test_pop_10.set_kills(7)

test_selection = ga.selection_of_pop()
test_selection(test_pop_1)
test_selection(test_pop_2)
test_selection(test_pop_3)
test_selection(test_pop_4)
test_selection(test_pop_5)
test_selection(test_pop_6)
test_selection(test_pop_7)
test_selection(test_pop_8)
test_selection(test_pop_9)
test_selection(test_pop_10)
value = 10
offset = 2
for obj in test_selection.get_population()['Population']:
    obj.set_fitnes_value(value*offset)

best_pop = test_selection.get_population()

test_new_pop = create_population(best_pop)

print(len(test_new_pop))