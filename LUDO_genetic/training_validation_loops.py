import ludopy
import numpy as np
import cv2

import gc
import os
import visual_stats as VS

import random
import genetic_alghorithm as ga

from tqdm import tqdm
import time 

def run_validation_game(path, show_matchs = False, print_data = False, save_data = False, how_many_runs = 100, weights_name = None):
    test_population = ga.load_weights(path)
    os.makedirs("LUDO_genetic/output/figures/"+weights_name+"/", exist_ok=True)
    if print_data:
        visual = VS.print_fit_win_all()
    count = 0
    for pop in tqdm(test_population, desc="Population Loop", leave=False):
        temp_testing = pop[0]
        if print_data:
            visual.reset_plot()
        for i in tqdm(range(0,how_many_runs), desc="Game Loop", leave=False):
            game = ludopy.Game()
            there_is_a_winner = False
            AI_controller = random.randint(0, 3) #Randomly select a player to be controlled by AI
            current_round = 0
            while not there_is_a_winner:
                current_round += 1
                (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = game.get_observation()
                if len(move_pieces) and player_i != AI_controller:
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                elif len(move_pieces) and player_i == AI_controller:
                    temp_testing(dice, move_pieces, player_pieces, enemy_pieces)
                    piece_to_move = temp_testing.get_piece_to_move()
                    #cv2.waitKey(0)
                else:
                    piece_to_move = -1
                #enviroment_image_rgb = game.render_environment() # RGB image of the enviroment
                #enviroment_image_bgr = cv2.cvtColor(enviroment_image_rgb, cv2.COLOR_RGB2BGR)
                #cv2.imshow("Enviroment", enviroment_image_bgr)
                #cv2.waitKey(1)
                a_dice, a_move_pieces, a_player_pieces, a_enemy_pieces, a_player_is_a_winner, there_is_a_winner = game.answer_observation(piece_to_move)
            if player_i == AI_controller:
                temp_testing.update_win()
                if print_data:
                    visual(temp_testing.get_fitness_value(), 1, i)
            else:
                if print_data:
                    visual(temp_testing.get_fitness_value(), 0, i)
            #print(f" It took {current_round} rounds to finish game {i}")
            #print(f"Player {player_i} won the game AI was {AI_controller}")
            #temp_testing.print_the_values()
            temp_testing.reset_fitness()
            #temp_testing.print_the_values()
        save_path = "LUDO_genetic/output/figures/"+weights_name+"/figure_"+str(count)+".png"
        visual.save_figure(save_path, count)
        count += 1
    #cv2.imshow("Enviroment", enviroment_image_bgr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
#run_validation_game("LUDO_genetic/output/weights/test_load/",print_data=True)



def run_training(save_path, generation = 0,load_weight_path = None, print_data = False,traning_limits = 5, validation_steps = 10, tries = 5):
    if load_weight_path == None:
        print("Creating the first generation")
        population_training = ga.create_first_generation()
        create_pop_fix = True
    else:
        print(f"Loading weights from: {load_weight_path}")
        population_training = ga.load_weights(load_weight_path)
        #population_training = ga.create_population(population_training,generation)
        population_training = ga.create_population_doubel(population_training,generation)
        create_pop_fix = False
    if print_data:
        visual = VS.print_fit_win_all()
        #print(test[0].get_kills())
    population_selection = ga.selection_of_pop()
    for i in tqdm(range(0, len(population_training)), desc="Population", leave=False):
        test_runner_pop = population_training[i]
        if create_pop_fix:
            test_runner_pop = test_runner_pop[0]
        #print(test_runner_pop)
        mean_fitness = 0
        fitness_value = []
        if print_data:
            visual.reset_plot()
        for k in tqdm(range(0,5), desc="Game Loop", leave=False):
            game = ludopy.Game()
            there_is_a_winner = False
            AI_controller = random.randint(0, 3) #Randomly select a player to be controlled by AI
            current_round = 0
            while not there_is_a_winner:
                current_round += 1
                (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = game.get_observation()
                if len(move_pieces) and player_i != AI_controller:
                    piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
                elif len(move_pieces) and player_i == AI_controller:
                    test_runner_pop(dice, move_pieces, player_pieces, enemy_pieces)
                    piece_to_move = test_runner_pop.get_piece_to_move()
                    #cv2.waitKey(0)
                else:
                    piece_to_move = -1
                #enviroment_image_rgb = game.render_environment() # RGB image of the enviroment
                #enviroment_image_bgr = cv2.cvtColor(enviroment_image_rgb, cv2.COLOR_RGB2BGR)
                #cv2.imshow("Enviroment", enviroment_image_bgr)
                #cv2.waitKey(1)
                a_dice, a_move_pieces, a_player_pieces, a_enemy_pieces, a_player_is_a_winner, there_is_a_winner = game.answer_observation(piece_to_move)
            if player_i == AI_controller:
                test_runner_pop.update_win()
                if print_data:
                    visual(test_runner_pop.get_fitness_value(), 1, generation)
            else:
                if print_data:
                    visual(test_runner_pop.get_fitness_value(), 0, generation)
            #print(f" It took {current_round} rounds to finish game {k}")
            #print(f"Player {player_i} won the game AI was {AI_controller}")
            #test_runner_pop.print_the_values()
            fitness_value.append(test_runner_pop.get_fitness_value())
            test_runner_pop.reset_fitness()
        mean_fitness = np.mean(fitness_value)
        tqdm.write(f"Mean fitness: {mean_fitness} for run {i} generation {generation}")
        #print(f"Mean fitness: {mean_fitness} for run {i}")
        test_runner_pop.set_fitnes_value(mean_fitness)
        population_selection(test_runner_pop)
    the_best_performer = population_selection.get_population()
    #print("the_best_performer where:")
    #print(the_best_performer)
    for l in range(0,len(the_best_performer)):
        #print(the_best_performer)
        the_best_performer[l][0].save_weight(save_path, generation,l)
        
    del population_selection
    del the_best_performer
    del test_runner_pop
    del population_training
    gc.collect()
    #cv2.imshow("Enviroment", enviroment_image_bgr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
#run_validation_game("LUDO_genetic/output/weights/test_load/",print_data=True)


def generational_training(load_path, save_path, generation_Start = 0, how_many_generations = 1, skip_load_path = False):
    print(f"Starting generational training for generation: {generation_Start} to generation: {how_many_generations}")
    #the_load_path = load_path
    for i in tqdm(range(generation_Start, generation_Start+how_many_generations), desc="Generation",leave=False):
        the_load_path = save_path+"/weights/generation_"+str(i)+"/"
        print("loadpath ",the_load_path)
        gc.collect()
        if skip_load_path:
            #This is for creating the first generation after it should be using the ohter as the load path should be useable
            run_training(save_path, i)
        else:
            run_training(save_path, i+1, the_load_path)
            skip_load_path = False
        #the_load_path = save_path+"/weights/generation_"+str(i)+"/"
        

generational_training("LUDO_genetic/output","LUDO_genetic/output",39,1)

#run_validation_game("LUDO_genetic/output/weights/generation_10/",print_data=True,weights_name="weights_10")

