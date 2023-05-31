import ludopy
import numpy as np

import cv2
from keras.backend import clear_session
import gc
import os
import visual_stats as VS

import random
#import genetic_alghorithm as ga
import GAS as ga

from qLearningMagn.MagnPlayer import MagnPlayer
from tqdm import tqdm

def run_validation_game(path, save_to_me, show_matchs = False, print_data = False, save_data = False, how_many_runs = 100, weights_name = None):
    test_population = ga.load_weights(path)
    os.makedirs(save_to_me+"/figures/"+weights_name+"/", exist_ok=True)
    if print_data:
        visual = VS.print_fit_win_all_2()
    count = 0
    for pop in tqdm(test_population, desc="Population Loop", leave=False):
        temp_testing = pop[0]
        if print_data:
            visual.reset_plot()
        for i in tqdm(range(0,how_many_runs), desc="Game Loop", leave=False):
            #game = ludopy.Game(ghost_players=[1,3])
            game = ludopy.Game()
            there_is_a_winner = False
            AI_controller = random.randint(0, 3) #Randomly select a player to be controlled by AI
            #AI_controller = 0
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
        save_path = save_to_me+"/figures/"+weights_name+"/figure_"+str(count)+".png"
        visual.save_figure(save_path, count)
        count += 1
    gc.collect()
    #cv2.imshow("Enviroment", enviroment_image_bgr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
#run_validation_game("LUDO_genetic/output/weights/test_load/",print_data=True)

def compare_game(path, save_to_me, print_data = False, how_many_runs = 100, figure_name = None):
    test_population = ga.load_weights(path,1)
    #os.makedirs(save_to_me+"/figures/"+figure_name+"/", exist_ok=True)
    wins_magn = 0
    wins_SWJ = 0
    no_player_A = 0
    no_player_B = 0
    no_player_C = 0
    sId = 2
    mId = 0
    AI_controller = [mId,sId]
    random_players = [x for x in range(0,4) if x not in AI_controller]
    if print_data:
        #visual = VS.print_comparions_1v3(how_many_runs,how_many_runs)
        visual = VS.print_comparions_1v1v2(how_many_runs,how_many_runs)
    for i in tqdm(range(0,how_many_runs), desc="Game Loop", leave=False):
        #game = ludopy.Game(ghost_players=[1,3])
        game = ludopy.Game()
        there_is_a_winner = False
        magnPlayer = MagnPlayer(mId, training = False, exploration = 0)
        SWJPlayer = test_population[0]
        SWJPlayer = SWJPlayer[0]
        #print(f"SWJ: {SWJPlayer}")
        #AI_controller = random.randint(0, 3) #Randomly select a player to be controlled by AI
        current_round = 0
        player_i = 0
        while not there_is_a_winner:
            current_round += 1
            (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = game.get_observation()
            if len(move_pieces) and player_i not in  AI_controller:
                piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
            elif len(move_pieces) and player_i == sId:
                SWJPlayer(dice, move_pieces, player_pieces, enemy_pieces)
                piece_to_move = SWJPlayer.get_piece_to_move()
                #cv2.waitKey(0)
            elif player_i == mId:
                piece_to_move = magnPlayer.update((dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i)
            else:
                piece_to_move = -1
            #enviroment_image_rgb = game.render_environment() # RGB image of the enviroment
            #enviroment_image_bgr = cv2.cvtColor(enviroment_image_rgb, cv2.COLOR_RGB2BGR)
            #cv2.imshow("Enviroment", enviroment_image_bgr)
            #cv2.waitKey(1)
            a_dice, a_move_pieces, a_player_pieces, a_enemy_pieces, a_player_is_a_winner, there_is_a_winner = game.answer_observation(piece_to_move)
        #print(f" It took {current_round} rounds to finish game {i}")
        #print(f"Player {player_i} won the game AI was {AI_controller}")
        #temp_testing.print_the_values()
        #temp_testing.print_the_values()
        if player_i == mId:
            wins_magn += 1
        elif player_i == sId:
            wins_SWJ += 1
        elif random_players[0] == player_i:
            no_player_A += 1
        elif random_players[1] == player_i:
            no_player_B += 1
        elif len(random_players) > 2:
            if random_players[2] == player_i:
                no_player_C += 1
        visual(wins_magn, wins_SWJ, no_player_A,no_player_B, i)
        print(f"Magn has won {wins_magn} vs SWJ {wins_SWJ}")
        print(f"Computer A has won {no_player_A} vs Computer B {no_player_B}, Computer C {no_player_C}")
    visual.save_figure("LUDO_genetic/comparison_test",save_to_me)
    #save_path = save_to_me+"/figures/"+weights_name+"/figure_"+str(count)+".png"
    #visual.save_figure(save_path, count)
    #count += 1
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
        visual = VS.print_fit_win_all_2()
        #print(test[0].get_kills())
    population_selection = ga.selection_of_pop()
    for i in tqdm(range(0, len(population_training)), desc="Population", leave=False):
        test_runner_pop = population_training[i]
        if create_pop_fix:
            test_runner_pop = test_runner_pop[0]
        mean_fitness = 0
        fitness_value = []
        if print_data:
            visual.reset_plot()
        for _ in tqdm(range(0,10), desc="Game Loop", leave=False):
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
    tqdm.write(f"Generation {generation} fitness values where")
    population_selection.print_fitness_values()
    the_best_performer = population_selection.get_population()
    #print("the_best_performer where:")
    #print(the_best_performer)
    for l in range(0,len(the_best_performer)):
        #print(the_best_performer)
        the_best_performer[l][0].save_weight(save_path, generation,l)
        
    #del population_selection
    #del the_best_performer
    #del test_runner_pop
    #del population_training
    #gc.collect()
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
        if skip_load_path:
            #This is for creating the first generation after it should be using the ohter as the load path should be useable
            run_training(save_path, i)
        else:
            run_training(save_path, i+1, the_load_path)
            skip_load_path = False
        #the_load_path = save_path+"/weights/generation_"+str(i)+"/"
        

#generational_training("LUDO_genetic/output_simple","LUDO_genetic/output_simple",105,5)
#generational_training("LUDO_genetic/output_last_try","LUDO_genetic/output_last_try",0,1)
#run_validation_game("LUDO_genetic/output_last_try/weights/generation_50/","LUDO_genetic/output_last_try/generation_50",print_data=True,weights_name="weights_50_1")

compare_game("LUDO_genetic/best_weight/","1v1v1v1_3", True, 100)

index_ = 42
steps = 2
run_me = False
while run_me:
    #est_path_A = "LLUDO_genetic/output_last_try/weights/generation_"+str(index_)+"/"
    #est_path_B = "LUDO_genetic/output_last_try/generation_"+str(index_)
    #eights_name = "weights_"+str(index_)
    #un_validation_game(test_path_A,test_path_B,print_data=True,weights_name=weights_name)
    #lear_session()
    generational_training("LUDO_genetic/output_last_try","LUDO_genetic/output_last_try",index_,steps)
    clear_session()
    index_ += steps