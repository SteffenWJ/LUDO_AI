import ludopy
import numpy as np
import cv2

import random
import genetic_alghorithm as ga

from tqdm import tqdm



def run_validation_game(path, show_matchs = False, print_data = False):
    test_population = ga.load_weights(path)   
    print(f"test_population: {len(test_population)}")
    for pop in tqdm(test_population, desc="Population Loop", leave=False):
        temp_testing = pop[0]
        for i in tqdm(range(0,20), desc="Game Loop", leave=False):
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
            print(f" It took {current_round} rounds to finish game {i}")
            print(f"Player {player_i} won the game AI was {AI_controller}")
            temp_testing.print_the_values()
            temp_testing.reset_fitness()
            temp_testing.print_the_values()
    #cv2.imshow("Enviroment", enviroment_image_bgr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
print("Testing validation")


run_validation_game("LUDO_genetic/output/weights/test_load/")