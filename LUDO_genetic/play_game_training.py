import ludopy
import numpy as np
import ann_model as ann
import cv2

import random
import genetic_alghorithm as ga

there_is_a_winner = False

#https://github.com/SimonLBSoerensen/LUDOpy
#game = ludopy.Game(ghost_players=[2,3])
game = ludopy.Game()
AI_controller = random.randint(0, 3)

first_run = True
if first_run:
    first_pop = ga.create_first_generation()
else:
    #Add the load weight after
    pass
    
    
for i in range(0, 30):
    game = ludopy.Game()
    there_is_a_winner = False
    if first_run:
        population = ga.Population_object(count,0)
    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = game.get_observation()

        if len(move_pieces) and player_i != AI_controller:
            piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
        elif len(move_pieces) and player_i == AI_controller:
            test_pop(dice, move_pieces, player_pieces, enemy_pieces)
            piece_to_move = test_pop.get_piece_to_move()
            #cv2.waitKey(0)
        else:
            piece_to_move = -1
        enviroment_image_rgb = game.render_environment() # RGB image of the enviroment
        enviroment_image_bgr = cv2.cvtColor(enviroment_image_rgb, cv2.COLOR_RGB2BGR)
        #cv2.imshow("Enviroment", enviroment_image_bgr)
        #cv2.waitKey(1)
        a_dice, a_move_pieces, a_player_pieces, a_enemy_pieces, a_player_is_a_winner, there_is_a_winner = game.answer_observation(piece_to_move)
    if player_i == AI_controller:
        test_pop.update_win()
    count = count +1
    #cv2.imshow("Enviroment", enviroment_image_bgr)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
print("ALL DONE")