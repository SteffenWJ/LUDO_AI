import ludopy
import numpy as np

import cv2

there_is_a_winner = False


played_game = []
current_round = []

game = ludopy.Game()
rounds = 0

AI_controller = np.random.randint(0, 4)

def get_input(dice, move_pieces, player_pieces, enemy_pieces):
    #Function needs to return a flatten array of the inputs
    print(move_pieces)
    if np.size(move_pieces) == 0:
        _move_pieces = np.array([0, 0, 0, 0])
    elif np.size(move_pieces) > 0:
        _move_pieces = np.array([0, 0, 0, 0])
        for num in move_pieces:
            _move_pieces[num] = 1
    _move_pieces = _move_pieces.flatten()
    print(f"Size: {np.size(_move_pieces)} and values where:\n {_move_pieces}")
    _player_pieces = player_pieces.flatten()
    #print(f"Size: {np.size(_player_pieces)} and values where:\n {_player_pieces}")
    _enemy_pieces = enemy_pieces.flatten()
    #print(f"Size: {np.size(_enemy_pieces)} and values where:\n {_enemy_pieces}")
    _dice = np.array([dice])
    return np.concatenate([_dice, _move_pieces, _player_pieces, _enemy_pieces])

while not there_is_a_winner:
    (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = game.get_observation()
    current_round.append(player_pieces)
    
    test = get_input(dice, move_pieces, player_pieces, enemy_pieces)
    print(test)
    #enviroment_image_rgb = game.render_environment() # RGB image of the enviroment
    #enviroment_image_bgr = cv2.cvtColor(enviroment_image_rgb, cv2.COLOR_RGB2BGR)
    #cv2.imshow("Enviroment", enviroment_image_bgr)
    #cv2.waitKey(0)
    if len(move_pieces):
        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
    else:
        piece_to_move = -1
    a_dice, a_move_pieces, a_player_pieces, a_enemy_pieces, a_player_is_a_winner, there_is_a_winner = game.answer_observation(piece_to_move)
    if player_i == 0:
        rounds += 1
        played_game.append(current_round)
        current_round = []
        
        