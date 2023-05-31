import ann_model as ann
import numpy as np
from numpy import random as rng

import GAS

import helper_functions as hf


def get_input(dice, move_pieces, player_pieces, enemy_pieces):
    #Function needs to return a flatten array of the inputs
    #print(move_pieces)
    _,_,_,enemy_pieces_converted = hf.get_enemy_list_convert(enemy_pieces)
    enemy_pieces_converted = enemy_pieces
    print(enemy_pieces_converted)
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
        elif hf.is_there_a_freindly(pawn+dice, player_pieces) and _move_pieces[idx] == 1:
            _glob_list[idx] = 1 #If there is a freind it will work like a globe
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
    for paw in pawns_on_same:
        _pawn_safe[paw] = 1
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

#Fitness is win+disA+DisB+DisC+DisD+kills+glob+star+safe+goal+in_goal-dies-dang
def update_the_score(pawn,dice,enemy, freinds):
    _,_,_,enemy_pieces_converted = hf.get_enemy_list_convert(enemy)
    if hf.is_it_a_glob(pawn+dice):
            if hf.is_it_a_kill(pawn+dice, enemy_pieces_converted) == 0:
                #Update the globe_score
                pass
            if hf.is_it_a_kill(pawn+dice, enemy_pieces_converted) == 1:
                 #Update the killed your self score
                pass
    elif hf.is_there_a_freindly(pawn+dice, freinds):
                #Update the globe_score #Pawns on top of each ohters works as globes
                pass
    else:
        if hf.is_it_a_kill(pawn+dice, enemy_pieces_converted) == 2:
            #Update Killed by an enemy
            pass
        elif hf.is_it_a_kill(pawn+dice, enemy_pieces_converted) == 1:
            #Update you killed enemy Check start
            if hf.is_it_a_star(pawn+dice):
                #Update star score
                pass
        else:
            if hf.is_it_a_star(pawn+dice):
                #Update star score
                pass
            if hf.is_it_goal_area(pawn,dice):
                #Update get to goal score
                pass
            if hf.is_it_the_goal(pawn+dice):
                #Update get in goal score
                pass
            if pawn+dice in freinds:
                #update safe score
                pass
        
            for i in range(1,7): #6 sided diceDice
                if hf.is_pawn_dang(pawn+dice, enemy_pieces_converted+i):
                #update score
                pass
                #break

def update_the_score_2(input):
    if input[1] == 1:
        if input[3] == 1:
            #Update the killed your self score
            pass
        #Update the globe_score
        pass
    
move_pices = np.array([0,2,3])
pawns = np.array([0,2,1,9])
enemies = np.array([[50,1,49,2],[2,5,5,1],[10,29,13,44]])
    


the_input = get_input(4, move_pices, pawns, enemies)

print(the_input)
print("[ pawn  | glob  | star  | kill  | dang  | dies  | safe  | to_goal | in_goal ]")

pawn_to_move = 3
the_input[3]


print("Answer : [1 0 0 1 1 0 1 0 0]")