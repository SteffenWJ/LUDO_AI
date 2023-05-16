#A collection of helper functions.


#STAR_INDEXS = [5, 12, 18, 25, 31, 38, 44, 51]
#GOAL_INDEX = 57
#GLOB_INDEXS = [9, 22, 35, 48]
import numpy as np
import os


#Checs if the position that it will move to have an enemy in it
def is_it_a_star(pos):
    STAR_INDEXS = [5, 12, 18, 25, 31, 38, 44, 51]
    return pos in STAR_INDEXS

#Is it a glob position mostly used as it was easier to use.
def is_it_a_glob(pos):
    GLOB_INDEXS = [9, 22, 35, 48]
    return pos in GLOB_INDEXS

#This is slightly redundant.
def is_it_the_goal(pos):
    GOAL_INDEX = 57
    return pos == GOAL_INDEX

def is_it_a_kill(pos, enemy_pos):
    #temp_enemy_pos = enemy_pos_at_pos_SWJ(enemy_pos)
    if pos in enemy_pos:
        return True
    else:
        return False    
#No longer used, switched out
def get_enemies_for_player_pos(enemies):
    temp_enemy_list = []
    for count, enemy_list in enumerate(enemies,1):
        #print(enemy_list)
        for enemy in enemy_list:
            _ = enemy_pos_at_pos_SWJ(enemy)
            temp_enemy_list.append(_[-count])
    return temp_enemy_list


def enemy_pos_at_pos_SWJ(pos):
    #Modifed version from the source code to check if the position that it will move to have an enemy in it
    #This is from the ludopy.player.Player class in the source code with some modifications from me
    HOME_INDEX = 0
    START_INDEX = 1
    HOME_AREAL_INDEXS = [52, 53, 54, 55, 56]
    ENEMY_1_GLOB_INDX = 14
    ENEMY_2_GLOB_INDX = 27
    ENEMY_3_GLOB_INDX = 40
    enemy_pos = []
    ENEMY_1_INDX_AT_HOME = 40  # HOME_AREAL_INDEXS[0] - 6 - i * 13 # i = 1
    ENEMY_2_INDX_AT_HOME = 27  # HOME_AREAL_INDEXS[0] - 6 - i * 13 # i = 2
    ENEMY_3_INDX_AT_HOME = 14  # HOME_AREAL_INDEXS[0] - 6 - i * 13 # i = 3
    for enemy_start_pos, enemy_pos_at_start in [[ENEMY_1_GLOB_INDX, ENEMY_1_INDX_AT_HOME],
                                                [ENEMY_2_GLOB_INDX, ENEMY_2_INDX_AT_HOME],
                                                [ENEMY_3_GLOB_INDX, ENEMY_3_INDX_AT_HOME]]:
        post_offset = enemy_start_pos - 1
        pre_offset = enemy_pos_at_start - 1
        if pos == enemy_start_pos:
            #pos_enemy = [START_INDEX, HOME_AREAL_INDEXS[0]] #Changed it to only take 1 of the [1,52] as it is the only one that maters from the player pawns view
            pos_enemy = [START_INDEX]
        elif pos < 0:
            pos_enemy = [max(enemy_pos_at_start + pos, -1)]
        elif START_INDEX <= pos < enemy_start_pos:
            pos_enemy = [pos + pre_offset]
        elif pos > HOME_AREAL_INDEXS[0] or pos == HOME_INDEX:
            pos_enemy = [-1]
        else:
            pos_enemy = [pos - post_offset]
        enemy_pos.append(pos_enemy)
    #print(enemy_pos)
    return enemy_pos

def get_enemy_list_convert(enemies):
    #This function converts the nemies to be the values of the current player
    off_1,off_2,off_3 = np.split(enemies,3) #Splits the matrix into 3 vectors
    
    enemy_offset_1 = []
    enemy_offset_2 = []
    enemy_offset_3 = []
    enemy_convert_full = []

    for i in range(0,4):
        temp_val = enemy_pos_at_pos_SWJ(off_1[0][i])
        if temp_val[-1] == [1,52]:
            enemy_offset_1.append([1])
        else:
            enemy_offset_1.append(temp_val[-1])
        temp_val = enemy_pos_at_pos_SWJ(off_2[0][i])
        if temp_val[-2] == [1,52]:
            enemy_offset_2.append([1])
        else:
            enemy_offset_2.append(temp_val[-2])
        temp_val = enemy_pos_at_pos_SWJ(off_3[0][i])
        if temp_val[-3] == [1,52]:
            enemy_offset_3.append([1])
        else:
            enemy_offset_3.append(temp_val[-3])
    enemy_offset_1 = np.concatenate(enemy_offset_1)
    enemy_offset_2 = np.concatenate(enemy_offset_2)
    enemy_offset_3 = np.concatenate(enemy_offset_3)
    enemy_convert_full = np.stack([enemy_offset_1,enemy_offset_2,enemy_offset_3])
        
    return enemy_offset_1, enemy_offset_2, enemy_offset_3, enemy_convert_full