import ludopy
import numpy as np
import cv2


import ann_model as ann

g = ludopy.Game()
there_is_a_winner = False

while not there_is_a_winner:
    (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()

    enviroment_image_rgb = g.render_environment() # RGB image of the enviroment
    enviroment_image_bgr = cv2.cvtColor(enviroment_image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow("Enviroment", enviroment_image_bgr)
    cv2.waitKey(1)

    if len(move_pieces):
        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
    else:
        piece_to_move = -1

    _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
cv2.destroyAllWindows()


ann_model = ann.ANN_network(4)
A,B,C,weights = ann_model.get_weights()

ann_model_2 = ann.ANN_network(4)
AB,BB,CB,weights_B = ann_model_2.get_weights()

print(f"weights: {A}")
print(f"weights: {AB}")