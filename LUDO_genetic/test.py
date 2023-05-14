import ludopy
import numpy as np
import cv2


import ann_model as ann


ann_model = ann.ANN_network()
A,B,C,weights = ann_model.get_weights()

ann_model_2 = ann.ANN_network()
AB,BB,CB,weights_B = ann_model_2.get_weights()

print(f"weights: {weights}")

weights_B[0] = A
weights_B[1] = B
weights_B[2] = C

ann_model_3 = ann.ANN_network(weights=weights_B)
