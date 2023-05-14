import ludopy
import numpy as np
import cv2


import ann_model as ann


ann_model = ann.ANN_network()
A,B,C,weights = ann_model.get_the_weights()

ann_model_2 = ann.ANN_network()
AB,BB,CB,weights_B = ann_model_2.get_the_weights()

#print(f"weights: {weights}")

weights_B[0] = A
weights_B[1] = B
weights_B[2] = C

ann_model_3 = ann.ANN_network(weights=weights_B)

the_weights_to_save = ann_model_2.get_all_the_weights()
np.save('weights.npy', the_weights_to_save)

test = np.load('weights.npy',allow_pickle=True)
print(the_weights_to_save)
print("")
print(test)

ann_model_4 = ann.ANN_network(weights=test)



#np.savetxt('weights.txt', the_weights_to_save)
