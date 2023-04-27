import numpy as np
import ann_model as ann

#These are the functions that helps to create the genetic alghorithm

#1. Create a population of random networks -> Done by ANN in play game
#2. Create a fitness function -> Done by play_game
#3. Create a selection function -> Done by genetic alghorithm
#4. Create a crossover function -> Done by genetic alghorithm
#5. Create a mutation function -> Done by genetic alghorithm


test_model = ann.ANN_network(4)
#print(test_model.get_weights())
A,B,test_weights = test_model.get_weights()


print(f"Layer 1 weights:{A}")
print(f" ")
#print(f"Layer 2 weights:{B}")
#print(f" ")
print(f"Model:{test_weights}")

_how_many_mutate = np.random.randint(1,(np.size(test_weights)*np.size(test_weights[0]))/2)
#print("Mutating: " ,_how_many_mutate)

_same_selection = []

#print(len(_same_selection))


def mutation_weights(test_weights):
    #This mutates random selectio of weights in the network
    pass
    
    
    
#mutation_weights(test_weights)