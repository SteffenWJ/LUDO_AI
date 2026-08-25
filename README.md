# LUDO_AI

This is a genetic algorithm that tries to train a neural network's weights. This is not the standard way to train a CNN it is done, but I thought it would be fun to see if a genetic algorithm could train a convolutional network to control a LUDO piece.

The Answer is yes, but not effectively, as there is too little information and too few options to make it viable as a whole, and when tested against more traditional methods like Q-Learning, it performed poorly.

In a 4-player game, it achieved around a 35% win rate overall, so better than the baseline of 25% vs random moves, but against better algorithms it lost almost every time.
