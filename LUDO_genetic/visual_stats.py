import matplotlib.pyplot as plt
import numpy as np




class print_comparions_1v3:
    def save_figure(self, path, name):
        save_path = path+"/"+name+".png"
        self.figure.savefig(save_path)
    def __init__(self, xLim = 1000, yLim = 1000, player_type = "Genetic Algorithm"):
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Win rate comparsion")
        self.ax.set_xlabel("Games played")
        self.ax.set_ylabel("Win rates")
        #Might need to change these to be dynamic
        self.ax.set_ylim(0, xLim)
        self.ax.set_xlim(1, yLim)
        self.ax.grid()
        self.label_AI = "Win rate "+player_type
        self.ax.plot([], [], color = '0.5' , linestyle=':', label='Random player A')
        self.ax.plot([], [], color = '0.5' , linestyle=':', label='Random player B')
        self.ax.plot([], [], color = '0.5' , linestyle=':', label='Random player C')
        self.ax.plot([], [], color = 'b' , linestyle=':', label=self.label_AI)
        
        self.magna_wins = []
        self.SWJ_wins = []
        self.random_A = []
        self.random_B = []
        
        self.games = []
        self.ax.legend()
        self.figure.show()
        
    def __call__(self, win_mag, win_swj,win_A,win_B, game):
        self.magna_wins.append(win_mag)
        self.SWJ_wins.append(win_swj)
        self.random_A.append(win_A)
        self.random_B.append(win_B)
        
        self.games.append(game+1)
        
        self.ax.plot(self.games, self.random_A, color = '0.5' , linestyle=':',  label='Random player A')
        self.ax.plot(self.games, self.random_B, color = '0.5' , linestyle=':',  label='Random player B')
        self.ax.plot(self.games, self.SWJ_wins, color = '0.5' , linestyle=':', label='Random player C')
        self.ax.plot(self.games, self.magna_wins, color = 'b' , linestyle=':', label=self.label_AI)
        #self.ax.plot(self.games, self.SWJ_wins, color = 'b' , linestyle=':', label='Win rate Genetic Algorithm')
        
        #self.figure.legend()  # Show the legend

        
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        
class print_comparions_1v1v2:
    def save_figure(self, path, name):
        save_path = path+"/"+name+".png"
        self.figure.savefig(save_path)
    def __init__(self, xLim = 1000, yLim = 1000, player_type_A = "Genetic Algorithm", player_type_B = "Q-learning"):
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Win rate comparsion")
        self.ax.set_xlabel("Games played")
        self.ax.set_ylabel("Win rates")
        #Might need to change these to be dynamic
        self.ax.set_ylim(0, xLim)
        self.ax.set_xlim(1, yLim)
        self.ax.grid()
        self.label_AI_A = "Win rate "+player_type_A
        self.label_AI_B = "Win rate "+player_type_B
        self.ax.plot([], [], color = '0.5' , linestyle=':', label='Random player A')
        self.ax.plot([], [], color = '0.5' , linestyle=':', label='Random player B')
        self.ax.plot([], [], color = 'b' , linestyle=':', label=self.label_AI_A)
        self.ax.plot([], [], color = 'r' , linestyle=':', label=self.label_AI_B)
        
        self.magna_wins = []
        self.SWJ_wins = []
        self.random_A = []
        self.random_B = []
        
        self.games = []
        self.ax.legend()
        self.figure.show()
        
    def __call__(self, win_mag, win_swj,win_A,win_B, game):
        self.magna_wins.append(win_mag)
        self.SWJ_wins.append(win_swj)
        self.random_A.append(win_A)
        self.random_B.append(win_B)
        
        self.games.append(game+1)
        
        self.ax.plot(self.games, self.random_A, color = '0.5' , linestyle=':',  label='Random player A')
        self.ax.plot(self.games, self.random_B, color = '0.5' , linestyle=':',  label='Random player B')
        self.ax.plot(self.games, self.SWJ_wins, color = 'b' , linestyle=':', label=self.label_AI_A)
        self.ax.plot(self.games, self.magna_wins, color = 'r' , linestyle=':', label=self.label_AI_B)
        #self.ax.plot(self.games, self.SWJ_wins, color = 'b' , linestyle=':', label='Win rate Genetic Algorithm')
        
        #self.figure.legend()  # Show the legend

        
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

class print_fit_win_all:
    #This class is used to visulise the different fitnesses value, the mean fitness and culumanative wins
    #As the fitness value will never go over 100 and i plan to run the validation to be a 100 this should be fine.
    def save_figure(self, path, weight_number):
        self.figure.savefig(path)
        
    def reset_plot(self):
        self.ax.cla()
        self.ax.set_title("Fitness and Wins for a Weight")
        self.ax.set_xlabel("Test run")
        self.ax.set_ylabel("Fitness/Wins")
        #Might need to change these to be dynamic
        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(0, 100)
        self.ax.grid()
        self.ax.plot([], [], color = 'g' ,label = "Mean Fitness")
        self.ax.plot([], [], color='r',linestyle=':', marker='.', label='Current Fitness')
        self.ax.plot([], [], color = 'b' ,  label = "Culumanative Wins")
        
        self.mean_fitness = []
        self.wins = []
        self.culumnative_wins = []
        self.current_fitness = []
        
        #Generation is a list from 1 to 100
        self.generation_list = []
        self.ax.legend()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
    
    
    def __init__(self):
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Fitness and Wins")
        self.ax.set_xlabel("Generations")
        self.ax.set_ylabel("Fitness/Wins")
        #Might need to change these to be dynamic
        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(0, 100)
        self.ax.grid()
        self.ax.plot([], [], color = 'g' , label = "Mean Fitness")
        self.ax.plot([], [], color='r',linestyle=':', marker='.', label='Current Fitness')
        self.ax.plot([], [], color = 'b' , label = "Culumanative Wins")
        
        self.mean_fitness = []
        self.wins = []
        self.culumnative_wins = []
        self.current_fitness = []
        
        #Generation is a list from 1 to 100
        self.generation_list = []
        self.ax.legend()
        self.figure.show()
        
    def __call__(self,fitness, win, generation):
        self.current_fitness.append(fitness)
        self.mean_fitness.append(np.mean(self.current_fitness))
        self.wins.append(win)
        self.culumnative_wins.append(np.sum(self.wins))
        self.generation_list.append(generation)
        
        self.ax.plot(self.generation_list,self.mean_fitness, color = 'g', label = "Mean Fitness")
        self.ax.plot(self.generation_list, self.current_fitness, color='r',linestyle=':', marker='.', label='Current Fitness')
        self.ax.plot(self.generation_list,self.culumnative_wins, color = 'b', label = "Culumanative Wins")
        
        #self.figure.legend()  # Show the legend

        
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        
class print_fit_win_all_2:
    #This class is used to visulise the different fitnesses value, the mean fitness and culumanative wins
    #As the fitness value will never go over 100 and i plan to run the validation to be a 100 this should be fine.
    def save_figure(self, path, weight_number):
        self.figure.savefig(path)
        
    def reset_plot(self):
        self.ax.cla()
        self.ax.set_title("Fitness and Wins for a Weight")
        self.ax.set_xlabel("Test run")
        self.ax.set_ylabel("Fitness/Wins")
        #Might need to change these to be dynamic
        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(0, 100)
        self.ax.grid()
        self.ax.axhline(y=25, color='0.5', linestyle='--', label = "25% Win Rate")
        self.ax.axhline(y=30, color='g', linestyle='--', label = "30% Win Rate")
        self.ax.plot([], [], color = 'b' ,  label = "Culumanative Wins")
        self.wins = []
        self.culumnative_wins = []
        self.current_fitness = []
        
        #Generation is a list from 1 to 100
        self.generation_list = []
        self.ax.legend()

        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
    
    
    def __init__(self):
        self.figure = plt.figure()
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Fitness and Wins")
        self.ax.set_xlabel("Generations")
        self.ax.set_ylabel("Fitness/Wins")
        #Might need to change these to be dynamic
        self.ax.set_ylim(0, 100)
        self.ax.set_xlim(0, 100)
        self.ax.grid()
        self.ax.axhline(y=25, color='0.5', linestyle='--', label = "25% Win Rate")
        self.ax.axhline(y=30, color='g', linestyle='--', label = "30% Win Rate")
        self.ax.plot([], [], color = 'b' , label = "Culumnative Wins")
        
        self.mean_fitness = []
        self.wins = []
        self.culumnative_wins = []
        self.current_fitness = []
        
        #Generation is a list from 1 to 100
        self.generation_list = []
        self.ax.legend()
        self.figure.show()
        
    def __call__(self,fitness, win, generation):
        self.current_fitness.append(fitness)
        self.mean_fitness.append(np.mean(self.current_fitness))
        self.wins.append(win)
        self.culumnative_wins.append(np.sum(self.wins))
        self.generation_list.append(generation)
        
        self.ax.axhline(y=25, color='0.5', linestyle='--', label = "25% Win Rate")
        self.ax.axhline(y=30, color='g', linestyle='--', label = "30% Win Rate")
        self.ax.plot(self.generation_list,self.culumnative_wins, color = 'b', label = "Culumnative Wins")
        
        #self.figure.legend()  # Show the legend

        
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()