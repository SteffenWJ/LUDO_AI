import matplotlib.pyplot as plt
import numpy as np




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
        #self.ax.plot([], [], 'ro', label = "Current Fitness")
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
        self.ax.plot([], [], color = 'g' ,label = "Mean Fitness")
        #self.ax.plot([], [], 'ro', label = "Current Fitness")
        self.ax.plot([], [], color = 'b' ,  label = "Culumanative Wins")
        
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
        #self.ax.plot(self.generation_list,self.current_fitness, 'ro', label = "Current Fitness")
        self.ax.plot(self.generation_list,self.culumnative_wins, color = 'b', label = "Culumanative Wins")
        
        #self.figure.legend()  # Show the legend

        
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()