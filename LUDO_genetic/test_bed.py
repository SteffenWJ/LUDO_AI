import numpy as np
import ann_model as ann
import helper_functions as hf
import genetic_alghorithm as ga

rng = np.random.default_rng()

test_pop_1 = ga.Population_object(1)
test_pop_2 = ga.Population_object(1)
test_pop_3 = ga.Population_object(1)
test_pop_4 = ga.Population_object(1)
test_pop_5 = ga.Population_object(1)
test_pop_6 = ga.Population_object(1)
test_pop_7 = ga.Population_object(1)
test_pop_8 = ga.Population_object(1)
test_pop_9 = ga.Population_object(1)
test_pop_10 = ga.Population_object(1)

test_pop_1.set_kills(1.4)
test_pop_2.set_kills(3)
test_pop_3.set_kills(30)
test_pop_4.set_kills(20)
test_pop_5.set_kills(1)
test_pop_6.set_kills(21)
test_pop_7.set_kills(4)
test_pop_8.set_kills(53)
test_pop_9.set_kills(6)
test_pop_10.set_kills(7)

test_selection = ga.selection_of_pop()
test_selection(test_pop_1)
test_selection(test_pop_2)
test_selection(test_pop_3)
test_selection(test_pop_4)
test_selection(test_pop_5)
test_selection(test_pop_6)
test_selection(test_pop_7)
test_selection(test_pop_8)
test_selection(test_pop_9)
test_selection(test_pop_10)
value = 10
offset = 2
for obj in test_selection.get_population()['Population']:
    obj.set_fitnes_value(value*offset)

best_pop = test_selection.get_population()
print(best_pop)
#test_new_pop = ga.create_population(best_pop)

#print(len(test_new_pop))

#test_2 = ga.create_first_generation(30)
#for i in range(0,10):
#    test_pop_1.save_weight("/home/steffen/uni/sem2mes/tools_AI/LUDO_AI/LUDO_genetic/output","test_load",i)
#test_pop_1.save_weight("/home/steffen/uni/sem2mes/tools_AI/LUDO_AI/LUDO_genetic/output","test_load",)



#test_get_values = ga.load_weights("LUDO_genetic/output/weights/test_load/")
#
#
#print(test_get_values)
#print(len(test_get_values))

#for pop in test_get_values:
#    temp_pop = pop[0]
#    print(temp_pop.get_kills())
#    print(temp_pop)


get_image = ann.ANN_network(36,64,64)


get_image.plot_model_all()