#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  2 10:03:19 2021

@author: cjrichier
"""

#import libraries
import random, math
import pandas as pd

######## Initialize connection class ########
class Connection(object):
    def __init__(self, recipient, sender, weight = 0):
        self.owner = recipient #set connection to have the unit connect to another unit
        self.sender = sender #set who is sending the connection 
        self.weight = weight #initialize the weight 
    def update_weight(self): #update the weight of this connection 
        #implementation of hopfield learning algorithim
        print('* * * * * * * * * * * * *')
        self.weight += (2 * self.owner.activation - 1) * (2 * self.sender.activation - 1)
        print('updated unit weight: ', self.weight)
        
######## Initialize unit class ########
class HopUnit(object):
    def __init__(self, my_list, threshold=0):
        self.index = len(my_list) # give the unit an index 
        self.threshold = threshold #set the threshold
        self.net_input = 0 #initialize net input to 0
        self.target_activation = 0
        self.connections = [] 
        self.activation = 0 #initialize activation to 0 #create an argument for which activation function to use when initializing the neuron
    def add_connection(self, sender, weight=0):
         self.connections.append(Connection(self, sender, weight)) #create connections
    def update_input(self):
         self.net_input = 0
         for connection in self.connections:
             self.net_input += connection.weight * connection.sender.activation
         return self.net_input
    def update_target_activation(self):
        #most basic activation function
      
            if self.net_input > self.threshold:
                self.target_activation = 1
            else:
                self.target_activation = 0
            return self.net_input
       
    def display_value(self):
        if self.activation == self.target_activation:
            return ' '+str(self.activation)
        else:
            if self.activation == 1:
                return ' i'
            else:
                return ' o'
    
    def settled(self):
        if self.activation == self.target_activation:
            return True
        else:
            return False
    def update_activation(self): 
        self.activation = self.target_activation
    def show(self):
        print('************************************')
        print ('\nUnit '+str(self.index)+':')
        print ('\tInput = '+str(self.net_input))
        print ('\tTarget activation = '+str(self.target_activation))
        print ('\tCurrent Activation =  '+str(self.activation))
        print ('\tHas connections from...')
        for connection in self.connections:
            print('\t Unit '+str(connection.sender.index)+' = '+str(connection.weight))

######## Initialize network class ########
class network(object):
    def __init__(self, num_units, default_threshold =0):
        self.units = []
        units = []
        for i in range(num_units):
            self.units.append(HopUnit(self.units, default_threshold))
     # and connect them
        for unit_i in self.units:
            for unit_j in self.units:
                if not unit_i is unit_j:
                    unit_i.add_connection(unit_j)
    def all_units_have_settled(self):
        for unit in self.units:
            if unit.settled() == False:
                return False 
        return True
    #This function is not working at all, gives an indexing error 
    def make_starting_state(self, training_pattern, noise):
        #adds noise to training pattern to hand to network as starting state
        starting_state = []
        for i in range(len(training_pattern)):
            if random.random() < noise:
                #flip the number in training pattern 
                if training_pattern[i] == 1:
                    starting_state.append(0)
                else:
                    starting_state.append(1)     
            else:
                starting_state.append(training_pattern[i])
        return starting_state
    def train_patterns(self, training_pattern):
        #impose the training pattern on the network
        for i in range(len(training_pattern)):
            self.units[i].activation = training_pattern[i]
            print(self.units[i].activation)
        for unit in self.units:
            for connection in unit.connections:
                connection.update_weight()
        #return self
    def run_network(self, starting_state, training_pattern, sync_update=False):
        print('************************************')
        print('Running network with starting activation: ', str(starting_state))
        print('Trying to reconstruct:                    ', str(training_pattern))
        print('************************************')
        list_of_units_to_update = []
        for unit in self.units:
            list_of_units_to_update.append(unit) 
        for i in range(len(starting_state)):
            self.units[i].activation = starting_state[i]
        iteration = 0
        all_done = False
        while not (all_done or iteration > 49):
            iteration += 1
            print('Iteration' , str(iteration), 'diagnostics and tracking:')
            ########################
            if sync_update == False:
            ########################
                for unit in self.units:
                    #update net input and print details
                    unit.update_input()
                    unit.update_target_activation()
                    #update target activations
                    unit_TAs = []
                    for unit in self.units:
                        unit_TAs.append(unit.target_activation)
                #check for settling
                settled = self.all_units_have_settled()
                print('The network is settled? ', settled)
                print('Target activations: ', unit_TAs)
                if settled == False:
                    #randomly choose one unit and update actual activation
                    random_unit_to_update = random.choice(list_of_units_to_update)
                    for unit in self.units:
                        if unit.index == random_unit_to_update.index:
                            unit.activation = unit.target_activation
                    iteration_activation = []
                    for unit in self.units:
                        iteration_activation.append(unit.activation)
                    print('unit randomly updated: ', random_unit_to_update.index)
                    print('Updated activations: ', iteration_activation)
                    iteration_energy = self.compute_energy()
                    print('This iteration has energy ' + str(iteration_energy))
                    print('************************************')
                elif settled == True:
                    iteration_activation = []
                    for unit in self.units:
                        iteration_activation.append(unit.activation)
                    all_done = True 
            #######################
            if sync_update == True:
            #######################
                unit_start = []
                for unit in self.units:
                    unit.update_input()
                    unit_start.append(unit.update_input())  
                print(unit_start)    
                for unit in self.units:
                    unit.update_target_activation()
                    #update net input and print details
                    #inputs.append(unit.update_input())
                #print('updated input: ', inputs)
                #inputs = []
                #for unit in self.units:
                    #inputs.append(unit.update_target_activation())
                #print('inputs as seen by target activations: ', inputs)
                    #update activations
                for unit in self.units:
                    unit.activation = unit.target_activation
                #check for settling
                
                settled = self.all_units_have_settled()
                print('The network is settled? ', settled)
                if settled == False:
                    iteration_activation = []
                    for unit in self.units:
                        iteration_activation.append(unit.activation)
                    print('Updated activations: ', iteration_activation)
                    iteration_energy = self.compute_energy()
                    print('This iteration has energy ' + str(iteration_energy))
                    print('************************************')
                elif settled == True:
                    iteration_activation = []
                    for unit in self.units:
                        iteration_activation.append(unit.activation)
                    all_done = True  
                    
        final_activation = iteration_activation   
        #now, calculate print some summary metrics
        #Keep track of the training state
        hamming_distance = self.hamming_distance(training_pattern, final_activation)
        energy = self.compute_energy()
        print('************************************')
        print('Training pattern to recover: ', str(training_pattern))
        print('Final network activation:    ', str(final_activation))
        print('************************************')
        print('After running, the hamming distance between patterns this network ran on is ' + str(hamming_distance))
        if hamming_distance == 0:
            print('Congrats! The network rebuilt the pattern.')
        else:
            print('A bit off the mark there, bucko.')
        print('The energy of this network in its final iteration is ' + str(energy))
        if iteration > 20:
            print('This network failed to settle and ran for the maximum number of alloted iterations: ' + str(iteration))
        else:
            print('The Network ran for ' + str(iteration)+' iterations before it settled.')
        print('End of network run.')
        print('************************************')
        return(hamming_distance, energy, iteration)
    def compute_energy(self):
         # E = -1/2 (*sum (over weights, i,j, of a[i] *w[i,j]))
        energy = 0.0
        for unit in self.units:
           for connection in unit.connections:
              energy += connection.weight * connection.sender.activation * connection.owner.activation
        return -.5 * energy
    def hamming_distance(self, pattern1, pattern2):
        distance = 0
        for i in range(len(pattern1)):
            if pattern1[i] != pattern2[i]:
                distance +=1
        return distance

####### Part 1 #######

#initialize network
Hop_network = network(16)    

## part a)
# Train the pattern on the walsh functions:
training_pattern_1 = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]
training_pattern_2 = [1,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0]
training_pattern_3 = [1,1,0,0,1,1,0,0,1,1,0,0,1,1,0,0]
training_pattern_4 = [1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]
list_of_training_patterns = [training_pattern_1, training_pattern_2, 
                            training_pattern_3, training_pattern_4]

#Train the network on those walsh functions:
Hop_network.train_patterns(training_pattern_1) 
Hop_network.train_patterns(training_pattern_2)
Hop_network.train_patterns(training_pattern_3)
Hop_network.train_patterns(training_pattern_4)

#Part b)
#run the network
list_0_runs = []
list_1_runs = []
list_2_runs = []
list_3_runs = []
list_4_runs = []
list_5_runs = []
for training_pattern in list_of_training_patterns:
    for noise_level in (0,.1,.2,.3,.4,.5): 
        for run_index in range(5):
            start_pattern = Hop_network.make_starting_state(training_pattern, noise_level)
            #list_of_runs_output.append(Hop_network.run_network(start_pattern, training_pattern))
            if noise_level == 0:
                list_0_runs.append(Hop_network.run_network(start_pattern, training_pattern))
            elif noise_level == .1:
                list_1_runs.append(Hop_network.run_network(start_pattern, training_pattern))
            elif noise_level == .2:
                list_2_runs.append(Hop_network.run_network(start_pattern, training_pattern))
            elif noise_level == .3:
                list_3_runs.append(Hop_network.run_network(start_pattern, training_pattern))
            elif noise_level == .4:
                list_4_runs.append(Hop_network.run_network(start_pattern, training_pattern))
            elif noise_level == .5:
                list_5_runs.append(Hop_network.run_network(start_pattern, training_pattern))
                
                  
#Get summary statistics

list_0_runs =  pd.DataFrame(list_0_runs)
list_1_runs =  pd.DataFrame(list_1_runs)
list_2_runs =  pd.DataFrame(list_2_runs)
list_3_runs =  pd.DataFrame(list_3_runs)
list_4_runs =  pd.DataFrame(list_4_runs)
list_5_runs =  pd.DataFrame(list_5_runs)
#############
list_0_runs = list_0_runs.rename(columns =  {0: 'Hamming Distance', 1: "Energy", 2: "# Iterations"})
list_1_runs = list_1_runs.rename(columns =  {0: 'Hamming Distance', 1: "Energy", 2: "# Iterations"})
list_2_runs = list_2_runs.rename(columns =  {0: 'Hamming Distance', 1: "Energy", 2: "# Iterations"})
list_3_runs = list_3_runs.rename(columns =  {0: 'Hamming Distance', 1: "Energy", 2: "# Iterations"})
list_4_runs = list_4_runs.rename(columns =  {0: 'Hamming Distance', 1: "Energy", 2: "# Iterations"})
list_5_runs = list_5_runs.rename(columns =  {0: 'Hamming Distance', 1: "Energy", 2: "# Iterations"})

part_1b_summary_stats = [list_0_runs, list_1_runs, list_2_runs, 
                         list_3_runs ,list_4_runs, list_5_runs]

#Print out the summary stats:
count = 0
for df in part_1b_summary_stats:
    print('* * * * * * * * * * * * * * * * * * * *')
    print('For noise value', str(count), ':')
    print('Number of times network failed to settle: ', df.loc[df['# Iterations'] == 50, '# Iterations'].count())
    print("Average energy: ", df['Energy'].mean())
    print("Average hamming Distance: ", df['Hamming Distance'].mean())
    count += .1

'''The network , with increasing error values in the starting pattern, had growing 
hamming distance and times it failed to settle. The energy also increased with each increasing
value of noise. In fact, when error is 0, it always can recreate the initial pattern.'''


############ Part 2 ###########
####### Random Patterns #######
###############################

#initialize network
Hop_network = network(16)   

#Part 2A
#Create three random training patterns. In each training pattern activate each unit with probability 0.5.
list_of_random_training_patterns = []
random_pattern_1 = []
for i in range(15):
    n = random.randint(0,1)
    random_pattern_1.append(n)
random_pattern_2 = []
for i in range(15):
    n = random.randint(0,1)
    random_pattern_2.append(n)
random_pattern_3 = []
for i in range(15):
    n = random.randint(0,1)
    random_pattern_3.append(n)
list_of_random_training_patterns = [random_pattern_1, random_pattern_2, random_pattern_3]

#Train the network on those walsh functions:
Hop_network.train_patterns(random_pattern_1) 
Hop_network.train_patterns(random_pattern_2)
Hop_network.train_patterns(random_pattern_3)

#Part 2B
#Run the network
list_0_runs = []
list_1_runs = []
list_2_runs = []
list_3_runs = []
list_4_runs = []
list_5_runs = []

for noise_level in (0,.1,.2,.3,.4,.5): 
    for training_pattern in list_of_random_training_patterns:
        for run_index in range(5):
            start_pattern = Hop_network.make_starting_state(training_pattern, noise_level)
            #list_of_runs_output.append(Hop_network.run_network(start_pattern, training_pattern))
            if training_pattern == random_pattern_1:
                list_0_runs.append(Hop_network.run_network(start_pattern, training_pattern))
            elif training_pattern == random_pattern_2:
                list_1_runs.append(Hop_network.run_network(start_pattern, training_pattern))
            elif training_pattern == random_pattern_3:
                list_2_runs.append(Hop_network.run_network(start_pattern, training_pattern))
          

#Get summary statistics
list_0_runs =  pd.DataFrame(list_0_runs)
list_1_runs =  pd.DataFrame(list_1_runs)
list_2_runs =  pd.DataFrame(list_2_runs)

#############
list_0_runs = list_0_runs.rename(columns =  {0: 'Hamming Distance', 1: "Energy", 2: "# Iterations"})
list_1_runs = list_1_runs.rename(columns =  {0: 'Hamming Distance', 1: "Energy", 2: "# Iterations"})
list_2_runs = list_2_runs.rename(columns =  {0: 'Hamming Distance', 1: "Energy", 2: "# Iterations"})

part_2b_summary_stats = [list_0_runs, list_1_runs, list_2_runs]

#Print out the summary stats:
count = 0
for df in part_2b_summary_stats:
    print('* * * * * * * * * * * * * * * * * * * *')
    count += 1
    print('For random pattern', str(count), ':')
    print('Number of times network failed to settle: ', df.loc[df['# Iterations'] == 50, '# Iterations'].count())
    print("Average energy: ", df['Energy'].mean())
    print("Average hamming Distance: ", df['Hamming Distance'].mean())
    
#How do the results differ from part 1?
'''because the patterns have totally random noise, there is no systematic increase or 
decrease in the different metrics of model performance'''

#add more random starting patterns twice over:
random_pattern_4 = []
for i in range(15):
    n = random.randint(0,1)
    random_pattern_4.append(n)
    
#now test the network with this new pattern:

random_pattern_4_ss = Hop_network.make_starting_state(training_pattern, .2)
Hop_network.train_patterns(random_pattern_4)
Hop_network.run_network(random_pattern_4_ss, random_pattern_4)
    
random_pattern_5 = []
for i in range(15):
    n = random.randint(0,1)
    random_pattern_5.append(n)
    
#now test the network with this new pattern:
random_pattern_5_ss = Hop_network.make_starting_state(training_pattern, .2)
Hop_network.train_patterns(random_pattern_5)
Hop_network.run_network(random_pattern_5_ss, random_pattern_5)

'''as you train more random patterns, the performance stays the same, more
or less making noisy predictions about patterns that are randomly generated. 
The energy also seems to increase, suggesting it falls into to some weird energy well.'''


#################### Part 3 ###################
####### Systematically Related Patterns #######
###############################################

#initialize network
Hop_network = network(16)   

#establish a base pattern and make noisy patterns based off it 
base_pattern = [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1]
part3_tp1 = Hop_network.make_starting_state(base_pattern, .125)
part3_tp2 = Hop_network.make_starting_state(base_pattern, .125)
part3_tp3 = Hop_network.make_starting_state(base_pattern, .125)
part3_tp4 = Hop_network.make_starting_state(base_pattern, .125)
part3_tp5 = Hop_network.make_starting_state(base_pattern, .125)
part3_tp6 = Hop_network.make_starting_state(base_pattern, .125)

system_related_patterns = [part3_tp1, part3_tp2 ,part3_tp3,
                           part3_tp4, part3_tp5 , part3_tp6]

#Train the network on the noisy patterns we made, but NOT base pattern:
Hop_network.train_patterns(part3_tp1) 
Hop_network.train_patterns(part3_tp2)
Hop_network.train_patterns(part3_tp3) 
Hop_network.train_patterns(part3_tp4) 
Hop_network.train_patterns(part3_tp5) 
Hop_network.train_patterns(part3_tp6) 



#Now run the network using the nosiy start patterns with no change

part_3_output = []
part_3_output.append(Hop_network.run_network(part3_tp1, base_pattern))
part_3_output.append(Hop_network.run_network(part3_tp2, base_pattern))
part_3_output.append(Hop_network.run_network(part3_tp3, base_pattern))
part_3_output.append(Hop_network.run_network(part3_tp4, base_pattern))
part_3_output.append(Hop_network.run_network(part3_tp5, base_pattern))
part_3_output.append(Hop_network.run_network(part3_tp6, base_pattern))


#Get summary statistics
part_3_output =  pd.DataFrame(part_3_output)
#############
part_3_output = part_3_output.rename(columns =  {0: 'Hamming Distance', 1: "Energy", 2: "# Iterations"})

#Print out the summary stats:
print('* * * * * * * * * * * * * * * * * * * *')
print('Number of times network failed to settle: ', part_3_output.loc[df['# Iterations'] == 50, '# Iterations'].count())
print("Average energy: ", part_3_output['Energy'].mean())
print("Average hamming Distance: ", part_3_output['Hamming Distance'].mean())
    


'''the network usually gets pretty close to rebuilding the base pattern, even
though it had never been shown it before. Hamming distances in the few times I tried running it
were either 0s or 1s. I'm really not sure what psychological 
phenomenon it might correspond to, but perhaps some sort of noise reduction? It can 
reconstruct a pattern it hadn't seen from many different instances of roughly that 
with noise. Perhaps it might be some sort of associative memory retrieval?
'''

########## Part 4 ###########
####### Sync updating #######
#############################

Hop_network = network(16)   


Hop_network.train_patterns(training_pattern_1) 
Hop_network.train_patterns(training_pattern_2)
Hop_network.train_patterns(training_pattern_3)
Hop_network.train_patterns(training_pattern_4)


#Run the network with asyncronous updating
list_0_runs = []
list_1_runs = []
list_2_runs = []
list_3_runs = []
list_4_runs = []
list_5_runs = []
for training_pattern in list_of_training_patterns:
    for noise_level in (0,.1,.2,.3,.4,.5): 
        for run_index in range(5):
            start_pattern = Hop_network.make_starting_state(training_pattern, noise_level)
            
            if noise_level == 0:
                list_0_runs.append(Hop_network.run_network(start_pattern, training_pattern, sync_update=True))
            elif noise_level == .1:
                list_1_runs.append(Hop_network.run_network(start_pattern, training_pattern, sync_update=True))
            elif noise_level == .2:
                list_2_runs.append(Hop_network.run_network(start_pattern, training_pattern, sync_update=True))
            elif noise_level == .3:
                list_3_runs.append(Hop_network.run_network(start_pattern, training_pattern, sync_update=True))
            elif noise_level == .4:
                list_4_runs.append(Hop_network.run_network(start_pattern, training_pattern, sync_update=True))
            elif noise_level == .5:
                list_5_runs.append(Hop_network.run_network(start_pattern, training_pattern, sync_update=True))
                
#Get summary statistics
list_0_runs =  pd.DataFrame(list_0_runs)
list_1_runs =  pd.DataFrame(list_1_runs)
list_2_runs =  pd.DataFrame(list_2_runs)
list_3_runs =  pd.DataFrame(list_3_runs)
list_4_runs =  pd.DataFrame(list_4_runs)
list_5_runs =  pd.DataFrame(list_5_runs)
#############
list_0_runs = list_0_runs.rename(columns =  {0: 'Hamming Distance', 1: "Energy", 2: "# Iterations"})
list_1_runs = list_1_runs.rename(columns =  {0: 'Hamming Distance', 1: "Energy", 2: "# Iterations"})
list_2_runs = list_2_runs.rename(columns =  {0: 'Hamming Distance', 1: "Energy", 2: "# Iterations"})
list_3_runs = list_3_runs.rename(columns =  {0: 'Hamming Distance', 1: "Energy", 2: "# Iterations"})
list_4_runs = list_4_runs.rename(columns =  {0: 'Hamming Distance', 1: "Energy", 2: "# Iterations"})
list_5_runs = list_5_runs.rename(columns =  {0: 'Hamming Distance', 1: "Energy", 2: "# Iterations"})

part_4_summary_stats = [list_0_runs, list_1_runs, list_2_runs, 
                         list_3_runs ,list_4_runs, list_5_runs]

#Print out the summary stats:
count = 0
for df in part_4_summary_stats:
    print('* * * * * * * * * * * * * * * * * * * *')
    print('For noise value', str(count), ':')
    print('Number of times network failed to settle: ', df.loc[df['# Iterations'] == 50, '# Iterations'].count())
    print("Average energy: ", df['Energy'].mean())
    print("Average hamming Distance: ", df['Hamming Distance'].mean())
    count += .1

