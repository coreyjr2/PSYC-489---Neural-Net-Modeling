#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:26:07 2021

@author: cjrichier
"""

### this is inspired by in class method 
inputs = [0.0, 0.0, 0.0]
activations = [0.0, 0.0, 0.0]

#initialize starting states 
state_1 = [1,1,0]
state_2 = [1,0,0]
state_3 = [.5,0,0] 

#Make them a list     
starting_states = [state_1, state_2, state_3]

#Let's define activation functions
#subtraction function
def basic_subtraction_function(input):
    return input - .5
#ReLU function
def relu_function(input):
    if input > 1:
        return 1
    elif input < .75:
        return 0
    else:
        return input
   
def update_inputs(activations):
    these_inputs = [0.0, 0.0, 0.0]
    for i in range(len(activations)):
        for j in range(len(activations)):
            if i != j:
                these_inputs[i] += activations[j]
    return these_inputs

def run_units(starting_pattern, activation_function, n_iterations):
    activations = list(starting_pattern)
    for iteration in range(n_iterations):
        inputs = update_inputs(activations)
        #Update all activations
        for i in range(len(activations)):
            if activation_function == "subtraction":
                activations[i] = basic_subtraction_function(inputs[i])
            else:
                if activation_function == "relu":
                    activations[i] = relu_function(inputs[i])
                    
        print(str(iteration) +str(activations))
    print('Model training completed after ' + str(n_iterations) +' iterations')
       
        
#Try the subtraction methods     
for state in starting_states:
       run_units(state, "subtraction", 10)
#Try ReLU
for state in starting_states:
       run_units(state, "relu", 10)




def temporal_integrator_unit(act, input):
    return act + (0.5 * input * (1 - act))



def leaky_integrator_unit(act, input):
    return act + (0.5 * input * (1 - act) - (.1 * act))
def grossberg_integrator_unit(act, ex_input, in_input):
    return act + ((.5 * ex_input * (1-act)) + (in_input * (1 + act ))) - (.1 * act)


# Part 2a1
act=0
for iteration in range(10):
    act = temporal_integrator_unit(act, 1.5)
    print(str(iteration)+') '+str(act))

# a2)
act=0
for iteration in range(10):
    act = temporal_integrator_unit(act, 5.0)
    print(str(iteration)+') '+str(act))

''''How did the unit's activation at the end of (a1) differ 
from its activation at the end of (a2).  Why?'''
#a1 asymptotically approached 1 with more iterations, where as a2 became unstable very quickly and bounded between 
#increasingly polar postive and negative values, which would progress unchecked

# a3)
act=0
for iteration in range(10):
    act = temporal_integrator_unit(act, 3.0)
    print(str(iteration)+') '+str(act))

'''How did the unit's activation at the end of (a3) differ from its activation at the end of (a2).  Why?'''
#unit a2 bounced back and forth in increasing polarity, whereas unit a3 stays relatively bounded around 1

# a4) 
act=0
for iteration in range(2):
    act = temporal_integrator_unit(act, 1)
    print(str(iteration)+') '+str(act))
for iteration in range(5):
    act = temporal_integrator_unit(act, -1)
    print(str(iteration)+') '+str(act))

# a5) 
act=0
for iteration in range(10):
    act = temporal_integrator_unit(act, 1)
    print(str(iteration)+') '+str(act))
for iteration in range(5):
    act = temporal_integrator_unit(act, -1)
    print(str(iteration)+') '+str(act))

'''What unfortunate thing happened on (a5) and why didn't this happen on (a4)?'''
#a4 was able to recover with changing inputs. a5 was not, and it remains resistant to change once the activation approaches 1

# b1) 
act=0
for iteration in range(10):
    act = leaky_integrator_unit(act, 1)
    print(str(iteration)+') '+str(act))
for iteration in range(5):
    act = leaky_integrator_unit(act, -1)
    print(str(iteration)+') '+str(act))


'''How did (b1) differ from (a5) and why?'''
#b1 is a leaky integrator, meaning that at each time step it loses some of its activation value and this does not increase infinitely 
#in this way, it is more flexible to change in inputs than a5 was, which got stuck as it got too close to 1
#even as its value approached 1, when the input changed to -1 it was able to adjust

'''What would have happened on (a4) and (b1) 
if you had run them longer with a negative input?  Why?'''
#They are unbounded on the lower threshold. Thus, they would continue decreasing towards infinity and would likely not recover

# c1) 
act=0
for iteration in range(5):
    act = grossberg_integrator_unit(act, 1, 0)
    print(str(iteration)+') '+str(act))
for iteration in range(5):
    act = grossberg_integrator_unit(act,0, -1)
    print(str(iteration)+') '+str(act))

'''How did (c1) differ from (b1) and why?'''
#grossberg adapts very quickly to its changes in input, where as the leaky integrator does not do so as much







############################################################################
### This is the janky way I tried to do it... only partially works ####
### Tread here if ye dare
############################################################################

#Create a connection class
class Connection(object): #define the connection object
    
    def __init__(self, recipient, sender, weight = 0): #initialize the properties of the object
        self.owner = recipient #this is unit i
        self.sender = sender #this is unit j
        
        self.weight = weight #this connection has a weighting, evaluating how strong it is and if it is excitatory or inhibitory
    def update_weight(self):
        #This is the hopfield learning algorithim 
        self.weight += (2*self.owner.activation - 1) * (2*self.sender.activation - 1)
#Create a unit class
class HopUnit(object):
    def __init__(self, my_list, activation_function, threshold=0 ):
        self.index = len(my_list) # give the unit an index 
        self.threshold = threshold #set the threshold
        self.net_input = 0 #initialize net input to 0
        self.activation = 0 #initialize activation to 0
        self.target_activation = 0
        self.activation_function = activation_function #create an argument for which activation function to use when initializing the neuron
        self.connections = [] 
    def add_connection(self, sender, weight=0):
         self.connections.append(Connection(self, sender, weight)) #create connections
    def update_input(self):
         self.net_input = 0
         for connection in self.connections:
             #There has to be some way to partion both inhibitory and excitatory input but
             #I cannot figure out how 
             self.net_input += connection.weight * connection.sender.activation
    def update_target_activation(self, activation_function):
        #most basic activation function
        if activation_function == 'binary':  
            if self.net_input > self.threshold:
                self.target_activation = 1
              
            else:
                self.target_activation = 0
                
        #initial activation function, subtract .5 from net input
        elif activation_function == 'subtraction': 
        
            self.target_activation = self.net_input - .5 
        #ReLu activation function
        elif activation_function == 'ReLu': 
            if self.net_input > 1:
                self.target_activation = 1
            elif self.net_input < .75:
                self.target_activation = 0
            else:
                self.net_input = self.net_input
        #part 2 activation functions
        elif activation_function == 'Temp_Integrator':
        #Temporal integrator
        #delta(a_sub_i) = gamma(1 - a_sub_i)n_sub_i, where gamma = 0.5
            self.target_activation += self.target_activation + .5*(1 - Connection.sender.activation) * self.net_input
        elif activation_function == 'Leaky_Integrator':
        #Leaky integrator
        #delta(a_sub_i) = gamma(1 - a_sub_i)n_sub_i - small_del(a_sub_i) where gamma = 0.5 and small_del = .1
            self.target_activation = .5 * (1 - Connection.sender.activation) * self.net_input - (.1 * self.activation)
        else:
            if activation_function == 'Grossberg': # this is hella broken, do not use
        #Grossberg's leaky integrator -- which requires inhibitory and excitatory input to be partitioned
        #delta(a_sub_i) = gamma[(1 - a_sub_i)e_sub_i + (1 + a_sub_i)i_sub_i] - small_del(a_sub_i) where gamma = 0.5 and small_del = .1
                self.target_activation = .5 * (((1 - Connection.sender.activation) * excitatory.activation) + ((1 + connection.sender.activation) * self.inhibitory.activation)) * self.net_input - (.1 * self.activation) 
        
    def settled(self):
        return self.target_activation == self.activation
    def update_activation(self): 
        self.activation= self.target_activation
    def show(self):
        
        print ('\nUnit '+str(self.index)+':')
        print ('\tInput = '+str(self.net_input))
        print ('\tTarget activation = '+str(self.target_activation))
        print ('\tCurrent Activation =  '+str(self.activation))
        print ('\tHas connections from...')
        for connection in self.connections:
            print('\t Unit '+str(connection.sender.index)+' = '+str(connection.weight))


#do some testing:
#initialize 3 units
units = []
for i in range(3):
    units.append(HopUnit(units, 'Binary'))
  
    
#initialize their connections: 
for unit_i in units:
    for unit_j in units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)
        
#Get the status of each upon initialization:
for unit in units:
    unit.show()

train_pattern(units, state_1)

#define a function to train some patterns
def train_pattern(network, pattern):
    if len(network) != len(pattern):
        print('Fatal error -- dimensions of training pattern and network do not match. Try again, you dolt.')
        return False
    else: #impose the training pattern on the network
        for i in range(len(pattern)):
            network[i].activation = pattern[i]
            
        for unit in network:
            for connection in unit.connections:
                connection.update_weight()
                
    #for unit in network:
        #unit.show()
    for unit in network:
        unit.activation = 0

#Create a function to iterate pattern training as many times as needed
def iterate_model(network, pattern, n_iterations):
    for iteration in range(n_iterations):
        train_pattern(network, pattern)
        i=0
        i+=1
        if i == n_iterations:
            for unit in network:
                unit.show()

def reset_model(network):
    for unit in network:
        unit.__init__(sub_units,'Binary')
     
        unit.show()

iterate_model(sub_units, state_1, 2)

'''Part 1 (4 points):  
Build a three-node network in 
which each node has a 
connection strength of 1.0 to 
and from every 
other node except itself.'''
         
#initialize 3 units
sub_units = []
for i in range(3):
    sub_units.append(HopUnit(sub_units, 'Subtraction'))
      
#initialize their connections: 
for unit_i in sub_units:
    for unit_j in sub_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)
        
#Get the status of each upon initialization:
for unit in sub_units:
    unit.show()


#1
#Run the network for 10 interations using the following starting configurations:
#define some starting states
state_1 = [1,1,0]
state_2 = [1,0,0]
state_3 = [.5,0,0] 
      
#Make the states into alist
starting_states = [state_1, state_2, state_3]

#Run the model for ten iterations on each configuration:
#for state in starting_states:
    #iterate_model(sub_units, state, 10) #This doesn't work like I want it to
#Need to figure out a way to just get the final output for each model, 
#instead of testing it over and over again because it just trains them over
# and over and preserves the weights

#train the first pattern
iterate_model(sub_units, state_1, 10)
#and reset
reset_model(sub_units)
for unit_i in sub_units:
    for unit_j in sub_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)
#train the second pattern
iterate_model(sub_units, state_2, 10)
#and reset
reset_model(sub_units)
for unit_i in sub_units:
    for unit_j in sub_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)
#train the third pattern
iterate_model(sub_units, state_3, 10)
#and reset
reset_model(sub_units)
for unit_i in sub_units:
    for unit_j in sub_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)
#What happened and why?
'''The weights just kept increasing or decreasing depending on the starting activation.
There really is nothing super exciting about what it is doing, it would progress until
infinity left unchecked if it kept iterating'''

#2
#Use the same starting pattern, but update activation function to be ReLu:
relu_units = []
for i in range(3):
    relu_units.append(HopUnit(relu_units, 'ReLu'))
       
#initialize their connections: 
for unit_i in relu_units:
    for unit_j in relu_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)
        
#train the first pattern
iterate_model(relu_units, state_1, 10)
#and reset
reset_model(relu_units)
for unit_i in relu_units:
    for unit_j in relu_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)
#train the second pattern
iterate_model(relu_units, state_2, 10)
#and reset
reset_model(relu_units)
for unit_i in relu_units:
    for unit_j in relu_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)
#train the third pattern
iterate_model(relu_units, state_3, 10)
#and reset
reset_model(relu_units)
for unit_i in relu_units:
    for unit_j in relu_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)


#How does this change the network?
'''more or less operates the same way?'''

##############################
         '''Part 2'''
##############################

state_0 = [0,0,0]

#a1) Run Temporal integrator for 10 iterations with ni = 1.5 on each iteration
temp_units = []
for i in range(3):
    temp_units.append(HopUnit(temp_units, 'Temp_Integrator'))
       
#initialize their connections: 
for unit_i in temp_units:
    for unit_j in temp_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)

#set ni to 1.5
for unit in temp_units:
    unit.net_input = 1.5

#iterate the model
iterate_model(temp_units, state_0, 10)
#and reset
reset_model(temp_units)
for unit_i in temp_units:
    for unit_j in temp_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)


#a2) Run Temporal integrator for 10 iterations with ni = 5.0 on each iteration
temp_units = []
for i in range(3):
    temp_units.append(HopUnit(temp_units, 'Temp_Integrator'))
       
#initialize their connections: 
for unit_i in temp_units:
    for unit_j in temp_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)

#set ni to 5.0
for unit in temp_units:
    unit.net_input = 5.0

#iterate the model
iterate_model(temp_units, state_0, 10)
#and reset
reset_model(temp_units)
for unit_i in temp_units:
    for unit_j in temp_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)

#a3) Run Temporal integrator for 10 iterations with ni = 3.0 on each iteration
temp_units = []
for i in range(3):
    temp_units.append(HopUnit(temp_units, 'Temp_Integrator'))
       
#initialize their connections: 
for unit_i in temp_units:
    for unit_j in temp_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)

#set ni to 5.0
for unit in temp_units:
    unit.net_input = 3.0

#iterate the model
iterate_model(temp_units, state_0, 10)
#and reset
reset_model(temp_units)
for unit_i in temp_units:
    for unit_j in temp_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)

#a4) Run Temporal integrator for 2 iterations with ni = 1 on each iteration and then for 5 iterations with ni = -1.
temp_units = []
for i in range(3):
    temp_units.append(HopUnit(temp_units, 'Temp_Integrator'))
       
#initialize their connections: 
for unit_i in temp_units:
    for unit_j in temp_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)

#set ni to 1
for unit in temp_units:
    unit.net_input = 1

#iterate the model for two iterations
iterate_model(temp_units, state_0, 2)

#now set ni to -1
for unit in temp_units:
    unit.net_input = -1

#iterate the model for five iterations
iterate_model(temp_units, state_0, 5)

#and reset
reset_model(temp_units)
for unit_i in temp_units:
    for unit_j in temp_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)
#a5) Run Temporal integrator for 10 iterations with ni = 1 on each iteration and then for 5 iterations with ni = -1.
temp_units = []
for i in range(3):
    temp_units.append(HopUnit(temp_units, 'Temp_Integrator'))
       
#initialize their connections: 
for unit_i in temp_units:
    for unit_j in temp_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)

#set ni to 1
for unit in temp_units:
    unit.net_input = 1

#iterate the model for ten iterations
iterate_model(temp_units, state_0, 10)

#now set ni to -1
for unit in temp_units:
    unit.net_input = -1

#iterate the model for five iterations
iterate_model(temp_units, state_0, 5)

#and reset
reset_model(temp_units)
for unit_i in temp_units:
    for unit_j in temp_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)


#b1) Run Leaky integrator for 10 iterations with ni = 1 on each iteration and then for 5 iterations with ni = -1.
leak_units = []
for i in range(3):
    leak_units.append(HopUnit(leak_units, 'Leaky_Integrator'))
       
#initialize their connections: 
for unit_i in leak_units:
    for unit_j in leak_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)

#set ni to 1
for unit in leak_units:
    unit.net_input = 1

#iterate the model for ten iterations
iterate_model(leak_units, state_0, 10)

#now set ni to -1
for unit in leak_units:
    unit.net_input = -1

#iterate the model for five iterations
iterate_model(leak_units, state_0, 5)

#and reset
reset_model(leak_units)
for unit_i in leak_units:
    for unit_j in leak_units:
        if not unit_i is unit_j:
            unit_i.add_connection(unit_j)

#c1) Run Grossberg's leaky integrator for 10 iterations with ei = 1 and ii = 0, and then for 5 iterations with ei = 0 and ii = -1.
#make a grossberg unit
class GrossbergUnit(object):
    def __init__(self, my_list, activation_function, threshold=0 ):
        self.index = len(my_list) # give the unit an index 
        self.threshold = threshold
        self.net_input = 0 
        self.activation = 0 
        self.target_activation = 0
        self.activation_function = activation_function
        self.connections = [] 
    def add_connection(self, sender, weight=0):
         self.connections.append(Connection(self, sender, weight))
    def update_input(self):
        self.net_input 
        self.inhibitory_input = []
         self.excitatory_input = [] 
         for connection in self.connections:
             #There has to be some way to partion both inhibitory and excitatory input but
             #I cannot figure out how 
             self.inhibitory_input += connection.weight * connection.sender.activation
             self.excitatory_input += connection.weight * connection.sender.activation
    def update_target_activation(self):
        #Grossberg's leaky integrator
        #delta(a_sub_i) = gamma[(1 - a_sub_i)e_sub_i + (1 + a_sub_i)i_sub_i] - small_del(a_sub_i) where gamma = 0.5 and small_del = .1
                self.target_activation = .5 * (((1 - Connection.sender.activation) * excitatory.activation) + ((1 + connection.sender.activation) * self.inhibitory.activation)) * self.net_input - (.1 * self.activation) 
        
    def settled(self):
        return self.target_activation == self.activation
    def update_activation(self): 
        self.activation= self.target_activation
    def show(self):
        
        print ('\nUnit '+str(self.index)+':')
        print ('\tInput = '+str(self.net_input))
        print ('\tTarget activation = '+str(self.target_activation))
        print ('\tCurrent Activation =  '+str(self.activation))
        print ('\tHas connections from...')
        for connection in self.connections:
            print('\t Unit '+str(connection.sender.index)+' = '+str(connection.weight))

class GrossbergConnection(object): #define the connection object
    
    def __init__(self, recipient, sender, weight = 0): #initialize the properties of the object
        self.owner = recipient #this is unit i
        self.sender = sender #this is unit j
        
        self.weight = weight #this connection has a weighting, evaluating how strong it is and if it is excitatory or inhibitory
    def update_weight(self):
        #This is the hopfield learning algorithim 
        
        self.weight += (2*self.owner.activation - 1) * (2*self.sender.activation - 1)

?????
#How did the unit's activation at the end of (a1) differ from its activation at the end of (a2).  Why?
????
#How did the unit's activation at the end of (a3) differ from its activation at the end of (a2).  Why?
?????
#What unfortunate thing happened on (a5) and why didn't this happen on (a4)?
'''all the weights on A5 ended up being postive. It is because it is unbounded and
 and will continue to inflate with too many iterations'''
#How did (b1) differ from (a5) and why?
'''In principle, I understand 



#What would have happened on (a4) and (b1) if you had run them longer with a negative input?  Why?

#How did (c1) differ from (b1) and why?




    








#For the future, ignore


def update_target_activation(self):
        
        #most basic activation function
        '''if self.net_input > self.threshold:
            self.target_activation = 1
        else:
            self.target_activation = 0'''
            
            
        #alternate activation functions:
        #initial activation function, subtract .5 from net input
        self.target_activation = self.net_input - .5 
        
        #ReLu activation function
        #if self.net_input > 1:
           # self.target_activation = 1
        #elif self.net_input < .75:
            #self.target_activation = 0
        #else:
            #self.net_input = self.net_input
            
            
        #part 2 activation functions
        #Assume that e sub i = excitatory input and i sub i is inhibitory input. 
        #net input n sub i = e sub i + i sub i
 
        #Temporal integrator
        #delta(a_sub_i) = gamma(1 - a_sub_i)n_sub_i, where gamma = 0.5
        '''self.target_activation = .5*(1 - connection.sender.activation) * self.net_input'''

        #Leaky integrator
        #delta(a_sub_i) = gamma(1 - a_sub_i)n_sub_i - small_del(a_sub_i) where gamma = 0.5 and small_del = .1
        '''self.target_activation = .5 * (1 - connection.sender.activation) * self.net_input - (.1 * self.activation)'''
        #Grossberg's leaky integrator
        #delta(a_sub_i) = gamma[(1 - a_sub_i)e_sub_i + (1 + a_sub_i)i_sub_i] - small_del(a_sub_i) where gamma = 0.5 and small_del = .1
        '''self.target_activation = .5 * (((1 - connection.sender.activation) * excitatory.activation) + ((1 + connection.sender.activation) * self.inhibitory.activation)) * self.net_input - (.1 * self.activation)''' 
        



    
    class MakeHopNet(object):
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
        #training patterns
        self.training_patterns = []
        
    def train_pattern(self):
        pass
    def run_network(self, starting_state):
    
        pass 
    def compute_energy(self):
        
    def hamming_distance(self, pattern1, pattern2):
        pass
        
        
           