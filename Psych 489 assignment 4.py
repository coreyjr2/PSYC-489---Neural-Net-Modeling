#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 10:01:28 2021

@author: cjrichier
"""

#import libraries
import random, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

###########################################
######## Start of connection class ########
###########################################
class Connection(object):
    def __init__(self, recipient, sender):
        self.recipient = recipient #set connection to have the unit connect to another unit
        self.sender = sender #set who is sending the connection 
        self.weight = (random.random()-0.5)*2 #initialize the weight
        self.delta_weight = 0
    
    def update_weight(self, momentum):#update the weight of this connection
        self.weight += self.delta_weight
        self.delta_weight *= momentum
        
    def update_deltaweight(self, learning_rate):
        self.delta_weight += learning_rate * self.sender.activation * self.recipient.corrected_error

#####################################
######## Start of input unit ########
#####################################
class InputUnit(object):
    def __init__(self, my_list):
        self.activation = 0 
        self.net_input = 0 #initialize net input to 0
        self.outgoing_connections = [] #used for calculating error
        self.index = len(my_list) # give the unit an index   
    def show(self):
        print('************************************')
        print ('\nIntput Unit '+str(self.index)+':')
        print ('\tCurrent Activation =  '+str(self.activation))
        if len(self.outgoing_connections) > 1:
            print('\tHas outgoing connections to...')
            for connection in self.outgoing_connections:
                print('\t Unit '+str(connection.recipient.index)+' = '+str(connection.weight))
        else: 
            print ('\tHas no outgoing connections')
        print('************************************')

######################################
######## Start of hidden unit ########
######################################
class HiddenUnit(object):
    def __init__(self, my_list):
        self.activation = 0 
        self.net_input = 0 #initialize net input to 0
        self.raw_error = 0
        self.corrected_error = 0   
        self.incoming_connections = [] 
        self.outgoing_connections = [] #used for calculating error
        self.index = len(my_list) # give the unit an index
    def update_activation(self): 
        self.activation = 1/(1+math.exp(-self.net_input))
    def add_connection(self, sender):
        new_connection = Connection(self, sender)
        self.incoming_connections.append(new_connection)
        sender.outgoing_connections.append(new_connection)   
    def update_input(self):
        self.net_input = 0
        for connection in self.incoming_connections:
            self.net_input += connection.weight * connection.sender.activation
    def update_error(self):
        self.raw_error = 0.0
        for connection in self.outgoing_connections:
            self.raw_error += connection.weight * connection.recipient.corrected_error #This is the backpropogation!!!!!!
        self.corrected_error = self.raw_error * self.activation * (1.0 - self.activation) #The error to backpropogate
    def show(self):
        print('************************************')
        print ('\nHidden Unit '+str(self.index)+':')
        print ('\tInput = '+str(self.net_input))
        print ('\tCurrent Activation =  '+str(self.activation))
        if len(self.incoming_connections) > 1:
            print('\tHas incoming connections from...')
            for connection in self.incoming_connections:
                print('\t Unit '+str(connection.sender.index)+' = '+str(connection.weight))
        else: 
            print ('\tHas no incoming connections')
        if len(self.outgoing_connections) > 1:
            print('\tHas outgoing connections to...')
            for connection in self.outgoing_connections:
                print('\t Unit '+str(connection.recipient.index)+' = '+str(connection.weight))
        else: 
            print ('\tHas no outgoing connections')
        print('************************************')

######################################
######## Start of output unit ########
######################################
class OutputUnit(object):
    def __init__(self, my_list):
        self.activation = 0 
        self.net_input = 0 #initialize net input to 0
        self.raw_error = 0
        self.corrected_error = 0   
        self.incoming_connections = [] 
        self.outgoing_connections = [] #used for calculating error
        self.index = len(my_list) # give the unit an index 

    def update_activation(self): 
        self.activation = 1/(1+math.exp(-self.net_input))
    def add_connection(self, sender):
        new_connection = Connection(self, sender)
        self.incoming_connections.append(new_connection)
        sender.outgoing_connections.append(new_connection)   
    def update_input(self):
        self.net_input = 0
        for connection in self.incoming_connections:
            self.net_input += connection.weight * connection.sender.activation
    def update_error(self, desired_activation):
        self.raw_error = desired_activation - self.activation #The error to evaluate network performance for output layer
        self.corrected_error = self.raw_error * self.activation * (1.0 - self.activation) #The error to backpropogate
    def show(self):
        print('************************************')
        print ('\nOutput Unit '+str(self.index)+':')
        print ('\tInput = '+str(self.net_input))
        print ('\tCurrent Activation =  '+str(self.activation))
        if len(self.incoming_connections) > 1:
            print('\tHas incoming connections from...')
            for connection in self.incoming_connections:
                print('\t Unit '+str(connection.sender.index)+' = '+str(connection.weight))
        else: 
            print ('\tHas no incoming connections')
        print ('\tHas no outgoing connections')
        print('************************************')


######################################## 
######## Start of network class ########
########################################
class network(object):
 
    ###############################
    def __init__(self, layer_list):
    ###############################
        self.error_list = []
        self.epoch_list = []
        self.layers= []
        #self.lists is a list of lists
        #Construct the net
        for layer_index in range(len(layer_list)): 
            self.layers.append([])
            if layer_index == 0:
                #This is making the input layer:
                for i in range(layer_list[layer_index]):
                    self.layers[-1].append(InputUnit(self.layers[-1]))
            elif layer_index == len(layer_list) -1: 
                #This is making the output layer:
                for i in range(layer_list[layer_index]):
                    self.layers[-1].append(OutputUnit(self.layers[-1]))
            else:
                #This is making the hidden layers:
                for i in range(layer_list[layer_index]):
                    self.layers[-1].append(HiddenUnit(self.layers[-1]))

        #Connect layers to one another
        for layer_j in range(len(layer_list)-1):
            layer_i = layer_j+1
            for unit_i in self.layers[layer_i]:
                for unit_j in self.layers[layer_j]:
                    unit_i.add_connection(unit_j)
        #Initialize error            
        self.global_error = 0.0  
        #initialize epochs
        self.epoch = 0
      

    ###########################################
    def forward_prop_only(self, input_pattern):
    ###########################################
        for i in range(len(input_pattern)): 
            self.layers[0][i].activation = input_pattern[i]     
    
        for layer in self.layers:
            if not layer == self.layers[0]:
                for unit in layer:
                    try:
                        unit.update_input()
                    except:
                        print('Update input choked for some reason')
                    unit.update_activation()

    #################################################################################################################
    def train_network(self, input_patterns, output_patterns, learning_rate, momentum, error_threshold, error_method):
    #################################################################################################################
        '''important variables to initalize'''
        self.epoch = 0 
        self.error_list = []
        self.global_error = 12345678.9 #just to start up, and to flag when something gets weird with this 
        #error messeges on arguments
        if len(input_patterns) == 0:
            print('Error! Length of lists is zero')
            
            sys.exit()
        
        ##################################################
        # Each step through the while loop runs one epoch! 
        ##################################################
        
        while self.epoch < 1000 and self.global_error > error_threshold:
            self.global_error = 0 # to make it run, we need to set it this way.
            
            ############################################
            # impose the training pattern on the network
            ############################################
            for pattern_index in range(len(input_patterns)):
                input_pattern = input_patterns[pattern_index]
                output_pattern = output_patterns[pattern_index]
                for i in range(len(input_pattern)): #It think that this is an int and not a nested list
                    self.layers[0][i].activation = input_pattern[i] 
                #############################
                #Propogate activation forward
                #############################
                for layer in self.layers:
                    if not layer == self.layers[0]:
                        for unit in layer:
                            unit.update_input()
                            unit.update_activation()
                #Show the activations if you'd like            
                #self.show_network()
                ######################################################################
                #Compare output layer activation to desired 
                #activation and propogate the error backward and compute pattern error
                ######################################################################
                pattern_error = 0.0
                #error_list = [] #Show the error for diagnostics
                for unit in self.layers[-1]:
                    unit.update_error(output_pattern[unit.index])
                    if error_method == 'mse':
                        pattern_error += unit.raw_error**2
                    elif error_method == 'max':
                        if unit.raw_error**2 > pattern_error: #for max error
                            pattern_error = unit.raw_error**2 #for max error
                    #Set pattern error
                    #Set the value of global error to be that of the highest pattern error
                if error_method == 'max':
                    if pattern_error > self.global_error: #for max error
                        self.global_error = pattern_error #for max error
                    #error_list.append(str(unit.raw_error))#Show the error for diagnostics
                #print('List of output errors on epoch ', str(self.epoch), 'pattern number', str(pattern_index), error_list) #Show the error for diagnostics
                    
                #calculate pattern error
                if error_method == 'mse':
                    pattern_error = pattern_error/len(self.layers[-1]) 
                self.global_error += pattern_error
                #print(self.global_error) 
                layer_index = len(self.layers)-2
                while layer_index > 0:
                    for unit in self.layers[layer_index]:
                        unit.update_error()
                    layer_index -= 1
                    
                #update the delta weights
                for layer in self.layers:
                    if layer != self.layers[0]:
                        for unit in layer:
                            for connection in unit.incoming_connections:
                                connection.update_deltaweight(learning_rate)
               
             
            #####################################
            #update weight changes on connections
            #####################################
            for layer in self.layers:
                if not layer == self.layers[0]:
                    for unit in layer:
                        for connection in unit.incoming_connections:
                            connection.update_weight(momentum)

            ########################################################
            #Increment the epoch number and cycle back through again
            ########################################################       
            #finalize error calculation
            if error_method == 'mse':
                self.global_error = self.global_error/len(input_patterns)
            
            self.error_list.append(self.global_error)
            #print(global_error_list)
            
            self.epoch += 1
            #self.global_error += pattern_error/len(input_patterns)
            #print('Epoch number: ', str(self.epoch))
            #print('Global Error: ', str(self.global_error))
            #print('Mean Squared Error: ', str(self.global_error))
            
            #append the global error to a list to graph later
            
            
        #######################################
        # Do some summarization of the training
        #######################################
        if len(self.error_list) > 0:
            self.error_list = pd.DataFrame(self.error_list)
            #print(global_error_list)
            
            plt.plot(self.error_list)
            #title = str(input_patterns[pattern_index])
            plt.title('Error change', fontsize=14)
            plt.xlabel('# of Epochs', fontsize=14)
            if error_method == 'mse':
                plt.ylabel('Mean Squared Error', fontsize=14)
            elif error_method == 'max':
                plt.ylabel('Mean Squared Error', fontsize=14)
            plt.grid(True)
            plt.show()
        else:
            print('Warning! Empty global error list.')
            print('Misbehaving list with input patterns ', str(input_patterns))
            print(self.error_list)
        #print('Global Error: ', str(self.global_error))
        if self.epoch > 1000:
            print('This network failed to settle and ran for the maximum number of alloted iterations: ' + str(self.epoch))
        else:
            print('The Network ran for ' + str(self.epoch)+' iterations before it settled.')
        print('Global error of training: ', str(self.global_error))
        print('End of network run.')
        #print(' ')
        
        
    ######################################
    def test_network(self, input_pattern):
    ######################################
        self.forward_prop_only(input_pattern)

    ########################
    def show_network(self):
    ########################
        print('')
        for layer in self.layers:
            unit_activation_list = []
            for unit in layer:
                unit_activation_list.append(str(unit.activation))
            print('unit activations: ', unit_activation_list)
            
    ######################
    def show_result(self):
    ######################
        unit_activation_list = []
        for unit in self.layers[-1]:
            unit_activation_list.append(str("{:.3f}".format(unit.activation)))
        print('unit activations: ', str(unit_activation_list))

###################################        
######## End network class ########
###################################       

#################################      
######## Homework part 1 ########
#################################     

################################
### Non overlapping patterns ###
################################

### Non-overlapping patterns
Non_overlap_input = [[0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,1], 
                     [0,0,0,0,1,0,0,0,1,0],[0,0,0,1,0,0,0,0,0,1],
                     [0,0,0,0,0,0,0,1,0,0],[0,0,1,0,0,0,0,0,0,0],
                     [0,1,0,0,0,0,1,0,0,0],[1,0,0,0,0,0,1,0,0,0]]
                     
Non_overlap_output = [[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],
                      [0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]]

#Initialize the networks 
no_hidden_net_layers = [10, 4]
no_hidden_net = network(no_hidden_net_layers)
one_layer_net_layers = [10, 3, 4]
one_layer_net = network(one_layer_net_layers)
two_layer_net_layers = [10, 3, 3, 4]
two_layer_net = network(two_layer_net_layers)


#train the networks and evaluate how they do 
no_hidden_net.train_network(Non_overlap_input, Non_overlap_output, .3, .7, .01, 'max')
one_layer_net.train_network(Non_overlap_input, Non_overlap_output, .3, .7, .01, 'max')
two_layer_net.train_network(Non_overlap_input, Non_overlap_output, .3, .7, .01, 'max')


#Test to see the activations
print(' ')
print('Non-overlapping results: ')
print('**************************')
print('No hidden layer network predicted activations: ')
for input_pattern in Non_overlap_input:
    print("input pattern: ", str(input_pattern))
    no_hidden_net.forward_prop_only(input_pattern)
    no_hidden_net.show_result() 
print('One layer network predicted activations: ')
for input_pattern in Non_overlap_input:
    print("input pattern: ", str(input_pattern))
    one_layer_net.forward_prop_only(input_pattern)
    one_layer_net.show_result() 
print('Two layer network predicted activations: ')
for input_pattern in Non_overlap_input:
    print("input pattern: ", str(input_pattern))
    two_layer_net.forward_prop_only(input_pattern)
    two_layer_net.show_result() 


'''*** Please note that the output tracks the summary statistics for each run, 
which are the error and number of iterations. ***'''


'''When I ran this, the no hidden layer net ran for 663 iterations, one layer at 1000 (maxed out), and the two layer at 963'''
'''Plots are generated by the code, but I am attaching as a word document in the case they do not appear.'''
'''all the networks can figure out how to forward prop and activate the correct neuron'''

###################################
### Linearly seperable patterns ###
###################################


Lin_sep_input = [[1,1,0,1,0,0,0,0,0,1],[1,1,0,0,1,0,0,0,0,1],
                 [1,1,0,0,0,1,0,0,1,0],[1,1,0,1,0,0,1,0,1,0],
                 [1,0,1,1,0,0,0,1,0,0],[1,0,1,0,1,0,0,1,0,0],
                 [1,0,1,0,0,1,1,0,0,0],[1,0,1,0,1,0,0,1,0,0]]
Lin_sep_output = [[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0],
                  [0,0,1,0],[0,0,1,0],[0,0,0,1],[0,0,0,1]]

#Initialize the networks 
no_hidden_net_layers = [10, 4]
no_hidden_net = network(no_hidden_net_layers)

one_layer_net_layers = [10, 3, 4]
one_layer_net = network(one_layer_net_layers)

two_layer_net_layers = [10, 3, 3, 4]
two_layer_net = network(two_layer_net_layers)

#train the networks and evaluate how they do 
no_hidden_net.train_network(Lin_sep_input, Lin_sep_output, .3, .7, .01, 'max')
one_layer_net.train_network(Lin_sep_input, Lin_sep_output, .3, .7, .01, 'max')
two_layer_net.train_network(Lin_sep_input, Lin_sep_output, .3, .7, .01, 'max')

#Test to see the activations
print(' ')
print('Linearly seperable results: ')
print('**************************')
print('No hidden layer network predicted activations: ')
for input_pattern in Lin_sep_input:
    print("input pattern: ", str(input_pattern))
    no_hidden_net.forward_prop_only(input_pattern)
    no_hidden_net.show_result() 
print('One layer network predicted activations: ')
for input_pattern in Lin_sep_input:
    print("input pattern: ", str(input_pattern))
    one_layer_net.forward_prop_only(input_pattern)
    one_layer_net.show_result() 
print('Two layer network predicted activations: ')
for input_pattern in Lin_sep_input:
    print("input pattern: ", str(input_pattern))
    two_layer_net.forward_prop_only(input_pattern)
    two_layer_net.show_result() 

'''When I ran this, the no hidden layer net ran for 663 iterations, one layer at 1000 (maxed out), and the two layer at 963'''

#######################################
### Non-linearly seperable patterns ###
#######################################

### Non-linearly seperable patterns
Non_lin_input = [[1,1,1,1,1,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1],
                 [1,1,0,1,1,0,0,1,0,0],[0,0,1,0,0,0,1,0,1,1]]
Non_lin_output = [[1,0,0,0],[1,0,0,0],[0,1,0,0],[0,1,0,0]]

#Initialize the networks 
no_hidden_net_layers = [10, 4]
no_hidden_net = network(no_hidden_net_layers)


one_layer_net_layers = [10, 3, 4]
one_layer_net = network(one_layer_net_layers)

two_layer_net_layers = [10, 3, 3, 4]
two_layer_net = network(two_layer_net_layers)


#train the networks and evaluate how they do 
no_hidden_net.train_network(Non_lin_input, Non_lin_output, .3, .7, .01, 'max')
one_layer_net.train_network(Non_lin_input, Non_lin_output, .3, .7, .01, 'max')
two_layer_net.train_network(Non_lin_input, Non_lin_output, .3, .7, .01, 'max')

'''In this set, the no hidden layer net ran for 1000 iterations (maxed out), one layer at 752, and the two layer at 590'''
'''one and two layer networks outperformed no hidden layer in terms of error.'''
#Test to see the activations
print(' ')
print('Non -Linearly seperable results: ')
print('**********************************')
print('No hidden layer network predicted activations: ')
for input_pattern in Non_lin_input:
    print("input pattern: ", str(input_pattern))
    no_hidden_net.forward_prop_only(input_pattern)
    no_hidden_net.show_result() 
print('One layer network predicted activations: ')
for input_pattern in Non_lin_input:
    print("input pattern: ", str(input_pattern))
    one_layer_net.forward_prop_only(input_pattern)
    one_layer_net.show_result() 
print('Two layer network predicted activations: ')
for input_pattern in Non_lin_input:
    print("input pattern: ", str(input_pattern))
    two_layer_net.forward_prop_only(input_pattern)
    two_layer_net.show_result() 


#Part 1 responses

'''I decided to make three networks, one with no hidden layers, another with one hidden
layer, and yet another with two hidden layers. Early on, the net with no hidden layers learns faster.
performs better, by learning quicker. However, with non linearly seperable patterns the network 
with no hidden layers generally can't figure out what to do (however, after many sessions with 
working with Dr. Hummel, somehow it can figure it out every now and then, just in case there is a 
spurious result. We have no idea why this might be the case, but I still grasp the principle a 
network with no hidden layers should not be able to map things in the input space that require
more complex logic to map - hence the need for extra layers to capture them.)'''


#################################      
######## Homework part 2 ########
#################################     
print(' ')
print('Start of Part 2')
print('* * * * * * * *')
##########
# Part 2A # 
##########
input_patterns = [[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0],
                  [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0]]
output_patterns = [[1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,0,1,0,0,0,0,0], [0,0,0,1,0,0,0,0],
                  [0,0,0,0,1,0,0,0], [0,0,0,0,0,1,0,0], [0,0,0,0,0,0,1,0]]


#Initialize the network     
layer_list = [8, 3, 8]
autoencoder = network(layer_list)
print(' ')
#train the network
print('Part 2A: training the autoencoder on input patterns')
print('* * * * * * * * * * * * * * * * * * * * * * * * * * *')
autoencoder.train_network(input_patterns, output_patterns, .3, .7, .1, 'max')



print('Predicted activations: ')
for input_pattern in input_patterns:
    print("input pattern: ", str(input_pattern))
    autoencoder.forward_prop_only(input_pattern)
    autoencoder.show_result() 

'''The network generally knows the input to output mapping based on the training.
However, if a few cases it was a little confused based on the parameters I chose. The first
and fifth unut got equally activated at a low value of .278 when trying to train the fifth
pattern.'''


###########
# Part 2B # 
###########
print(' ')
autoencoder.test_network([0,0,0,0,0,0,0,1])
print('Part 2B: Result of testing new pattern ', str([0,0,0,0,0,0,0,1]))
autoencoder.show_result()        
     
'''The network did not know what to make of the new pattern it was shown on the eight unit,
it instead gave a low value of activation of all units, indicating it has no idea. 
Can't generalize.'''

print(' ') 
###########
# Part 2C # 
###########

additional_input_patterns = [[1,1,1,0,0,0,0,0], [0,0,0,1,1,1,0,0]]
additional_output_patterns = [[1,1,1,0,0,0,0,0], [0,0,0,1,1,1,0,0]]

print('Part 2C: training the autoencoder on additional input patterns')
autoencoder.train_network(additional_input_patterns, additional_output_patterns, .3, .7, .01, 'mse')
print(' ')
print('Try on some patterns it does not know formally via training: ')
autoencoder.test_network([0,0,1,1,0,0,0,0])
print('Result of testing new pattern ', str([0,0,1,1,0,0,0,0]))
autoencoder.show_result()  
print(' ')
print('Try on some patterns it does not know formally via training: ')
autoencoder.test_network([0,1,0,0,0,1,0,0])
print('Result of testing new pattern ', str([0,1,0,0,0,1,0,0]))
autoencoder.show_result()  


#Part 2C reponse:
'''The network here ran for 100 iterations and did not settle below the error threshold of .1,
the global error was .06. When given more patterns, it doesn't really figure out how 
to map every single input-output pair. It usually can get one unit reasonably correct
but then is confused about the others. Still not generalizing this rule very well.'''

#################################      
######## Homework part 3 ########
#################################     
print('Start of Part 3')
print('* * * * * * * *')

autoencoder.test_network([0,0,0,0,0,0,0,1])
print('Result of testing new pattern ', str([0,0,0,0,0,0,0,1]))
autoencoder.show_result()  

'''The network still fails to determine which neuron in the output layer should be most active
in this case I ran the code, the network flagged the fifth neuron as being most active.
This suggests it doesn't really know how to generalize to patterns it hasn't seen, even if in
a sense we have given it the most likely pattern for it to figure out. Additonaly training did 
not help that much.'''





