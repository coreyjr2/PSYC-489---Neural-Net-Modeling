#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 11:26:07 2021

@author: cjrichier
"""

########################
# dice rolling problem #
########################

import random
import sys


class Dice(object):
    def __init__(self, num_dice, num_sides):
        self.num_dice = num_dice
        self.num_sides = num_sides
        
    def roll_dice(self): #This line defines the function and its base arguments. 
        dice_list = []
        for dice in range(self.num_dice): #iterate over the indices in the length of num_dice 
            dice_list.append(random.randint(1, self.num_sides))
        print("These dice rolled to ", str(dice_list)) #Set the boundaries for the dice randomization, and then print
        print("The sum of these dice are ", str(sum(dice_list)))
    
#Test two normal dice
dice = Dice(2,6)
dice.roll_dice()

#Does it generalize?
dice = Dice(4,8)
dice.roll_dice()

dice = Dice(10,20)
dice.roll_dice()

#################################
# Odd integer counting function #
#################################

def count_odd_numbers(list_to_check): #This line defines the function and its base arguments. 
    odd_count = 0 
    non_int_count = 0
    for value in list_to_check:
        if isinstance(value, int) == True:
            if value % 2 != 0:
                odd_count += 1
        else:
            print("These numbers aren't all integers, you moron.")
            non_int_count += 1
            
    print('This list has', str(odd_count), 'odd numbers in it. Nice.')
    if non_int_count > 0:
        print("You included", str(non_int_count), 'values that are not all integers.')
        
count_odd_numbers([1,2,3,4,5])    
count_odd_numbers([1,2,3,4,5,7,8,9,234,54634,7345734,374,3745,75])         
count_odd_numbers([1,2,3.3,4,5])   
        
###################
# Make some Units #
###################

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
class Unit(object):
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
             
    def update_target_activation(self):
        #most basic activation function
            if self.net_input > self.threshold:
                self.target_activation = 1
            else:
                self.target_activation = 0
                
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

##### make a network to wire the units up
class network(object):
    def __init__(self, num_units, default_threshold = 0):
        self.units = []
        for i in range(num_units):
            self.units.append(Unit(self.units, default_threshold))
     # and connect them
        for unit_i in self.units:
            for unit_j in self.units:
                if not unit_i is unit_j:
                    unit_i.add_connection(unit_j)


#Print them to see if they connected
net = network(8, default_threshold = 0)
for unit in net.units:
    unit.show()
