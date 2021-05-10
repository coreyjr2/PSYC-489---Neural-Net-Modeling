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


#Load the needed libraries
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
# Necessary for visualization
from nilearn import plotting, datasets

import csv
import urllib.request as urllib2
# matplotlib
import matplotlib.pyplot as plt # For changing the color maps
from matplotlib import cm # cm=colormap
from matplotlib.colors import ListedColormap, LinearSegmentedColormap



'''Set the path, and create some variables'''
# The download cells will store the data in nested directories starting here:
HCP_DIR = "/Volumes/Byrgenwerth/Datasets/HCP/"
HCP_DIR_REST = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_rest/subjects/"
HCP_DIR_TASK = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_task/subjects/"
HCP_DIR_EVS = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_task/"
HCP_DIR_BEHAVIOR = "/Volumes/Byrgenwerth/Datasets/HCP/hcp_behavior/"
if not os.path.isdir(HCP_DIR): os.mkdir(HCP_DIR)
# The data shared for NMA projects is a subset of the full HCP dataset
N_SUBJECTS = 339
# The data have already been aggregated into ROIs from the Glasesr parcellation
N_PARCELS = 360
# The acquisition parameters for all tasks were identical
TR = 0.72  # Time resolution, in sec
# The parcels are matched across hemispheres with the same order
HEMIS = ["Right", "Left"]
# Each experiment was repeated multiple times in each subject
N_RUNS_REST = 4
N_RUNS_TASK = 2
# Time series data are organized by experiment, with each experiment
# having an LR and RL (phase-encode direction) acquistion
BOLD_NAMES = [ "rfMRI_REST1_LR", 
              "rfMRI_REST1_RL", 
              "rfMRI_REST2_LR", 
              "rfMRI_REST2_RL", 
              "tfMRI_MOTOR_RL", 
              "tfMRI_MOTOR_LR",
              "tfMRI_WM_RL", 
              "tfMRI_WM_LR",
              "tfMRI_EMOTION_RL", 
              "tfMRI_EMOTION_LR",
              "tfMRI_GAMBLING_RL", 
              "tfMRI_GAMBLING_LR", 
              "tfMRI_LANGUAGE_RL", 
              "tfMRI_LANGUAGE_LR", 
              "tfMRI_RELATIONAL_RL", 
              "tfMRI_RELATIONAL_LR", 
              "tfMRI_SOCIAL_RL", 
              "tfMRI_SOCIAL_LR"]
# This will use all subjects:
subjects = range(N_SUBJECTS)


'''You may want to limit the subjects used during code development. This 
will only load in 10 subjects if you use this list.'''
SUBJECT_SUBSET = 10
test_subjects = range(SUBJECT_SUBSET)


'''import the demographics and bheavior data'''
demographics = pd.read_csv('/Volumes/Byrgenwerth/Datasets/HCP/HCP_demographics/demographics_behavior.csv')
#What is our gender breakdown?
demographics['Gender'].value_counts()
demographics['Age'].value_counts()



'''this is useful for visualizing:'''
with np.load(f"{HCP_DIR}hcp_atlas.npz") as dobj:
  atlas = dict(**dobj)

'''let's generate some information about the regions using the rest data'''
regions = np.load("/Volumes/Byrgenwerth/Datasets/HCP/hcp_rest/regions.npy").T
region_info = dict(
    name=regions[0].tolist(),
    network=regions[1],
    myelin=regions[2].astype(np.float),
)

print(region_info)

'''Now let's define a few helper functions'''
def get_image_ids(name):
  """Get the 1-based image indices for runs in a given experiment.

    Args:
      name (str) : Name of experiment ("rest" or name of task) to load
    Returns:
      run_ids (list of int) : Numeric ID for experiment image files

  """
  run_ids = [
    i for i, code in enumerate(BOLD_NAMES, 1) if name.upper() in code
  ]
  if not run_ids:
    raise ValueError(f"Found no data for '{name}''")
  return run_ids

def load_rest_timeseries(subject, name, runs=None, concat=True, remove_mean=True):
  """Load timeseries data for a single subject.
  
  Args:
    subject (int): 0-based subject ID to load
    name (str) : Name of experiment ("rest" or name of task) to load
    run (None or int or list of ints): 0-based run(s) of the task to load,
      or None to load all runs.
    concat (bool) : If True, concatenate multiple runs in time
    remove_mean (bool) : If True, subtract the parcel-wise mean

  Returns
    ts (n_parcel x n_tp array): Array of BOLD data values

  """
  # Get the list relative 0-based index of runs to use
  if runs is None:
    runs = range(N_RUNS_REST) if name == "rest" else range(N_RUNS_TASK)
  elif isinstance(runs, int):
    runs = [runs]

  # Get the first (1-based) run id for this experiment 
  offset = get_image_ids(name)[0]

  # Load each run's data
  bold_data = [
      load_single_rest_timeseries(subject, offset + run, remove_mean) for run in runs
  ]

  # Optionally concatenate in time
  if concat:
    bold_data = np.concatenate(bold_data, axis=-1)

  return bold_data


def load_task_timeseries(subject, name, runs=None, concat=True, remove_mean=True):
  """Load timeseries data for a single subject.
  
  Args:
    subject (int): 0-based subject ID to load
    name (str) : Name of experiment ("rest" or name of task) to load
    run (None or int or list of ints): 0-based run(s) of the task to load,
      or None to load all runs.
    concat (bool) : If True, concatenate multiple runs in time
    remove_mean (bool) : If True, subtract the parcel-wise mean

  Returns
    ts (n_parcel x n_tp array): Array of BOLD data values

  """
  # Get the list relative 0-based index of runs to use
  if runs is None:
    runs = range(N_RUNS_REST) if name == "rest" else range(N_RUNS_TASK)
  elif isinstance(runs, int):
    runs = [runs]

  # Get the first (1-based) run id for this experiment 
  offset = get_image_ids(name)[0]

  # Load each run's data
  bold_data = [
      load_single_task_timeseries(subject, offset + run, remove_mean) for run in runs
  ]

  # Optionally concatenate in time
  if concat:
    bold_data = np.concatenate(bold_data, axis=-1)

  return bold_data



def load_single_rest_timeseries(subject, bold_run, remove_mean=True):
  """Load timeseries data for a single subject and single run.
  
  Args:
    subject (int): 0-based subject ID to load
    bold_run (int): 1-based run index, across all tasks
    remove_mean (bool): If True, subtract the parcel-wise mean

  Returns
    ts (n_parcel x n_timepoint array): Array of BOLD data values

  """
  bold_path = f"{HCP_DIR}/hcp_rest/subjects/{subject}/timeseries"
  bold_file = f"bold{bold_run}_Atlas_MSMAll_Glasser360Cortical.npy"
  ts = np.load(f"{bold_path}/{bold_file}")
  if remove_mean:
    ts -= ts.mean(axis=1, keepdims=True)
  return ts

def load_single_task_timeseries(subject, bold_run, remove_mean=True):
  """Load timeseries data for a single subject and single run.
  
  Args:
    subject (int): 0-based subject ID to load
    bold_run (int): 1-based run index, across all tasks
    remove_mean (bool): If True, subtract the parcel-wise mean

  Returns
    ts (n_parcel x n_timepoint array): Array of BOLD data values

  """
  bold_path = f"{HCP_DIR}/hcp_task/subjects/{subject}/timeseries"
  bold_file = f"bold{bold_run}_Atlas_MSMAll_Glasser360Cortical.npy"
  ts = np.load(f"{bold_path}/{bold_file}")
  if remove_mean:
    ts -= ts.mean(axis=1, keepdims=True)
  return ts

def load_evs(subject, name, condition):
  """Load EV (explanatory variable) data for one task condition.

  Args:
    subject (int): 0-based subject ID to load
    name (str) : Name of task
    condition (str) : Name of condition

  Returns
    evs (list of dicts): A dictionary with the onset, duration, and amplitude
      of the condition for each run.

  """
  evs = []
  for id in get_image_ids(name):
    task_key = BOLD_NAMES[id - 1]
    ev_file = f"{HCP_DIR_EVS}subjects/{subject}/EVs/{task_key}/{condition}.txt"
    ev_array = np.loadtxt(ev_file, ndmin=2, unpack=True)
    ev = dict(zip(["onset", "duration", "amplitude"], ev_array))
    evs.append(ev)
  return evs


####################################
 ####### Taks Data Analysis #######
####################################

'''Make a list of the task names. This will be helpful in the future'''
tasks_names = ["motor", "wm", "gambling", "emotion", "language", "relational", "social"]

'''Now let's switch to doing some task-based 
analysis. Here are some helper functions for that.'''
def condition_frames(run_evs, skip=0):
  """Identify timepoints corresponding to a given condition in each run.

  Args:
    run_evs (list of dicts) : Onset and duration of the event, per run
    skip (int) : Ignore this many frames at the start of each trial, to account
      for hemodynamic lag

  Returns:
    frames_list (list of 1D arrays): Flat arrays of frame indices, per run

  """
  frames_list = []
  for ev in run_evs:

    # Determine when trial starts, rounded down
    start = np.floor(ev["onset"] / TR).astype(int)

    # Use trial duration to determine how many frames to include for trial
    duration = np.ceil(ev["duration"] / TR).astype(int)

    # Take the range of frames that correspond to this specific trial
    frames = [s + np.arange(skip, d) for s, d in zip(start, duration)]

    frames_list.append(np.concatenate(frames))

  return frames_list


def selective_average(timeseries_data, ev, skip=0):
  """Take the temporal mean across frames for a given condition.

  Args:
    timeseries_data (array or list of arrays): n_parcel x n_tp arrays
    ev (dict or list of dicts): Condition timing information
    skip (int) : Ignore this many frames at the start of each trial, to account
      for hemodynamic lag

  Returns:
    avg_data (1D array): Data averagted across selected image frames based
    on condition timing

  """
  # Ensure that we have lists of the same length
  if not isinstance(timeseries_data, list):
    timeseries_data = [timeseries_data]
  if not isinstance(ev, list):
    ev = [ev]
  if len(timeseries_data) != len(ev):
    raise ValueError("Length of `timeseries_data` and `ev` must match.")

  # Identify the indices of relevant frames
  frames = condition_frames(ev, skip)

  # Select the frames from each image
  selected_data = []
  for run_data, run_frames in zip(timeseries_data, frames):
    run_frames = run_frames[run_frames < run_data.shape[1]]
    selected_data.append(run_data[:, run_frames])

  # Take the average in each parcel
  avg_data = np.concatenate(selected_data, axis=-1).mean(axis=-1)

  return avg_data





'''load in the timeseries for each task'''
timeseries_motor = []
for subject in subjects:
    timeseries_motor.append(load_task_timeseries(subject, "motor", concat=True))
print(timeseries_motor)
timeseries_wm = []
for subject in subjects:
  timeseries_wm.append(load_task_timeseries(subject, "wm", concat=True))
print(timeseries_wm)
timeseries_gambling = []
for subject in subjects:
  timeseries_gambling.append(load_task_timeseries(subject, "gambling", concat=True))
print(timeseries_gambling)
timeseries_emotion = []
for subject in subjects:
  timeseries_emotion.append(load_task_timeseries(subject, "emotion", concat=True))
print(timeseries_emotion)
timeseries_language = []
for subject in subjects:
  timeseries_language.append(load_task_timeseries(subject, "language", concat=True))
print(timeseries_language)
timeseries_relational = []
for subject in subjects:
  timeseries_relational.append(load_task_timeseries(subject, "relational", concat=True))
print(timeseries_relational)
timeseries_social = []
for subject in subjects:
  timeseries_social.append(load_task_timeseries(subject, "social", concat=True))
print(timeseries_social)


'''now let's make FC matrices for each task'''

'''Initialize the matrices'''
fc_matrix_task = []
fc_matrix_motor = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_wm = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_gambling = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_emotion = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_language = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_relational = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))
fc_matrix_social = np.zeros((N_SUBJECTS, N_PARCELS, N_PARCELS))


'''calculate the correlations (FC) for each task'''
for subject, ts in enumerate(timeseries_motor):
  fc_matrix_motor[subject] = np.corrcoef(ts)
for subject, ts in enumerate(timeseries_wm):
  fc_matrix_wm[subject] = np.corrcoef(ts)
for subject, ts in enumerate(timeseries_gambling):
  fc_matrix_gambling[subject] = np.corrcoef(ts)
for subject, ts in enumerate(timeseries_emotion):
  fc_matrix_emotion[subject] = np.corrcoef(ts)
for subject, ts in enumerate(timeseries_language):
  fc_matrix_language[subject] = np.corrcoef(ts)
for subject, ts in enumerate(timeseries_relational):
  fc_matrix_relational[subject] = np.corrcoef(ts)
for subject, ts in enumerate(timeseries_social):
  fc_matrix_social[subject] = np.corrcoef(ts)

'''Initialize the vector form of each task, 
where each row is a participant and each column is a connection'''
vector_motor = np.zeros((N_SUBJECTS, 64620))
vector_wm = np.zeros((N_SUBJECTS, 64620))
vector_gambling = np.zeros((N_SUBJECTS, 64620))
vector_emotion = np.zeros((N_SUBJECTS, 64620))
vector_language = np.zeros((N_SUBJECTS, 64620))
vector_relational = np.zeros((N_SUBJECTS, 64620))
vector_social = np.zeros((N_SUBJECTS, 64620))

'''import a package to extract the diagonal of the correlation matrix, as well as
initializing a list of the subset of subjects. It is a neccesary step in appending the list 
of subjects to the connection data'''
from nilearn.connectome import sym_matrix_to_vec
subject_list = np.array(np.unique(range(339)))


for subject in range(subject_list.shape[0]):
    vector_motor[subject,:] = sym_matrix_to_vec(fc_matrix_motor[subject,:,:], discard_diagonal=True)
    vector_motor[subject,:] = fc_matrix_motor[subject][np.triu_indices_from(fc_matrix_motor[subject], k=1)]
for subject in range(subject_list.shape[0]):
    vector_wm[subject,:] = sym_matrix_to_vec(fc_matrix_wm[subject,:,:], discard_diagonal=True)
    vector_wm[subject,:] = fc_matrix_wm[subject][np.triu_indices_from(fc_matrix_wm[subject], k=1)]
for subject in range(subject_list.shape[0]):
    vector_gambling[subject,:] = sym_matrix_to_vec(fc_matrix_gambling[subject,:,:], discard_diagonal=True)
    vector_gambling[subject,:] = fc_matrix_gambling[subject][np.triu_indices_from(fc_matrix_gambling[subject], k=1)]
for subject in range(subject_list.shape[0]):
    vector_emotion[subject,:] = sym_matrix_to_vec(fc_matrix_emotion[subject,:,:], discard_diagonal=True)
    vector_emotion[subject,:] = fc_matrix_emotion[subject][np.triu_indices_from(fc_matrix_emotion[subject], k=1)]
for subject in range(subject_list.shape[0]):
    vector_language[subject,:] = sym_matrix_to_vec(fc_matrix_language[subject,:,:], discard_diagonal=True)
    vector_language[subject,:] = fc_matrix_language[subject][np.triu_indices_from(fc_matrix_language[subject], k=1)]
for subject in range(subject_list.shape[0]):
    vector_relational[subject,:] = sym_matrix_to_vec(fc_matrix_relational[subject,:,:], discard_diagonal=True)
    vector_relational[subject,:] = fc_matrix_relational[subject][np.triu_indices_from(fc_matrix_relational[subject], k=1)]
for subject in range(subject_list.shape[0]):
    vector_social[subject,:] = sym_matrix_to_vec(fc_matrix_social[subject,:,:], discard_diagonal=True)
    vector_social[subject,:] = fc_matrix_social[subject][np.triu_indices_from(fc_matrix_social[subject], k=1)]



'''remove stuff we don't need to save memory'''
del timeseries_motor
del timeseries_wm
del timeseries_gambling
del timeseries_emotion
del timeseries_language
del timeseries_relational
del timeseries_social

del fc_matrix_motor 
del fc_matrix_wm 
del fc_matrix_gambling 
del fc_matrix_emotion
del fc_matrix_language
del fc_matrix_relational
del fc_matrix_social


'''make everything pandas dataframes'''
emotion_brain = pd.DataFrame(vector_emotion)
gambling_brain = pd.DataFrame(vector_gambling)
language_brain = pd.DataFrame(vector_language)
motor_brain = pd.DataFrame(vector_motor)
relational_brain= pd.DataFrame(vector_relational)
social_brain = pd.DataFrame(vector_social)
wm_brain = pd.DataFrame(vector_wm)


'''Delete the old vectors to save space'''
del vector_motor 
del vector_wm 
del vector_gambling 
del vector_emotion 
del vector_language 
del vector_relational 
del vector_social 


emotion_brain['task'] =1
gambling_brain['task'] =2
language_brain['task'] =3
motor_brain['task'] =4
relational_brain['task'] =5
social_brain['task'] =6
wm_brain['task'] =7


task_data = pd.DataFrame(np.concatenate((emotion_brain, gambling_brain,  language_brain,
          motor_brain, relational_brain, social_brain, wm_brain), axis = 0))
X = task_data.iloc[:, :-1]
y = task_data.iloc[:,-1]


list_of_task_labels = []
for value in y:
    print(value)
    if value == 1:
        list_of_task_labels.append([1,0,0,0,0,0,0])
    if value == 2:
        list_of_task_labels.append([0,1,0,0,0,0,0])
    if value == 3:
        list_of_task_labels.append([0,0,1,0,0,0,0])
    if value == 4:
        list_of_task_labels.append([0,0,0,1,0,0,0])
    if value == 5:
        list_of_task_labels.append([0,0,0,0,1,0,0])
    if value == 6:
        list_of_task_labels.append([0,0,0,0,0,1,0])
    if value == 7:
        list_of_task_labels.append([0,0,0,0,0,0,1])
    



'''make more space'''
del emotion_brain
del gambling_brain
del language_brain
del motor_brain
del relational_brain
del social_brain
del wm_brain                 
        

#Use PCA to reduce dimensions of brain data

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


#Test optimal number of components
pca2 = PCA().fit(X)
plt.plot(pca2.explained_variance_, linewidth=2)
plt.xlabel('Components')
plt.ylabel('Explained Variaces')
plt.show()
#Looks like it may only be about 50 or so

pca = PCA(n_components = 50)
scaled_data = StandardScaler().fit_transform(X)
scaled_data = pd.DataFrame(scaled_data)
pca_result = pca.fit_transform(scaled_data)
pca_result = pd.DataFrame(pca_result)
pca_result = StandardScaler().fit_transform(pca_result)

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

pca_result = NormalizeData(pca_result)

components_data = pd.DataFrame(pca_result)
components_data['First PCA Component'] = pca_result[:,0]
components_data['Second PCA Component'] = pca_result[:,1] 



list_of_brain_data = pca_result.tolist()



##################################################
##### Time to build and train the model ##########
##################################################

#Build a network with two hidden layers
two_layer_net_layers = [50, 10, 10, 7]
network = network(two_layer_net_layers)


#train the networks and evaluate how they do 
network.train_network(list_of_brain_data, list_of_task_labels, .3, .7, .01, 'max')
#uh oh... doesn't look so good

#Try to run it with random instances of 
network.forward_prop_only(random.choice(list_of_brain_data))
network.show_result() 
#confirmed to be crap. But it runs!





