# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 09:03:04 2023

@author: bront
"""

import random
import csv
import os
import copy

# Set the seed value to ensure reproducibility
seed_value = 69
random.seed(seed_value)

def check_spacing(numbers, min_participant_break, probe_interval):
    """
    This function ensures that within an order, probe durations do not repeat 
    and that there is sufficient spacing between probes.

    Parameters
    ----------
    numbers : list
        list of probe timings
    min_participant_break : integer
        min amount of time between probes for a person (in 'order' space)
    probe_interval : integer
        how many seconds between probes across participants

    Returns
    -------
    bool
        True or false; have conditions been met
    encountered_spacings
        This is essential 'durations' between probes that we use later on

    """
    spacings = set()
    encountered_spacings = []  # List to store encountered spacings

    # Calculate first spacing by taking first probe and adding pre_clip secs
    first_spacing = abs(numbers[0] - 0)
    
    spacings.add(first_spacing)
    encountered_spacings.append(first_spacing)
    
    for i in range(len(numbers) - 1):
        spacing = abs(numbers[i] - numbers[i+1])
        
        if spacing < (min_participant_break ) or spacing in spacings:
            return False, []  # Return an empty list along with False
        
        spacings.add(spacing)
        encountered_spacings.append(spacing)  # Save the encountered spacing
    
    return True, encountered_spacings  # Return True and the list of encountered spacings

def initialize_flexible_dict(num_keys):
    """
    Creates dictionary based on number of keys (which is num_participants needed
    for min coverage.

    Parameters
    ----------
    num_keys : int
        num_participants needed for min coverage

    Returns
    -------
    order_dict : dictionary
        Dictionary for adding orders to.

    """
    order_dict = {}
    for i in range(1, num_keys + 1):
        key = str(i)
        order_dict[key] = []
    return order_dict

def generate_order_dict(value_mapping, preclip_secs, num_participants, num_samples_per_interval, min_participant_break, probe_interval):
    condition = False # set condition to false so that while loop continues until conditions met
    spacing_dict = {}  # Initialize spacing dictionary for returning durations

    while condition == False:
        # while conditions are not met...
        order_dict = initialize_flexible_dict(num_participants) # create empty dict using flexible function
        spacing_dict.clear()  # Clear spacing_dict for each attempt
    
        # for as many segments as there are (i.e., n of probes per participant)
        for segment in range(num_probes):
            # available orders are 1 to number of participants needed for min coverage
            available_numbers = list(range(1, num_participants + 1))
            
            # loop over empty dict, randomly select number from available numbers
            # without replacement
            for key in order_dict:
                assigned_number = random.sample(available_numbers, 1)[0]
                available_numbers.remove(assigned_number)
                order_dict[key].append(assigned_number)
    
        # condition = all(check_consecutive_same(order_dict[key]) for key in order_dict)
        # if condition:
        # spacing_dict = {}  # Clear spacing_dict to ensure it only stores spacings for successful order
        
        # put into seconds space here for evaluating spaces
        # but also keep normal order space
        order_dict_num = copy.deepcopy(order_dict)
        
        # Apply value_mapping to order_dict to get into 'seconds space'
        for key in order_dict:
            for i in range(len(order_dict[key])):
                if order_dict[key][i] in value_mapping:
                    # print (order_dict[key][i])
                    order_dict[key][i] = (value_mapping[order_dict[key][i]] + i * probe_coverage_duration_secs) + preclip_secs
                    
        # adjust so includes first probe interval and omits last one essentially
        for key in order_dict:
            # print ("key:", key)
            for i in range(len(order_dict[key])):
                order_dict[key][i] -= probe_interval
    
        # check spacing and only return true if both conditions are met (see function above)
        condition = all(check_spacing(order_dict[key],  min_participant_break, probe_interval)[0] for key in order_dict)
        
        if condition: # if condition met, save durations
            for key in order_dict:
                _, spacings = check_spacing(order_dict[key],  min_participant_break, probe_interval)
                spacing_dict[key] = spacings  # Store the spacings in spacing_dict
        
        #print("Attempting to generate valid probe schedule...") #debug
                
        # spacing_dict_num = copy.deepcopy(spacing_dict)
                
        # for key in spacing_dict:
        #     for i in range(len(spacing_dict[key])):
        #         if i == 0:  # Adjust the first item in the list
        #             spacing_dict[key][i] = (spacing_dict[key][i] * probe_interval)-probe_interval + preclip_secs
        #         else:
        #             spacing_dict[key][i] = spacing_dict[key][i]* probe_interval
                
    return order_dict_num, order_dict, spacing_dict
    

def create_value_mapping(num_participants, probe_interval):
    """
    Parameters
    ----------
    num_participants : int
        how many participants needed for full coverage
    probe_interval : int
        how many seconds bewteen probes across participants

    Returns
    -------
    value_mapping : Dictionary
        mapping between order dictionary in 'order space' to 'seconds space'

    """
    num_keys_value_mapping = num_participants
    value_mapping = {}
    for i in range(1, num_keys_value_mapping + 1):
        value_mapping[i] = probe_interval + (i - 1) * probe_interval

    return value_mapping


############################## User Input #####################################
# set absolute path for saving order and duration csv files
# directory = "C:\\Users\\raven\\Documents\\Repositories\\FiveMovieParadigm\\Tasks\\taskScripts\\resources\\Movie_Task\\csv"

# Experiment parameters
num_clips = 4 # set to how many clips you have
preclip_min = 1 # set to how many minutes before 1st probe
clip_min = 10 # set to how many minutes of probing in each clip
probe_coverage_duration_min = 2 # how often, roughly, do you want to interupt each person (in minutes)
probe_interval = 10  # how often do you want probes across participants (in seconds)
min_participant_break = 60  # what is the min amount of time between a probe for a person (in seconds)
num_samples_per_interval = 10 # how many participants do you want at each interval

############################## Calculations ###################################
preclip_secs = preclip_min * 60 # how many seconds before 1st probe
total_duration = clip_min * 60  # how many seconds of probing in each clip
probe_coverage_duration_secs = probe_coverage_duration_min * 60  # how often, roughly, do you want to interupt each person (in seconds)
num_probes = int(clip_min / probe_coverage_duration_min) # how many probes per person
num_participants = int(probe_coverage_duration_secs / probe_interval) # how many participants needed for full coverage
num_participants_full_sample = num_participants * num_samples_per_interval # how many participants needed for full coverage and enough power

# print out calculations
print ("Total clip duration in seconds:", total_duration, "\n")
print (f"""If you want to sample each participant roughly every {probe_coverage_duration_min} minutes / 
{probe_coverage_duration_secs} seconds, each participant can provide {num_probes} probes \n""")
print (f"""If you want to sample each participant roughly every {probe_coverage_duration_min} minutes,
/ {probe_coverage_duration_secs} seconds and you want to collect a probe across 
participants every {probe_interval} seconds, you need {num_participants} participants and
therefore, {num_participants} different orders. \n""")
print (f"""Therefore, if you want {num_samples_per_interval} observations every {probe_interval} seconds,
you need {num_participants_full_sample} participants. \n""")

######################## Create orders and durations ##########################

# Create value_mapping for going from 'order space' to actual seconds
value_mapping = create_value_mapping(num_participants, probe_interval)

# create order dictionary and duration dictionary (in 'order space')
order_dict_num, order_dict_secs, spacing_dict_secs = generate_order_dict(value_mapping, preclip_secs, 
                                           num_participants, num_samples_per_interval,
                                           min_participant_break, probe_interval)
    
print("Order dictionary:\n", order_dict_num, "\n")

print("Order dictionary in seconds:\n", order_dict_secs, "\n")
print("Duration dictionary in seconds:\n", spacing_dict_secs, "\n")

######################## Save orders and durations ############################

# Construct the full path for the CSV file
# csv_file_path_orders = os.path.join(directory, "probetimes_orders.csv")
# csv_file_path_durations = os.path.join(directory, "probe_durations.csv")

csv_file_path_orders = "probetimes_orders.csv"
csv_file_path_durations = "probe_durations.csv"

# Convert order_dict to a list of lists for CSV
csv_data_orders = []
csv_data_durs = []
for key in order_dict_secs:
    csv_data_orders.append(order_dict_secs[key])
    csv_data_durs.append(spacing_dict_secs[key])

# Create column names for the order CSV
column_names = [f"Probe {i+1}" for i in range(num_probes)]

# Save the order data to the CSV file
with open(csv_file_path_orders, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile)
    
    # Write the column names
    csv_writer.writerow(column_names)
    
    # Write the data rows
    csv_writer.writerows(csv_data_orders)

print(f"Order dictionary saved to {csv_file_path_orders}")

# Save the duration data to the CSV file
with open(csv_file_path_durations, "w", newline="") as csvfile:
    csv_writer = csv.writer(csvfile) 

    # Write the data rows
    csv_writer.writerows(csv_data_durs)
    
print(f"Duration dictionary saved to {csv_file_path_durations}")

############################# Counterbalancing ################################
# Get the list of clip numbers 
clip_numbers = list(range(1, num_clips + 1))

# Assuming order_keys is a list of strings
order_keys_from1 = list(order_dict_num.keys())

# Convert keys to integers, subtract 1, and convert back to strings in a single list comprehension
# this is so that numbers to be inputted match up to probe_order indexing
order_keys = [str(int(key) - 1) for key in order_keys_from1]

## This first bit selects orders per clip across the min number of participants
# for full coverage (e.g., 6 participants)
# ensuring that for any one person, clip orders are not the same between clips
# and that, for each clip every N participants, one of the orders is used

while True:
    # Create the new dictionary to store selected orders for each clip and participant
    selected_orders_dict = {}
    
    # Create a list of available orders for each clip
    available_orders = {clip_num: list(order_keys) for clip_num in clip_numbers}
    
    # Iterate over num_participants
    for participant_num in range(1, num_participants + 1):
        selected_orders_dict[participant_num] = {}
        
        # Select an order for each clip
        for clip_num in clip_numbers:
            # Select a random order from available_orders for the current clip
            selected_order = random.choice(available_orders[clip_num])
            
            # Remove the selected order from the available options
            available_orders[clip_num].remove(selected_order)
            
            # Add the selected order to the participant's entry in the dictionary
            selected_orders_dict[participant_num][clip_num] = selected_order
        
        # Check for duplicates within the current participant's selected orders
        selected_orders_set = set()
        has_duplicates = False
        for selected_order in selected_orders_dict[participant_num].values():
            if selected_order in selected_orders_set:
                has_duplicates = True
                break
            else:
                selected_orders_set.add(selected_order)
        
        # If duplicates are found, restart the loop to generate new orders
        if has_duplicates:
            break
    else:
        # If no duplicates are found for any participant, exit the loop
        break

print("Selected orders dictionary:\n")
print(selected_orders_dict)

# now to loop over num_samples_per_interval
# randomly shuffle the values of selected_orders_dict
# modify keys according to iteration, so, e.g., second iteration = keys + num_participants * 2

# Create a dictionary to store shuffled and modified dictionaries
shuffled_dicts = {}

# Loop over num_samples_per_interval
for iteration in range(1, num_samples_per_interval + 1):
    # Shuffle the values in selected_orders_dict
    shuffled_dict = {}
    shuffled_values = list(selected_orders_dict.values())
    random.shuffle(shuffled_values)
    
    # Reassign shuffled values to each participant while keeping keys intact
    shuffled_dict = {participant_num: shuffled_values[i] for i, participant_num in enumerate(selected_orders_dict)}
    print (shuffled_dict)
    
    # Modify keys based on the iteration
    if iteration > 1:
        new_shuffled_dict = {}
        for participant_num, participant_data in shuffled_dict.items():
            new_participant_num = participant_num + num_participants * (iteration - 1)
            new_shuffled_dict[new_participant_num] = participant_data
        shuffled_dict = new_shuffled_dict
    
    shuffled_dicts[iteration] = shuffled_dict
    
# Save shuffled_dicts as a CSV file
# csv_filename = os.path.join(directory, f"counterbalanced_orders_n{num_participants_full_sample}.csv")
csv_filename = f"counterbalanced_orders_n{num_participants_full_sample}.csv"


# Flatten the shuffled_dicts structure for easier processing
flattened_data = []
for participant_group, participant_data in shuffled_dicts.items():
    for participant_num, clip_data in participant_data.items():
        padded_participant_num = str(participant_num).zfill(3)
        row = {
            "participant_number": padded_participant_num,
            "Clip 1": clip_data[1],
            "Clip 2": clip_data[2],
            "Clip 3": clip_data[3],
            "Clip 4": clip_data[4]
        }
        flattened_data.append(row)
        
with open(csv_filename, 'w', newline='') as csvfile:
    fieldnames = ["participant_number", "Clip 1", "Clip 2", "Clip 3", "Clip 4"]
    csv_writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    
    # Write header row
    csv_writer.writeheader()
    
    # Write data rows
    csv_writer.writerows(flattened_data)

print("CSV file saved:", csv_filename)
    
# TO DO:        
    # once all that is done, can consider whether you want to automate
    # so only ID needs to be entered, then probes selected from file instead