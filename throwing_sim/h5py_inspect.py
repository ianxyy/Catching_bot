import h5py
import numpy as np

# # Open your HDF5 file
with h5py.File('filtered_graspnet_data.h5', 'r') as file:
    # List all groups
    print("Keys: %s" % file.keys())
    a_group_key = list(file.keys())[1]

    # Get the data
    data = list(file[a_group_key])
    print(data)
    print(len(file.keys()))
    # If you know the specific path to your dataset, you can directly read
    # Example: file['group/subgroup/dataset']
    # dataset = file['time_data_after_0']
    # # with open('visualize.txt', "a") as text_file:
    # #         # Write the information to the file
    # #         text_file.write(f'{file.keys()}')
    # print((np.array(dataset)))

# # Initialize a list to keep track of dataset identifiers where the result is 1

# result_ids_with_1 = []

# # Open the original HDF5 file to inspect the 'result_x' datasets
# with h5py.File('graspnet_data.h5', 'r') as file:
#     # Iterate over datasets starting with 'result_'
#     for name in file:
#         if name.startswith('result_'):
#             # Read the dataset
#             result_data = file[name][...]
            
#             # Check if the result data contains 1
#             if 1 in result_data:
#                 # Extract and keep the identifier for datasets where the result is 1
#                 identifier = name.split('_')[-1]
#                 if identifier != 327:
#                     result_ids_with_1.append(identifier)

# # Open a new HDF5 file to save the filtered data
# with h5py.File('filtered_graspnet_data.h5', 'w') as filtered_file:
#     # Re-open the original file to copy the necessary data
#     with h5py.File('graspnet_data.h5', 'r') as original_file:
#         # Sort the result identifiers to maintain the order
#         sorted_identifiers = sorted(result_ids_with_1, key=int)
#         # Iterate over the result identifiers we kept earlier, reindexing as we go
#         for new_index, identifier in enumerate(sorted_identifiers):
#             # Construct the new dataset names with the new index
#             new_traj_data_name = f'traj_data_{new_index}'
#             new_pc_data_name = f'pc_data_{new_index}'
#             new_time_data_name = f'time_data_{new_index}'
#             new_X_WG1_name = f'X_WG1_{new_index}'
#             new_X_WG2_name = f'X_WG2_{new_index}'
#             new_obj_catch_t_name = f'obj_catch_t_{new_index}'
#             new_result_name = f'result_{new_index}'
            
#             # Original dataset names
#             traj_data_name = f'traj_data_{identifier}'
#             pc_data_name = f'pc_data_{identifier}'
#             time_data_name = f'time_data_{identifier}'
#             X_WG1_name = f'X_WG1_{identifier}'
#             X_WG2_name = f'X_WG2_{identifier}'
#             obj_catch_t_name = f'obj_catch_t_{identifier}'
#             result_name = f'result_{identifier}'
            
#             # Check and copy each dataset to the new file with reindexed names
#             if traj_data_name in original_file:
#                 filtered_file.create_dataset(new_traj_data_name, data=original_file[traj_data_name][...])
#                 filtered_file.create_dataset(new_pc_data_name, data=original_file[pc_data_name][...])
#                 filtered_file.create_dataset(new_time_data_name, data=original_file[time_data_name][...])
#                 filtered_file.create_dataset(new_X_WG1_name, data=original_file[X_WG1_name][...])
#                 filtered_file.create_dataset(new_X_WG2_name, data=original_file[X_WG2_name][...])
#                 filtered_file.create_dataset(new_obj_catch_t_name, data=original_file[obj_catch_t_name][...])
#                 filtered_file.create_dataset(new_result_name, data=original_file[result_name][...])
