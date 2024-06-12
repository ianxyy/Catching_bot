import h5py
import numpy as np

# # Open your HDF5 file

with h5py.File('graspnet9_data_ring_clear.h5', 'r') as file:
    # List all keys in the file to understand the structure
    keys = list(file.keys())
    print("Keys:", keys)
    
    # Example of how to properly access a dataset and handle data
    dataset_name = 'X_WG1_rot_1011'  # Make sure this dataset exists in your file
    if dataset_name in file:
        dataset = file[dataset_name]
        data_array = np.array(dataset)  # Convert the dataset to a numpy array for easier handling
        print(data_array) 

# import h5py
# import numpy as np

# # Replace with your actual file path
# file_path = 'transformer_test.h5'

# # Define the range of indices to check
# start_index = 0  # Start from the beginning or the smallest index you expect
# end_index = 224  # Replace with the maximum index you expect

# # Open the HDF5 file in read/write mode
# with h5py.File(file_path, 'r+') as hf:
#     for idx in range(start_index, end_index + 1):
#         # Construct the dataset name for the current index
#         current_key = f'traj_data_{idx}'
        
#         # Check if the current dataset exists
#         if current_key not in hf:
#             # Find the previous dataset that exists
#             prev_idx = idx - 1
#             while f'traj_data_{prev_idx}' not in hf and prev_idx >= start_index:
#                 prev_idx -= 1
            
#             # Check if a valid previous dataset was found
#             if prev_idx >= start_index:
#                 # Copy all related datasets for the current index from the previous index
#                 for suffix in ['traj_data', 'pc_data', 'time_data', 'traj_data_after', 
#                                'time_data_after', 'X_WG1', 'X_WG2', 'obj_catch_t', 
#                                'result', 'obj_pose_at_catch']:
#                     src_key = f'{suffix}_{prev_idx}'
#                     dest_key = f'{suffix}_{idx}'
                    
#                     # Copy the dataset
#                     hf.copy(src_key, dest_key)
#                     print(f"Copied data from {src_key} to {dest_key}")
#             else:
#                 print(f"No valid previous dataset found for index {idx}")

