import h5py
import torch
import numpy as np
import os
import json
import tqdm
from torch.utils.data import DataLoader, Dataset, TensorDataset
from model_archs import *
from model_archs import SingleStream3D
from model_archs.ResNetSeparableOptimized2 import FocalLoss

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


def combined_stats(mean, std_devs):
    # Step 1: Compute the total number of observations (N)

    # counts = 32*256 * np.ones(len(mean))
    counts = 64*256 * np.ones(len(mean))

    N = np.sum(counts)

    # Step 2: Compute the combined mean (X_bar)
    weighted_means = np.sum(np.array(mean) * np.array(counts)) / N

    # Step 3: Compute the sum of squares for each group (SS_i)
    SS_i = [(counts[i] - 1) * std_devs[i]**2 + counts[i] * (mean[i] - weighted_means)**2 for i in range(len(counts))]

    # Step 4: Compute the combined sum of squares (SS)
    SS = np.sum(SS_i)

    # Step 5: Compute the combined variance (s^2)
    combined_variance = SS / (N - 1)

    # Step 6: Compute the combined standard deviation (s)
    combined_std_dev = np.sqrt(combined_variance)

    # return np.round(weighted_means,3), np.round(combined_std_dev,3)
    return weighted_means, combined_std_dev


class RadarHDF5DatasetNoStat():
    def __init__(self, hdf5_path, model_arch):
        # str_vars = ['packet_type','SENSOR_ADDRESS', 'MAC', 'IP', 'numchirps', 'chirpsamples', 'max_speed_m_s', 
        #             'max_range_m', 'range_resolution_m', 'folder', 'radar', 'date', 'label', 'Name', 
        #             'position', 'speed']
        str_vars = ['packet_type','SENSOR_ADDRESS', 'MAC', 'IP', 'numchirps', 'chirpsamples', 'max_speed_m_s', 'max_range_m', 'range_resolution_m','folder','radar', 'date', 'label', 'Name', 'position','speed']

        self.str_vars = str_vars
        self.str_dict = dict()
        self.radar_vars = ['Azimuth', 'Elevation', 'Doppler_Map']
        self.radar_dict = dict()

        self.predictions = []

        # Define encoding for different activities
        self.encoding = {
            'layonfloor': 0,
            'layonsofa': 1,
            'sitonfloor': 2,
            'sitting': 3,
            'standing': 4,
            'walking': 5,
            'movesitonchair': 6,
            'movestandingpickupfromfloor': 7,
            'movestandingpickupfromtable': 8
        }

        # Merging similar activities
        self.merge = [['movesitonchair', 'movestandingpickupfromfloor', 'movestandingpickupfromtable']]

        # Keep only the following activities
        self.to_keep = [
            'layonfloor',
            'layonsofa',
            'sitonfloor',
            'sitting',
            'standing',
            'walking'
        ]

        # Load the data from the .h5 file
        with h5py.File(hdf5_path, 'r') as hdf5_file:
            self.length = len(hdf5_file['label'])
            self.orig_length = len(hdf5_file['label'])

            # Store radar data without normalization
            for radar_var in self.radar_vars:
                vals = [x.reshape((2, 32, 256)) for x in hdf5_file[radar_var][:]]
                self.radar_dict[radar_var] = dict(zip(range(self.length), vals))
            
            # Store string variables
            for str_var in self.str_vars:
                self.str_dict[str_var] = dict(zip(range(self.length), hdf5_file[str_var][:]))

            # Clean and store labels
            str_cleaned = [s.strip().lower().decode("utf-8") for s in hdf5_file['label'][:]]
            self.str_dict['label_str'] = dict(zip(range(self.length), str_cleaned))

        # Apply custom filtering to the dataset
        self.custom_filter()

        # Encode labels after filtering
        self.encoding = {k: i for i, k in enumerate(self.to_keep)}
        self.encoding_reverse = {i: k for i, k in enumerate(self.to_keep)}

        encoded_labels = [*map(self.encoding.get, self.str_dict['label_str'].values())]
        self.str_dict['label'] = dict(zip(range(self.length), encoded_labels))


    def custom_filter(self):

        merge_encoded = []

        for m in self.merge:
            merge_encoded.append([self.encoding[i] for i in m])

        for merge_tuple in self.merge:
            for k,v in self.str_dict['label_str'].items():
                if v in merge_tuple:
                    self.str_dict['label_str'][k] = merge_tuple[0]

        dict1 = list(self.str_dict['label_str'].items()).copy()

        for k, v in dict1:
            if v not in self.to_keep:
              del self.str_dict['label_str'][k]
              for str_var in self.str_vars:
                  del self.str_dict[str_var][k]
              for radar_var in self.radar_vars:
                  del self.radar_dict[radar_var][k]


        self.length = len(self.str_dict['label_str'])

    def __len__(self):
        return self.length

    def __getmetadata__(self, idx):
        # Get metadata (string variables) for a given index
        metadata = {}
        for str_var in self.str_vars:
            metadata[str_var] = self.str_dict[str_var][idx]
        return metadata

    def get_whole_data(self):
        # Retrieve the entire dataset (radar data and labels)
        data = []
        for i in range(self.__len__()):
            datapoint = []
            for radar_var in self.radar_dict.keys():
                datapoint.append(self.radar_dict[radar_var][i])
            data.append(datapoint)

        labels = list(self.str_dict['label'].values())
        return torch.tensor(data), torch.tensor(labels)

    def __getitem__(self, idx):
        # Get radar data (Azimuth, Elevation, Doppler) and label for a given index
        x = []
        for radar_var in self.radar_vars:
            radar_data = self.radar_dict[radar_var][idx]
            x.append(radar_data.copy())  # Return a copy of the data
        return x, self.str_dict['label'][idx]


class RadarHDF5Dataset():
    def __init__(self, hdf5_path, map_stats, model_arch):
        str_vars = ['SENSOR_ADDRESS', 'MAC', 'IP', 'numchirps', 'chirpsamples', 'max_speed_m_s', 'max_range_m', 'range_resolution_m', 'date', 'Name', 'scene_num']
        str_vars = ['packet_type','SENSOR_ADDRESS', 'MAC', 'IP', 'numchirps', 'chirpsamples', 'max_speed_m_s', 'max_range_m', 'range_resolution_m','folder','radar', 'date', 'label', 'Name', 'position','speed']
        self.str_vars = str_vars
        self.str_dict = dict()
        self.radar_vars = ['Azimuth','Elevation','Doppler_Map']
        self.radar_dict = dict()

        self.predictions = []

        self.encoding = { 'layonfloor':0,
                          'layonsofa':1,
                          'sitonfloor':2,
                          'sitting':3,
                          'standing':4,
                          'walking':5,
                          'movesitonchair':6,
                          'movestandingpickupfromfloor':7,
                          'movestandingpickupfromtable':8
                          }

        self.merge = [['movesitonchair','movestandingpickupfromfloor', 'movestandingpickupfromtable']]

        self.to_keep = [
            'layonfloor',
            'layonsofa',
            'sitonfloor',
            'sitting',
            'standing',
            'walking',
            # 'movesitonchair',
            # 'movestandingpickupfromfloor',
            # 'movestandingpickupfromtable',

        ]



class RadarHDF5Dataset2():
    def __init__(self, hdf5_path, model_arch):
        str_vars = ['SENSOR_ADDRESS', 'MAC', 'IP', 'numchirps', 'chirpsamples', 'max_speed_m_s', 'max_range_m', 'range_resolution_m', 'date', 'Name', 'scene_num']
        str_vars = ['packet_type','SENSOR_ADDRESS', 'MAC', 'IP', 'numchirps', 'chirpsamples', 'max_speed_m_s', 'max_range_m', 'range_resolution_m','folder','radar', 'date', 'label', 'Name', 'position','speed']
        self.str_vars = str_vars
        self.str_dict = dict()
        self.radar_vars = ['Azimuth','Elevation','Doppler_Map']
        self.radar_dict = dict()

        self.predictions = []

        self.encoding = { 'layonfloor':0,
                          'layonsofa':1,
                          'sitonfloor':2,
                          'sitting':3,
                          'standing':4,
                          'walking':5,
                          'movesitonchair':6,
                          'movestandingpickupfromfloor':7,
                          'movestandingpickupfromtable':8
                          }

        self.merge = [['movesitonchair','movestandingpickupfromfloor', 'movestandingpickupfromtable']]

        self.to_keep = [
            'layonfloor',
            'layonsofa',
            'sitonfloor',
            'sitting',
            'standing',
            'walking',
            # 'movesitonchair',
            # 'movestandingpickupfromfloor',
            # 'movestandingpickupfromtable',

        ]

#        self.to_keep = ['enter the room and walk',
#                    'lay down','lay down on the floor','sit on the bed',
#                    'sit on the chair']


        with h5py.File(hdf5_path, 'r') as hdf5_file:

            self.length = len(hdf5_file['label'])
            self.orig_length = len(hdf5_file['label'])

            for radar_var in self.radar_vars:
                vals = [x.reshape((2, 32,256)) for x in hdf5_file[radar_var][:]]
                self.radar_dict[radar_var] = dict(zip(range(self.length), vals ))
            for str_var in self.str_vars:
                self.str_dict[str_var] = dict(zip(range(self.length), hdf5_file[str_var][:]))

            str_cleaned = [s.strip().lower().decode("utf-8") for s in hdf5_file['label'][:] ]

            self.str_dict['label_str'] = dict(zip(range(self.length), str_cleaned ))

        self.custom_filter()

        self.encoding = {k:i for i,k in enumerate(self.to_keep)}
        self.encoding_reverse = {i:k for i,k in enumerate(self.to_keep)}

        encoded_labels =  [*map(self.encoding.get, self.str_dict['label_str'].values())]
        self.str_dict['label'] = dict(zip(range(self.length), encoded_labels ))


    def custom_filter(self):

        merge_encoded = []

        for m in self.merge:
            merge_encoded.append([self.encoding[i] for i in m])

        for merge_tuple in self.merge:
            for k,v in self.str_dict['label_str'].items():
                if v in merge_tuple:
                    self.str_dict['label_str'][k] = merge_tuple[0]

        dict1 = list(self.str_dict['label_str'].items()).copy()

        for k, v in dict1:
            if v not in self.to_keep:
              del self.str_dict['label_str'][k]
              for str_var in self.str_vars:
                  del self.str_dict[str_var][k]
              for radar_var in self.radar_vars:
                  del self.radar_dict[radar_var][k]


        self.length = len(self.str_dict['label_str'])

    def __len__(self):
        return self.length


    def __getmetadata__(self, idx):
        # Convert image bytes back to PIL Image
        x = dict()

        for str_var in self.str_vars:
            d = self.str_dict[str_var][idx]
            x[str_var] = d
        return x

    def get_whole_data(self):

        data = []
        for i in range(self.__len__()):
            datapoint = []
            for k,v in self.radar_dict.items():
                datapoint.append((list(v.values())[i]))
            data.append(datapoint)

        labels = list(self.str_dict['label'].values())
        return torch.tensor(data), torch.tensor(labels)

    def __getitem__(self, idx):
        # Convert image bytes back to PIL Image
        x = []

        for radar_var in self.radar_vars:
            image_bytes = self.radar_dict[radar_var][idx]
#            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
#            image = np.float32(image)
            x.append(image_bytes.copy())
        return x, self.str_dict['label'][idx]


    def get_whole_data_old(self):

        data = []

        for k,v in self.radar_dict.items():
            data.append(list(v.values()))

        labels = list(self.str_dict['label'].values())
        return data, labels



# class RadarHDF5Dataset():
#     def __init__(self, hdf5_path, map_stats, model_arch, transform=None):
        
#         self.model_arch = model_arch
#         self.transform = transform
#         str_vars = ['packet_type','SENSOR_ADDRESS', 'MAC', 'IP', 'numchirps', 'chirpsamples', 'max_speed_m_s', 'max_range_m', 'range_resolution_m','folder','radar', 'date', 'label', 'Name', 'position','speed']
#         self.str_vars = str_vars
#         self.str_dict = dict()
#         self.radar_vars = ['Azimuth','Elevation','Doppler_Map']
#         self.radar_dict = dict()

#         self.predictions = []

#         self.encoding = { 'layonfloor':0,
#                           'layonsofa':1,
#                           'sitonfloor':2,
#                           'sitting':3,
#                           'standing':4,
#                           'walking':5,
#                           'movesitonchair':6,
#                           'movestandingpickupfromfloor':7,
#                           'movestandingpickupfromtable':8
#                         }

#         # self.merge = [['sit on the chair','keep sitting']]
#         self.merge = [['movesitonchair', 
#                        'movestandingpickupfromtable',
#                        'movestandingpickupfromfloor']]
        

#         self.to_keep = [
#             'layonfloor',
#             'layonsofa',
#             'sitonfloor',
#             'sitting',
#             'standing',
#             'walking',
#             'movesitonchair',
#         ]

#         self.to_keep = [
#             '0',
#             '1',
#             '2',
#             '3',
#             '4',
#             '5',
#             '6',
#         ]



# #        self.to_keep = ['enter the room and walk',
# #                    'lay down','lay down on the floor','sit on the bed',
# #                    'sit on the chair']

#         # print(f"hdf5_path: {hdf5_path}")
#         with h5py.File(hdf5_path, 'r') as hdf5_file:

#             self.length = len(hdf5_file['label'])
#             self.orig_length = len(hdf5_file['label'])
#             # print(self.length)

#             # tmp = 'Azimuth'
#             # print(np.shape(hdf5_file))
#             # print(hdf5_file)
#             # print(np.shape(hdf5_file[tmp][:]))


#             # for i in hdf5_file:
#             # print(f"Label: {hdf5_file['label']}")


#             for radar_var in self.radar_vars:
#                 # print(f"Original shape of {radar_var} data: {np.shape(hdf5_file[radar_var][0])}")
#                 if model_arch in ["VerticalStackConv2DModel", "VerticalStackConv2DModelV1", "VerticalStackConv2DModelV2"] :
#                     print("HERE_____________")
#                     vals = [(x.reshape((160,256)) - map_stats[radar_var]['mean'])/map_stats[radar_var]['std'] for x in hdf5_file[radar_var][:]]
#                 else:
#                     vals = [(x.reshape((32,256)) - map_stats[radar_var]['mean'])/map_stats[radar_var]['std'] for x in hdf5_file[radar_var][:]]
#                 # print(vals)
#                 # print()
#                 # print(f"Reshaped data of {radar_var}: {vals[0].shape}")

#                 self.radar_dict[radar_var] = dict(zip(range(self.length), vals ))
#             for str_var in self.str_vars:
#                 self.str_dict[str_var] = dict(zip(range(self.length), hdf5_file[str_var][:]))

#             # str_cleaned = [s.strip().lower().decode("utf-8") for s in hdf5_file['label'][:] ]
#             str_cleaned = [str(s).strip().lower().decode("utf-8") if isinstance(s, bytes) else str(s) for s in hdf5_file['label'][:]]


#             self.str_dict['label_str'] = dict(zip(range(self.length), str_cleaned ))

#         self.custom_filter()

#         self.encoding = {k:i for i,k in enumerate(self.to_keep)}
#         self.encoding_reverse = {i:k for i,k in enumerate(self.to_keep)}

#         encoded_labels =  [*map(self.encoding.get, self.str_dict['label_str'].values())]
#         # print(f"encoded_labels: {encoded_labels}")
#         self.str_dict['label'] = dict(zip(range(self.length), encoded_labels ))
#         # print(f"self.str_dict: {self.str_dict}")


#     # def custom_filter(self):

#     #     merge_encoded = []

#     #     for m in self.merge:
#     #         merge_encoded.append([self.encoding[i] for i in m])

#     #     for merge_tuple in self.merge:
#     #         for k,v in self.str_dict['label_str'].items():
#     #             if v in merge_tuple:
#     #                 self.str_dict['label_str'][k] = merge_tuple[0]

#     #     dict1 = list(self.str_dict['label_str'].items()).copy()

#     #     for k, v in dict1:
#     #         if v not in self.to_keep:
#     #           del self.str_dict['label_str'][k]
#     #           for str_var in self.str_vars:
#     #               del self.str_dict[str_var][k]
#     #           for radar_var in self.radar_vars:
#     #               del self.radar_dict[radar_var][k]


#     #     self.length = len(self.str_dict['label_str'])
#     #     print(f"Filtered dataset length: {self.length}")


#     def custom_filter(self):
#         merge_encoded = []

#         # print("Before filtering:")
#         # for k, v in self.str_dict['label_str'].items():
#             # print(f"Index: {k}, Label: {v}")
#         # return -1
#         # Apply merging
#         for merge_tuple in self.merge:
#             for k, v in self.str_dict['label_str'].items():
#                 if v in merge_tuple:
#                     self.str_dict['label_str'][k] = merge_tuple[0]

#         dict1 = list(self.str_dict['label_str'].items()).copy()

#         # Filter based on string labels before encoding
#         for k, v in dict1:
#             # print(f"Index: {k}, Label: {v}")
#             if v not in self.to_keep:
#                 # print(f"Removing label {v} at index {k} from dataset")  # Debugging: see which labels are being removed
#                 del self.str_dict['label_str'][k]
#                 for str_var in self.str_vars:
#                     del self.str_dict[str_var][k]
#                 for radar_var in self.radar_vars:
#                     del self.radar_dict[radar_var][k]

#         # Recalculate length after filtering
#         self.length = len(self.str_dict['label_str'])
#         # print(f"Filtered dataset length: {self.length}")

#         # Now encode the labels into integers after filtering
#         self.encoding = {k: i for i, k in enumerate(self.to_keep)}
#         self.encoding_reverse = {i: k for i, k in enumerate(self.to_keep)}

#         encoded_labels = [*map(self.encoding.get, self.str_dict['label_str'].values())]
#         self.str_dict['label'] = dict(zip(range(self.length), encoded_labels))

#     def __len__(self):
#         return self.length


#     def __getmetadata__(self, idx):
#         # Convert image bytes back to PIL Image
#         x = dict()

#         for str_var in self.str_vars:
#             d = self.str_dict[str_var][idx]
#             x[str_var] = d
#         return x

#     def get_whole_data(self):

#         data = []
#         for i in range(self.__len__()):
#             datapoint = []
#             for k,v in self.radar_dict.items():
#                 datapoint.append((list(v.values())[i]))
#             data.append(datapoint)

#         # labels = list(self.str_dict['label'].values())
#         # data = np.array(data)
#         # labels = np.array(list(self.str_dict['label'].values()))  # Convert labels to a numpy arrayreturn torch.tensor(data), torch.tensor(labels)
#         # # print(f"data : {np.shape(data)}")     
#         # # data : (2048, 3, 32, 256)
#         # # 
#         # # print(f"labels: {np.shape(labels)}")
#         # # labels: (2048,)
        
        
#         data = np.array(data)

#         if self.model_arch == "SingleStream3D":
#             data = data.reshape(-1, 3, 1, 32, 256)  # Reshape to [batch_size, channels, depth, height, width]

#         labels = np.array(list(self.str_dict['label'].values()))  # Convert labels to a numpy array

#         print(f"HERE::data : {np.shape(data)}")     
#         # # data : (2048, 3, 32, 256)
#         # # 
#         print(f"HERE::labels: {np.shape(labels)}")
#         # # labels: (2048,)

#         return torch.tensor(data), torch.tensor(labels)

#     def __getitem__(self, idx):
#         # Convert image bytes back to PIL Image
#         x = []

#         for radar_var in self.radar_vars:
#             image_bytes = self.radar_dict[radar_var][idx]
# #            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
# #            image = np.float32(image)
#             x.append(image_bytes.copy())
#         return x, self.str_dict['label'][idx]


#     def get_whole_data_old(self):

#         data = []

#         for k,v in self.radar_dict.items():
#             data.append(list(v.values()))

#         labels = list(self.str_dict['label'].values())
#         return data, labels
    

# Function to save training and validation history
def save_history(history_epochs, save_path):
    history_path = os.path.join(save_path, 'history')
    os.makedirs(history_path, exist_ok=True)
    
    with open(os.path.join(history_path, 'training_history_epochs.json'), 'w') as f:
        json.dump(history_epochs, f)
    print(f"History saved at {history_path}/training_history_epochs.json")

import time



def create_data_loaders_no_stat(folder_train, folder_val, batch_size, model_arch):
    # Load all train data at once
    X_train_all = []
    y_train_all = []
    print(folder_train)
    print(folder_val)

    # Metrics you want to normalize separately
    metrics = ['Azimuth', 'Elevation', 'Doppler_Map']
    
    # Dictionaries to hold means and stds for each metric
    train_means = {metric: None for metric in metrics}
    train_stds = {metric: None for metric in metrics}

    # Load and concatenate the training data
    for train_file in os.listdir(folder_train):
        train_data = RadarHDF5DatasetNoStat(folder_train + '/' + train_file, model_arch=model_arch)
        print(train_data)
        X_train, y_train = train_data.get_whole_data()
        X_train_all.append(X_train)
        y_train_all.append(y_train)

    # Concatenate all training data into tensors
    X_train_torch = torch.cat(X_train_all, dim=0)  # Shape should be [num_samples, num_metrics, 2, 32, 256]
    y_train_torch = torch.cat(y_train_all, dim=0)

    # Separate data by metrics and compute mean and std for each metric (e.g., Azimuth, Elevation, Doppler)
    for idx, metric in enumerate(metrics):
        # Extract data corresponding to this metric
        metric_data = X_train_torch[:, idx, :, :, :].reshape(-1, 32 * 256)  # Reshape for mean/std computation

        # Compute mean and std for this metric
        train_means[metric] = torch.mean(metric_data, dim=0)
        train_stds[metric] = torch.std(metric_data, dim=0)

        # Normalize the data for this metric
        X_train_torch[:, idx, :, :, :] = (X_train_torch[:, idx, :, :, :] - train_means[metric].view(1, 1, 32, 256)) / train_stds[metric].view(1, 1, 32, 256)

    # Load all validation data at once
    X_val_all = []
    y_val_all = []

    for val_file in os.listdir(folder_val):
        val_data = RadarHDF5DatasetNoStat(folder_val + '/' + val_file, model_arch=model_arch)
        print(val_data)
        X_val, y_val = val_data.get_whole_data()
        X_val_all.append(X_val)
        y_val_all.append(y_val)

    # Concatenate validation data into tensors
    X_val_torch = torch.cat(X_val_all, dim=0)
    y_val_torch = torch.cat(y_val_all, dim=0)

    # Normalize validation data using the same statistics from the training data for each metric
    for idx, metric in enumerate(metrics):
        X_val_torch[:, idx, :, :, :] = (X_val_torch[:, idx, :, :, :] - train_means[metric].view(1, 1, 32, 256)) / train_stds[metric].view(1, 1, 32, 256)

    # Create DataLoaders for training and validation
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    val_dataset = TensorDataset(X_val_torch, y_val_torch)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader




def create_data_loaders(folder_train, folder_val, map_stats, batch_size, model_arch, transform):
    # Load all train data at once
    X_train_all = []
    y_train_all = []
    print(folder_train)
    print(folder_val) 
    # time.sleep(10)
    for train_file in os.listdir(folder_train):
        train_data = RadarHDF5Dataset(folder_train + '/' + train_file, map_stats, model_arch)
        print(train_data)
        X_train, y_train = train_data.get_whole_data()
        X_train_all.append(X_train)
        y_train_all.append(y_train)

    # Concatenate all training data
    X_train_torch = torch.cat(X_train_all, dim=0)
    y_train_torch = torch.cat(y_train_all, dim=0)

    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load all validation data at once
    X_val = []
    y_val = []
    
    for val_file in os.listdir(folder_val):
        val_data = RadarHDF5Dataset(folder_val + '/' + val_file, map_stats, model_arch)
        print(val_data)
        X_v, y_v = val_data.get_whole_data()
        X_val.append(X_v)
        y_val.append(y_v)

    # Concatenate validation data
    X_val_torch = torch.cat(X_val, dim=0)
    y_val_torch = torch.cat(y_val, dim=0)

    val_dataset = TensorDataset(X_val_torch, y_val_torch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader





def extract_model_arch_from_path(weight_path):
    # Extract the directory name that contains the model architecture part
    model_arch = os.path.basename(os.path.dirname(weight_path)).split('_b')[0]
    config = f"b{os.path.basename(os.path.dirname(weight_path)).split('_b')[1]}"

    return model_arch, config

def create_test_loaders(folder_test, map_stats, batch_size, model_arch, model):
    labels_list = []
    preds_list = []
    with torch.no_grad():
        for test_file in os.listdir(folder_test):
            # print(test_file)
            test_data = RadarHDF5Dataset(folder_test + '/' + test_file,map_stats, model_arch=model_arch)
            # val_data = RadarHDF5Dataset(folder_val + '/' + val_file, map_stats, model_arch)

            X_test, y_test = test_data.get_whole_data()
            y_pred = model(X_test.to(device))
            y_pred = y_pred.argmax(axis=1).tolist()
            preds_list.extend(y_pred)
            labels_list.extend(y_test)
            # print(preds_list)

    return labels_list, preds_list, test_data
 
# def create_test_loaders(folder_test, map_stats, batch_size, model_arch, model):
#     labels_list = []
#     preds_list = []
#     with torch.no_grad():
#         model.eval()  # Set model to evaluation mode
#         for test_file in os.listdir(folder_test):
#             print(f"Processing file: {test_file}")
#             test_data = RadarHDF5Dataset(folder_test + '/' + test_file, map_stats, model_arch=model_arch)
#             X_test, y_test = test_data.get_whole_data()

#             # Print shape of inputs and labels
#             print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

#             # Ensure data is on the same device (cuda if available)
#             X_test = X_test.to(device)
#             y_test = y_test.to(device)

#             # Run inference
#             y_pred = model(X_test)

#             # Print prediction shape
#             print(f"y_pred shape: {y_pred.shape}")

#             y_pred = y_pred.argmax(dim=1).cpu().tolist()  # Convert to list after moving to CPU
#             y_test = y_test.cpu().tolist()  # Move labels to CPU

#             preds_list.extend(y_pred)
#             labels_list.extend(y_test)

#     return labels_list, preds_list, test_data

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
def plot_corr_matrix(labels_list, preds_list, test_data, model_path, model_arch, config, major_vote=False):
    # Set up font properties for better readability
    font = {'weight': 'bold',
            'size': 16}  # Increased size for better readability in presentations
    plt.rc('font', **font)
    print("========================")
    print(test_data.to_keep)
    cm = confusion_matrix(labels_list, preds_list, labels=range(len(test_data.to_keep)))

    # Normalize the confusion matrix
    row_sums = cm.sum(axis=1, keepdims=True)  # Keepdims ensures the dimensions are preserved
    cm_normalized = np.divide(cm.astype('float'), row_sums, where=row_sums!=0)

    # Handle any remaining NaNs or infinities
    cm_normalized = np.nan_to_num(cm_normalized)

    # Create confusion matrix display
    fig, ax = plt.subplots(figsize=(14, 12))

    # Use seaborn heatmap for better control over aesthetics
    sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='viridis', cbar=True, 
                xticklabels=test_data.to_keep, yticklabels=test_data.to_keep, 
                linewidths=.5, linecolor='black', ax=ax)

    # Rotate x-axis labels slightly and adjust the position
    plt.xticks(rotation=30, ha='right', fontsize=14)
    plt.yticks(rotation=0, fontsize=14)

    # Label axes with larger, bolder text
    plt.xlabel('Predicted label', fontsize=18, fontweight='bold')
    plt.ylabel('True label', fontsize=18, fontweight='bold')

    # Add a title with the calculated accuracy
    accuracy = 100 * np.trace(cm) / np.sum(cm)
    accuracy = round(float(accuracy), 2)

    # Save the figure with higher resolution
    absolute_model_path = os.path.abspath(model_path)
    model_dir = os.path.dirname(absolute_model_path)  
    output_filename = os.path.basename(absolute_model_path).replace('.pth', '_confusion_matrix.png')
    output_path = os.path.join(model_dir, output_filename)

    if major_vote:
        prefix = "(Major Vote)"
    else:
        prefix = ""

    title_line1 = f"{prefix} Correlation Matrix for {model_arch}_{config}"   
    title_line2 = f"with weight {os.path.basename(output_filename).replace('_confusion_matrix.png','.pth')}"
    title_line3 = f"Accuracy: {accuracy} %"

    combined_title = f"{title_line1}\n{title_line2}\n{title_line3}"
    plt.title(combined_title, fontsize=20, fontweight='bold', pad=20)

    # plt.title(f"Correlation Matrix for {model_arch}_{config} with weight {os.path.basename(output_path)}")
    plt.savefig(output_path, dpi=400, bbox_inches='tight')

# def plot_corr_matrix(labels_list, preds_list, test_data, model_path, model_arch, config, major_vote=False):
#     # Set up font properties for better readability
#     font = {'weight': 'bold',
#             'size': 16}  # Increased size for better readability in presentations
#     plt.rc('font', **font)

#     print("========================")
#     print(f"Labels (True Classes): {test_data.to_keep}")
#     cm = confusion_matrix(labels_list, preds_list, labels=range(len(test_data.to_keep)))

#     # Normalize the confusion matrix
#     row_sums = cm.sum(axis=1, keepdims=True)  # Keepdims ensures the dimensions are preserved
#     cm_normalized = np.divide(cm.astype('float'), row_sums, where=row_sums!=0)

#     # Handle any remaining NaNs or infinities
#     cm_normalized = np.nan_to_num(cm_normalized)

#     # Create confusion matrix display
#     fig, ax = plt.subplots(figsize=(14, 12))

#     # Use seaborn heatmap for better control over aesthetics
#     sns.heatmap(cm_normalized, annot=True, fmt=".2f", cmap='viridis', cbar=True, 
#                 xticklabels=test_data.to_keep, yticklabels=test_data.to_keep, 
#                 linewidths=.5, linecolor='black', ax=ax)

#     # Rotate x-axis labels slightly and adjust the position
#     plt.xticks(rotation=30, ha='right', fontsize=14)
#     plt.yticks(rotation=0, fontsize=14)

#     # Label axes with larger, bolder text
#     plt.xlabel('Predicted label', fontsize=18, fontweight='bold')
#     plt.ylabel('True label', fontsize=18, fontweight='bold')

#     # Add a title with the calculated accuracy
#     total_predictions = np.sum(cm)
#     correct_predictions = np.trace(cm)
#     accuracy = 100 * correct_predictions / total_predictions
#     accuracy = round(float(accuracy), 2)

#     # Save the figure with higher resolution
#     absolute_model_path = os.path.abspath(model_path)
#     model_dir = os.path.dirname(absolute_model_path)  
#     output_filename = os.path.basename(absolute_model_path).replace('.pth', '_confusion_matrix.png')
#     output_path = os.path.join(model_dir, output_filename)

#     if major_vote:
#         prefix = "(Major Vote)"
#     else:
#         prefix = ""

#     title_line1 = f"{prefix} Correlation Matrix for {model_arch}_{config}"   
#     title_line2 = f"with weight {os.path.basename(output_filename).replace('_confusion_matrix.png','.pth')}"
#     title_line3 = f"Accuracy: {accuracy} %"

#     combined_title = f"{title_line1}\n{title_line2}\n{title_line3}"
#     plt.title(combined_title, fontsize=20, fontweight='bold', pad=20)

#     # plt.title(f"Correlation Matrix for {model_arch}_{config} with weight {os.path.basename(output_path)}")
#     plt.savefig(output_path, dpi=400, bbox_inches='tight')

# Train and validate function
def train_model(model, optimizer, criterion, train_loader, val_loader, epochs, scheduler, device, save_path,patience=5):
    model.to(device)
    
    history_epochs = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }
    
    min_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        # Training Loop
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Accumulate metrics
            running_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)

        epoch_loss = running_loss / total_preds
        epoch_acc = 100 * correct_preds / total_preds

        # Record training metrics
        history_epochs['train_loss'].append(epoch_loss)
        history_epochs['train_accuracy'].append(epoch_acc)

        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct_val_preds = 0
        total_val_preds = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                preds = outputs.argmax(dim=1)
                correct_val_preds += (preds == labels).sum().item()
                total_val_preds += labels.size(0)

        val_loss /= total_val_preds
        val_acc = 100 * correct_val_preds / total_val_preds

        # Record validation metrics
        history_epochs['val_loss'].append(val_loss)
        history_epochs['val_accuracy'].append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}: Train Loss = {epoch_loss:.4f}, Train Acc = {epoch_acc:.2f}%, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")

        # Check for early stopping and save best model
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), os.path.join(save_path, f'model_best_val.pth'))
            print("Model saved with best validation loss.")
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
        
        # Step the scheduler
        scheduler.step(val_loss)
    
    # save_history(history_epochs, save_path)
    return model, history_epochs


import time

def simple_train_model(model, train_loader, val_loader, criterion, optimizer, epochs, device, patience=10, log_interval=100):
    # History tracking
    history_epochs = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    best_val_loss = float('inf')
    early_stop_counter = 0

    # Learning Rate Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    model.to(device)

    for epoch in range(epochs):
        epoch_start_time = time.time()  # Track time for each epoch
        model.train()
        total_train_loss = 0.0
        correct_train_preds = 0
        total_train_points = 0

        # Training phase
        print(f"\nStarting Epoch {epoch + 1}/{epochs}")
        for batch_idx, (train_features, train_labels) in enumerate(train_loader):
            train_features, train_labels = train_features.to(device), train_labels.to(device)

            optimizer.zero_grad()
            outputs = model(train_features)
            loss = criterion(outputs, train_labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * train_labels.size(0)
            correct_train_preds += (outputs.argmax(axis=1) == train_labels).sum().item()
            total_train_points += train_labels.size(0)

            if (batch_idx + 1) % log_interval == 0:
                print(f"Epoch [{epoch + 1}/{epochs}] | Batch [{batch_idx + 1}/{len(train_loader)}] | "
                      f"Loss: {loss.item():.4f}")

        train_loss = total_train_loss / total_train_points
        train_acc = 100 * correct_train_preds / total_train_points

        # Validation phase
        model.eval()
        total_val_loss = 0.0
        correct_val_preds = 0
        total_val_points = 0

        with torch.no_grad():
            for val_features, val_labels in val_loader:
                val_features, val_labels = val_features.to(device), val_labels.to(device)
                val_outputs = model(val_features)

                loss = criterion(val_outputs, val_labels)
                total_val_loss += loss.item() * val_labels.size(0)
                correct_val_preds += (val_outputs.argmax(axis=1) == val_labels).sum().item()
                total_val_points += val_labels.size(0)

        val_loss = total_val_loss / total_val_points
        val_acc = 100 * correct_val_preds / total_val_points

        epoch_end_time = time.time()  # End time for the epoch

        # Log results
        history_epochs['train_loss'].append(train_loss)
        history_epochs['train_accuracy'].append(train_acc)
        history_epochs['val_loss'].append(val_loss)
        history_epochs['val_accuracy'].append(val_acc)

        print(f"\nEpoch Summary [{epoch + 1}/{epochs}]: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f} | "
              f"Time: {epoch_end_time - epoch_start_time:.2f} seconds")

        # Early stopping and model saving based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), f"{model.get_dir()}/model_epoch_{epoch}_best.pth")
            print(f"Best model saved at epoch {epoch + 1} with validation loss {val_loss:.4f}")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

        # Update learning rate
        scheduler.step(val_loss)

    return model, history_epochs


from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim
import torch.nn as nn

def train_model_Cosine_Annealing(model, train_loader, val_loader, epochs, lr=1e-4, patience=20):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(patience)
    # Define the optimizer with L2 regularization (weight decay)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=lr, momentum = 0.9, weight_decay=1e-3)

    # Initialize model, loss function, and optimizer

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
    # Cosine annealing LR scheduler
    
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)


    # optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    # Use Cyclic Learning Rate (CLR)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode="triangular")

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # Early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0

    # Initialize history tracking dictionary
    history_epochs = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        correct_train_preds = 0
        total_train_points = 0

        # Training loop
        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            targets = targets.long()

            optimizer.zero_grad()
            data = data.to(torch.float32)
            outputs = model(data)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            correct_train_preds += (outputs.argmax(dim=1) == targets).sum().item()
            total_train_points += targets.size(0)

        train_loss = total_train_loss / len(train_loader)
        train_acc = 100 * correct_train_preds / total_train_points

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        correct_val_preds = 0
        total_val_points = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                data = data.to(torch.float32)
                targets = targets.long()

                outputs = model(data)
                loss = criterion(outputs, targets)

                total_val_loss += loss.item()
                correct_val_preds += (outputs.argmax(dim=1) == targets).sum().item()
                total_val_points += targets.size(0)

        val_loss = total_val_loss / len(val_loader)
        val_acc = 100 * correct_val_preds / total_val_points

        # Log results into the history
        history_epochs['train_loss'].append(train_loss)
        history_epochs['train_accuracy'].append(train_acc)
        history_epochs['val_loss'].append(val_loss)
        history_epochs['val_accuracy'].append(val_acc)

        # Cosine annealing step
        scheduler.step()

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            print("New best weight based on validation loss")
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), f"{model.get_dir()}/best_model.pth")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return model, history_epochs

def train_model_optimized(model, train_loader, val_loader, epochs, lr=1e-4, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the optimizer with L2 regularization (weight decay)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Use Cyclic Learning Rate (CLR)
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-5, max_lr=1e-3, step_size_up=10, mode="triangular")

    # Focal Loss for better handling of imbalanced classes
    criterion = FocalLoss().to(device)

    # Early stopping
    best_val_loss = float('inf')
    early_stop_counter = 0

    history_epochs = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        correct_train_preds = 0
        total_train_points = 0

        # Training loop
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device).to(torch.float32)
            targets = targets.to(device)
            targets = targets.long()

            optimizer.zero_grad()
            data = data.to(torch.float32)
            outputs = model(data)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            correct_train_preds += (outputs.argmax(dim=1) == targets).sum().item()
            total_train_points += targets.size(0)

        train_loss = total_train_loss / len(train_loader)
        train_acc = 100 * correct_train_preds / total_train_points

        model.eval()
        total_val_loss = 0.0
        correct_val_preds = 0
        total_val_points = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)

                total_val_loss += loss.item()
                correct_val_preds += (outputs.argmax(dim=1) == targets).sum().item()
                total_val_points += targets.size(0)

        val_loss = total_val_loss / len(val_loader)
        val_acc = 100 * correct_val_preds / total_val_points

        # Log the history
        history_epochs['train_loss'].append(train_loss)
        history_epochs['train_accuracy'].append(train_acc)
        history_epochs['val_loss'].append(val_loss)
        history_epochs['val_accuracy'].append(val_acc)

        # Step the scheduler
        scheduler.step()

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            torch.save(model.state_dict(), f"{model.get_dir()}/best_model.pth")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    return model, history_epochs


# This function is used for EnhancedCrossAttention model
def train_model_with_contrastive_loss(model, train_loader, val_loader, epochs, lr=1e-4, patience=10, lambda_contrastive=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the optimizer with L2 regularization (weight decay)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Cosine annealing LR scheduler with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

    # Classification loss function with label smoothing
    classification_criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(device)

    # Early stopping variables
    best_val_loss = float('inf')
    early_stop_counter = 0

    # Initialize history tracking dictionary
    history_epochs = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": []
    }

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        correct_train_preds = 0
        total_train_points = 0

        # Training loop
        for batch_idx, (data, targets) in enumerate(train_loader):
            data = data.to(device, dtype=torch.float32)
            targets = targets.to(device, dtype=torch.long)

            optimizer.zero_grad()

            # Forward pass with labels to compute contrastive loss
            outputs, contrastive_loss = model(data, labels=targets)

            # Classification loss
            classification_loss = classification_criterion(outputs, targets)

            # Total loss
            loss = classification_loss + lambda_contrastive * contrastive_loss

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_train_preds += (predicted == targets).sum().item()
            total_train_points += targets.size(0)

        train_loss = total_train_loss / total_train_points
        train_acc = 100 * correct_train_preds / total_train_points

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        correct_val_preds = 0
        total_val_points = 0

        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(device, dtype=torch.float32)
                targets = targets.to(device, dtype=torch.long)

                # Forward pass without labels (contrastive loss not computed during validation)
                outputs = model(data, labels=None)

                # Classification loss
                loss = classification_criterion(outputs, targets)

                total_val_loss += loss.item() * data.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct_val_preds += (predicted == targets).sum().item()
                total_val_points += targets.size(0)

        val_loss = total_val_loss / total_val_points
        val_acc = 100 * correct_val_preds / total_val_points

        # Log results into the history
        history_epochs['train_loss'].append(train_loss)
        history_epochs['train_accuracy'].append(train_acc)
        history_epochs['val_loss'].append(val_loss)
        history_epochs['val_accuracy'].append(val_acc)

        # Cosine annealing step
        scheduler.step()

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            # Save the best model
            torch.save(model.state_dict(), f"{model.get_dir()}/best_model.pth")
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        # Print progress
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Load the best model before returning
    model.load_state_dict(torch.load(f"{model.get_dir()}/best_model.pth"))

    return model, history_epochs


from PIL import Image
def reshape_for_conv2d(x):
    # x has shape [batch_size, channels, depth, height, width]
    # We reshape it to [batch_size, channels * depth, height, width]
    batch_size, channels, depth, height, width = x.shape
    return x.view(batch_size, channels * depth, height, width)

from PIL import Image  # Import PIL for conversion

class RadarHDF5DatasetTransform(Dataset):
    def __init__(self, hdf5_path, map_stats, model_arch, transform=None):
        self.model_arch = model_arch
        self.transform = transform
        str_vars = ['packet_type', 'SENSOR_ADDRESS', 'MAC', 'IP', 'numchirps', 'chirpsamples', 'max_speed_m_s',
                    'max_range_m', 'range_resolution_m', 'folder', 'radar', 'date', 'label', 'Name', 'position', 'speed']
        self.str_vars = str_vars
        self.str_dict = dict()
        self.radar_vars = ['Azimuth', 'Elevation', 'Doppler_Map']
        self.radar_dict = dict()

        self.predictions = []

        self.encoding = {
            'layonfloor': 0,
            'layonsofa': 1,
            'sitonfloor': 2,
            'sitting': 3,
            'standing': 4,
            'walking': 5,
            'movesitonchair': 6,
            'movestandingpickupfromfloor': 7,
            'movestandingpickupfromtable': 8
        }

        self.to_keep = ['layonfloor', 'layonsofa', 'sitonfloor', 'sitting', 'standing', 'walking', 'movesitonchair']

        with h5py.File(hdf5_path, 'r') as hdf5_file:
            self.length = len(hdf5_file['label'])
            self.orig_length = len(hdf5_file['label'])

            for radar_var in self.radar_vars:
                vals = [(x.reshape((32, 256)) - map_stats[radar_var]['mean']) / map_stats[radar_var]['std'] for x in
                        hdf5_file[radar_var][:]]
                self.radar_dict[radar_var] = dict(zip(range(self.length), vals))

            for str_var in self.str_vars:
                self.str_dict[str_var] = dict(zip(range(self.length), hdf5_file[str_var][:]))

            str_cleaned = [s.strip().lower().decode("utf-8") for s in hdf5_file['label'][:]]
            self.str_dict['label_str'] = dict(zip(range(self.length), str_cleaned))

        self.custom_filter()

        self.encoding = {k: i for i, k in enumerate(self.to_keep)}
        self.encoding_reverse = {i: k for i, k in enumerate(self.to_keep)}

        encoded_labels = [*map(self.encoding.get, self.str_dict['label_str'].values())]
        self.str_dict['label'] = dict(zip(range(self.length), encoded_labels))

    def custom_filter(self):
        dict1 = list(self.str_dict['label_str'].items()).copy()

        valid_indices = []
        for k, v in dict1:
            if v in self.to_keep:
                valid_indices.append(k)
            else:
                del self.str_dict['label_str'][k]
                for str_var in self.str_vars:
                    del self.str_dict[str_var][k]
                for radar_var in self.radar_vars:
                    del self.radar_dict[radar_var][k]

        # Regenerate contiguous indices for radar_dict
        new_radar_dict = {radar_var: {} for radar_var in self.radar_vars}
        for new_idx, old_idx in enumerate(valid_indices):
            for radar_var in self.radar_vars:
                new_radar_dict[radar_var][new_idx] = self.radar_dict[radar_var][old_idx]

        self.radar_dict = new_radar_dict
        self.length = len(self.str_dict['label_str'])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = []
        for radar_var in self.radar_vars:
            image_bytes = self.radar_dict[radar_var][idx]
            x.append(image_bytes.copy())

        # Stack along channel dimension
        x = np.stack(x, axis=0)  # Shape: [3, 1, 32, 256]

        # Remove the extra dimension if it exists
        if x.shape[1] == 1:
            x = np.squeeze(x, axis=1)  # Shape: [3, 32, 256]

        # Convert the NumPy array to PIL Image
        pil_images = []
        for i in range(x.shape[0]):
            pil_image = Image.fromarray((x[i] * 255).astype(np.uint8))  # Convert float to uint8 for image
            pil_images.append(pil_image)

        # Apply the transformation if provided
        if self.transform:
            pil_images = [self.transform(img) for img in pil_images]

        x_transformed = torch.stack(pil_images, dim=0)  # Stack transformed images into a single tensor
        label = self.str_dict['label'][idx]
        
        return x_transformed, torch.tensor(label, dtype=torch.long)

    def get_whole_data(self):
        """
        Get all data (features and labels) for the entire dataset.
        """
        data = []
        labels = []

        for idx in range(self.__len__()):
            x, label = self.__getitem__(idx)
            # Squeeze the extra depth dimension (if exists)
            if x.shape[1] == 1:
                x = x.squeeze(1)  # Shape: [3, 32, 256]

            data.append(x.unsqueeze(0))  # Add batch dimension
            labels.append(label)

        data = torch.cat(data, dim=0)  # Concatenate all data along the batch dimension
        labels = torch.tensor(labels)
        print(f"shape data : {data.shape}")
        print(f"shape label : {labels.shape}")

        return data, labels



from torchvision import transforms

def create_data_loaders_transform(folder_train, folder_val, map_stats, batch_size, model_arch, transform=None):
    # Load all train data at once
    X_train_all = []
    y_train_all = []
    
    for train_file in os.listdir(folder_train):
        train_data = RadarHDF5DatasetTransform(os.path.join(folder_train, train_file), map_stats, model_arch, transform=transform)
        X_train, y_train = train_data.get_whole_data()
        X_train_all.append(X_train)
        y_train_all.append(y_train)

    # Concatenate all training data
    X_train_torch = torch.cat(X_train_all, dim=0)
    y_train_torch = torch.cat(y_train_all, dim=0)

    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Load all validation data at once
    X_val = []
    y_val = []
    
    for val_file in os.listdir(folder_val):
        val_data = RadarHDF5DatasetTransform(os.path.join(folder_val, val_file), map_stats, model_arch, transform=transform)
        print(val_data)
        X_v, y_v = val_data.get_whole_data()
        X_val.append(X_v)
        y_val.append(y_v)

    # Concatenate validation data
    X_val_torch = torch.cat(X_val, dim=0)
    y_val_torch = torch.cat(y_val, dim=0)

    val_dataset = TensorDataset(X_val_torch, y_val_torch)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


# Function to load and visualize the first 30 sample points
def val_load_train_image(train_path, map_stats):
    os.makedirs('val_image_load', exist_ok=True)
    os.makedirs('val_image_norm', exist_ok=True)

    hdf5_path = os.listdir(train_path)[0]
    hdf5_path = '/home/h3trinh/paper/data/multi_radar/train_data_30/train_chunk_num_100.h5'
    with h5py.File(hdf5_path, 'r') as h5_file:
        # Extract the relevant datasets and reshape properly
        # azimuth = h5_file['Azimuth'][:].reshape(-1, 2, 32, 256)  # Reshape to (1024, 2, 32, 256)
        # elevation = h5_file['Elevation'][:].reshape(-1, 2, 32, 256)  # Reshape to (1024, 2, 32, 256)
        # doppler = h5_file['Doppler_Map'][:].reshape(-1, 2, 32, 256)  # Reshape to (1024, 2, 32, 256)
        # labels = h5_file['label'][:]
        print(np.shape(h5_file['Azimuth']))


        for idx in range(30):

            azimuth = h5_file['Azimuth'][idx].reshape(2,32,256)
            elevation = h5_file['Elevation'][idx].reshape(2,32,256)
            doppler = h5_file['Doppler_Map'][idx].reshape(2,32,256)
            label = h5_file['label'][idx].decode("utf-8")


            print(f"Azimuth shape: {azimuth.shape}")
            print(f"Elevation shape: {elevation.shape}")
            print(f"Doppler Map shape: {doppler.shape}")

            view_1 = [azimuth[0], elevation[0], doppler[0]]  
            view_2 = [azimuth[1], elevation[1], doppler[1]]  #
            print(f"Sample {idx}: View 1 shapes: {[v.shape for v in view_1]}, View 2 shapes: {[v.shape for v in view_2]}")

            plot_views(view_1, view_2, label, idx)


def plot_views(view_1, view_2, label, idx):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))  # 2 rows, 3 columns for 3 metrics in each view

    metrics = ['Azimuth', 'Elevation', 'Doppler Map']

    # Plot for View 1
    for i, metric in enumerate(view_1):
        print(f"Plotting View 1, metric {metrics[i]} shape: {metric.shape}")  # Print shape before plotting
        axes[0, i].imshow(metric, cmap='viridis', aspect='auto')
        axes[0, i].set_title(f"View 1 {metrics[i]}")
        axes[0, i].axis('off')

    # Plot for View 2
    for i, metric in enumerate(view_2):
        print(f"Plotting View 2, metric {metrics[i]} shape: {metric.shape}")  # Print shape before plotting
        axes[1, i].imshow(metric, cmap='viridis', aspect='auto')
        axes[1, i].set_title(f"View 2 {metrics[i]}")
        axes[1, i].axis('off')

    fig.suptitle(f"Sample {idx} - Label: {label}", fontsize=16)

    plt.savefig(f'val_image_load/sample_{idx}.png')
    plt.close()


def validate_normalization_via_class(map_stats, hdf5_path='/home/h3trinh/paper/data/multi_radar/train_data_30/train_chunk_num_100.h5', save_dir='val_image_norm'):
    

    os.makedirs(save_dir, exist_ok=True)

    # Load the data using the RadarHDF5Dataset class (which performs normalization)
    train_data = RadarHDF5Dataset(hdf5_path, map_stats, model_arch=None)

    # Get the whole dataset (normalized data and labels)
    X_train, y_train = train_data.get_whole_data()

    # Check shapes of the training data and labels
    print(f"X_train shape: {np.shape(X_train)}")
    print(f"y_train shape: {np.shape(y_train)}")

    # Validate normalization for the first 30 samples
    for idx in range(min(30, len(y_train))):  # Limit to the first 30 samples or the dataset length
        data, label = X_train[idx], y_train[idx]  

        azimuth_view_1 = data[0,0,:,:].numpy()   
        azimuth_view_2 = data[0,1,:,:].numpy()  
        elevation_view_1 = data[1,0,:,:].numpy()  
        elevation_view_2 = data[1,1,:,:].numpy()  
        doppler_view_1 = data[2,0,:,:].numpy()  
        doppler_view_2 = data[2,1,:,:].numpy()  

        print(f"Sample {idx}: Azimuth View 1 shape: {azimuth_view_1.shape}, View 2 shape: {azimuth_view_2.shape}")
        print(f"Sample {idx}: Elevation View 1 shape: {elevation_view_1.shape}, View 2 shape: {elevation_view_2.shape}")
        print(f"Sample {idx}: Doppler View 1 shape: {doppler_view_1.shape}, View 2 shape: {doppler_view_2.shape}")

        # Plot and save normalized images for each view
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns

        # View 1 Plots (Azimuth, Elevation, Doppler)
        axes[0, 0].imshow(azimuth_view_1, cmap='viridis', aspect='auto')
        axes[0, 0].set_title(f'Normalized Azimuth View 1')

        axes[0, 1].imshow(elevation_view_1, cmap='viridis', aspect='auto')
        axes[0, 1].set_title(f'Normalized Elevation View 1')

        axes[0, 2].imshow(doppler_view_1, cmap='viridis', aspect='auto')
        axes[0, 2].set_title(f'Normalized Doppler View 1')

        # View 2 Plots (Azimuth, Elevation, Doppler)
        axes[1, 0].imshow(azimuth_view_2, cmap='viridis', aspect='auto')
        axes[1, 0].set_title(f'Normalized Azimuth View 2')

        axes[1, 1].imshow(elevation_view_2, cmap='viridis', aspect='auto')
        axes[1, 1].set_title(f'Normalized Elevation View 2')

        axes[1, 2].imshow(doppler_view_2, cmap='viridis', aspect='auto')
        axes[1, 2].set_title(f'Normalized Doppler View 2')

        # Save the figure for the current sample
        save_path = os.path.join(save_dir, f"sample_{idx}_normalized.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        # Convert torch.Tensor to numpy before calculating statistics
        azimuth_mean_1, azimuth_std_1 = np.mean(azimuth_view_1), np.std(azimuth_view_1)
        azimuth_mean_2, azimuth_std_2 = np.mean(azimuth_view_2), np.std(azimuth_view_2)

        elevation_mean_1, elevation_std_1 = np.mean(elevation_view_1), np.std(elevation_view_1)
        elevation_mean_2, elevation_std_2 = np.mean(elevation_view_2), np.std(elevation_view_2)

        doppler_mean_1, doppler_std_1 = np.mean(doppler_view_1), np.std(doppler_view_1)
        doppler_mean_2, doppler_std_2 = np.mean(doppler_view_2), np.std(doppler_view_2)

        print(f"Sample {idx} - Azimuth View 1: Mean={azimuth_mean_1:.3f}, Std={azimuth_std_1:.3f}")
        print(f"Sample {idx} - Azimuth View 2: Mean={azimuth_mean_2:.3f}, Std={azimuth_std_2:.3f}")
        print(f"Sample {idx} - Elevation View 1: Mean={elevation_mean_1:.3f}, Std={elevation_std_1:.3f}")
        print(f"Sample {idx} - Elevation View 2: Mean={elevation_mean_2:.3f}, Std={elevation_std_2:.3f}")
        print(f"Sample {idx} - Doppler View 1: Mean={doppler_mean_1:.3f}, Std={doppler_std_1:.3f}")
        print(f"Sample {idx} - Doppler View 2: Mean={doppler_mean_2:.3f}, Std={doppler_std_2:.3f}")

    print(f"Saved normalized images to {save_dir}")

    # Optionally, print out overall statistics for the entire dataset to validate normalization
    for radar_var, radar_idx in zip(['Azimuth', 'Elevation', 'Doppler'], [0, 1, 2]):
        all_view_1 = np.concatenate([X_train[i][radar_idx, 0].numpy().reshape(-1) for i in range(len(X_train))])
        all_view_2 = np.concatenate([X_train[i][radar_idx, 1].numpy().reshape(-1) for i in range(len(X_train))])
        
        print(f"{radar_var} View 1 - Overall Mean: {np.mean(all_view_1):.3f}, Std: {np.std(all_view_1):.3f}")
        print(f"{radar_var} View 2 - Overall Mean: {np.mean(all_view_2):.3f}, Std: {np.std(all_view_2):.3f}")

def validate_normalization_via_class_no_stat(hdf5_path='/home/h3trinh/paper/data/multi_radar/train_data_30/train_chunk_num_100.h5', save_dir='val_image_norm'):
    os.makedirs(save_dir, exist_ok=True)

    # Load the data using the RadarHDF5DatasetNoStat class (which performs normalization without stats)
    train_data = RadarHDF5DatasetNoStat(hdf5_path, model_arch=None)

    # Get the whole dataset (normalized data and labels)
    X_train, y_train = train_data.get_whole_data()

    # Check shapes of the training data and labels
    print(f"X_train shape: {np.shape(X_train)}")
    print(f"y_train shape: {np.shape(y_train)}")

    # Validate normalization for the first 30 samples (or as many as available)
    for idx in range(min(30, len(y_train))):  
        data, label = X_train[idx], y_train[idx]  

        azimuth_view_1 = data[0, 0, :, :].numpy()  # Azimuth, View 1  
        azimuth_view_2 = data[0, 1, :, :].numpy()  # Azimuth, View 2  
        elevation_view_1 = data[1, 0, :, :].numpy()  # Elevation, View 1  
        elevation_view_2 = data[1, 1, :, :].numpy()  # Elevation, View 2  
        doppler_view_1 = data[2, 0, :, :].numpy()  # Doppler, View 1  
        doppler_view_2 = data[2, 1, :, :].numpy()  # Doppler, View 2  

        # Print shapes of the views
        print(f"Sample {idx}: Azimuth View 1 shape: {azimuth_view_1.shape}, View 2 shape: {azimuth_view_2.shape}")
        print(f"Sample {idx}: Elevation View 1 shape: {elevation_view_1.shape}, View 2 shape: {elevation_view_2.shape}")
        print(f"Sample {idx}: Doppler View 1 shape: {doppler_view_1.shape}, View 2 shape: {doppler_view_2.shape}")

        # Plot and save normalized images for each view
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))  # 2 rows, 3 columns

        # View 1 Plots (Azimuth, Elevation, Doppler)
        axes[0, 0].imshow(azimuth_view_1, cmap='viridis', aspect='auto')
        axes[0, 0].set_title(f'Normalized Azimuth View 1')

        axes[0, 1].imshow(elevation_view_1, cmap='viridis', aspect='auto')
        axes[0, 1].set_title(f'Normalized Elevation View 1')

        axes[0, 2].imshow(doppler_view_1, cmap='viridis', aspect='auto')
        axes[0, 2].set_title(f'Normalized Doppler View 1')

        # View 2 Plots (Azimuth, Elevation, Doppler)
        axes[1, 0].imshow(azimuth_view_2, cmap='viridis', aspect='auto')
        axes[1, 0].set_title(f'Normalized Azimuth View 2')

        axes[1, 1].imshow(elevation_view_2, cmap='viridis', aspect='auto')
        axes[1, 1].set_title(f'Normalized Elevation View 2')

        axes[1, 2].imshow(doppler_view_2, cmap='viridis', aspect='auto')
        axes[1, 2].set_title(f'Normalized Doppler View 2')

        # Save the figure for the current sample
        save_path = os.path.join(save_dir, f"sample_{idx}_normalized.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()

        # Compute mean and std for each view
        azimuth_mean_1, azimuth_std_1 = np.mean(azimuth_view_1), np.std(azimuth_view_1)
        azimuth_mean_2, azimuth_std_2 = np.mean(azimuth_view_2), np.std(azimuth_view_2)

        elevation_mean_1, elevation_std_1 = np.mean(elevation_view_1), np.std(elevation_view_1)
        elevation_mean_2, elevation_std_2 = np.mean(elevation_view_2), np.std(elevation_view_2)

        doppler_mean_1, doppler_std_1 = np.mean(doppler_view_1), np.std(doppler_view_1)
        doppler_mean_2, doppler_std_2 = np.mean(doppler_view_2), np.std(doppler_view_2)

        print(f"Sample {idx} - Azimuth View 1: Mean={azimuth_mean_1:.3f}, Std={azimuth_std_1:.3f}")
        print(f"Sample {idx} - Azimuth View 2: Mean={azimuth_mean_2:.3f}, Std={azimuth_std_2:.3f}")
        print(f"Sample {idx} - Elevation View 1: Mean={elevation_mean_1:.3f}, Std={elevation_std_1:.3f}")
        print(f"Sample {idx} - Elevation View 2: Mean={elevation_mean_2:.3f}, Std={elevation_std_2:.3f}")
        print(f"Sample {idx} - Doppler View 1: Mean={doppler_mean_1:.3f}, Std={doppler_std_1:.3f}")
        print(f"Sample {idx} - Doppler View 2: Mean={doppler_mean_2:.3f}, Std={doppler_std_2:.3f}")

    print(f"Saved normalized images to {save_dir}")

    # Optionally, compute and print overall statistics for the entire dataset
    for radar_var, radar_idx in zip(['Azimuth', 'Elevation', 'Doppler'], [0, 1, 2]):
        all_view_1 = np.concatenate([X_train[i][radar_idx, 0].numpy().reshape(-1) for i in range(len(X_train))])
        all_view_2 = np.concatenate([X_train[i][radar_idx, 1].numpy().reshape(-1) for i in range(len(X_train))])
        
        print(f"{radar_var} View 1 - Overall Mean: {np.mean(all_view_1):.3f}, Std: {np.std(all_view_1):.3f}")
        print(f"{radar_var} View 2 - Overall Mean: {np.mean(all_view_2):.3f}, Std: {np.std(all_view_2):.3f}")
    



def val_create_data_loaders_no_stat(folder_train, folder_val, batch_size, model_arch):
    # Load all train data at once
    X_train_all = []
    y_train_all = []
    print(folder_train)
    print(folder_val)

    # Metrics you want to normalize separately (Azimuth, Elevation, Doppler)
    metrics = ['Azimuth', 'Elevation', 'Doppler_Map']
    
    # Load and concatenate the training data
    for train_file in os.listdir(folder_train):
        train_data = RadarHDF5Dataset2(folder_train + '/' + train_file, model_arch=model_arch)
        print(train_data)
        X_train, y_train = train_data.get_whole_data()
        X_train_all.append(X_train)
        y_train_all.append(y_train)

    # Concatenate all training data into tensors
    X_train_torch = torch.cat(X_train_all, dim=0)  # Shape should be [num_samples, num_metrics, 2, 32, 256]
    y_train_torch = torch.cat(y_train_all, dim=0)

    # Normalize each sample (image) independently
    for i in range(X_train_torch.size(0)):  # Iterate over each sample
        for idx, metric in enumerate(metrics):  # For each metric (Azimuth, Elevation, Doppler)
            # Extract the views for the current sample
            sample_data = X_train_torch[i, idx, :, :, :]  # Shape [2, 32, 256] for each metric's views
            
            # Compute mean and std for the current sample
            sample_mean = torch.mean(sample_data)
            sample_std = torch.std(sample_data)
            
            # Normalize the current sample
            X_train_torch[i, idx, :, :, :] = (sample_data - sample_mean) / (sample_std + 1e-6)  # Add epsilon to avoid division by zero

    # Validation step: Check that each sample is normalized correctly
    for i in range(100):  # Check the first 5 samples for validation
        for idx, metric in enumerate(metrics):
            sample_mean_check = torch.mean(X_train_torch[i, idx, :, :, :])
            sample_std_check = torch.std(X_train_torch[i, idx, :, :, :])
            print(f"Training Sample {i} - {metric}: Mean={sample_mean_check:.5f}, Std={sample_std_check:.5f}")
            assert torch.abs(sample_mean_check) < 1e-2, f"Mean is not zero for {metric} in sample {i}!"
            assert torch.abs(sample_std_check - 1) < 1e-3, f"Std is not one for {metric} in sample {i}!"

    # Load all validation data at once
    X_val_all = []
    y_val_all = []

    for val_file in os.listdir(folder_val):
        val_data = RadarHDF5Dataset2(folder_val + '/' + val_file, model_arch=model_arch)
        print(val_data)
        X_val, y_val = val_data.get_whole_data()
        X_val_all.append(X_val)
        y_val_all.append(y_val)

    # Concatenate validation data into tensors
    X_val_torch = torch.cat(X_val_all, dim=0)
    y_val_torch = torch.cat(y_val_all, dim=0)

    # Normalize each validation sample using per-sample statistics
    for i in range(X_val_torch.size(0)):  # Iterate over each validation sample
        for idx, metric in enumerate(metrics):  # For each metric
            # Extract the views for the current sample
            sample_data = X_val_torch[i, idx, :, :, :]
            
            # Compute mean and std for the current sample
            sample_mean = torch.mean(sample_data)
            sample_std = torch.std(sample_data)
            
            # Normalize the current sample
            X_val_torch[i, idx, :, :, :] = (sample_data - sample_mean) / (sample_std + 1e-6)

    # Validation step: Check that each validation sample is normalized correctly
    for i in range(100):  # Check the first 5 validation samples for validation
        for idx, metric in enumerate(metrics):
            sample_mean_check = torch.mean(X_val_torch[i, idx, :, :, :])
            sample_std_check = torch.std(X_val_torch[i, idx, :, :, :])
            print(f"Validation Sample {i} - {metric}: Mean={sample_mean_check:.5f}, Std={sample_std_check:.5f}")
            assert torch.abs(sample_mean_check) < 1e-2, f"Mean is not zero for {metric} in validation sample {i}!"
            assert torch.abs(sample_std_check - 1) < 1e-3, f"Std is not one for {metric} in validation sample {i}!"

    # Create DataLoaders for training and validation
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    val_dataset = TensorDataset(X_val_torch, y_val_torch)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


from sklearn.preprocessing import StandardScaler
import torch

def create_data_loaders_with_scaler(folder_train, folder_val, batch_size, model_arch):
    # Load all train data at once
    X_train_all = []
    y_train_all = []
    metrics = ['Azimuth', 'Elevation', 'Doppler_Map']
    
    # Load and concatenate the training data
    for train_file in os.listdir(folder_train):
        train_data = RadarHDF5Dataset2(folder_train + '/' + train_file, model_arch=model_arch)
        X_train, y_train = train_data.get_whole_data()
        X_train_all.append(X_train)
        y_train_all.append(y_train)

    # Concatenate all training data into tensors
    X_train_torch = torch.cat(X_train_all, dim=0)  # Shape should be [num_samples, num_metrics, 2, 32, 256]
    y_train_torch = torch.cat(y_train_all, dim=0)

    # Initialize a scaler for each metric
    scalers = {metric: StandardScaler() for metric in metrics}

    # Apply the scaler on each metric
    for idx, metric in enumerate(metrics):
        # Reshape data for scaler
        metric_data = X_train_torch[:, idx, :, :, :].reshape(-1, 32 * 256)
        # Fit scaler on training data and transform the data
        scaled_data = scalers[metric].fit_transform(metric_data)  # Fit on training data
        # Reshape back to original shape and assign it
        X_train_torch[:, idx, :, :, :] = torch.tensor(scaled_data).reshape(-1, 2, 32, 256)

    # Load all validation data at once
    X_val_all = []
    y_val_all = []

    for val_file in os.listdir(folder_val):
        val_data = RadarHDF5Dataset2(folder_val + '/' + val_file, model_arch=model_arch)
        X_val, y_val = val_data.get_whole_data()
        X_val_all.append(X_val)
        y_val_all.append(y_val)

    # Concatenate validation data into tensors
    X_val_torch = torch.cat(X_val_all, dim=0)
    y_val_torch = torch.cat(y_val_all, dim=0)

    # Normalize validation data using the same scalers from training data
    for idx, metric in enumerate(metrics):
        # Reshape validation data for the scaler
        metric_data_val = X_val_torch[:, idx, :, :, :].reshape(-1, 32 * 256)
        # Transform validation data (using the scaler fitted on training data)
        scaled_data_val = scalers[metric].transform(metric_data_val)
        # Reshape back to original shape and assign it
        X_val_torch[:, idx, :, :, :] = torch.tensor(scaled_data_val).reshape(-1, 2, 32, 256)

    # Create DataLoaders for training and validation
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    val_dataset = TensorDataset(X_val_torch, y_val_torch)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
