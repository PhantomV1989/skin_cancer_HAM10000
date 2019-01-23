import os
import numpy as np
import time
from itertools import compress
import pickle

'''
I am not using pandas because pandas has some performance issues with regarding to process_data_path(), almost 
INF slower compared using normal python array. So bye bye pandas...
'''

np.random.seed(int(time.time()))


class sc_mnist_data_handler():
    def __init__(self, data_path):
        self.data_path = data_path
        try:
            self.metadata
        except:
            self.process_data_path()
        return

    def process_data_path(self):
        items = get_items_in_folder(self.data_path)

        metadata_file = list(filter(lambda x: x.__contains__('metadata'), items))[0]
        self.metadata = read_csv_as_dict(metadata_file)

        # remapping metadata images to image directories
        img_folders = list(filter(lambda x: x.__contains__('HAM10000_images_part_'), items))
        tmp_img = list(self.metadata['image_id'])
        img_paths = []
        for f in img_folders:
            img_paths += get_items_in_folder(f)
        img_cnt = len(img_paths)
        for i, p in enumerate(img_paths):
            if i % 1000 == 0:
                print("Remapping metadata's images names to images paths...", i, ' of ', img_cnt)
            img_name = p.split('/')[-1].split('.')[0]
            try:
                img_pos = tmp_img.index(img_name)
                self.metadata['image_id'][img_pos] = p
            except:
                print('Image ', img_name, ' not found in image folders')

        self.metadata['dx'], self.lookup_dx = self.to_lookup_values(self.metadata['dx'])
        self.metadata['dx_type'], self.lookup_dx_type = self.to_lookup_values(self.metadata['dx_type'])
        self.metadata['localization'], self.lookup_localization = self.to_lookup_values(self.metadata['localization'])
        self.metadata['age'] = [float(x) if x != '' else -1.0 for x in self.metadata['age']]
        self.metadata['sex'] = [0 if x == 'male' else 1 for x in self.metadata['sex']]
        return

    def get_imgs_by_unique_lesion_id(self, mode='first'):
        '''
        mode='first', returns 1st image id associated with particular lesion_id
        mode='random', returns a random image id associated with particular lesion id
        '''
        unique_lesion_ids = list(set(self.metadata['lesion_id']))
        unique_image_paths = []  # 'HAM_0002644'
        if mode == 'first':
            unique_image_paths = [self.metadata['lesion_id'].index(lid) for lid in unique_lesion_ids]
        elif mode == 'random':
            raise NotImplementedError()
            '''
            for lid in unique_lesion_ids:
                img_indices = []
                appended = False
                for i, v in enumerate(self.metadata['lesion_id']):
                    if v == lid:
                        img_indices.append(i)
                    elif len(img_indices) > 0:
                        break

                unique_image_paths.append(np.random.choice(img_indices))
            '''
        data = {'lesion_id': unique_lesion_ids}

        mask = np.zeros(len(self.metadata['lesion_id']))
        for i in unique_image_paths:
            mask[i] = 1

        for k in self.metadata.keys():
            if k != 'lesion_id':
                data[k] = list(compress(self.metadata[k], mask))
        return data

    @staticmethod
    def to_lookup_values(values):
        keys = list(set(values))
        lookup_table = {}
        for i, k in enumerate(keys):
            lookup_table[k] = i
        values = [lookup_table[x] for x in values]
        return values, lookup_table


def get_items_in_folder(folder):
    return [folder + '/' + x for x in os.listdir(folder)]


def get_data_path(current_path):
    # handles windows or linux systems (haven test for windows though)
    separator = '/' if current_path.__contains__('/') else '\\'
    data_path = separator.join(current_path.split(separator)[:-1] + ['skin-cancer-mnist-ham10000'])
    if not os.path.exists(data_path):
        print('Data path ', data_path,
              ' does not exist!! Please download data from https://www.kaggle.com/kmader/skin-cancer-mnist-ham10000 ' +
              'and unzip the data to ', data_path)
        raise FileNotFoundError(data_path, ' not found.')
    return data_path

def get_save_path(current_path):
    # handles windows or linux systems (haven test for windows though)
    separator = '/' if current_path.__contains__('/') else '\\'
    save_path = separator.join(current_path.split(separator)[:-1] + ['save'])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    return save_path

def read_csv(path, delimiter=','):
    import csv
    data = []
    with open(path) as csvfile:
        spamreader = csv.reader(csvfile, delimiter=delimiter)
        data = [x for x in spamreader]
    return data


def read_csv_as_dict(path, delimiter=','):
    arr = read_csv(path=path, delimiter=delimiter)
    keys = arr[0]
    data = arr[1:]
    result = {}
    for i, k in enumerate(keys):
        result[k] = [x[i] for x in data]
    return result


def save_load_helper(save, load, path, constructor):
    obj = []
    if load and os.path.exists(path):
        with open(path, 'rb') as file_object:  # load
            obj = pickle.load(file_object)
    else:
        obj = constructor()
        if save:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
    return obj


def split_data(data, ratio):
    a, b = {}, {}
    a_len = -1
    for k in data.keys():
        if a_len == -1:
            a_len = int(ratio * len(data[k]))
        a[k] = data[k][:a_len]
        b[k] = data[k][a_len:]
    return a, b


def shuffle_dict(data):
    arr, keys = dict_to_array(data)
    np.random.shuffle(arr)
    return arr_to_dict(arr, keys)


def dict_to_array(dict_data):
    keys = list(dict_data.keys())
    arr = []
    length = len(dict_data[keys[0]])
    for i in range(length):
        t = []
        for k in keys:
            t.append(dict_data[k][i])
        arr.append(t)
    return arr, keys


def arr_to_dict(arr, keys, mode='multi'):
    result = {}
    for i, k in enumerate(keys):
        if mode == 'multi':
            result[k] = [x[i] for x in arr]
        else:
            result[k] = [x[i] for x in arr][0]
    return result


def fragment_dict_data(data):
    arr, keys = dict_to_array(data)
    result = [arr_to_dict([x], keys, mode='first') for x in arr]
    return result
