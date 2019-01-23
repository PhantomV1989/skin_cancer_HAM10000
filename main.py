from sc_mnist_data import *
from sc_mnist_model import *
import os
import pickle
import cv2

# note: only tested on linux system, file separator is /. If you are using windows, please change separator to \

mode = 'train'  # use, train, validate
# mode = 'train'

# for training only
load_from_save_state = False
save_mode = False

current_dir = os.getcwd()
data_dir = get_data_path(current_dir)
save_dir = get_save_path(current_dir)

data_path = save_dir + '/sc_mnist_data'
model_path = save_dir + '/sc_mnist_model'
device = 'cpu'

iter = 1000
lr = 1e-7
batch_size = 6
if __name__ == '__main__':
    sc_mnist_data = save_load_helper(save=save_mode, load=True, path=data_path,
                                     constructor=lambda: sc_mnist_data_handler(data_path=data_dir))

    if mode == 'train':
        # i am using images by unique lesion id so no lesion will have over-representation due to multiple images
        unique_lesion_data = sc_mnist_data.get_imgs_by_unique_lesion_id(mode='first')
        train_data, test_data = split_data(shuffle_dict(unique_lesion_data), 0.8)

        for label in range(len(sc_mnist_data.lookup_dx)):
            ml_model = sc_mnist_model(
                cnn_sizes=[{'filters': 10, 'kernel': 3, 'stride': 1}, {'filters': 12, 'kernel': 3, 'stride': 1},
                           {'filters': 14, 'kernel': 4, 'stride': 1}, {'filters': 24, 'kernel': 4, 'stride': 1},
                           {'filters': 34, 'kernel': 5, 'stride': 2}],  # depth~5, kernel stable, filters stable
                fc_sizes=[20, 40, 80],  # <--increasing pyramid
                img_dim=[450, 600],
                dx_cnt=len(sc_mnist_data.lookup_dx.keys()),
                dx_type_cnt=len(sc_mnist_data.lookup_dx_type.keys()),
                localization_cnt=len(sc_mnist_data.lookup_localization.keys()),
                device=device
            )
            if save_mode:
                ml_model.train_model_single_label(data=train_data, label=label, batch_size=batch_size, lr=lr,
                                                  save_path=model_path, iter=iter, load=load_from_save_state)
            else:
                ml_model.train_model_single_label(data=train_data, label=label, batch_size=batch_size, lr=lr, iter=iter,
                                                  load=load_from_save_state)

    elif mode == 'validate':
        unique_lesion_data = sc_mnist_data.get_imgs_by_unique_lesion_id(mode='first')
        train_data, test_data = split_data(shuffle_dict(unique_lesion_data), 0.990)
        sc_mnist_model.multiclass_get_results(test_data, model_path=model_path, device=device)
