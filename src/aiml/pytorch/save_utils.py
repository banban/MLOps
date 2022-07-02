import os
import scipy.io as sio
import torchvision
import numpy as np


def save_mat(save_path, info):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    matfile_path = os.path.join(save_path, 'info.mat')
    sio.savemat(matfile_path, info)


def load_mat(load_path, variable_names=None):

    matfile_path = os.path.join(load_path, 'info.mat')
    return sio.loadmat(matfile_path, variable_names=variable_names)


def save_cam_map(save_path, augmented_images, name, disp_epoch_no):
    dname = '{}'.format(name)
    dname = os.path.join(save_path, dname)
    if not os.path.exists(dname):
        os.makedirs(dname)

    fname = 'e{}.jpg'.format(disp_epoch_no)
    fpath = os.path.join(dname, fname)
    torchvision.utils.save_image(augmented_images, fp=fpath,
                                 nrow=int(round(np.sqrt(len(augmented_images)))))
