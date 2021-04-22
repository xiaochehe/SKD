import os
import sys
import pickle

import argparse
import numpy as np
from numpy.lib.format import open_memmap

from utils.ntu_read_skeleton import read_xyz, read_xy

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body = 2
num_joint = 25
max_frame = 100
toolbar_width = 30

C = 3
T = 100
V = 25
M = 2

def mean_impute(skeleton, window_size = 1):
    C, T, V, M = skeleton.shape
    skeleton = skeleton.transpose((1,0,2,3)).reshape((T, -1)).transpose((1,0))
    NaN_series = [np.nan for i in range(window_size)]
    for m in range(skeleton.shape[0]):
        T_series = skeleton[m]
        T_series_cat = np.concatenate([NaN_series, T_series, NaN_series])
        for n in range(T_series.shape[0]):
            if np.isnan(T_series[n]):
                window_series =T_series_cat[n:n + 1 + 2* window_size]
                window_series[window_size] = np.nan
                T_series[n] = np.average(window_series[np.invert(np.isnan(window_series))])
        skeleton[m] = T_series
    skeleton = skeleton.transpose((1,0)).reshape((T, C, V, M)).transpose((1,0,2,3))
    return skeleton

def mean_filter(skeleton, window_size = 1):
    C, T, V, M = skeleton.shape
    skeleton = skeleton.transpose((1,0,2,3)).reshape((T, -1)).transpose((1,0))
    NaN_series = [np.nan for i in range(window_size)]
    for m in range(skeleton.shape[0]):
        T_series = skeleton[m]
        T_series_cat = np.concatenate([NaN_series, T_series, NaN_series])
        for n in range(T_series.shape[0]):
            window_series =T_series_cat[n:n + 1 + 2* window_size]
            T_series[n] = np.average(window_series[np.invert(np.isnan(window_series))])
        skeleton[m] = T_series
    skeleton = skeleton.transpose((1,0)).reshape((T, C, V, M)).transpose((1,0,2,3))
    return skeleton

def knn_impute(skeleton, K = 3):
    C, T, V, M = skeleton.shape
    skeleton = skeleton.transpose((1,0,2,3)).reshape((T, -1)).transpose((1,0))
    skeleton = KNN(k=K).fit_transform(skeleton)
    skeleton = skeleton.transpose((1,0)).reshape((T, C, V, M)).transpose((1,0,2,3))
    return skeleton


def print_toolbar(rate, annotation=''):
    # setup toolbar
    sys.stdout.write("{}[".format(annotation))
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(' ')
        else:
            sys.stdout.write('-')
        sys.stdout.flush()
    sys.stdout.write(']\r')


def end_toolbar():
    sys.stdout.write("\n")


def gendata(data_path,
            out_path,
            ignored_sample_path=None,
            benchmark='xview',
            part='eval'):
    if ignored_sample_path != None:
        with open(ignored_sample_path, 'r') as f:
            ignored_samples = [
                line.strip() + '.skeleton' for line in f.readlines()
            ]
    else:
        ignored_samples = []
    sample_name = []
    sample_label = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(
            filename[filename.find('A') + 1:filename.find('A') + 4])
        subject_id = int(
            filename[filename.find('P') + 1:filename.find('P') + 4])
        camera_id = int(
            filename[filename.find('C') + 1:filename.find('C') + 4])

        if benchmark == 'xview':
            istraining = (camera_id in training_cameras)
        elif benchmark == 'xsub':
            istraining = (subject_id in training_subjects)
        else:
            raise ValueError()

        if part == 'train':
            issample = istraining
        elif part == 'val':
            issample = not (istraining)
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open('{}/{}_label.pkl'.format(out_path, part), 'wb') as f:
        pickle.dump((sample_name, list(sample_label)), f)
    # np.save('{}/{}_label.npy'.format(out_path, part), sample_label)

    fp = open_memmap(
        '{}/{}_data.npy'.format(out_path, part),
        dtype='float32',
        mode='w+',
        shape=(len(sample_label), 6, max_frame, num_joint, max_body))

    for i, s in enumerate(sample_name):
        print_toolbar(i * 1.0 / len(sample_label),
                      '({:>5}/{:<5}) Processing {:>5}-{:<5} data: '.format(
                          i + 1, len(sample_name), benchmark, part))
        data = read_xyz(
            os.path.join(data_path, s), max_body=max_body, num_joint=num_joint)

        data_teacher = data

        dc, dt, dv, dm = data.shape
        occlusion_rand = 0.6
        mask = np.random.rand(dt, dv, dm)
        mask[mask > occlusion_rand] = 1
        mask[mask <= occlusion_rand] = 0
        data = data * mask[np.newaxis, :, :, :]
        data[data == 0] = np.nan
        data = mean_impute(data, 3)
        data[np.isnan(data)] = 0
        ture_shape = data.shape[1] if data.shape[1] < max_frame else max_frame
        data_slice = [int(a * (data.shape[1]/ture_shape)) for a in range(ture_shape)]
        # print(data[:, data_slice,:,:])
        fp[i, :3, 0:ture_shape, :, :] = data[:, data_slice,:,:]
        fp[i, 3:, 0:ture_shape, :, :] = data_teacher[:, data_slice,:,:]
    end_toolbar()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='NTU-RGB-D Data Converter.')
    parser.add_argument(
        '--data_path', default='../data/nturgb+d_skeletons')
    parser.add_argument(
        '--ignored_sample_path',
        default='../resource/NTU-RGB-D/samples_with_missing_skeletons.txt')
    parser.add_argument('--out_folder', default='../data/NTURGBDCompleteTeachOccludedMeanImputer_0.6')

    benchmark = ['xsub', 'xview']
    part = ['train', 'val']
    arg = parser.parse_args()

    for b in benchmark:
        for p in part:
            out_path = os.path.join(arg.out_folder, b)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(
                arg.data_path,
                out_path,
                arg.ignored_sample_path,
                benchmark=b,
                part=p)
