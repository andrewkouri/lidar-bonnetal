#!/usr/bin/env python3
# Copyright, 2020 DoorDash, Inc. All Rights Reserved.
# Code imported from open source projects not copyright.

import argparse
import os
import torch

import yaml
from tasks.semantic.dataset.kitti.parser import SemanticKitti
from tasks.semantic.modules.segmentator import Segmentator
import torch.backends.cudnn as cudnn
import numpy as np


def infer_and_save_labels(model, data_loader, to_orig_fn, out_directory, use_gpu=False):
    for i, (proj_in, proj_mask, _, _, path_seq, path_name,
            p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(data_loader):
        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if use_gpu:
            proj_in = proj_in.cuda()
            proj_mask = proj_mask.cuda()
            p_x = p_x.cuda()
            p_y = p_y.cuda()

        # compute output
        proj_output = model(proj_in, proj_mask)
        proj_argmax = proj_output[0].argmax(dim=0)

        unproj_argmax = proj_argmax[p_y, p_x]

        if use_gpu:
            torch.cuda.synchronize()

        print("Infered seq", path_seq, "scan", path_name)

        # save scan
        # get the first scan in batch and project scan
        pred_np = unproj_argmax.cpu().numpy()
        pred_np = pred_np.reshape((-1)).astype(np.int32)

        # map to original label
        pred_np = to_orig_fn(pred_np)

        # save scan
        path = os.path.join(out_directory, path_name)
        pred_np.tofile(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("./infer_directory.py")
    parser.add_argument(
        '--input-directory', '-i',
        type=str,
        required=True,
        help='Input Directory to .bin(s) (lidar scans). Place the input files inside this folder AND \
                nested under `/00/velodyne/`. No default',
    )
    parser.add_argument(
        '--output-directory', '-o',
        type=str,
        required=True,
        default=None,
        help='Directory to put the predictions .label(s). No default'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        default=None,
        help='Directory to get the trained model.'
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("----------")
    print("Argument Summary:")
    print("input_directory", FLAGS.input_directory)
    print("output_directory", FLAGS.output_directory)
    print("model", FLAGS.model)
    print("----------\n")

    # open arch config file
    try:
        print("Opening arch config file from %s" % FLAGS.model)
        ARCH = yaml.safe_load(open(FLAGS.model + "/arch_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file from %s" % FLAGS.model)
        DATA = yaml.safe_load(open(FLAGS.model + "/data_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # open model dir
    if os.path.isdir(FLAGS.model):
        print("model folder exists! Using model from %s" % (FLAGS.model))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()

    kitti_dataset = SemanticKitti(root=FLAGS.input_directory,
                                  sequences=[0],
                                  labels=DATA['labels'],
                                  color_map=DATA['color_map'],
                                  learning_map=DATA['learning_map'],
                                  learning_map_inv=DATA['learning_map_inv'],
                                  sensor=ARCH['dataset']['sensor'],
                                  max_points=ARCH['dataset']['max_points'],
                                  gt=True)

    data_loader = torch.utils.data.DataLoader(kitti_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              num_workers=0,
                                              pin_memory=True,
                                              drop_last=True)
    model = None

    # concatenate the encoder and the head
    with torch.no_grad():
        model = Segmentator(ARCH,
                            len(DATA['learning_map_inv']),
                            FLAGS.model)
        model.eval()
        # use knn post processing?
        # self.post = None
        # if self.ARCH["post"]["KNN"]["use"]:
        #     self.post = KNN(self.ARCH["post"]["KNN"]["params"],
        #                     self.parser.get_n_classes())

        # GPU?
        use_gpu = False
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Infering on device: ", device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            cudnn.benchmark = True
            cudnn.fastest = True
            gpu = True
            model.cuda()
            torch.cuda.empty_cache()

        infer_and_save_labels(model, data_loader,
                              lambda label: SemanticKitti.map(label, DATA['learning_map_inv']),
                              FLAGS.output_directory)
