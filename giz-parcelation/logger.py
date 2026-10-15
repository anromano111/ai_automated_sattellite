# Author: Jingyu Wang https://github.com/ccetaw, ETH Zurich
# A universal logger for printing and writing files.

import torch
import os
import imageio 
import json
# import pyexr
from icecream import ic

import numpy as np

class Logger:
    """
    Class Logger. An integration icecream debug print, as well as other ostream utilities. 
    This class is to be declared globally. You must call set_logdir() method to set the log directory.
    """

    def __init__(self, debug) -> None:
        """
        Input:
        - debug: bool. 
        """
        # Init parameters
        self.logdir = '.'
        self.debug = debug
        
        # ic debug_print
        self.debug_print = ic
        self.debug_print.configureOutput(prefix="Debug | ", includeContext=True)
        if debug:
            self.debug_print.enable()
        else:
            self.debug_print.disable()
        
    def info_print(self, info):
        print(f"- {info}")

    def set_logdir(self, basedir):
        """
        Set the log directory.
        ----
        Input:
        - basedir: str. Path to directory. 
        """
        self.logdir = basedir
        os.makedirs(self.logdir, exist_ok=True)

    def set_mode(self, debug=True):
        """
        Enable printing debug information. 
        """
        if debug:
            self.debug_print.enable()
        else:
            self.debug_print.disable()

    def _prepare_dir(self, path):
        """
        Make directory if path contains a dir.
        """
        prefix = os.path.dirname(path)
        if prefix != '':
            os.makedirs(os.path.join(self.logdir, prefix), exist_ok=True)
        write_path = os.path.join(self.logdir, path)
        
        return write_path

    def save_ckpt(self, path, ckpt):
        write_path = self._prepare_dir(path)
        torch.save(ckpt, write_path)

        if self.debug:
            info = f"Save checkpoint to {write_path}"
            self.debug_print(info)

    def write_image(self, path, *imgs):
        """
        Write a set of numpy array image to designated path. Convert it to uint8 in the first place.
        Note that all images must have same number of channels.
        """
        write_path = self._prepare_dir(path)
        image = np.concatenate(imgs, axis=1)
        if image.dtype == np.float32 or image.dtype == np.float64:
            image = (image * 255).astype(np.uint8)
        imageio.imwrite(write_path, image)

        if self.debug:
            info = f"Write image to {write_path}"
            self.debug_print(info)

    def write_pil_image(self, path, img):
        write_path = self._prepare_dir(path)
        img.save(write_path)
        if self.debug:
            info = f"Write image to {write_path}"
            self.debug_print(info)


    # def write_hdr(self, path, hdr):
    #     write_path = self._prepare_dir(path)
    #     pyexr.write(write_path, hdr)

    def write_yolo_label(self, path, labels, polygons):
        write_path = self._prepare_dir(path)
        assert len(labels) == len(polygons)
        with open(write_path, 'w') as file:
            for i, polygon in enumerate(polygons):
                points_str = " ".join([f"{p[0]:.6f} {p[1]:.6f}" for p in polygon])
                file.write(f"{labels[i]} " + points_str + "\n")

        

    def write_pddf2txt(self, path, pddf, **args):
        write_path = self._prepare_dir(path)
        pddf.to_csv(write_path, **args)

    def write_dict2txt(self, path, dict_):
        write_path = self._prepare_dir(path)
        # try:
        #     dict_.pop('config')
        # except:
        #     pass
        #
        # try:
        #     dict_.pop('train')
        # except:
        #     pass
        #
        with open(write_path, 'w') as f:
            for key, value in dict_.items(): 
                f.write('%s = %s\n' % (key, value))

        if self.debug:
            info = f"Write txt file to {write_path}"
            self.debug_print(info)

    def write_dict2json(self, path, dict_):
        write_path = self._prepare_dir(path)
        with open(write_path, 'w') as f:
            f.write(json.dumps(dict_, indent=4))

        if self.debug:
            info = f"Write json file to {write_path}"
            self.debug_print(info)

logger = Logger(debug=True)
