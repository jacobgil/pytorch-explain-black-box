# coding: utf-8
"""
main script
"""

import argparse

from explain import run

__author__ = "nyk510"


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("img_path", type=str, help="path to image")
    parser.add_argument("--tv_beta", default=3, type=int)
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
    parser.add_argument("--max_iteration", default=500, type=int, help="max iteration")
    parser.add_argument("--l1_coefficient", default=0.01, type=float, help="l1 loss weight")
    parser.add_argument("--tv_coefficient", default=0.2, type=float, help="tv loss weight")
    return parser.parse_args()


if __name__ == '__main__':
    args = vars(get_args())
    run(**args)
