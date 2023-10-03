import numpy as np
from scipy.signal import butter, lfilter


def rotate_around(center, angle, values):
    angle = -(angle - np.deg2rad(90))

    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])

    return (R @ (values - center).T).T


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def is_point_in_triangle(pt, v1, v2, v3):
    d1 = sign(pt, v1, v2)
    d2 = sign(pt, v2, v3)
    d3 = sign(pt, v3, v1)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not(has_neg and has_pos)


def rp_to_trans(R, p):
    return np.r_[np.c_[R, p], [[0, 0, 0, 1]]]


def trans_to_rp(T):
    T = np.array(T)
    return T[0: 3, 0: 3], T[0: 3, 3]


def trans_inv(T):
    R, p = trans_to_rp(T)
    Rt = np.array(R).T
    return np.r_[np.c_[Rt, -np.dot(Rt, p)], [[0, 0, 0, 1]]]


def butter_lowpass(cutoff, fs, order=2):
    return butter(order, cutoff, fs=fs, btype='low', analog=False)


def butter_lowpass_filter(data, cutoff, fs, order=2):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y