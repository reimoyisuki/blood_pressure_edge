#!/usr/bin/env python
# coding: utf-8

#
# MIMIC-BP - (C) 2024
# https://doi.org/10.7910/DVN/DBM1NF
#
# Open Data Commons Open Database License (ODbL) v1.0
# https://opendatacommons.org/licenses/odbl/1-0/
#

"""
Example usage:
> python read_data.py -d mimic-bp -p p093833 -i 29 -g
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def showBP(dbPath, patient, idx):
    "Prints SBP and DBP of segment idx from patient"

    if not os.path.isdir(dbPath):
        raise Exception(f'Check if dbPath "{dbPath}" is correct')

    labels_fn = patient + "_labels.npy"
    labels = np.load(os.path.join(dbPath, labels_fn))
    assert labels.shape == (30, 2), 'Problem reading "{labels_fn}"'

    # systolic blood pressure, sbp, and diastolic blood pressure, dbp
    sbp, dbp = labels[idx, :]
    print(f"Patient {patient}, segment {idx}")
    print(f"SBP = {sbp} mmHg\nDBP = {dbp} mmHg")


def plot_abp_ppg(dbPath, patient, idx):
    "Plots first 5 seconds of waveforms"
    print(f"\nFirst 5 seconds of waveforms\n")
    waves = ["abp", "ecg", "ppg", "resp"]
    for wav in waves:
        # Load PPG
        ppg_fn = patient + "_ppg.npy"
        ppg = np.load(os.path.join(dbPath, ppg_fn))
        assert ppg.shape == (30, 3750), 'Problem reading "{ppg_fn}"'
        
        # Load ABP
        abp_fn = patient + "_abp.npy"
        abp = np.load(os.path.join(dbPath, abp_fn))
        assert abp.shape == (30, 3750), 'Problem reading "{abp_fn}"' 

        fs = 125  # sampling frequency
        # N = len(abp[idx])  # number of samples in a segment
        N = abp.shape[1]
        t = np.arange(N) / fs  # time index
        tidx = t < 5  # plot first five seconds
        alpha=0.8 
        fig, axs = plt.subplots(2, 1, figsize=(12,8), sharex=True)
        # sub plot (1) abp
        # subplot 1: ppg
        axs[0].plot(t[tidx], ppg[idx][tidx], color="blue")
        axs[0].set_ylabel("PPG")
        axs[0].grid()

        # subplot 2: ABP
        axs[1].plot(t[tidx], abp[idx][tidx], color="red")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("ABP (mmHg)")
        axs[1].grid()

        fig.suptitle(f"Patient {patient}, segment {idx} - ABP & PPG", fontsize=14)
        plt.tight_layout()
        plt.show()

        plt.plot(t[tidx], ppg[idx][tidx], label="PPG")
        plt.plot(t[tidx], abp[idx][tidx], label="ABP  [mmHg]")
        plt.xlabel("time (s)")
        if "abp" == wav:
            plt.ylabel("PPG")
        else:
            plt.ylabel("ABP [mmHg]")
        plt.title(f"Patient {patient}, segment {idx}")
        plt.grid()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="read_data",
        description="How to read files from MIMIC-BP",
        epilog="Last update: 20Jun2023",
    )
    parser.add_argument(
        "-d", "--dbPath", help="path to .npy files", required=True
    )
    parser.add_argument(
        "-p", "--patient", help="patient ID (eg, p093833)", required=True
    )
    parser.add_argument(
        "-i", "--idx", type=int, help="segment index", required=True
    )
    parser.add_argument(
        "-g", "--graph", action="store_true", help="flag for graphs"
    )
    args = parser.parse_args()

    showBP(args.dbPath, args.patient, args.idx)

    if args.graph:
        plot_abp_ppg(args.dbPath, args.patient, args.idx)
