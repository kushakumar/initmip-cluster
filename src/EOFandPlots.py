#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 14 15:09:38 2018

@author: rtwalker
"""

# imports #####################################################################

import sys
import os

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))
print(current_dir)

# Add the project root to the PYTHONPATH
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(project_root)

# Add the src directory to the PYTHONPATH
src_dir = os.path.abspath(os.path.join(current_dir, '.'))
sys.path.append(src_dir)

# Add ghub-utils to the PYTHONPATH
ghub_utils_dir = os.path.abspath(os.path.join(project_root, 'ghub-utils'))
sys.path.append(ghub_utils_dir)

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from numpy.linalg import svd, norm
from collections import defaultdict
import seaborn as sns
import pickle
from functools import partial
from typing import List
from pathlib import Path, PosixPath

import ReadFinalModule as read
from ghub_utils import files as gfiles
import analysis

def check_for_invalid_values(matrix, name):
    if np.isnan(matrix).any():
        print(f"NaNs detected in {name}")
    if np.isinf(matrix).any():
        print(f"Infs detected in {name}")
    if not np.all(np.isfinite(matrix)):
        print(f"Non-finite values detected in {name}")
        
        
def eof_and_plot(
        fig_dist: plt.Figure,
        fig_weight: plt.Figure,
        paths: List[Path]
):
    ### Read netCDF Files
    data = read.read_data(paths)
    step = 21

    models = data.models
    fields = data.fields
    eof_fields = data.eof_fields
    experiment = data.exps[0]

    X, Y = data.xraw, data.yraw
    variables = data.variables
    miss_data = data.miss_data

    bm = set()
    for m in miss_data.values():
        bm.update({mi[0] for mi in m})

    # remove data associated with the bad models
    models = sorted(list(set(models).difference(bm)))
    for f in fields:
        for m in bm:
            try:
                del variables[f][m]
            except KeyError:
                continue

    ### Mask Invalid Model Outputs
    mask2d, xclean, yclean = analysis.mask_invalid(
        X, Y, models, variables, list(fields)
    )

    matrix = defaultdict(partial(np.ndarray, 0))

    for field in fields:
        matrix[field] = np.empty((len(models), len(xclean)), float)

        for i, m in enumerate(models):
            matrix[field][i, :] = ma.masked_array(
                variables[field][m],
                mask2d
            ).compressed()


        # Check for invalid values in the input matrix
        check_for_invalid_values(matrix[field], f"matrix[{field}]")

    """------------------------Prepare for EOF Analysis--------------------------"""
    if {'uvelsurf', 'vvelsurf'}.issubset(set(fields)):
        matrix['uvelsurf'] *= 31557600  # m/a from m/s
        matrix['vvelsurf'] *= 31557600
        matrix['velsurf'] = np.sqrt(
            matrix['uvelsurf'] ** 2 + matrix['vvelsurf'] ** 2
        )
        fields.append('velsurf')
        for i in range(len(models)):
            variables['velsurf'][models[i]] = matrix['velsurf'][i, :]

    if {'uvelbase', 'vvelbase'}.issubset(set(fields)):
        matrix['uvelbase'] *= 31557600  # m/a from m/s
        matrix['vvelbase'] *= 31557600
        matrix['velbase'] = np.sqrt(
            matrix['uvelbase'] ** 2 + matrix['vvelbase'] ** 2
            )
        fields.append('velbase')
        for i in range(len(models)):
            variables['velbase'][models[i]] = matrix['velbase'][i, :]

    # scale the matrices
    anomaly_matrix = defaultdict(partial(np.ndarray, 0))
    scaled_matrix = defaultdict(partial(np.ndarray, 0))

    for field in eof_fields:
        std = np.std(matrix[field], axis=0)  # normalizing factors
        std[std == 0] = 1  # To avoid division by zero

        anomaly_matrix[field] = matrix[field] - np.mean(matrix[field], axis=0)
        scaled_matrix[field] = anomaly_matrix[field] / std

        # Debugging prints
        #print(f"Field: {field}")
        #print(f"Anomaly Matrix (first 5 elements): {anomaly_matrix[field].flatten()[:5]}")
        #print(f"Scaled Matrix (first 5 elements): {scaled_matrix[field].flatten()[:5]}")
        check_for_invalid_values(scaled_matrix[field], f"scaled_matrix[{field}]")

    """------------------------------EOF Analysis--------------------------------"""
    M = np.hstack([scaled_matrix[field] for field in eof_fields])

    # Check for NaNs or Infs in M
    check_for_invalid_values(M, "M before SVD")

    UU, svals, VT = svd(M, full_matrices=False)

    var_frac = svals / np.sum(svals)
    var_cum = np.cumsum(var_frac)

    # choose how many eof (i.e., cols) to keep (arbitrary cutoff)
    ncol = np.where(var_cum > 0.95)[0][0] if np.any(var_cum > 0.95) else len(var_cum) - 1

    UU = UU[:, :ncol]

    """---------------------------Find Intermodel Distances----------------------"""
    distance = defaultdict(dict)
    distance_matrix = np.empty((len(models), len(models)))

    for i in range(len(models)):
        for j in range(len(models)):
            distance[models[i]][models[j]] = distance_matrix[i, j] = norm(
                UU[i, :] - UU[j, :]
            )

    """--------------------------------Find Weights------------------------------"""
    num_std = [1.0, 2.0, 3.0]
    similarity_matrix = defaultdict(dict)
    weights = defaultdict(dict)

    for radius in num_std:
        similarity_radius = radius * np.std(distance_matrix[distance_matrix != 0])
        similarity_matrix[radius] = np.exp(
            -(distance_matrix / similarity_radius) ** 2
        )

        effective_repetition = np.zeros((len(models), 1))
        effective_repetition.fill(1.0)
        for i in range(len(models)):
            for j in range(len(models)):
                if j != i:
                    effective_repetition[i] += similarity_matrix[radius][i, j]

        weights[radius] = 1.0 / effective_repetition

    """--------------------------------Save Arrays-------------------------------"""
    pickle_name = f'output-{eof_fields}-{experiment}.p'

    with open(gfiles.DIR_OUT / pickle_name, 'wb') as f:
        pickle.dump([distance, distance_matrix, weights], f)

    """-----------------------------------Plot-----------------------------------"""
    plotmask = np.zeros_like(distance_matrix)
    plotmask[np.triu_indices_from(plotmask)] = True

    fig_dist.clear()
    ax1 = fig_dist.add_subplot()

    lim = 0.8 * np.max(np.abs(distance_matrix - np.median(distance_matrix)))
    ax1 = sns.heatmap(
        distance_matrix - np.median(distance_matrix), mask=plotmask,
        square=True, cmap='RdBu',
        linewidths=0.25, linecolor='white', vmin=-lim, vmax=lim,
        ax=ax1
    )

    ax1.xaxis.set_ticks(np.arange(0.5, len(models) + 0.5, 1))
    ax1.xaxis.set_ticklabels(models, rotation=90)
    ax1.yaxis.set_ticks(np.arange(0.5, len(models) + 0.5, 1))
    ax1.yaxis.set_ticklabels(models, rotation=0)

    fields_string = ', '.join(eof_fields)
    fields_string = '(' + fields_string + ')'

    if experiment == 'init':
        time_string = ' at INIT'
    else:
        time_string = ' at ' + experiment.upper() + ' time step ' + str(step)

    fig_dist.suptitle(
        'EOF inter-model distances vs median for fields ' + fields_string + time_string
    )

    if len(miss_data) > 0:
        cap_miss_d = [m[0] for m in list(miss_data.values())[0]]
    else:
        cap_miss_d = 'None'

    caption = f'Following models were excluded: '\
              f'\n- models with invalid data: {cap_miss_d}'
    fig_dist.text(
        x=0.5, y=-0.05, s=caption, horizontalalignment='center', fontsize=12
    )

    fig_dist.tight_layout()

    fname = f'distances-{fields_string}-{experiment.upper()}'
    fig_dist.savefig(gfiles.DIR_OUT / fname, bbox_inches='tight')

    display(fig_dist)
    plt.show()

    """---------------------------------Plot Weights-----------------------------"""
    fig_weight.clear()
    ax2 = fig_weight.add_subplot()

    for ix, key in enumerate(weights):
        ax2.plot(weights[key], marker='o', linestyle='none', alpha=0.75,
                 color=plt.cm.tab10(ix), label='R = ' + str(key) + r'$\sigma$'
                 )

    ax2.xaxis.set_ticks(np.arange(len(models)))
    ax2.xaxis.set_ticklabels(models, rotation=90)
    ax2.grid(which='both')

    fields_string = ', '.join(eof_fields)
    fields_string = '(' + fields_string + ')'

    if experiment == 'init':
        time_string = ' at INIT'
    else:
        time_string = ' at ' + experiment.upper() + ' time step ' + str(step)

    fig_weight.suptitle(
        'Weights by similarity radius R for fields ' + fields_string + time_string
    )
    ax2.legend()

    fig_weight.text(
        x=0.5, y=-0.05, s=caption, horizontalalignment='center', fontsize=12
    )

    fig_weight.tight_layout()

    fname = f'weights-{fields_string}-{experiment.upper()}'
    fig_weight.savefig(gfiles.DIR_OUT / fname, bbox_inches='tight')

    display(fig_weight)
    plt.show()


if __name__ == '__main__':
    paths = [PosixPath('data/models/libmassbf_AIS_ARC_PISM2_abmb.nc')]
    eof_and_plot(
        plt.Figure(figsize=(10,10)),
        plt.Figure(figsize=(10,10)),
        paths
    )
