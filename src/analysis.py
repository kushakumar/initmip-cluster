import yaml
import regex as re
from pathlib import Path
from collections import defaultdict, namedtuple
from typing import List
import traceback
import numpy as np
import numpy.ma as ma

from ghub_utils import files


with open(files.DIR_PROJECT / 'conf.yml', 'r') as f:
    y = yaml.safe_load(f)

    MODELS = list(y['model_paths'].keys())
    # possible model exclusions
    EXCLUDE = y['exclude']
    FIELD_PAIRS = y['field_pairs']
    EXPS = y['experiments']
    STEPS = y['steps']

    RUN_FIELDS = y['run_fields']


def group_fields(fields: List[str]):
    """
    Group component @fields into corresponding eof fields
    :param fields: list of fields (ungrouped)
    """
    eof_fields = set()
    used = set() # component fields to discard

    for eof, fs in FIELD_PAIRS.items():
        if isinstance(fs, list):
            if set(fs).issubset(set(fields)):
                eof_fields.add(eof)
                used.update(fs)
        else:
            if fs in fields:
                eof_fields.add(eof)
                used.add(fs)

    # discard used component fields
    eof_fields.update(set(fields).difference(used))
    return eof_fields


def format_data_exc(e: BaseException):
    """Process exceptions during data reading and put into a legible format"""
    Exc = namedtuple('Exc', 'name full simple')

    if isinstance(e, FileNotFoundError):
        fname_pat = r'No such file .+: b\'(.+.nc)\'' # filename regex pattern

        exc = traceback.format_exc(limit=0)
        file = re.search(fname_pat, exc).group(1)
        file = Path(file)
        fname = file.name

        outcome = Exc(name=e.__class__.__name__, full=exc, simple=fname)
    else:
        exc = traceback.format_exc()
        outcome = Exc(name=e.__class__.__name__, full=exc, simple=None)

    return outcome


def test_valid_data(models, fields, variables):
    """Test for invalid data in @models"""
    invalid = defaultdict(list) # MINE

    for field in fields:
        #print(f"Checking field: {field}")
        for m in models:
            #print(f"Checking model: {m}")
            if field in variables and m in variables[field]:
                uniq = np.unique(variables[field][m])
                num_uniq = len(uniq)
                if num_uniq <= 2:
                    # data is invalid and only has nans and/or mask values
                    invalid[field].append((m, ValueError))
            else:
                continue
                #print(f"KeyError: '{m}' not found in variables[{field}]")
                #raise KeyError(f"'{m}' not found in variables[{field}]")
    
    '''for field in fields:
        for m in models:
            print('field is:', field)
            print('model is:', m)
            uniq = np.unique(variables[field][m])
            num_uniq = len(uniq)
            if num_uniq <= 2:
                # data is invalid and only has nans and/or mask values
                invalid[field].append((m, ValueError))'''

    return invalid


def mask_invalid(xraw, yraw, models, variables, fields: list):
    mask2d = np.full(xraw.shape, False)
    print('mask2d is:', mask2d)

    for model in models:
        # Me: masking points that are nan, or missing (filler values), then
        #   combining them into single matrix mask2d
        # Ryan: masked points should be True due to some nan's, can't assume
        #   all fields have the same mask
        for i, field in enumerate(fields):
            print('field is:', field)
            print('model is:', model)
            if model in variables[field]:
                mask_nan = np.isnan(variables[fields[i]][model])
                print('mask_nan is:',mask_nan)
                mask_missing = variables[fields[i]][model] > 9e36
                print('mask_missing is:', mask_missing)
                mask2d = mask2d | mask_nan | mask_missing
        # for i in range(0, len(fields)):
        #     mask_nan = np.isnan(variables[fields[i]][model])
        #     mask_missing = variables[fields[i]][model] > 9e36
        #     mask2d = mask2d | mask_nan | mask_missing

    # Me: remove masked values and squeeze matrix into a 1d array
    # Ryan: meshgrid to coordinate lists
    xclean = ma.masked_array(xraw, mask2d).compressed()
    yclean = ma.masked_array(yraw, mask2d).compressed()

    return mask2d, xclean, yclean