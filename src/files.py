from typing import List
from pathlib import Path
import regex as re
from enum import Enum
from collections import namedtuple, defaultdict

from ghub_utils.types import FileType
from ghub_utils import files as gfiles


DIR_MODELS = gfiles.DIR_SAMPLE_DATA / 'models'


class FileParams(Enum):
    """netCDF file parameters"""
    MODEL = 'model'
    EXP = 'exp'
    FIELD = 'field'


def filter_paths_terms(paths: List[Path], search: list, param: FileParams) -> list:
    """
    Search for a list of terms @search in file names @paths
    :param paths:
    :param search: list of terms to search for
    :param param: parameter type of @search
    :return:
    """
    #print('paths in filter_paths_terms is:', paths)
    if param == FileParams.MODEL:
        search = map(lambda x: fr'_{x.replace("-", "_")}_', search)
    elif param == FileParams.EXP:
        search = map(lambda x: fr'_{x}.nc', search)
    elif param == FileParams.FIELD:
        search = map(lambda x: fr'\b{x}_', search)
    else:
        raise ValueError(f'Bad input for @param: {param}')

    search_str = '|'.join(search)

    filtered = []
    for p in paths:
        mat = re.search(search_str, p.name)
        if mat is not None:
            filtered.append(gfiles.get_path_relative_to(p, gfiles.DIR_PROJECT))

    return filtered


def netcdf_file_params(path: Path) -> namedtuple:
    """Get field, model, and experiment from .nc filename"""
    #print('path in netcdf_file_params:', path)
    pat_file_params = r'(?P<field>\w+)_\w+_(?P<model>\w+_\w+)_(?P<exp>\w+).nc'
    #pat_file_params = r'(?P<field>\w+)_GIS_(?P<model>\w+_\w+)_(?P<exp>\w+).nc'
    #print('path.name is:', path.name)
    Params = namedtuple('Params', 'field model exp')

    mat = re.search(pat_file_params, path.name)
    
    if mat and '_' in mat.group('field'):
        underscore_count = len(mat.group('field').split('_')) - 1
        
        # Adjust the pattern for 'model' to include more words
        model_word_count = underscore_count + 2  # Base model count is 2, add underscore_count
        
        # Build the new pattern for 'model'
        model_pattern = '_'.join([r'\w+'] * model_word_count)
        pat_file_params = rf'(?P<field>\w+)_\w+_(?P<model>{model_pattern})_(?P<exp>\w+).nc'
        print('Adjusted pattern:', pat_file_params)
        
        # Re-match with the updated pattern
        mat = re.search(pat_file_params, path.name)

    #mat = re.search(pat_file_params, path.name)
    if mat is not None:
        model = mat.group('model').replace('_', '-')

        return Params(mat.group('field'), model, mat.group('exp'))
    else:
        return None
    
def netcdf_file_params_new(path: Path) -> namedtuple:
    """Get field, model, and experiment from .nc filename based on folder structure and filename"""
    #print('type of the path is:', type(path))
    #print('path in netcdf_file_params:', path)
    Params = namedtuple('Params', 'field model exp')

    # Split the path into parts
    path_parts = path.parts
    #print('path_parts:', path_parts)

    # Extract the model from the folder names
    model_folder1 = path_parts[-4]  # ILTS_PIK
    model_folder2 = path_parts[-3]  # SICOPOLIS2
    model = f"{model_folder1}-{model_folder2}"

    # Extract the exp from the folder name just before the .nc file
    exp = path_parts[-2]  # ctrl_proj_05

    # Extract the field from the first word of the .nc file name
    filename = path.stem  # zvelbase_GIS_ILTS_PIK_SICOPOLIS2_ctrl_proj
    field = filename.split('_')[0]

    return Params(field, model, exp)


def intersect_netcdf_model_params(paths: List[Path]) -> namedtuple:
    """
    Extract all common models found in .nc filenames across fields

    :param paths: list of netCDF4 file paths;
      NOTE: must follow pattern: <field>_GIS_<modelA>_<modelB>_<experiment>.nc
    :return: namedtuple of sets of models, experiments, and fields
    """
    fields = defaultdict(set)
    for p in paths:
        if p.suffix != '.nc':
            continue

        params = netcdf_file_params(p)
        
        if params is not None:
            fields[params.field].add(params.model)

    models = set.intersection(*fields.values())
    return models


def union_netcdf_params(paths: List[Path]) -> namedtuple:
    """
    Extract all unique fields, experiments, and models found in .nc filenames

    :param paths: list of netCDF4 file paths;
      NOTE: must follow pattern: <field>_GIS_<modelA>_<modelB>_<experiment>.nc
    :return: namedtuple of sets of models, experiments, and fields
    """
    #print('paths in union_netcdf_params fn is:', paths)
    models = set()
    exps = set()
    fields = set()

    for p in paths:
        if p.suffix != '.nc':
            continue

        params = netcdf_file_params(p)
        #print('params for %s is:%s' %(p,params(models, exps, fields)))
        print('params for %s is: %s' %(p,params))
        if params is not None:
            models.add(params.model)
            exps.add(params.exp)
            fields.add(params.field)

    Params = namedtuple('Params', 'models exps fields')
    print('Overall Params is:',Params(models, exps, fields))
    return Params(models, exps, fields)


def get_dirs_union(
        dirs: List[Path],
        ftype: FileType = None,
        regex: str = None
) -> list:
    """
    TODO 8/14 (1): a lot of repetition in code

    Get the union of directory elements in @dirs;
      optionally use regular expression @regex to group by part of filenames
    """
    elems_all = []

    for d in dirs:
        if not d.is_dir():
            continue

        elems = set()

        for elem in d.iterdir():
            if ftype == FileType.DIR:
                if not elem.is_dir():
                    continue

                if not regex:
                    elems.add(elem.name)
                else:
                    try:
                        match = re.search(regex, elem.name).group(1)
                    except (AttributeError, IndexError):
                        # no regex match
                        continue

                    elems.add(match)
            elif ftype == FileType.FILE:
                if not elem.is_file():
                    continue

                if not regex:
                    elems.add(elem.name)
                else:
                    try:
                        match = re.search(regex, elem.name).group(1)
                    except (AttributeError, IndexError):
                        # no regex match
                        continue

                    elems.add(match)
            else:
                raise NotImplementedError('Not implemented')

        elems_all.append(elems)

    return sorted(set.union(*elems_all))


def get_dirs_intersect(
        dirs: List[Path],
        ftype: FileType = None,
        regex: str = None
) -> list:
    """
    # TODO 8/14 (2): a lot of code repetition
    Get the union of directory elements in @dirs;
      optionally use regular expression @regex to group by part of filenames
    """
    elems_all = []

    for d in dirs:
        if not d.is_dir():
            continue

        elems = set()

        for elem in d.iterdir():
            if ftype == FileType.DIR:
                if not elem.is_dir():
                    continue

                if not regex:
                    elems.add(elem.name)
                else:
                    try:
                        match = re.search(regex, elem.name).group(1)
                    except (AttributeError, IndexError):
                        # no regex match
                        continue

                    elems.add(match)
            elif ftype == FileType.FILE:
                if not elem.is_file():
                    continue

                if not regex:
                    elems.add(elem.name)
                else:
                    try:
                        match = re.search(regex, elem.name).group(1)
                    except (AttributeError, IndexError):
                        # no regex match
                        continue

                    elems.add(match)
            else:
                raise NotImplementedError('Not implemented')

        elems_all.append(elems)

    return sorted(set.intersection(*elems_all))


if __name__ == '__main__':
    import analysis

    print(analysis.FIELD_PAIRS)

    grouped = analysis.group_fields(['litempbot','uvelbase','vvelbase','uvelsurf','vvelsurf'])
    print(grouped)


# if __name__ == '__main__':
#     model_path = gfiles.DIR_SAMPLE_DATA / 'models'
#     files = list(model_path.rglob('*.nc'))
#
#     filtered = filter_paths_terms(files, ['lithk', 'acabf'], FileParams.FIELD)
#     [print(f) for f in filtered]