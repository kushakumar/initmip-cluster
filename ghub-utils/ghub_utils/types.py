from typing import Union
from pathlib import Path
from enum import Enum


class FileType(Enum):
    DIR = 'dir'
    FILE = None
    NETCDF = '.nc'
    CSV = '.csv'
    PICKLE = '.p'


class SelectOption:
    def __init__(self,
                 option: str,
                 content: Union[Path, list],
                 ftype: FileType,
                 multi: bool):
        self.option = option
        self.content = self._read_content(content, ftype)
        self.file_type = ftype
        self.multi = multi

    def _read_content(self, content, ftype: FileType):
        """
        Parse and return the sorted contents of @content depending on its type
        """
        if isinstance(content, list):
            return sorted(content)

        if not isinstance(content, Path):
            raise TypeError(f'content must be pathlib.Path or list; '
                            f'got: {content} ({type(content)})')

        if ftype == FileType.DIR:
            opts = sorted([c for c in content.iterdir() if c.is_dir()])
        elif ftype is FileType.ANY:
            opts = sorted([c for c in content.iterdir() if c.is_file()])
        else:
            opts = sorted([c for c in content.iterdir() if c.suffix == ftype.value])

        return opts