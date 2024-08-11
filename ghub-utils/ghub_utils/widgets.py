import ipywidgets as widg
import matplotlib.pyplot as plt
from IPython.display import display, Javascript
#from hublib import ui
from pathlib import Path
from typing import List, Dict, Callable

from . import files
from .types import SelectOption


class OptionToggle(widg.HBox):
    """Widget with multiple buttons that trigger a function on press"""
    def __init__(self, label: str, options: list, **kwargs):
        """
        :param options: list of button names
        :param kwargs:
        """
        label_desc = widg.Label(label)

        children = [label_desc,]
        for o in options:
            btn = widg.Button(description=o)
            children.append(btn)

        super().__init__(children=children, **kwargs)

    def on_click(self, btn_map: Dict[str, Callable]):
        """
        Register a function to trigger on click
        :param btn_map: dictionary of {button description: on click function}
        """
        for bname, func in btn_map.items():
            for b in self.children:
                if b.description == bname:
                    b.on_click(func)

    def click(self, btn_desc: str):
        for b in self.children:
            if b.description == btn_desc:
                b.click()

    def disable(self, btn_desc):
        for c in self.children:
            if c.description == btn_desc:
                c.disabled = True

    def enable(self, btn_desc):
        for c in self.children:
            if c.description == btn_desc:
                c.disabled = False


class DataSelector(widg.VBox):
    """
    Widget handling the selection of data for the project; allows for selecting
      from sample data (packaged with project), existing (previously uploaded),
      and upload data (upload your own)
    """
    OPTIONS = ['Sample', 'Personal']

    def __init__(self, **kwargs):
        self.data = None
        self.data_path: Path = None

        # uploaded/selected data
        label_instr = 'Select data source:'
        box_options = OptionToggle(label_instr, self.OPTIONS)

        sel_file = widg.Select(
            options=[],
            description='Select files:',
        )
        sel_file.layout.display = 'none' # hide selector upon init

        btn_up = widg.FileUpload(
            desc='Upload',
            accept='.p,.csv',
            multiple=True,
            dir=files.DIR_SESS,
        )
        btn_submit = widg.Button(description='Select')
        btn_submit.disabled = True # disable upon init
        box_pick = widg.HBox((btn_submit, btn_up))
        box_pick.layout.display = 'none' # hide upon init

        out_selected = widg.Output()
        out_selected.layout.visibility = 'hidden'

        def source_sample(b):
            """Sample data dir selected as source"""
            sel_file.layout.display = 'block'
            btn_up.layout.visibility = 'hidden'
            box_pick.layout.display = 'block'
            out_selected.layout.visibility = 'visible'

            samples = [
                str(files.get_path_relative_to(p, files.DIR_PROJECT))
                for p in Path(files.DIR_SAMPLE_DATA).iterdir()
                if p.suffix in files.FORMAT_DATA_IN
            ]
            sel_file.options = samples

        def source_own(b):
            """Personal data dir selected as source"""
            sel_file.layout.display = 'block'
            btn_up.layout.visibility = 'visible'
            box_pick.layout.display = 'block'
            out_selected.layout.visibility = 'visible'

            personal = [
                str(files.get_path_relative_to(p, files.DIR_PROJECT))
                for p in Path(files.DIR_TEMP).iterdir()
                if p.suffix in files.FORMAT_DATA_IN
            ]
            sel_file.options = personal

        box_options.on_click(
            {self.OPTIONS[0]: source_sample,
             self.OPTIONS[1]: source_own}
        )

        def select(change):
            """File selected from selector"""
            v = change['new']

            if (v is not None) and (len(v) > 0):
                btn_submit.disabled = False
                # save data path from project root
                self.data_path = files.DIR_PROJECT / v
            else:
                btn_submit.disabled = True

        sel_file.observe(select, names='value')

        def submit(b):
            """Read data in @self.data_path"""
            self.data = files.load_data(self.data_path)

            with out_selected:
                print(f'Loaded data from: {self.data_path.name}\n'
                      f'*** Access using "data" attribute of DataSelector. ***')
                display(self.data)

            out_selected.clear_output(wait=True)

        btn_submit.on_click(submit)

        def upload(change):
            v = change['new']

            if (v is not None) and (len(v) > 0):
                for name, meta in v.items():
                    # files.dump_data() requires a dictionary
                    d = {name: meta['content']}
                    files.dump_data(files.DIR_SESS_TDATA / name, d, bytes=True)
                    # reload list of existing files
                    box_options.click(self.OPTIONS[1])

        btn_up.observe(upload, names='value')

        super().__init__(
            [box_options, sel_file, box_pick, out_selected],
            **kwargs
        )


class ResultsDownloader(widg.HBox):
    """
    Abstract widget object for downloading some data with a field for specifying filename and a
      dropdown filetype selection
    """
    def __init__(self,
                 placeholder_filename: str,
                 download_formats: List,
                 download_name: str):
        txt_filename = widg.Text(
            value='',
            placeholder=placeholder_filename,
        )
        drop_file_format = widg.Dropdown(
            options=download_formats,
            value=download_formats[0]
        )
        btn_down = widg.Button(
            description=download_name,
            icon='download',
            disabled=True
        )

        def toggle(change):
            """Disable download button if no filename specified"""
            if change['name'] == 'value':
                if len(change['new']) > 0:
                    btn_down.disabled = False
                else:
                    btn_down.disabled = True

        txt_filename.observe(toggle)

        super().__init__((txt_filename, drop_file_format, btn_down))

    def on_click(self, func):
        btn_down = self.children[2]
        btn_down.on_click(func)

    def download(self, **kwargs):
        """Function that downloads respective data"""
        raise NotImplementedError('Abstract method; must override!')

    def disable(self):
        """
        Disable everything but the download button -- that gets controlled by
          the filename field
        """
        # TODO 6/27: could see this not working as intended if order of the
        #   children is changed
        for c in self.children[:-1]:
            c.disabled = True

    def enable(self):
        """
        Enable everything but the download button -- that gets controlled by
          the filename field
        """
        # TODO 6/27: could see this not working as intended if order of the
        #   children is changed
        for c in self.children[:-1]:
            c.disabled = False

    def hide(self):
        """Hide the widget"""
        if self.layout.display == 'block':
            self.layout.display = 'none'

    def show(self):
        """Show the widget"""
        if self.layout.display == 'none':
            self.layout.display = 'block'


class PlotDownloader(ResultsDownloader):
    """Button to download plots"""
    def __init__(self):
        super().__init__(
            placeholder_filename='enter plot filename',
            download_formats=files.FORMAT_IMG_OUT,
            download_name='Plot'
        )

    def download(self, fig: plt.Figure):
        """Download image to browser"""
        txt_filename = self.children[0]
        drop_file_format = self.children[1]

        filename = f'{txt_filename.value}.{drop_file_format.value}'
        path = files.upload_plt_plot(fig, filename)
        # need to make path relative to '.' for javascript windows
        path = files.get_path_relative_to(path, files.DIR_SRC).as_posix()

        # trigger a browser popup that will download the image to browser
        js = Javascript(
            f"window.open('{path}', 'Download', '_blank')"
        )
        display(js)


class DataDownloader(ResultsDownloader):
    """Widget for transformed data downloading"""
    def __init__(self):
        super().__init__(
            placeholder_filename='enter output data filename',
            download_formats=files.FORMAT_DATA_OUT,
            download_name='Data'
        )

    def download(self, data: dict):
        """Download @data to host results directory and to user system"""
        txt_filename = self.children[0]
        drop_file_format = self.children[1]

        fname = f'{txt_filename.value}.{drop_file_format.value}'
        fpath = files.DIR_RESULTS / fname
        files.dump_data(fpath, data, bytes=False)

        # need to make path relative to '.' for javascript windows
        path = files.get_path_relative_to(fpath, files.DIR_SRC).as_posix()

        # trigger a browser popup that will download the image to browser
        js = Javascript(
            f"window.open('{path}', 'Download', '_blank')"
        )
        display(js)


class FormConfigIO(widg.VBox):
    """
    Extension of hublib.ui.Form which appends submit and download buttons,
      as well as an Output widget that can print optional (test) messages
    """
    def __init__(self,
                 wlist,
                 update_func,
                 submit_text: str = "Submit",
                 download=True,
                 test=False,
                 test_msg: str = None,
                 **kwargs):
        """
        :param wlist:
        :param update_func:
        :param submit_text:
        :param download: should a download button be displayed to save output?
        :param test:
        :param test_msg:
        :param kwargs:
        """
        btn_submit = widg.Button(description=submit_text)
        down_plot = PlotDownloader()
        down_data = DataDownloader()
        if download:
            down_plot.disable() # nothing to download yet; disable
            down_data.disable()  # nothing to download yet; disable
        else:
            down_plot.hide() # @download == False
            down_data.hide()  # @download == False
        box_down = widg.VBox([down_plot, down_data])
        box_btns = widg.HBox([btn_submit, box_down])
        out_test = widg.Output()

        # @update_func output
        output = None

        @out_test.capture(clear_output=True, wait=True)
        def update(b):
            """Call the @update_func passed above"""
            if test:
                # print test output
                with out_test:
                    print('Updated parameters:\n')
                    for widget in wlist:
                        # only read new input from NumValue and its children
                        if isinstance(widget, widg.FloatText):
                             print(f'{widget.name} {widget.value}')

                    # print any additional message if passed
                    if test_msg is not None: print(test_msg)

            nonlocal output
            output = update_func()

            # if @update_func returns some results, enable downloading
            if output is None:
                down_plot.disable()
                down_data.disable()
            else:
                down_plot.enable()
                down_data.enable()

        def save_plot(b):
            nonlocal output
            fig = output.get('fig')

            down_plot.download(fig)

        def save_data(b):
            nonlocal output

            data = output.get('data')
            down_data.download(data)

        btn_submit.on_click(update)
        down_plot.on_click(save_plot)
        down_data.on_click(save_data)

        wlist.extend([box_btns, out_test])
        super().__init__(wlist, **kwargs)
