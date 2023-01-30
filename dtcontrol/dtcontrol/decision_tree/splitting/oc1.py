import logging
import os
import re
import shutil
import subprocess
from os.path import exists

import numpy as np

import dtcontrol.dtcontrol
import dtcontrol.dtcontrol.globals

from ...util import log_without_newline
from ..determinization.label_powerset_determinizer import LabelPowersetDeterminizer
from .axis_aligned import AxisAlignedSplit
from .linear_split import LinearSplit
from .splitting_strategy import SplittingStrategy


class OC1SplittingStrategy(SplittingStrategy):
    def __init__(self, determinizer=LabelPowersetDeterminizer(), num_restarts=10, num_jumps=5, delete_tmp=True):
        super().__init__()
        self.determinizer = determinizer
        self.oc1_path = 'decision_tree/OC1_source/mktree'
        self.header_file = 'decision_tree/OC1_source/oc1.h'
        self.tmp_path = '.dtcontrol_tmp'
        self.output_file = f'{self.tmp_path}/output'
        self.data_file = f'{self.tmp_path}/data.csv'
        self.dt_file = f'{self.tmp_path}/dt'
        self.log_file = f'{self.tmp_path}/log'
        self.num_restarts = num_restarts
        self.num_jumps = num_jumps
        self.delete_tmp = delete_tmp
        self.regex_pattern = re.compile(r'IMPURITY (.+) \(\)')
        if not os.path.exists(self.oc1_path):
            self.compile_oc1()

    def compile_oc1(self):
        for path in dtcontrol.__path__:
            oc1_src = f"{path}/decision_tree/OC1_source"
            if os.path.exists(oc1_src):
                if os.path.exists(oc1_src + "/mktree"):
                    self.oc1_path = oc1_src + "/mktree"
                    return
                try:
                    log_without_newline("Compiling OC1... ")
                    subprocess.call("make", cwd=oc1_src)
                    self.oc1_path = oc1_src + "/mktree"
                    logging.info("Compiled OC1")
                except subprocess.CalledProcessError:
                    raise EnvironmentError("Compiling OC1 failed")
            else:
                raise EnvironmentError("Could not find OC1 files")

    def find_split(self, dataset, impurity_measure, **kwargs):
        x_numeric = dataset.get_numeric_x()
        if x_numeric.shape[1] == 0:
            return None
        y = self.determinizer.determinize(dataset)
        if not os.path.exists(self.tmp_path):
            os.mkdir(self.tmp_path)
        self.save_data_to_file(x_numeric, y)
        self.set_impurity_measure(impurity_measure)
        self.execute_oc1()
        split = self.parse_oc1_dt(dataset)
        if self.delete_tmp:
            shutil.rmtree(self.tmp_path)
        return split

    def set_impurity_measure(self, impurity_measure):
        for path in dtcontrol.__path__:
            header = f'{path}/decision_tree/OC1_source/oc1.h'
            if os.path.exists(header):
                self.header_file = header
        with open(self.header_file, 'r') as infile:
            content = infile.read()
        match = re.search(self.regex_pattern, content)
        assert match
        if match.group(1) == impurity_measure.get_oc1_name():
            return
        new_content = re.sub(self.regex_pattern, f'IMPURITY {impurity_measure.get_oc1_name()} ()', content)
        with open(self.header_file, 'w') as outfile:
            outfile.write(new_content)
        logging.disable(logging.INFO)
        self.compile_oc1()
        logging.disable(logging.NOTSET)

    def save_data_to_file(self, x, y):
        data = np.c_[x, y]
        num_float_columns = data.shape[1] - 1
        np.savetxt(self.data_file, data, fmt=' '.join(['%f'] * num_float_columns + ['%d']), delimiter='\t')

    def execute_oc1(self):
        command = f'{self.oc1_path} -t {self.data_file} -D {self.dt_file} -p0 -i{self.num_restarts} ' \
                  f'-j{self.num_jumps} -l {self.log_file}'
        with open(self.output_file, 'w+') as out:
            p = subprocess.Popen(command.split(' '), stdout=out)
            dtcontrol.globals.oc1_pid = p.pid
            p.wait()
            dtcontrol.globals.oc1_pid = None

    def parse_oc1_dt(self, dataset):
        if not exists(self.dt_file):
            return None
        with open(self.dt_file) as infile:
            while not infile.readline().startswith('Root'):
                pass
            hyperplane_str = infile.readline()

        summands = hyperplane_str[:-4].split(' + ')
        intercept = float(summands[-1])
        dim = dataset.get_numeric_x().shape[1]
        j = 0
        coefficients = []
        for i in range(1, dim + 1):
            if f"x[{i}]" in summands[j]:
                coefficients.append(float(summands[j].split()[0]))
                j = j + 1
            else:
                coefficients.append(0)
        if len([c for c in coefficients if c != 0]) == 1:
            for i in range(len(coefficients)):
                if coefficients[i] != 0:
                    return AxisAlignedSplit(dataset.map_numeric_feature_back(i), -intercept)
        real_coefficients = LinearSplit.map_numeric_coefficients_back(coefficients, dataset)
        return LinearSplit(coefficients, intercept, real_coefficients, dataset.numeric_columns)
