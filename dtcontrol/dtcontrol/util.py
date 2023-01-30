import json
import logging
import queue
from enum import Enum
from os.path import basename, splitext

import numpy as np
import sympy as sp


# Queue for use in interactive tree building between web ui and backend
class InteractiveQueue:
    def __init__(self):
        self.backend_to_frontend = queue.Queue()
        self.frontend_to_backend = queue.Queue()
        self.ready = False

    def set_ready(self):
        self.ready = True

    def set_done(self):
        self.ready = False

    def get_ready(self):
        return self.ready

    def get_done(self):
        return not self.ready

    def send_to_front(self, item):
        self.backend_to_frontend.put(item)

    def send_to_back(self, item):
        self.frontend_to_backend.put(item)

    def get_from_front(self):
        return self.frontend_to_backend.get()

    def get_from_back(self):
        return self.backend_to_frontend.get()

    def reset(self):
        for q in [self.backend_to_frontend, self.frontend_to_backend]:
            for _ in range(q.qsize()):
                q.get_nowait()
                q.task_done()


interactive_queue = InteractiveQueue()


class Caller(Enum):
    # Indicates that dtControl is run using Python code
    PYTHON = 1

    # Indicates that dtControl is run from the CLI
    CLI = 2

    # Indicates that dtControl is run from the Web UI
    WEBUI = 3


def ignore_convergence_warnings():
    logging.captureWarnings(capture=True)
    logger = logging.getLogger("py.warnings")
    handler = logging.StreamHandler()
    logger.addHandler(handler)
    logger.addFilter(lambda record: "ConvergenceWarning" not in record.getMessage())


def format_seconds(sec):
    m, s = divmod(sec, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    pattern = '%%02d:%%02d:%%0%d.%df' % (6, 3)
    if d == 0:
        return pattern % (h, m, s)
    return ('%d days, ' + pattern) % (d, h, m, s)


def split_relevant_extension(filename):
    if filename.endswith('.storm.json'):
        return filename.split('.storm.json')[0], '.storm.json'
    return splitext(filename)


def get_filename_and_relevant_extension(filename):
    path, ext = split_relevant_extension(filename)
    return basename(path), ext


def make_set(v):
    if v is None:
        return set()
    if isinstance(v, tuple):
        return {v}
    try:
        return set(v)
    except TypeError:
        return {v}


def objround(obj, precision):
    if isinstance(obj, list) or isinstance(obj, np.ndarray):
        return [objround(o, precision) for o in obj]
    if isinstance(obj, tuple):
        return tuple(round(o, precision) for o in obj)
    # if just a float
    return round(obj, precision)


def print_tuple(t):
    return f'({", ".join([str(e) for e in t])})'


def print_list(l):
    return f'[{", ".join([str(e) for e in l])}]'


def print_set(s):
    return f'{{{", ".join([str(e) for e in s])}}}'


def split_into_lines(l):
    if not isinstance(l, list) or len(l) < 5:
        return print_list(l)
    i = 0
    l2 = []
    while i < len(l):
        l2.append(', '.join([str(j) for j in l[i:min(i + 5, len(l))]]))
        i += 5
    return '[' + '\\n'.join(l2) + ']'


def peek_line(file):
    pos = file.tell()
    line = file.readline()
    file.seek(pos)
    return line


def log_without_newline(msg):
    old_terminator = logging.StreamHandler.terminator
    logging.StreamHandler.terminator = ""
    logging.info(msg)
    logging.StreamHandler.terminator = old_terminator


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def convert(o):
    if isinstance(o, np.int64):
        return int(o)
    if isinstance(o, np.float32):
        return float(o)
    if isinstance(o, sp.Basic):
        return str(o)
    raise TypeError


def error_wrapper(msg):
    ret = {"type": "error", "body": msg}
    return json.dumps(ret)


def success_wrapper(msg):
    ret = {"type": "success", "body": msg}
    return json.dumps(ret)