"""
Tools for communicating structure data back to Palmreader.

This module monkey-patches `tqdm` and must be imported before any module which uses `tqdm`.

Code can send events to Palmreader which will be shown to the user via the `Palmreader` class's static methods.

To manage + send progress reporting messages, use the `PalmreaderProgress` class. This class is what uses the monkey-patched `tqdm` to send progress messages to Palmreader using any progress bars created by our code or any downstream libraries.
"""

import json
import traceback
from typing import Union, Literal, TypedDict
import tqdm

class Event(TypedDict):
    tag: Literal["event"]
    level: Literal["info", "warning", "error"]
    title: str
    message: Union[str, None]
    backtrace: Union[str, None]

class SingleProgress(TypedDict):
    tag: Literal["progress"]
    progress: float
    message: str

class MultiProgress(TypedDict):
    tag: Literal["multiprogress"]
    total: int
    current: int
    progress: float
    message: str

class MultiProgressState:
    def __init__(self, total: int, message: str, autoincrement: bool = False):
        self.total = total
        self.current = 0
        self.message = message
        self.last_progress = -1.0
        self.autoincrement = autoincrement
    
    def inc(self):
        self.current += 1
    
    def as_message(self, progress: float) -> MultiProgress:
        if self.autoincrement:
            # some processes which we hook iterate the outer loop themselves, so we need a way to detect if the next iteration has started. we do this by checking if the progress has wrapped back around to zero.
            if progress < self.last_progress:
                self.inc()
            
            self.last_progress = progress

        return {
            'tag': 'multiprogress',
            'total': self.total,
            'current': self.current,
            'progress': progress,
            'message': self.message
        }

class PalmreaderProgress:
    _multi = None
    _single = None

    # used for managing parallel mode
    _single_current_id = 0
    _single_id = None
    _single_parallel = False

    @staticmethod
    def start_multi(total: int, message: str, autoincrement: bool = False):
        PalmreaderProgress._multi = MultiProgressState(total, message, autoincrement)

        PalmreaderProgress._send()

    @staticmethod
    def increment_multi():
        if PalmreaderProgress._multi is not None:
            PalmreaderProgress._multi.inc()
            PalmreaderProgress._send()
    
    @staticmethod
    def start_single(message: str, parallel: bool = False):
        PalmreaderProgress._single = message
        PalmreaderProgress._single_parallel = parallel

        PalmreaderProgress._send()
    
    @staticmethod
    def _provision_id():
        PalmreaderProgress._single_current_id += 1

        return PalmreaderProgress._single_current_id

    @staticmethod
    def _should_report(hook_id: int):
        if not PalmreaderProgress._single_parallel:
            # if parallel mode is off, every bar should report progress
            return True

        # otherwise, we need to assign the task of reporting progress to a single bar to prevent weird behavior
        if PalmreaderProgress._single_id is None:
            # whoever calls this function first gets to report progress
            PalmreaderProgress._single_id = hook_id
            return True

        return PalmreaderProgress._single_id == hook_id

    @staticmethod
    def _send(progress: float = 0):
        if PalmreaderProgress._multi is not None:
            Palmreader._message(PalmreaderProgress._multi.as_message(progress))
        elif PalmreaderProgress._single is not None:
            Palmreader._message({
                'tag': 'progress',
                'progress': progress,
                'message': PalmreaderProgress._single
            })

class PalmreaderProgressHook(tqdm.tqdm):
    _last_progress = -1.0
    _hook_id = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._hook_id = PalmreaderProgress._provision_id()

    def display(self, msg = None, pos = None):
        orig = super().display(msg, pos)

        if PalmreaderProgress._should_report(self._hook_id):
            # if we're in a parallel context, only one bar will run this block
            progress = self.n / self.total if self.total else 0
            if progress != self._last_progress:
                PalmreaderProgress._send(progress)

        return orig

def _trange(*args, **kwargs):
    return PalmreaderProgressHook(range(*args), **kwargs)

# Monkey-patch tqdm
tqdm.tqdm = PalmreaderProgressHook
tqdm.trange = _trange

MESSAGE = Union[Event, SingleProgress, MultiProgress]

class Palmreader:
    """
    Context for passing messages to Palmreader.

    Palmreader is constantly scanning stdout for messages produced by this
    context, and will display messages to the user as necessary.

    This also allows backtraces to be hidden from users, which is currently
    not feasible with API v1.

    Methods do nothing unless `Palmreader.set_enabled` is called.
    """

    _enabled = False
    
    @staticmethod
    def set_enabled(enabled: bool = True):
        """
        Set whether or not messages are actually emitted. They are disabled by
        default to maintain backwards compatibility.
        """
        Palmreader._enabled = enabled

    @staticmethod
    def _message(message: MESSAGE):
        if not Palmreader._enabled:
            return
        
        print(json.dumps(message), flush=True)

    @staticmethod
    def _event(level: Literal["info", "warning", "error"], title: str, message: Union[str, None] = None, exception: Union[Exception, None] = None):
        backtrace = None
        if exception is not None:
            backtrace = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))

        Palmreader._message({
            'tag': 'event',
            'level': level,
            'title': title,
            'message': message,
            'backtrace': backtrace
        })

    @staticmethod
    def info(title: str, message: Union[str, None] = None):
        Palmreader._event("info", title, message)
    
    @staticmethod
    def warning(title: str, message: Union[str, None] = None):
        Palmreader._event("warning", title, message)
    
    @staticmethod
    def error(title: str, message: Union[str, None] = None):
        Palmreader._event("error", title, message)
    
    @staticmethod
    def exception(title: str, exception: Exception):
        Palmreader._event("error", title, str(exception), exception)
    
    @staticmethod
    def nonfatal(title: str, exception: Exception):
        Palmreader._event("warning", title, str(exception), exception)