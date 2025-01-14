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
    type: Literal["info", "warning", "error"]
    message: str
    backtrace: Union[str, None]

class SingleProgress(TypedDict):
    tag: Literal["progress"]
    progress: float
    message: str

class MultiProgress(TypedDict):
    tag: Literal["multi"]
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
            'tag': 'multi',
            'total': self.total,
            'current': self.current,
            'progress': progress,
            'message': self.message
        }

class PalmreaderProgress:
    _multi = None
    _single = None

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
    def start_single(message: str):
        PalmreaderProgress._single = message
        PalmreaderProgress._send()
    
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

    def display(self, msg = None, pos = None):
        orig = super().display(msg, pos)

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
        
        print(json.dumps(message))

    @staticmethod
    def _event(kind: Literal["info", "warning", "error"], message: str, exception: Union[Exception, None] = None):
        backtrace = None
        if exception is not None:
            backtrace = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))

        Palmreader._message({
            'tag': 'event',
            'type': kind,
            'message': message,
            'backtrace': backtrace
        })

    @staticmethod
    def info(message: str):
        Palmreader._event("info", message)
    
    @staticmethod
    def warning(message: str):
        Palmreader._event("warning", message)
    
    @staticmethod
    def error(message: str):
        Palmreader._event("error", message)
    
    @staticmethod
    def exception(message: str, exception: Exception):
        Palmreader._event("error", message, exception)
    
    @staticmethod
    def nonfatal(message: str, exception: Exception):
        Palmreader._event("warning", message, exception)