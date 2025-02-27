import warnings


class WarningContext:
    """
    A context manager that allows adding additional information onto warning messages. This is useful for debugging when some code downstream ours is throwing a warning and we want to know what we were doing when this warning was thrown, especially when the code creating a warning is called many times with different data and we want to know which data caused the warning.
    """

    extra_context: str
    _old_showwarning = None

    def __init__(self, extra_context: str):
        self.extra_context = extra_context

    def __enter__(self):
        self._old_showwarning = warnings.showwarning
        warnings.showwarning = self._showwarning

    def __exit__(self, exc_type, exc_val, exc_tb):
        warnings.showwarning = self._old_showwarning

    def _showwarning(self, message, category, filename, lineno, file=None, line=None):
        extra_context = f"Context: {self.extra_context}"
        full_message = f"{message}\n{extra_context}"
        self._old_showwarning(full_message, category, filename, lineno, file, line)
