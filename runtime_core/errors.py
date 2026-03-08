class RuntimeErrorBase(Exception):
    """Base class for runtime-core errors."""


class UnknownTaskKindError(RuntimeErrorBase):
    """Raised when no handler is registered for a task kind."""


class TaskNotFoundError(RuntimeErrorBase):
    """Raised when repository operation targets an unknown task id."""
