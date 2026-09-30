class IntelifError(Exception):
    pass


class IntelifValidationError(IntelifError, ValueError):
    pass


class IntelifUnsupportedError(IntelifError):
    pass


class IntelifModelError(IntelifError):
    pass
