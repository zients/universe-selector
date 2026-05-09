from __future__ import annotations


class UniverseSelectorError(Exception):
    exit_code = 7


class ValidationError(UniverseSelectorError):
    exit_code = 1


class NotFoundError(UniverseSelectorError):
    exit_code = 2


class SchemaError(UniverseSelectorError):
    exit_code = 3


class BusyError(UniverseSelectorError):
    exit_code = 4


class DuckDbBusyError(BusyError):
    pass


class DataIntegrityError(UniverseSelectorError):
    exit_code = 5


class ProviderDataError(UniverseSelectorError):
    exit_code = 6
