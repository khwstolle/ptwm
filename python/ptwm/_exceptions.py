import logging

# Standard logger for the package
logger = logging.getLogger("ptwm")
# Default handler setup could be left to the user, but we can provide a NullHandler
logger.addHandler(logging.NullHandler())


class Error(Exception):
    """Base exception for all errors originating from the PTWM library."""


class CompressionMethodNotSupportedError(Error):
    """Raised when an unsupported compression method is requested."""


class InvalidDTypeError(Error):
    """Raised when a tensor or numpy array has an unsupported dtype."""


class HeaderParseError(Error):
    """Raised when the binary header is invalid or corrupt."""


class UnsupportedQuantConfigError(ValueError):
    """Raised when ``hf_quant_config.json`` carries an unsupported algo."""
