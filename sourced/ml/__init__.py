"""MLonCode research playground."""
try:
    import modelforge.configuration

    modelforge.configuration.refresh()
except ImportError:
    pass

__version__ = "0.8.1"
