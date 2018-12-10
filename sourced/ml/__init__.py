try:
    import modelforge.configuration

    modelforge.configuration.refresh()
except ImportError:
    pass

__version__ = 0, 7, 2
