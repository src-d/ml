import os


class SymlinkToNotExistingFile(Exception):
    """
    Exception for resolve_symlink function when symlink converts to nonexistent file
    """
    pass


def resolve_symlink(path: str) -> str:
    """
    Resolve symlink if path is a symbolic link
    Check file existence. If it does not exist raise SymlinkToNotExistingFile Exception
    :param path: path to check
    :return: filepath to existing file
    """
    # Check if file path is path
    # sometimes you have path to nonexistent files. We need to check it
    islink = os.path.islink(path)
    filepath = os.readlink(path) if islink else path
    if not os.path.exists(filepath):
        err_msg = "File %s does not exist.\n\t" % filepath + \
                  "Get it from %s path." % path if islink else "And it is not a path..."
        raise SymlinkToNotExistingFile(err_msg)
    return filepath
