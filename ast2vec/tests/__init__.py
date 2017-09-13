import os
import shutil
import sys
import tempfile

from modelforge.logs import setup_logging

from ast2vec import ensure_bblfsh_is_running_noexc, install_enry


ENRY = None

utmain = sys.modules['__main__']
if utmain.__package__ == "unittest" and utmain.__spec__ is None:
    from collections import namedtuple
    ModuleSpec = namedtuple("ModuleSpec", ["name"])
    utmain.__spec__ = ModuleSpec("unittest.__main__")
    del ModuleSpec
del utmain


def setup():
    setup_logging("INFO")
    global ENRY
    if ENRY is not None:
        return
    ENRY = os.path.join(tempfile.mkdtemp(), "enry")
    if os.path.isfile("enry"):
        shutil.copy("enry", ENRY)
    else:
        install_enry(target=ENRY)
    ensure_bblfsh_is_running_noexc()
