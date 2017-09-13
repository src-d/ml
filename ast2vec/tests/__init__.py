import multiprocessing
import multiprocessing.forkserver as forkserver
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


if "grpc" in sys.modules:
    # use lazy_grpc as in bblfsh_roles.py if you really need it above
    raise RuntimeError("grpc may not be imported before fork()")
if multiprocessing.get_start_method() != "forkserver":
    try:
        multiprocessing.set_start_method("forkserver", force=True)
        forkserver.ensure_running()
    except ValueError:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        raise RuntimeError("multiprocessing start method is already set to \"%s\"" %
                           multiprocessing.get_start_method()) from None


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
