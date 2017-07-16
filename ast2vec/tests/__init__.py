import os
import shutil
import tempfile

from modelforge.logs import setup_logging

from ast2vec import ensure_bblfsh_is_running_noexc, install_enry


ENRY = None


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
