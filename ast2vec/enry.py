import logging
import os
import shutil
import subprocess
import tempfile

MIN_GO_VERSION = 1, 8, 0


def install_enry(args=None, target="./enry", tempdir=None):
    log = logging.getLogger("enry")
    if args is not None:
        tempdir = args.tempdir
        target = os.path.join(args.output, "enry")
    if shutil.which(os.path.basename(target)) is not None:
        log.warning("enry is in the PATH, no-op.")
        return 0
    if shutil.which(target, path=os.getcwd()):
        log.warning("%s exists, no-op.", target)
        return 0
    parent_dir = os.path.dirname(target)
    os.makedirs(parent_dir, exist_ok=True)
    if not os.path.isdir(parent_dir):
        log.error("%s is not a directory.", parent_dir)
        return 1
    try:
        version = subprocess.check_output(["go", "version"]).decode("utf-8")
    except (subprocess.SubprocessError, FileNotFoundError):
        log.error("Please install a Go compiler, refer to https://golang.org")
        return 2
    version = tuple(int(n) for n in version.split()[2][2:].split("."))
    if version < MIN_GO_VERSION:
        log.error("Your Go compiler version is %d.%d.%d, at least %d.%d.%d is "
                  "required." % (version + MIN_GO_VERSION))
        return 3
    with tempfile.TemporaryDirectory(prefix="enry-", dir=tempdir) as tmp:
        log.info("Building src-d/enry in %s...", tmp)
        env = os.environ.copy()
        env["GOPATH"] = tmp
        # FIXME(vmarkovtsev): change to gopkg.in when we fix https://github.com/src-d/enry/issues/37
        os.makedirs(os.path.join(tmp, "src", "gopkg.in", "src-d"))
        os.symlink(os.path.join(tmp, "src", "github.com", "src-d", "enry"),
                   os.path.join(tmp, "src", "gopkg.in", "src-d", "enry.v1"),
                   target_is_directory=True)
        subprocess.check_call(
            ["go", "get", "-v", "-ldflags=-s -w",
             "github.com/src-d/enry/cli/enry"], env=env)
        shutil.copyfile(os.path.join(tmp, "bin", "enry"), target)
        os.chmod(target, 0o777)
    log.info("Installed %s", target)
