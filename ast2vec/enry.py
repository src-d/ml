import logging
import io
import os
import platform
import shutil
import subprocess
import tarfile
import tempfile

import requests


MIN_GO_VERSION = 1, 8, 0


def download_enry(target):
    log = logging.getLogger("enry")
    if platform.machine() != "x86_64":
        return False
    system = platform.system().lower()
    release = system + "_" + "amd64.tar.gz"
    try:
        latest_url = "https://api.github.com/repos/src-d/enry/releases/latest"
        log.info("Fetching %s", latest_url)
        assets = requests.get(latest_url).json()["assets"]
        for asset in assets:
            name = asset["name"]
            if name.endswith(release):
                log.info("Latest release resolved to %s", name)
                break
        else:
            return False
        log.info("Fetching %s", asset["browser_download_url"])
        response = requests.get(asset["browser_download_url"])
        tarbin = io.BytesIO(response.content)
        log.info("Extracting the binary")
        with tarfile.open(fileobj=tarbin) as tar:
            for member in tar.getmembers():
                if member.size > 0 and member.mode & 0o755 == 0o755:
                    with open(target, "wb") as fout:
                        shutil.copyfileobj(tar.extractfile(member), fout)
                    os.chmod(target, 0o755)
        log.info("Downloaded %s", os.path.abspath(target))
        return True
    except Exception as e:
        log.warning("Failed to download enry: %s: %s", type(e).__name__, e)
        return False


def install_enry(args=None, target="./enry", tempdir=None, warn_exists=True,
                 force_build=False):
    """
    Deploys src-d/enry at the specified path.

    :param args: :class:`argparse.Namespace` with "output" and "tmpdir". \
                 "output" sets the target directory, "tmpdir" sets \
                 the temporary directory which is used to clone src-d/enry \
                 and build it.
    :param target: The path to the built executable. If args is not None, it \
                   becomes overridden.
    :param tempdir: The temporary directory where to clone and build \
                    src-d/enry. If args is not None, it becomes overridden.
    :param force_build: Unconditionally build enry instead of downloading.
    :return: None if successful; otherwise, the error code (can be 0!).
    """
    log = logging.getLogger("enry")
    if args is not None:
        tempdir = args.tmpdir
        target = os.path.join(args.output, "enry")
    if shutil.which(os.path.basename(target)) or shutil.which(target, path=os.getcwd()):
        if warn_exists:
            log.warning("enry is in the PATH, no-op.")
        return 0
    parent_dir = os.path.dirname(target)
    os.makedirs(parent_dir, exist_ok=True)
    if not os.path.isdir(parent_dir):
        log.error("%s is not a directory.", parent_dir)
        return 1
    if not force_build and download_enry(target):
        return
    try:
        version = subprocess.check_output(["go", "version"]).decode("utf-8")
    except (subprocess.SubprocessError, FileNotFoundError):
        log.error("Please install a Go compiler, refer to https://golang.org")
        return 2
    version = tuple(int(n) for n in version.split()[2][2:].split("."))
    if len(version) == 2:
        version += 0,
    if version < MIN_GO_VERSION:
        log.error("Your Go compiler version is %d.%d.%d, at least %d.%d.%d is "
                  "required." % (version + MIN_GO_VERSION))
        return 3
    with tempfile.TemporaryDirectory(prefix="enry-", dir=tempdir) as tmp:
        log.info("Building src-d/enry in %s...", tmp)
        env = os.environ.copy()
        env["GOPATH"] = tmp
        subprocess.check_call(
            ["go", "get", "-v", "-ldflags=-s -w",
             "gopkg.in/src-d/enry.v1/..."], env=env)
        shutil.copyfile(os.path.join(tmp, "bin", "enry"), target)
        os.chmod(target, 0o777)
    log.info("Installed %s", os.path.abspath(target))
