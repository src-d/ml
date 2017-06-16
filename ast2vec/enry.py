import os
import shutil
import subprocess
import sys
import tempfile

MIN_GO_VERSION = 1, 8, 0


def install_enry(args):
    target = os.path.join(args.output, "enry")
    if os.path.exists(target):
        print("%s exists, no-op." % target)
        return 0
    os.makedirs(args.output, exist_ok=True)
    if not os.path.isdir(args.output):
        print("%s is not a directory." % args.output, file=sys.stderr)
        return 1
    try:
        version = subprocess.check_output(["go", "version"]).decode("utf-8")
    except subprocess.SubprocessError:
        print("Please install Go compiler, refer to https://golang.org",
              file=sys.stderr)
        return 2
    version = tuple(int(n) for n in version.split()[2][2:].split("."))
    if version < MIN_GO_VERSION:
        print("Your Go compiler version is %d.%d.%d, at least %d.%d.%d is "
              "required." % (version + MIN_GO_VERSION), file=sys.stderr)
        return 3
    with tempfile.TemporaryDirectory(prefix="enry-", dir=args.tempdir) as tmp:
        print("Building src-d/enry in %s..." % tmp)
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
    print("Installed %s" % target)
