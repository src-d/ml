from bblfsh.client import BblfshClient
from distutils.version import StrictVersion

BBLFSH_VERSION_LOW = "2.2"
BBLFSH_VERSION_HIGH = "3.0"


def check_version(host: str="0.0.0.0", port: str="9432") -> bool:
    """
    Check if the bblfsh server version matches module requirements.

    :param host: bblfsh server host
    :param port: bblfsh server port
    :return: True if bblfsh version specified matches requirements
    """
    # get version and remove leading 'v'
    version = StrictVersion(BblfshClient("%s:%s" % (host, port)).version().version.lstrip("v"))
    return StrictVersion(BBLFSH_VERSION_LOW) <= version < StrictVersion(BBLFSH_VERSION_HIGH)
