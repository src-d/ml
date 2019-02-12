import errno
import random
import socket
import time
import unittest

import docker.client

from sourced.ml.utils.bblfsh import BBLFSH_VERSION_HIGH, BBLFSH_VERSION_LOW, check_version


class BblfshUtilsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.docker_client = docker.from_env()
        # ensure docker is running
        try:
            cls.docker_client.containers.list()
        except Exception:
            raise Exception("docker not running properly")
        cls.er_msg = "supported bblfshd versions: " \
                     ">=%s,<%s" % (BBLFSH_VERSION_LOW, BBLFSH_VERSION_HIGH)

    def __check_bblfsh_version_support(self, version: str) -> bool:
        """
        :param version: version of bblfshd to check
        :return: True if version is supported, False otherwise
        """
        with socket.socket() as s:
            for _ in range(3):
                try:
                    port = random.randint(10000, 50000)
                    s.connect(("localhost", port))
                except socket.error as e:
                    if e.errno == errno.ECONNREFUSED:
                        break

        container = self.docker_client.containers.run(
            image="bblfsh/bblfshd:%s" % version,
            privileged=True,
            detach=True,
            ports={"9432": port},
        )

        assert container is not None, "failed to create bblfsh container"

        for _ in range(10):
            try:
                res = check_version(port=port)
                break
            except Exception:
                time.sleep(.1)
                pass

        container.stop()
        container.remove()
        return res

    def test_v200(self):
        self.assertFalse(self.__check_bblfsh_version_support("v2.0.0"), self.er_msg)

    def test_v210(self):
        self.assertFalse(self.__check_bblfsh_version_support("v2.1.0"), self.er_msg)

    def test_v220(self):
        self.assertTrue(self.__check_bblfsh_version_support("v2.2.0"), self.er_msg)

    def test_v230(self):
        self.assertTrue(self.__check_bblfsh_version_support("v2.3.0"), self.er_msg)

    def test_v240(self):
        self.assertTrue(self.__check_bblfsh_version_support("v2.4.0"), self.er_msg)

    def test_v250(self):
        self.assertTrue(self.__check_bblfsh_version_support("v2.5.0"), self.er_msg)

    @classmethod
    def tearDownClass(cls):
        cls.docker_client.close()


if __name__ == "__main__":
    unittest.main()
