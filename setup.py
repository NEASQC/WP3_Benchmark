import os, sys
from setuptools import setup, find_packages
from setuptools.command.test import test as TestCommand

#sys.path.append("tnbs/BTC_01_PL")
#sys.path.append("tnbs/BTC_02_AE")
#sys.path.append("tnbs/BTC_03_QPE")

class PyTest(TestCommand):
    """
    A test command to run pytest on a the full repository.
    This means that any function name test_XXXX
    or any class named TestXXXX will be found and run.
    """
    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = []

    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = True

    def run_tests(self):
        import pytest
        errno = pytest.main([".", "-vv"])
        sys.exit(errno)


setup(
    name="TNBS",
    version="0.0.1",
    author="Gonzalo Ferro",
    license="European Union Public License 1.2",
    packages=find_packages(),
    install_requires=["numpy", "pandas", "myqlm", "matplotlib", "scipy"],
    # Don't change these two lines
    tests_require=["pytest"],
    cmdclass={'test': PyTest},
)
