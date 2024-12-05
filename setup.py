from setuptools import setup

with open('VERSION') as f:
    version = f.read().strip()

setup(
    name='driftbench',
    version=version,
    packages=['driftbench'],
    url='',
    author='Edgar Wolf',
    author_email='edgar.wolf@hs-kempten.de',
    description='A package to benchmark process drift detection'
)
