from setuptools import setup
from pathlib import Path

with open('VERSION') as f:
    version = f.read().strip()

setup(
    name='driftbench',
    version=version,
    packages=['driftbench'],
    url='',
    author='Edgar Wolf',
    author_email='edgar.wolf@hs-kempten.de',
    description='A package to benchmark process drift detection',
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type='text/markdown',
)
