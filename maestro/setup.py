from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_deps = f.read().splitlines()

setup(
    name='maestro',
    version='1.0.0',
    description='Toolset for conducting LLM applications',
    url='https://github.com/Satish-Chilloji/content_summarization',
    author='Satish Chilloji',
    author_email='m22aie242@iitj.ac.in',
    packages=find_packages(),
    install_requires=required_deps
)
