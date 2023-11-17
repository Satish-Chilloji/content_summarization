from setuptools import setup, find_packages

with open('requirements.txt') as f:
    required_deps = f.read().splitlines()

setup(
    name='maestro',
    version='1.0.0',
    description='Toolset for conducting LLM applications',
    url='https://github.com/6si/magicshop/tree/main/projects',
    author='Skyler Dale',
    author_email='skyler.dale@6sense.com',
    maintainer_email='rohit.kewalramani@6sense.com',
    packages=find_packages(),
    install_requires=required_deps
)
