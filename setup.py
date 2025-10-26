
from setuptools import find_packages,setup
from typing import List



HYPEN_E_DOT = '-e .'


def get_requirements(file_path: str) -> List[str]:
    requirements = []
    with open(file_path) as file:
        for line in file:
            req = line.strip()
            if req and req != HYPEN_E_DOT:
                requirements.append(req)
    return requirements

setup(
    name="SarcasmDetection",
    version='0.0.1',
    author="Aniket Sinha",
    author_email="sinhaaniket010@gmail.com",
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)