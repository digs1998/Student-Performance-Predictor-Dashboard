## create machine learning applications as a package, even deploy in pypy
from setuptools import find_packages, setup
from typing import List

hyphen_e_dot = '-e .'
def get_requirements(filepath:str)->List[str]:
    '''
    Returns list of required libraries
    '''
    requirements = []
    with open(filepath) as file_obj:
        requirements = file_obj.readlines()
        for req in requirements:
            req = req.replace("\n", "")
        
        if hyphen_e_dot in requirements:
            requirements.remove(hyphen_e_dot)
            
    return requirements

setup(
    name = "starter-ml-project",
    version='0.0.1',
    author='Digvijay Yadav',
    author_email='digvijayyadav48@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)