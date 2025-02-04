from setuptools import find_packages, setup
from typing import List
hypen="-e ."
def get_requirements(path:str)->List[str]:
    req=[]
    with open(path) as f:
        req=f.read().splitlines()
        req=[r.replace('/n','') for r in req]
        if hypen in req:
            req.remove(hypen)
    return req
setup(
    name='mlproject',
    version='0.0.1',
    author='Lakshya',
    author_email='lakshyabhatnagar1@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt'),
)