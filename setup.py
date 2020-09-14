'''
Created on 12 Aug 2020

@author: Tobias Pielok
'''

from setuptools import setup, find_packages

setup(
    name='PKRL',
    version='0.1',
    packages=find_packages(exclude=['tests*','simulations*','showcases*']),
    license='MIT',
    description='Probabiistic Koopman based Representation Learning',
    long_description=open('README.md').read(),
    install_requires=['numpy', 'matplotlib', 'sklearn', 'scipy',
                      'tensorflow==2.1.0',
                      'tensorflow_probability'],
    url='https://github.com/pkmtum/Probabilistic_Koopman_Learning',
    author='Tobias Pielok',
    author_email='t.pielok@tum.de'
)