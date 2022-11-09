from importlib.metadata import entry_points
from setuptools import setup, find_packages

setup(
    name='birb',
    version='0.0.1',
    description='birb',
    author='Marcel Gietzmann-Sanders',
    author_email='marcelsanders96@gmail.com',
    packages=find_packages(include=['birb', 'birb*']),
    install_requires=[
        'librosa==0.9.2',
        'click==8.1.3',
        'tqdm==4.64.1',
    ],
    entry_points={
        'console_scripts': [
            'build_spectrograms=birb.preprocess.build_spectrogram:main'
        ]
    }
)