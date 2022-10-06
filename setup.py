#!/usr/bin/env python3

import os
from setuptools import setup, find_packages

directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(directory, 'README.md'), encoding='utf-8') as f:
  long_description = f.read()

setup(name='easy_inference',
      version='0.0.3',
      description='Got a working detection model file? Want to quickly setup inference pipelines? You are in the right place!',
      author='Chadi Salmi',
      license='MIT',
      long_description=long_description,
      long_description_content_type='text/markdown',
      packages = find_packages(),
      classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License"
      ],
      install_requires=['numpy', 'pyrealsense2', 'opencv-python', 'scipy'],
      python_requires='>=3.8',
      extras_require={
        'ros': ['rospkg', 'pyyaml'],
      },
      include_package_data=True)
