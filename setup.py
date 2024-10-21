from setuptools import setup, find_packages

setup(
    name='snapsort',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'opencv-python',
        'click',
        'ultralytics',
        'torch',
        'transformers',
        'numpy',
        'sentence-transformers'
    ],
    entry_points={
        'console_scripts': [
            'snapsort=snap_sort.cli:snapsort',
        ],
    },
    package_data={
        'snap_sort': ['snap_sort/models/yolov8s.pt'],  
    },
    exclude_package_data={
        '': ['README.md', 'snapsort.egg-info/'],
    },
    author="Jiaming Liu",
    description="A CLI tool to classify photos",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url="https://github.com/Jiaaming/snapsort",
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
