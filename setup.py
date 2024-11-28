from setuptools import setup, find_packages

setup(
    name='cli-snapsort',
    version='0.1.4',
    packages=find_packages(),
    install_requires=[
        "click>=8.1.0",
        "numpy>=1.21.0",
        "torch>=1.9.0",
        "setuptools>=50.0.0",
        "ultralytics>=8.0.0",
        "transformers>=4.0.0",
        "imagehash>=4.2.1",
        "pillow>=8.3.0",
        "opencv-python>=4.5.0",
        "sentence-transformers>=3.3.0",
    ],
    entry_points={
        'console_scripts': [
            'snapsort=snap_sort.cli:snapsort',
        ],
    },
    package_data={
        'snap_sort': ['models/*'],
    },
    exclude_package_data={
        '': ['README.md', 'snapsort.egg-info/', 'assets/', 'snap_sort/__pycache__', 'dist/', '.github/', '.idea/', '.vscode/', 'snap_sort/utils/__pycache__'],
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
    python_requires='>=3.8',
    include_package_data=True,
)
