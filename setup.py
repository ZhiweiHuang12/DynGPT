from setuptools import setup, find_packages
setup(
    name='dyngpt',  # Package name
    version='0.1.0',  # Version number
    author='Zhiwei Huang',  # Author name
    author_email='huangzhw59@mail2.sysu.edu.cn',  # Author email
    description='...',  # Short description
    long_description=open('README.md').read(),  # Detailed description
    long_description_content_type='text/markdown',  # Description format
    url='https://github.com/yourusername/my_package',  # Project URL
    packages=find_packages(),   # Automatically discover packages
    package_data={'dyngpt': ['data/*.json','weights/*.pt']},  #Include all JSON files in the data directory and .pt files in weights
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',  # License type
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',  # Required Python version
)
