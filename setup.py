from importlib.metadata import entry_points
from setuptools import find_packages, setup


with open('requirements.txt', 'rt') as f:
    required_packages = f.read().split()

setup(
    name='wemoney_recommender',
    version='0.0.1',
    description='',
    url='',
    author='Anton Grishin',
    author_email='grishin3@outlook.com',
    license='',
    package_data={'wemoney_recommender': ['data/*.csv']},
    include_package_data=True,
    install_requires=required_packages,
    python_requires='>=3.8',
    extras_require={
    'dev': [],
    'docs': [],
    'testing': []
    },
    entry_points={'console_scripts': [
        'sort_posts = wemoney_recommender.main:main'
    ]
    }



)