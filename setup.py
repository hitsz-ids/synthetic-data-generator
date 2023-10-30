from setuptools import setup
    
setup(
    name='sdgx',
    version='alpha-0.1',
    packages=['sdgx'],
    url='https://github.com/hitsz-ids/synthetic-data-generator',
    entry_points= "sdgx/",
    license='Apache2',
    author='hitsz-ids',
    author_email='sjaqyjy@hit.edu.cn',
    description='A framework focused on quickly generating structured tabular synthetic data',
    install_requires=[
    "setproctitle",
    "PyMySQL",
    "pandas",
    "numpy",
    "scikit-learn",
    "torch",
    "torchvision",
    "rdt",
    "joblib",
    "dython",
    "seaborn",
    "table-evaluator",
    "copulas",
    ]
)