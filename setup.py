from setuptools import setup, find_packages

setup(
    name='CarPricePredictor',
    version='0.1',
    packages=find_packages(),
    description='Package to predict car price.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Hasmik Mnatsakanyan',
    author_email='hasmik.mnatsakanyan2000@gmail.com',
    url='https://github.com/HasmikMnatsakanyann/test_1/tree/main',
    package_data={"CarPricePredictor": ["weights.json"]},
    include_package_data=True,
    install_requires=[
        'pandas==2.1.4',
        'scikit-learn==1.3.2',
        'xgboost==2.0.2',
        'numpy==1.26.2',
        'torch==2.1.2'
    ]
)