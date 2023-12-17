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
    url='https://github.com/yourusername/my_package',
    package_data={"CarPricePredictor": ["weights/*"]},
    include_package_data=True,
)
