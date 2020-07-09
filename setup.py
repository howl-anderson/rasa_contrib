from setuptools import setup, find_packages

setup(
    name='rasa_contrib',
    version='0.3.1',
    packages=find_packages(),
    url='https://github.com/howl-anderson/rasa_contrib',
    license='Apache 2.0',
    author='Xiaoquan Kong',
    author_email='u1mail2me@gmail.com',
    description='Addons for Rasa',
    install_requires=["rasa", "paddlepaddle", "seq2label", "tensorflow", "seq2annotation", "MicroTokenizer"]
)
