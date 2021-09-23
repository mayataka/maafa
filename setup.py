from setuptools import find_packages, setup

setup(
    name='maafa',
    version='0.0.1',
    description="MPC as a function approximator.",
    author='Sotaro Katayama',
    author_email='katayama.25w@st.kyoto-u.ac.jp',
    platforms=['any'],
    license="MIT",
    url='https://github.com/mayataka/maafa',
    packages=find_packages(),
    install_requires=['numpy', 'torch']
)