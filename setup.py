import setuptools

setuptools.setup(
    name="timediffeq",
    version="0.0.1",
    author="Ian Carter",
    author_email="carterian8@gmail.com",
    description="Multivariate time series analysis with neural ODE.",
    url="https://github.com/carterian8/timediffeq",
    packages=['timediffeq', 'timediffeq._impl'],
    install_requires=['tensorflow>=1.13.1'], 
    classifiers=(
        "Programming Language :: Python :: 3"),)
