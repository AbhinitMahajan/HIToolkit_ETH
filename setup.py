from setuptools import setup, find_packages

setup(
    name='HI_package',
    version='0.0.1',
    description='A toolkit which offers the user to implement different preprocessing techniques on hyperspectral images of flowing waste water. It also helps the user to implement and fine tune various ML models.',
    author='Abhinit Mahajan',
    author_email='abhinit81@gmail.com',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'scipy',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'xgboost',
        'seaborn',
        'sqlite3'
    ],
    python_requires='>=3.6, <4',
    keywords='Hyperspectral Images, Modelling waste water, Machine Learning',
    url='https://github.com/AbhinitMahajan/HIToolkit_ETH',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition'
    ]
)
