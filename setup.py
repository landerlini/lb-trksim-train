"""Setup file for lb-trksim-train
Original template at:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    name='LbTrksimTrain',  # Required
    version='0.1.0',  # Required
    description='Training pipeline for the parametrization of the tracking',  
    long_description=long_description,  
    long_description_content_type='text/markdown',  # Optional (see note above)
    url='https://github.com/landerlini/lb-trksim-train',  # Optional
    author='Lucio Anderlini',  # Optional
    author_email='Lucio.Anderlini@fi.infn.it',  # Optional
    classifiers=[  # Optional
      # Project maturity
      #   3 - Alpha
      #   4 - Beta
      #   5 - Production/Stable
      'Development Status :: 3 - Alpha',

      'Intended Audience :: Developers',
      'Intended Audience :: Science/Research', 
      'Topic :: Software Development :: Code Generators', 

      'License :: OSI Approved :: MIT License',

      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.6',
      'Programming Language :: Python :: 3.7',
      'Programming Language :: Python :: 3.8',
      #'Programming Language :: Python :: 3.9',
      'Programming Language :: Python :: 3 :: Only',
      ],

    keywords='machine-learning, simulation, lhcb, hep, high-energy-physics',  # Optional

    #package_dir={'': 'scikinC'},  # Optional

    packages=find_packages(),  # Required

    python_requires='>=3.6, <4',

    install_requires=[
      'numpy', 
      'scipy', 
      'scikit-learn', 
      'tensorflow',
      'keras',
      'scikinC[keras]>=0.1',
      'html-reports>=0.2',
    ],  # Optional


    entry_points={  # Optional
      'console_scripts': [
        'LbTrksimTrain=LbTrksimTrain.__main__:main',
        ],
      },
    )

