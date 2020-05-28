from setuptools import setup, find_packages

setup(
    name='MorphX',
    version='0.0.1',
    description='Toolkit for exploration and segmentation of 3D morphologies '
                'in form of point clouds or meshes.',
    url='https://github.com/StructuralNeurobiologyLab/MorphX',
    download_url='https://github.com/StructuralNeurobiologyLab/MorphX.git',
    author='Jonathan Klimesch, Philipp Schubert',
    author_email='jklimesch@neuro.mpg.de',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 1 - Planning',
        'License :: OSI Approved :: GPL-2.0 License',
        'Intended Audience :: Science/Research'
        'Topic :: Scientific/Engineering :: Morphology Analysis',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=[],
    tests_require=['pytest', 'pytest-cov', ],
)
