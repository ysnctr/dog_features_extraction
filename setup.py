from setuptools import setup

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(name='dog_features_extraction',
      version='0.1',
      description='Features extraction for dog accelerometer data',
      url='https://github.com/mda14/dog_features_extraction',
      author='Rocio Alvarez',
      author_email='mda14@ic.ac.uk',
      license='Imperial College London',
      packages=['dog_features_extraction'],
      install_requires=required,
      zip_safe=False)
