#!/usr/bin/env python

from distutils.core import setup
import glob

setup(name='dialog_graph_processer',
      version='0.1.0',
      description='Dialog Graph Processing',
      author='@kudep',
      packages=['dialog_graph_processer'],
      entry_points={
    'console_scripts': [
        f'{cli_script[:-3].split("/")[-1].replace("_", "-")}={cli_script[:-3].replace("/", ".")}:cli'
        for cli_script in glob.glob('dialog_graph_processer/cli/*.py')
    ]
}
     )

