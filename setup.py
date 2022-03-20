#!/usr/bin/env python3
#
# 2021 - 2022 Jan Provaznik (jan@provaznik.pro)
#
# Let's see how poorly this goes.

import setuptools

VERSION = '0.1'
DESCRIPTION = 'Witnessing multi-partite entanglement.'

# Yes, yes, yes!

setuptools.setup(
    name = 'witnessme',
    version = VERSION,
    description = DESCRIPTION,
    author = 'Jan Provaznik',
    author_email = 'jan@provaznik.pro',
    url = 'https://provaznik.pro/witnessme',
    license = 'LGPL',
    packages = [ 'witnessme' ]
)

