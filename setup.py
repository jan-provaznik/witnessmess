#!/usr/bin/env python3
#
# 2019 - 2023 Jan Provaznik (jan@provaznik.pro)
#
# Let's see how poorly this goes.

import setuptools

VERSION = '0.4.0'
DESCRIPTION = 'Witnessing multi-partite entanglement.'

# Yes, yes, yes!

setuptools.setup(
    name = 'witnessmess',
    version = VERSION,
    description = DESCRIPTION,
    long_description = DESCRIPTION,
    long_description_content_type = 'text/plain',
    author = 'Jan Provaznik',
    author_email = 'jan@provaznik.pro',
    url = 'https://provaznik.pro/witnessmess',
    license = 'LGPL',

    install_requires = [
        'picos >= 2.4',
        'numpy >= 1.22',
        'scipy >= 1.8'
    ],
    packages = [ 'witnessmess' ]
)

