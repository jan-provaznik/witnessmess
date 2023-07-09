#!/usr/bin/env python3
#
# 2019 - 2023 Jan Provaznik (jan@provaznik.pro)

import setuptools

VERSION = '0.4.3'
DESCRIPTION = 'Witnessing multi-partite entanglement.'

with open('./README', encoding = 'utf-8') as file:
    README = file.read()

# Yes, yes, yes!

setuptools.setup(
    name = 'witnessmess',
    version = VERSION,
    description = DESCRIPTION,
    long_description = README,
    long_description_content_type = 'text/plain',
    author = 'Jan Provaznik',
    author_email = 'jan@provaznik.pro',
    url = 'https://provaznik.pro/witnessmess',
    license = 'LGPL',

    install_requires = [
        'picos >= 2.4',
        'numpy',
        'scipy'
    ],
    packages = [ 'witnessmess' ]
)

