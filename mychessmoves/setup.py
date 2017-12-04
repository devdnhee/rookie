from distutils.core import setup, Extension

module1 = Extension(
        'mychessmoves',
        sources = [
                'Source/mychessmovesmodule.c',
                'Source/format.c',
                'Source/moves.c',
                'Source/polyglot.c',
                'Source/stringCopy.c' ],
        extra_compile_args = ['-O3', '-std=c99', '-Wall', '-pedantic'],
        undef_macros = ['NDEBUG']
)

setup(
        name         = 'mychessmoves',
        version      = '1.0b',
        description  = 'Package to generate chess positions and moves (FEN/SAN/UCI)',
        author       = 'Marcel van Kervinck',
        author_email = 'marcelk@bitpit.net',
        url          = 'http://marcelk.net/chessmoves',
        ext_modules  = [module1])
