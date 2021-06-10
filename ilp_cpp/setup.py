from setuptools import setup, Extension

setup(
    #...
    ext_modules=[Extension('ilp_matching', [
        'ilp_matching.cpp',
        'GurobiBackend.cpp',
        'LinearConstraint.cpp',
        'LinearConstraints.cpp',
        'QuadraticObjective.cpp',
        'Solution.cpp',
    ],
                           include_dirs=['/misc/local/gurobi-9.0.3/include', "."],
                           library_dirs=['/misc/local/gurobi-9.0.3/lib'],
                           # extra_compile_args=['-std=c++20', '-O3', '-funroll-loops'],
                           extra_compile_args=['-std=c++20', '-O3'],
                           # extra_compile_args=['-std=c++20', '-O0', '-g'],
                           libraries=['gurobi90', 'gurobi_c++'],
                           ),],
)
