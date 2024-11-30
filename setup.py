
import pybind11
from setuptools import setup, Extension

# Get the pybind11 include path
cpp_include_path = pybind11.get_include()

# Create Extension object
kmeans_cpp_module = Extension(
    'kmeans',
    sources=['main.cpp'],
    include_dirs=[cpp_include_path],
    language='c++',
    extra_compile_args=['-fopenmp'], 
    extra_link_args=['-fopenmp']  
)

# Call setup function
setup(
    name='kmeans',
    version='1.0',
    ext_modules=[kmeans_cpp_module],
)