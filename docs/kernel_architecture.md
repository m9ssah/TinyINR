# Kernel Architecture Guide

not really an architecture, more so an explaination of how the kernels are meant to be modularized

```cuda_utils.cuh```: utility file for kernels, contains macros for error checking and testing

```*.cuh```: create one for each kernel, just like cpp files are laid out (ie cuda header file)

```*cu```: actual kernel, keep clean and concise, no testing

```test_*.cu```: one for each kernel, test kernel outputs against their cpu counterparts. basically the "main" block that is usually in kernel files 

