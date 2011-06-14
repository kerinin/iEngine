Immediate NOTES:
=========

Update prediction per SVR
Shift the decomposition into the GPU using scan()
BOC needs to incoporate model risk
Mask SV's after calculation


Once that's done
=========

Split into subsets for nu
Use the nodes & pipes library so you can use both cores
When splitting the kernel matrix, save output to file to free RAM - see if libsvm can read directly from the file



