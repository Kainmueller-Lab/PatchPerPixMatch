rgl.useNULL = TRUE

library(nat)
library(nat.nblast)

# get arguments
args <- commandArgs(trailing = TRUE)
print(args)
em_folder <- args[1]
result_folder <- args[2]

# read em neurons
print("read em")
print(em_folder)
ems = read.neurons(em_folder)

# flip / mirror:
print("mirror")
reg=t(translationMatrix(1427, 0, 0)) %*% scaleMatrix(-1,1,1) 
print(reg)
ems.flip=xform(ems, reg)

print("write")
write.neurons(ems.flip,result_folder,format='swc',files=names(ems.flip))

