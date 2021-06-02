rgl.useNULL = TRUE

library(nat)

# get arguments
args <- commandArgs(trailing = TRUE)
print(args)
em_folder <- args[1]
result_folder <- args[2]
stepsize <- as.numeric(args[3])

# read em neurons
print("read em")
print(em_folder)
ems = read.neurons(em_folder)

# resample:
print("resample")
ems_rsmp = lapply(ems,nat::resample,stepsize=stepsize)

print("write")
write.neurons(ems_rsmp,result_folder,format='swc',files=names(ems_rsmp))

# warnings()

