rgl.useNULL = TRUE
library(nat)
library(nat.nblast)

# get arguments
args <- commandArgs(trailing = TRUE)
swc_folder <- args[1]
result_folder <- args[2]

print(swc_folder)
print(result_folder)

# read neurons
print("reading neurons")
neurons = read.neurons(swc_folder)
neuronnames = names(neurons)

# spinify
print("spinifying neurons")
neuron_spines = as.neuronlist(lapply(neurons,spine,rval='neuron'))

# write
print("writing spines")
write.neurons(neuron_spines,result_folder,format='swc',files=neuronnames)

