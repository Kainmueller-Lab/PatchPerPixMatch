library(igraph)

rgl.useNULL = TRUE
library(nat)
library(nat.nblast)

# get arguments
args <- commandArgs(trailing = TRUE)
lm_folder <- args[1]
em_folder <- args[2]
result_folder <- args[3]

if (em_folder=="hemibrain") {
    bbox = matrix( 
       c(302.6390, 1.6348, 18.2223, 870.4659, 529.5118, 442.0632), 
       nrow=2, 
       ncol=3, 
       byrow = TRUE)
} else if (em_folder=="hemibrain_flipped") {
    bbox = matrix( 
       c(556.5341, 1.6348, 18.2223, 1124.3610, 529.5118, 442.0632), 
       nrow=2, 
       ncol=3, 
       byrow = TRUE)
} else {
    # read em neurons
    print("read em")
    print(em_folder)
    ems = read.neurons(em_folder)
    bbox = boundingbox(ems)
}

print("em bbox")
print(bbox)

# read ppp instance segmentations and create dotprops
print("read lm")
print(lm_folder)
preds = read.neurons(lm_folder)

try_subset <- function(neuron, expression) {
   return(tryCatch(subset(neuron, expression), error=function(e) NULL))
}

print("restricting to em bbox")
preds = nlapply(preds, subset, X>bbox[1,1], OmitFailures=TRUE)
preds = nlapply(preds, subset, X<bbox[2,1], OmitFailures=TRUE)
preds = nlapply(preds, subset, Y>bbox[1,2], OmitFailures=TRUE)
preds = nlapply(preds, subset, Y<bbox[2,2], OmitFailures=TRUE)
preds = nlapply(preds, subset, Z>bbox[1,3], OmitFailures=TRUE)
preds = nlapply(preds, subset, Z<bbox[2,3], OmitFailures=TRUE)

print("new lm bbox")
bbox = boundingbox(preds)
print(bbox)

prednames = names(preds)
# write
print("write")
write.neurons(preds,result_folder,format='swc',files=prednames)

warnings()

