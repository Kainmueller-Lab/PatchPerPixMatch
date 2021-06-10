# install.packages("igraph")
# library(igraph)
# install.packages("nat")
# install.packages("nat.nblast")
# if (!require("devtools")) install.packages("devtools")
# devtools::install_github(c("natverse/nat", "natverse/nat.nblast"))

start=Sys.time()

rgl.useNULL = TRUE
library(nat)
library(nat.nblast)
library(stringr)

# get arguments
args <- commandArgs(trailing = TRUE)
lm_folder <- args[1]
em_folder <- args[2]
base_result_dir <- args[3]
result_dir <-args[4]
result_fn <- args[5]
nblast_thresh <- as.numeric(args[6])
do_pre_filter_near <- as.logical(as.numeric(args[7]))

sprintf("nblast thresh: %f", nblast_thresh)
sprintf("do_pre_filter_near: %s", do_pre_filter_near)

# read ppp instance segmentations and create dotprops
print(lm_folder)
end=Sys.time()
print(paste("preliminaries: ", difftime(end, start, units = "secs"), sep=""))

start=Sys.time()
# read em neurons
# check if R object exists at path
rfilename = paste(substr(em_folder, 1, nchar(em_folder)-1), ".rda", sep="")
if (file.exists(rfilename)) {
    load(rfilename)
    print(ls())
    print("loaded ems from .rda")
} else if (dir.exists(em_folder)) {
    ems = read.neurons(em_folder)

    em_dps = nlapply(ems, dotprops, OmitFailures=TRUE)
    em_names = names(em_dps)
}
end=Sys.time()
print(paste("reading ems, and computing dotprops, for ", toString(length(em_dps)), " ems: ", difftime(end, start, units = "secs"), " ", sep=""))

start=Sys.time()
lms = read.neurons(lm_folder)
preds = nlapply(lms, dotprops, k=3, OmitFailures=TRUE)
pred_idcs = seq_along(preds)
numpoints = as.vector(unlist(lapply(pred_idcs, function(i, ...) nrow(preds[[pred_idcs[i]]]$points))))
preds = preds[which(numpoints>10)]
lm_samples = names(preds)
end=Sys.time()
print(paste("loading frags, and computing dotprops, for ", toString(length(preds)), " frags: ", difftime(end, start, units = "secs"), " ", sep=""))

print(paste("starting nblasting at ",Sys.time(),sep=""))
# process em_dps in batches of 1000:
by_em=1000
by_lm=500
for (start_em_idx in seq(1, length(em_dps), by=by_em)) {
    start=Sys.time()
    
    stop_em_idx = min(start_em_idx+by_em-1,length(em_dps))
    em_dps_batch = em_dps[start_em_idx:stop_em_idx]
    em_names_batch = em_names[start_em_idx:stop_em_idx]

    scorelist = list()
    for (start_lm_idx in seq(1, length(preds), by=by_lm)) {
        stop_lm_idx = min(start_lm_idx+by_lm-1, length(preds))
        print(paste("nblasting lm batch ",toString(start_lm_idx)," to ",toString(stop_lm_idx)," of ",toString(length(preds))))
        preds_batch = preds[start_lm_idx:stop_lm_idx]
        scorelist[[(start_lm_idx%/%by_lm)+1]] = nblast(preds_batch, em_dps_batch, version=2, normalised = FALSE, do_pre_filter_near=do_pre_filter_near)
    }
    scores = do.call(cbind,scorelist)
    # free workspace:
    end=Sys.time()
    print(paste("nblasting ems ",toString(start_em_idx)," to ",toString(stop_em_idx)," of ",toString(length(em_dps)),": ", difftime(end, start, units = "secs"), " ", sep=""))

    start=Sys.time()
    # write to json dict to be read in python:

    for (em_idx in seq_along(em_names_batch)) {
        # assemble filename and create dirs if necessary:
        em_name = em_names_batch[[em_idx]]
        em_name = str_extract(em_name, "[^.]+")
        em_group_number = str_pad(toString(as.numeric(str_extract(str_extract(em_name, "[^_]+"), "[^-]+")) %% 100), 2, side="left", pad="0")
        em_group_path = paste(base_result_dir, "/", em_group_number, sep="")
        dir.create(em_group_path, showWarnings = FALSE)
        em_path = paste(em_group_path, "/", em_name, sep="")
        dir.create(em_path, showWarnings = FALSE)
        em_suffix_path = paste(em_path, "/", result_dir, sep="")
        dir.create(em_suffix_path, showWarnings = FALSE)

        result_fn_per_em = paste(em_suffix_path, "/", result_fn, sep="")
        
        lines = c()
        line = "{"
        lines = c(lines, line)
        
        # assemble line:
        line = sprintf("\"%s\": {", (em_name))
        em_lines = c()
        em_lines = c(em_lines, line)
        found=FALSE
        for (lm_idx in seq_along(lm_samples)){
           score1 = scores[2*em_idx-1,lm_idx]
           score2 = scores[2*em_idx,lm_idx]
           # print(score)
           if (score2 > nblast_thresh | score1 > nblast_thresh){
             found=TRUE
             lm_name =  unlist(strsplit(lm_samples[[lm_idx]], "[.]"))[[1]]
             line = sprintf("    \"%s\": [%f, %f],", lm_name, score1, score2)
             em_lines = c(em_lines, line)
           }
        }
        if (found==FALSE) next

        # remove last comma:
        lastLine = em_lines[length(em_lines)]
        dummy = substr(lastLine,1,nchar(lastLine)-1)
        em_lines[length(em_lines)] = dummy
       
        line = "},"
        em_lines = c(em_lines, line)
        lines = c(lines, em_lines)
        
        # remove last comma:
        lastLine = lines[length(lines)]
        lastLine = substr(lastLine,1,nchar(lastLine)-1)
        lines[length(lines)] = lastLine

        line = "}"
        lines = c(lines, line)
        
        writeLines(lines, result_fn_per_em)
    }

    end=Sys.time()
    print(paste("writing: ", difftime(end, start, units = "secs"), " ", sep=""))
}

