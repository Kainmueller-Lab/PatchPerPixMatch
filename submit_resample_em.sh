#!/bin/bash

export PATH=$HOME/R-4.0.3/bin:$PATH

if [ -z "$NRS" ]
then
    NRS="/nrs/saalfeld/kainmuellerd"
fi

if [ -z "$DATASET" ]
then
    em_set=Tanya_2021_02 # "ground_truth_set" # "Gerry_2021_01_02" # ground_truth_set
else
    em_set=$DATASET
fi


stepsize=1

for em_id in "" "_flipped"
do
    root=$NRS/data/hemibrain
    em_foldername=40x_iter_3_len_30${em_id}
    em_dir=$NRS/data/hemibrain/${em_set}/$em_foldername
    em_resampled_dir=${em_dir}_resmp_$stepsize/
    em_dir=${em_dir}/ # _not_resampled/

    echo $em_dir $em_resampled_dir

    logbasedir=${root}/logs
    logdir=$logbasedir/resample_em
    mkdir -p $logbasedir
    mkdir -p $logdir
    mkdir -p $em_resampled_dir

    log_file=$logdir/resample_em.out
    echo $log_file

    Rscript resample_em.R $em_dir $em_resampled_dir $stepsize > $log_file
done
