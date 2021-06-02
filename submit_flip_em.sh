#!/bin/bash

export PATH=$HOME/R-4.0.3/bin:$PATH

if [ -z "$NRS" ]
then
    NRS="/nrs/saalfeld/kainmuellerd"
fi

if [ -z "$DATASET" ]
then
    em_set=all_NB # Tanya_2021_02 # Gerry_2021_01_03 # Gerry_2020_12 #  # Gerry_2020_12 # Barry_2021_01 #  #  all_cat_2_3
else
    em_set=$DATASET
fi

root=$NRS/data/hemibrain
em_foldername=40x_iter_3_len_30
em_dir=$NRS/data/hemibrain/${em_set}/$em_foldername
em_flipped_dir=${em_dir}_flipped/
em_dir=${em_dir}/

echo $em_dir $em_flipped_dir

logbasedir=${root}/logs
logdir=$logbasedir/flip_em
mkdir -p $logbasedir
mkdir -p $logdir
mkdir -p $em_flipped_dir

log_file=$logdir/flip_em.out
echo $log_file

Rscript flip_em.R $em_dir $em_flipped_dir > $log_file
