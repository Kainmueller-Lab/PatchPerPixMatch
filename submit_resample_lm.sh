#!/bin/bash

export PATH=/groups/kainmueller/home/kainmuellerd/R-4.0.3/bin:$PATH

stepsize=3

billing=kainmueller

for lm_suffix in '_cropped'
do
  for em_suffix in '' '_flipped'
  do
    setup=setup22
    exp=${setup}_200511_00
    root=/nrs/saalfeld/kainmuellerd/ppp_all_cat_2_3
    source_foldername=skeletons_vote_instances_rm_by_bbox_20_min_length_20$lm_suffix$em_suffix
    source_dir=$root/$exp/test/400000/$source_foldername
    
    output_dir=${source_dir}_resmp_$stepsize/
    source_dir=${source_dir}/

    mkdir $output_dir

    logbasedir=${root}/${exp}/logs
    logdir=$logbasedir/resample_lm_${source_foldername}_resmp_$stepsize
    mkdir $logbasedir
    mkdir $logdir

    lines=($(ls -d ${source_dir}*))

    for l in "${lines[@]}";
    do
      echo $l
      line=${l##*/}
      echo $line
      
      mkdir $output_dir$line
      
      log_file=$logdir/resample_lm${source_foldername}_resmp_${stepsize}_${line}.out
      echo $source_dir$line $output_dir$line

      # run on cluster:
      bsub -n 1 -W 2:00 -P $billing -o $log_file Rscript resample_em.R $source_dir$line $output_dir$line $stepsize
      # exit
      # sleep 1

    done
  done
done

