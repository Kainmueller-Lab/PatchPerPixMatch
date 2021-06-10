#!/bin/bash

export PATH=$HOME/R-4.0.3/bin:$PATH

if [ -z "$NRS" ]
then
    NRS="/nrs/saalfeld/kainmuellerd"
fi

if [ -z "$DATASET" ]
then
    em_sets=(all_hemibrain_1.2_NB)
else
    em_sets=($DATASET)
fi

for em_set in ${em_sets[@]}
do
mkdir -p $NRS/flymatch/${em_set}
for lm_suffix in '_cropped'
do
  for em_suffix in '' '_flipped'
  do
    setup=setup22
    exp=${setup}_200511_00
    root=/nrs/saalfeld/kainmuellerd/ppp_all_cat_2_3
    lm_resmp_suffix=""
    em_resmp_suffix=""
    min_length=20
    nblast_suffix="_"${min_length}
    source_foldername=skeletons_vote_instances_rm_by_bbox_${min_length}_min_length_${min_length}$lm_suffix$em_suffix$lm_resmp_suffix
    source_dir=$root/$exp/test/400000/$source_foldername/
    target_foldername=40x_iter_3_len_30$em_suffix${em_resmp_suffix}
    target_dir=$NRS/data/hemibrain/${em_set}/${target_foldername}/
    base_base_output_dir=$NRS/flymatch/${em_set}/${setup}_nblast${nblast_suffix}
    base_output_dir=$base_base_output_dir/results

    do_pre_filter_near=1
    output_dir=nblastScores${em_suffix}
    nblast_thresh=-0.5

    mkdir -p $base_output_dir
    logdir=$base_base_output_dir/logs
    mkdir -p $logdir

    lines=($(ls -d ${source_dir}*))

    for l in "${lines[@]}";
    do
      echo $l
      line=${l##*/}
      echo $line
      log_file=$logdir/nblast_lm${source_foldername}_to_em${target_foldername}_thresh${nblast_thresh}_${line}_${em_set}.out
      output_filename=nblastScores_thresh${nblast_thresh}_${line}.json

      echo "running, log $log_file"

      # run on cluster:
      bsub -n 1 -W 96:00 -o $log_file Rscript --no-restore --no-save nblast_lm_to_em_both.R $source_dir$line $target_dir $base_output_dir $output_dir $output_filename $nblast_thresh $do_pre_filter_near

    done
  done
done
done

