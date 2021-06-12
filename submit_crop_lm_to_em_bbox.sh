#!/bin/bash

export PATH=$HOME/R-4.0.3/bin:$PATH

for lm_suffix in '' 
do
    for em_suffix in '' '_flipped'
    do
        setup=setup22
        exp=${setup}_200511_00
        root=/nrs/saalfeld/maisl/ppp_test
        min_frag_length=20
        lm_id=skeletons_vote_instances_rm_by_bbox_${min_frag_length}_min_length_${min_frag_length}
        lm_dir=$root/$exp/test/400000/$lm_id$lm_suffix/
        em_dir=hemibrain$em_suffix
        output_dir=$root/$exp/test/400000/$lm_id${lm_suffix}_cropped${em_suffix}/

        echo $exp $data_dir
        logbasedir=${root}/${exp}/logs
        logdir=$logbasedir/crop_lm_to_em_bbox
        mkdir $logbasedir
        mkdir $logdir
        mkdir $output_dir

        lines=($(ls -d $lm_dir*))

        for l in "${lines[@]}";
        do
          echo $l
          line=${l##*/}
          echo $line
          log_file=$logdir/crop_lm_$lm_id${lm_suffix}_to_em${em_suffix}_bbox_${line}.out
          
          if [ -f $log_file ]
          then
            echo Skipping ${lm_dir}${line}, already ran...
            continue
          fi

          echo $log_file
          mkdir $output_dir$line

          # run on cluster:
          bsub -n 1 -W 2:00 -o $log_file Rscript --no-restore --no-save crop_lm_to_em_bbox.R $lm_dir$line $em_dir $output_dir$line
      done
    done
done
