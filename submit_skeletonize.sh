#!/bin/bash

setup=setup22
exp=${setup}_200511_00
root=/nrs/saalfeld/kainmuellerd/ppp_all_cat_2_3
data_dir=$root/$exp/test/400000/instanced
min_cable_length=20
in_key=vote_instances_rm_by_bbox_$min_cable_length
output_dir=$root/$exp/test/400000/skeletons_${in_key}_min_length_$min_cable_length 

echo $exp $data_dir
mkdir ${root}/${exp}/logs
logdir=${root}/${exp}/logs/skeletonize
mkdir $logdir
mkdir $output_dir

samples=($(echo $data_dir/*.hdf))

all_jobs=$(bjobs -l | tr -d '\n' | tr -d ' ' | tr '/' '\n' )

for s in "${samples[@]}";
do
  echo $s
  sname=$(basename $s .hdf)
  log_file=$logdir/skeletonize_${in_key}_${min_cable_length}_${sname}.out
  
  line=$(echo $sname | cut -d- -f1)
  skel_file=$output_dir/$line/$sname
  echo $skel_file

  l=$(ls $skel_file* | wc -l)
  if [ $((l)) -gt 1 ]
  then
    echo Skipping ${data_dir}/${s}, skels already exist...
    continue
  fi

  jobs_running=$(echo $all_jobs | grep $sname | wc -l)
  
  if [ $jobs_running -gt 0 ]
  then
    echo "job for $s running or pending, skipping" 
    continue
  fi

  echo $log_file

  # run on cluster:
  # don't use multiple workers on the cluster; inefficient in case of one catastrophic merger / huge skeleton
  bsub -n 1 -W 8:00 -P flylight -o $log_file python skeletonize.py --in-folder $data_dir --sample $sname --in-key $in_key --min-cable-length $min_cable_length --out-dir $output_dir --num-worker 1

done
