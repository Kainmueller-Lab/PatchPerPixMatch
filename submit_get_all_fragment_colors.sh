#!/bin/bash

setup=setup22
exp=${setup}_200511_00
root=/nrs/saalfeld/kainmuellerd/ppp_all_cat_2_3
lm_base_id=vote_instances_rm_by_bbox_20_min_length_20
lm_id=skeletons_$lm_base_id

raw_path=/nrs/saalfeld/kainmuellerd/data/flylight/all_cat_2_3/
base_lm_path=/nrs/saalfeld/kainmuellerd/ppp_all_cat_2_3/setup22_200511_00/test/400000
inst_path=${base_lm_path}/instanced/
inst_key="vote_instances_rm_by_bbox_20"

output_dir=${base_lm_path}/frag_colors_$lm_base_id

mkdir $output_dir
echo $exp $output_dir
logbasedir=${root}/${exp}/logs
logdir=$logbasedir/frag_colors_all_lm_${lm_base_id}
mkdir $logbasedir
mkdir $logdir

vols=($(echo ${inst_path}*.hdf))
# OMP_NUM_THREADS=1

for v in "${vols[@]}";
do
  vol=${v##*/}
  vol=${vol%".hdf"}
  echo $vol
  
  log_file=$logdir/frag_colors_all_lm_${lm_base_id}_${vol}.out
  output_file=$output_dir/fragment_colors_${vol}.json
  
  if [ -f $output_file ]
  then
    echo "Skipping ${vol}, color results already exist in $output_file" ...
    continue
  fi

  echo "running, log $log_file, output $output_file"

  bsub -n 2 -W 03:00 -o $log_file python -W ignore get_all_fragment_colors.py \
                --raw-path ${raw_path} \
                --inst-path ${inst_path} \
                --lm-vol ${vol} \
                --inst-key $inst_key \
                --output-color-json $output_file
done
