#!/bin/bash

setup=setup22
exp=${setup}_200511_00
root=/nrs/saalfeld/kainmuellerd/ppp
input_dir=/nrs/saalfeld/kainmuellerd/ppp/$exp/test/400000/skeletons_min_length_200/
output_dir=/nrs/saalfeld/kainmuellerd/ppp/$exp/test/400000/skeletons_min_length_200_spines/

echo $exp $data_dir
mkdir ${root}/${exp}/logs
mkdir ${root}/${exp}/logs/compute_skeleton_spine
mkdir $output_dir

lines=($(ls -d $input_dir*))

for l in "${lines[@]}";
do
  echo $l
  line=${l##*/}
  echo $line
  log_file=${root}/${exp}/logs/compute_skeleton_spine/compute_skeleton_spine_${line}.out
  
  if [ -f $log_file ]
  then
    echo Skipping ${input_dir}${line}, already ran...
    continue
  fi

  echo $log_file
  mkdir $output_dir$line
  
  while true 
  do
    numprocs=$(ps | grep R | wc -l)
    if [ $numprocs -lt 50 ] 
    then
      Rscript compute_skeleton_spines.R $input_dir$line $output_dir$line > $log_file &
      break
    fi
    sleep 3
  done

done

# one-time call for em neurons:
# em_input_dir=/nrs/saalfeld/kainmuellerd/data/hemibrain/40x_iter_3_len_30
# em_output_dir=/nrs/saalfeld/kainmuellerd/data/hemibrain/40x_iter_3_len_30_spines
# mkdir em_output_dir
# Rscript compute_skeleton_spines.R $em_input_dir $em_output_dir 



