#!/bin/bash

setup=setup22
exp=${setup}_200511_00
root=/nrs/saalfeld/kainmuellerd/ppp_all_cat_2_3
# root=/nrs/saalfeld/kainmuellerd/ppp
# root=/nrs/saalfeld/kainmuellerd/ppp_Barry
data_dir=$root/$exp/test/400000/instanced
mode='bbox' # 'counts' # 
min_component_size=20 # 40 # 10 # 2000
in_key=vote_instances

out_key=${in_key}_rm_by_${mode}_$min_component_size

echo $exp $data_dir
mkdir ${root}/${exp}/logs
mkdir ${root}/${exp}/logs/remove_small_components

samples=($(ls $data_dir/BJD_100A* | grep .hdf))

for s in "${samples[@]}";
do
  # echo $s
  sname=$(basename $s .hdf)
  log_file=${root}/${exp}/logs/remove_small_components/remove_small_components_by_${mode}_${in_key}_${min_component_size}_${sname}.out
  mip_file=$data_dir/${sname}_${out_key}_${min_component_size}.png
  
  if [ -f ${mip_file} ]
  then
    echo "${mip_file} exists, skipping..."
    # continue
  fi

  l=$(h5dump --header ${data_dir}/$s | grep $out_key | wc -l)
  if [ $((l)) -gt 0 ]
  then
    echo "${data_dir}/${s}, key $out_key already exists, but not mip, running..."
  fi

  echo "Processing ${data_dir}/${s}..."
  echo $log_file
  
  # run locally:
  # python remove_small_components.py --in-folder $data_dir --sample $sname --in-key $in_key --out-key $out_key --mode $mode --small-comps $min_component_size --show-mip --verbose > $log_file &
  # exit
  
  # run on cluster:
  bsub -P flylight -n 1 -W 0:10 -o $log_file python remove_small_components.py --in-folder $data_dir --sample $sname --in-key $in_key --out-key $out_key --mode $mode --small-comps $min_component_size --show-mip --verbose
  # sleep 1
  # exit

done





