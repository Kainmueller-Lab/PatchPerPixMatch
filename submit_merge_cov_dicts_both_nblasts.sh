#!/bin/bash

show_n_best=1

if [ -z "$1" ]
then
    exp_id="_v4_adj_by_cov_numba_agglo_aT"
else
    exp_id=$1
fi

if [ -z "$NRS" ]
then
    NRS="/nrs/saalfeld/kainmuellerd"
fi

if [ -z "$DATASET" ]
then
    em_sets=(gt_and_suspected_set_1.2)
else
    em_sets=($DATASET)
fi

for em_set in ${em_sets[@]}
do
    echo $em_set
    lm_suffix="_cropped"

    prune_iter=3
    prune_length=30
    min_lm_length=20

    lm_resmp_suffix=""
    em_resmp_suffix=""

    lm_id=skeletons_vote_instances_rm_by_bbox_${min_lm_length}_min_length_${min_lm_length}${lm_suffix}_flipped$lm_resmp_suffix
    base_em_id=40x_iter_${prune_iter}_len_${prune_length}
    em_id=${base_em_id}$em_resmp_suffix
    em_id_flipped=${base_em_id}_flipped$em_resmp_suffix
    base_em_swc_folder=$NRS/data/hemibrain/${em_set}/
    em_swc_folder=$base_em_swc_folder$em_id
    em_swc_folder_flipped=$base_em_swc_folder$em_id_flipped
    
    base_output_path=$NRS/flymatch/$em_set/setup22_nblast_$min_lm_length

    raw_path=/nrs/saalfeld/kainmuellerd/data/flylight/all_cat_2_3/
    inst_path=/nrs/saalfeld/kainmuellerd/ppp_all_cat_2_3/setup22_200511_00/test/400000/instanced/
    inst_key="vote_instances_rm_by_bbox_${min_lm_length}"

    for em in ${em_swc_folder}/*.swc
    do
        em_name=${em%.*}
        em_name=${em_name##*/}
        
        em_number=${em_name%%_*}
        em_number=${em_number%%'-'*}
        echo "em_number" $em_number
        em_group_number=$(echo "$em_number % 100" | bc)
        printf -v em_group_number "%02d" $em_group_number

        base_results_path=${base_output_path}/results/$em_group_number/$em_name
        echo "base_results_path" $base_results_path
        
        results_path=${base_results_path}/lm_cable_length_$min_lm_length$exp_id

        log_file=${results_path}/merge_cov_dicts_both_nblasts${em_name}.log

        em_flipped_swc_file=${em_swc_folder_flipped}/${em_name}.swc
        echo $em
        echo ${em_flipped_swc_file}

        bsub -n 3 -W 4:00 -o $log_file python merge_cov_dicts_both_nblasts.py \
                --em-name ${em_name} \
                --em-swc-file ${em} \
                --em-flipped-swc-file ${em_flipped_swc_file} \
                --input-folder $results_path \
                --nblast-ext "_0" \
                --nblast-ext-pruned "_1" \
                --show-num-best $show_n_best \
                --raw-path $raw_path \
                --inst-path $inst_path \
                --inst-key $inst_key \
                --output-folder $results_path 

    done
done
