#!/bin/bash

while true
do
numprocs=$(ps aux | grep find_best_matches.py | wc -l)
if [ $numprocs -lt 2 ]
then
    break
fi
echo "waiting..."
sleep 600
done

print_best=0

if [ -z "$1" ]
then
    min_num_frag_points=0
    cov_version=""
    max_matches=6
    cov_dist=25
    exp_id="_v4_adj_by_cov_numba_agglo_aT"
else
    exp_id=$1
    cov_version=$2
    max_matches=$3
    cov_dist=$4
fi

if [ -z "$NRS" ]
then
   export PATH=/misc/local/gcc-10.2/bin/:$PATH
   export LD_LIBRARY_PATH=/misc/local/gurobi-9.0.3/lib:$LD_LIBRARY_PATH
   export GRB_LICENSE_FILE=/misc/local/gurobi-9.0.3/gurobi.lic
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
    echo $em_set
    lm_suffix="_cropped"
    em_suffix="_flipped"

    prune_iter=3
    prune_length=30
    min_lm_length=20

    lm_resmp_suffix=""
    em_resmp_suffix=""

    lm_id=skeletons_vote_instances_rm_by_bbox_${min_lm_length}_min_length_${min_lm_length}$lm_suffix$em_suffix$lm_resmp_suffix
    em_id=40x_iter_${prune_iter}_len_${prune_length}${em_suffix}$em_resmp_suffix
    em_swc_base_folder=$NRS/data/hemibrain/$em_set/
    em_swc_folder=${em_swc_base_folder}${em_id}
    
    base_output_path=$NRS/flymatch/$em_set/setup22_nblast_$min_lm_length

    raw_path=/nrs/saalfeld/kainmuellerd/data/flylight/all_cat_2_3/
    base_lm_swc_path=/nrs/saalfeld/kainmuellerd/ppp_all_cat_2_3/setup22_200511_00/test/400000/
    inst_path=${base_lm_swc_path}instanced/
    inst_key="vote_instances_rm_by_bbox_${min_lm_length}"
    skel_color_folder=${base_lm_swc_path}frag_colors_vote_instances_rm_by_bbox_20_min_length_20/

    echo "base_output_path" $base_output_path

    for name in 1072063538 5813050455 669325882
    do
    for em in ${em_swc_folder}/$name*.swc
    # 887195902*.swc ${em_swc_folder}/1135160387*.swc ${em_swc_folder}/517587356*.swc
    do
        em_name=${em%.*}
        em_name=${em_name##*/}
        echo "em_name" $em_name
        
        em_number=${em_name%%_*}
        em_number=${em_number%%'-'*}
        echo "em_number" $em_number
        em_group_number=$(echo "$em_number % 100" | bc)
        printf -v em_group_number "%02d" $em_group_number
        
        base_results_path=${base_output_path}/results/$em_group_number/$em_name
        echo "base_results_path" $base_results_path
        
        results_path=${base_results_path}/lm_cable_length_$min_lm_length$exp_id

        mkdir -p ${results_path}
        echo $min_lm_length ${results_path}

        show_n_best=150
        xlsx_filename=PatchPerPixMatch_top_${show_n_best}_ranks_${em_name}.xlsx
        if [ -f ${results_path}/${xlsx_filename} ]
        then
            echo "Skipping find_best.., already ran for $em_name"
            continue
        fi 

        cmd_file=${results_path}/cmd.sh
        echo "" > $cmd_file
        for em_suffix in "" "_flipped"
        do
              echo -e python prepare_scores_per_em.py \
                  --input-folder $( sed "s/'/\\\'/g" <<< ${base_results_path}/nblastScores$em_suffix) \
                  --output-folder $(sed "s/'/\\\'/g" <<< ${results_path}) \
                  --output-suffix \"$em_suffix\" \
              >> $cmd_file
        done

        log_file=${results_path}/find_best_matches_by_cov_score_${em_name}.log

        echo "logfile" $log_file
        echo python -W ignore find_best_matches_by_cov_score.py \
                --em-name $(sed "s/'/\\\'/g" <<< ${em_name}) \
                --nblast-json-path $( sed "s/'/\\\'/g" <<< ${results_path}) \
                --exp-id ${exp_id} \
                --lm-base-id ${lm_id} \
                --em-base-id ${em_id} \
                --base-lm-swc-path ${base_lm_swc_path} \
                --show-mip $print_best\
                --raw-path $raw_path \
                --inst-path $inst_path \
                --inst-key $inst_key \
                --em-swc-base-folder $em_swc_base_folder \
                --skel-color-folder $skel_color_folder \
                --cov-score-thresh -40 \
                --nblast-score-thresh -0.5 \
                --min-aggregate-coverage 1 \
                --min-lm-cable-length $min_lm_length \
                --max-matches $max_matches \
                --max-coverage-dist $cov_dist \
                --min-num-frag-points-hack $min_num_frag_points \
                --both-sides \
                --get-cov-version \"${cov_version}\" \
                --adaptive-score-thresh-factor 200 \
                --adaptive-score-thresh-offset -0.1 \
                --resmp-factor 1 \
                --clustering-algo \'agglomerative\' \
        >> $cmd_file
        sed -i "s/(/\\\(/g" $cmd_file
        sed -i "s/)/\\\)/g" $cmd_file
        
        chmod ugo+rwx $cmd_file

        bsub -n 1 -W 40:00 -o $log_file sh $cmd_file
        sleep 0.1
    done
done
done

