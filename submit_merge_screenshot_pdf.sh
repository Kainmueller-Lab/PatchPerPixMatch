#!/bin/bash

show_n_best=150
target_n_screenshots=$(($show_n_best * 6))

if [ -z "$1" ]; then
  exp_id="_v4_adj_by_cov_numba_agglo_aT"
else
  exp_id=$1
fi

if [ -z "$NRS" ]; then
  NRS="/nrs/saalfeld/maisl"
fi

if [ -z "$DATASET" ]; then
  em_sets=(all_hemibrain_1.2_NB)
else
  em_sets=($DATASET)
fi

for em_set in ${em_sets[@]}; do
  #
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

  raw_path=/nrs/saalfeld/maisl/data/flylight/all_cat_2_3/
  inst_path=/nrs/saalfeld/maisl/ppp_test/setup22_200511_00/test/400000/instanced/
  inst_key="vote_instances_rm_by_bbox_${min_lm_length}"

  for redoname in 514850616 5813063587; do
    for em in ${em_swc_folder}/$redoname*.swc; do
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

      cmd_file=${results_path}/cmd_merge_screenshot_pdf.sh
      echo "" >$cmd_file

      pdf_filename=PatchPerPixMatch_top_${show_n_best}_ranks_${em_name}.pdf
      xlsx_filename=PatchPerPixMatch_top_${show_n_best}_ranks_${em_name}.xlsx

      cov_json_name=cov_scores_$em_name.json # todo: check for 150 screenshots instead
      num_screenshots=$(ls ${results_path}/screenshots/ | wc -l)
      if [[ $num_screenshots -eq $target_n_screenshots ]]; then
        echo "Skipping merge_cov_dicts_both_nblasts, already ran for $em_name"
      else
        echo "got $num_screenshots screenshots for $em_name"
        rm ${results_path}/$xlsx_filename
        if [[ $num_screenshots -gt $target_n_screenshots ]]; then
          echo "removing screenshots, too many"
          rm -r ${results_path}/screenshots/
        fi
        echo python -W ignore merge_cov_dicts_both_nblasts.py \
          --em-name $(sed "s/'/\\\'/g" <<<${em_name}) \
          --em-swc-file $(sed "s/'/\\\'/g" <<<${em}) \
          --em-flipped-swc-file $(sed "s/'/\\\'/g" <<<${em_flipped_swc_file}) \
          --input-folder $(sed "s/'/\\\'/g" <<<${results_path}) \
          --nblast-ext "_0" \
          --nblast-ext-pruned "_1" \
          --show-num-best $show_n_best \
          --raw-path $raw_path \
          --inst-path $inst_path \
          --inst-key $inst_key \
          --output-folder $(sed "s/'/\\\'/g" <<<${results_path}) \
          >>$cmd_file
        echo 'num_screenshots=$(ls '$(sed "s/'/\\\'/g" <<<${results_path})'/screenshots/  | wc -l)' >>$cmd_file
        echo if [[ '$num_screenshots' -ne $target_n_screenshots ]] >>$cmd_file
        echo then >>$cmd_file
        echo echo 'not the right amount of screenshots to proceed: $num_screenshots' >>$cmd_file
        echo rm $(sed "s/'/\\\'/g" <<<${results_path}/$xlsx_filename) >>$cmd_file
        echo exit 0 >>$cmd_file
        echo fi >>$cmd_file

      fi

      if [ -f ${results_path}/${xlsx_filename} ]; then
        echo "Skipping pngs_to_mipp_pdf, already ran for $em_name"
        continue
      else
        echo python -W ignore pngs_to_mipp_pdf.py \
          --png-path $(sed "s/'/\\\'/g" <<<${results_path})/screenshots/ \
          --result-path $(sed "s/'/\\\'/g" <<<${results_path})/ \
          --result-filename $(sed "s/'/\\\'/g" <<<${pdf_filename}) \
          --num-pages ${show_n_best} \
          >>$cmd_file

        echo rm $(sed "s/'/\\\'/g" <<<${results_path}/dummy_both.pdf) >>$cmd_file
        echo rm $(sed "s/'/\\\'/g" <<<${results_path}/*.csv) >>$cmd_file
        echo 'sed -i' \''10,${/Count/d}'\' $(sed "s/'/\\\'/g" <<<${results_path}/${pdf_filename}) >>$cmd_file
      fi

      sed -i "s/(/\\\(/g" $cmd_file
      sed -i "s/)/\\\)/g" $cmd_file

      sed -i "s/\$\\\(ls/\$(ls/g" $cmd_file
      sed -i "s/l\\\)/l)/g" $cmd_file

      chmod ugo+rwx $cmd_file
      echo $log_file

      bsub -n 3 -W 4:00 -o $log_file bash $cmd_file

    done
  done
done
