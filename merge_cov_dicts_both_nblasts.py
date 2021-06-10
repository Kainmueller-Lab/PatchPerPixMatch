import numpy as np
import pandas as pd
import os
import argparse
import json
from visualize import visualize_frags
from prune_skeleton import create_graph_from_swc
import utils
import visualize


def get_arguments():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-folder", type=str, dest="input_folder",
        required=True
    )
    parser.add_argument(
        "--nblast-ext", type=str, dest="nblast_ext",
        default="_0"
    )
    parser.add_argument(
        "--nblast-ext-pruned", type=str, dest="nblast_ext_pruned",
        default="_1"
    )
    parser.add_argument("--json-basename",
                        type=str,
                        dest="json_basename",
                        default="cov_scores_"
                        )
    parser.add_argument("--em-name",
                        type=str,
                        dest="em_name"
                        )
    parser.add_argument("--raw-path", type=str, dest="raw_path",
                        default="/nrs/saalfeld/kainmuellerd/data/flylight/test/"
                        )
    parser.add_argument("--inst-path", type=str, dest="inst_path",
                        default="/nrs/saalfeld/kainmuellerd/ppp/setup22_200511_00/test/400000/instanced/"
                        )
    parser.add_argument("--inst-key", type=str, dest="inst_key",
                        default="vote_instances_rm_by_bbox_20"
                        )
    parser.add_argument("--em-swc-file",
                        type=str,
                        dest="em_swc_file"
                        )
    parser.add_argument("--em-flipped-swc-file",
                        type=str,
                        dest="em_flipped_swc_file"
                        )
    parser.add_argument("--show-num-best",
                        type=int,
                        dest="show_n_best"
                        )
    parser.add_argument("--output-folder",
                        type=str,
                        dest="output_folder"
                        )

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    print(args)

    em = args.em_name

    input_file = args.input_folder + "/" + args.json_basename + em + args.nblast_ext + ".json"
    print("loading input_file %s" % input_file)
    with open(input_file, 'r') as f:
        input_dict = json.load(f)

    lm_vols = list(input_dict[em].keys())
    for rank, lm_vol in enumerate(lm_vols):
        input_dict[em][lm_vol]['nblast_pruned'] = False
        input_dict[em][lm_vol]['rank'] = rank

    input_file_pruned = args.input_folder + "/" + args.json_basename + em + args.nblast_ext_pruned + ".json"
    print("loading input_file_pruned %s" % input_file_pruned)
    with open(input_file_pruned, 'r') as f:
        input_dict_pruned = json.load(f)

    lm_vols_pruned = list(input_dict_pruned[em].keys())

    for rank_pruned, lm_vol in enumerate(lm_vols_pruned):
        if lm_vol in lm_vols:
            # keep the better one:
            rank = lm_vols.index(lm_vol)
            if rank_pruned + 10 < rank:  # smaller is better, but prefer non-pruned
                input_dict[em][lm_vol] = input_dict_pruned[em][lm_vol]
                input_dict[em][lm_vol]['nblast_pruned'] = True
                input_dict[em][lm_vol]['rank'] = rank_pruned + 0.5

    # sort:
    input_dict[em] = {k: v for k, v in sorted((input_dict[em]).items(),
                                              key=lambda item: item[1]['rank'])}

    output_file = args.output_folder + "/" + args.json_basename + em + ".json"
    with open(output_file, 'w') as fp:
        json.dump(input_dict, fp, indent=4)

    lm_vols = list(input_dict[em].keys())

    # visualize best:
    did_create_masks = False
    for idx in range(min(args.show_n_best, len(lm_vols))):
        lm_vol = lm_vols[idx]
        skel_ids = eval(input_dict[em][lm_vol]['skel_ids'])
        if len(skel_ids) == 0:
            break
        scores = eval(input_dict[em][lm_vol]['nblast_scores'])
        colors = np.array(eval(input_dict[em][lm_vol]['colors']))
        coverages = np.array(eval(input_dict[em][lm_vol]['coverages']))
        mirrored = bool(input_dict[em][lm_vol]['mirrored'])
        pruned = bool(input_dict[em][lm_vol]['nblast_pruned'])
        # visualize coverage matches in mips
        cov_score = -1 * input_dict[em][lm_vol]['cov_score']
        raw_file = args.raw_path + lm_vol + ".zarr"
        inst_file = args.inst_path + lm_vol + ".hdf"

        if not did_create_masks:
            raw = utils.load_raw(raw_file)

            em_neuron = create_graph_from_swc(args.em_swc_file)
            # TODO: clean handling of downsample_by
            radius = 2
            prune_radius = 50
            downsample_by = 1
            prune_radius = int(prune_radius / downsample_by)
            halo_radius = radius + 4
            em_swc_mask = visualize.rasterize_skeleton(
                np.zeros(raw[0].shape, dtype=np.bool), em_neuron, radius=radius,
                downsample_by=downsample_by)
            em_halo_mask = visualize.rasterize_skeleton(
                np.zeros(raw[0].shape, dtype=np.bool), em_neuron,
                radius=halo_radius, downsample_by=downsample_by)
            em_prune_mask = visualize.rasterize_skeleton(
                np.zeros(raw[0].shape, dtype=np.bool), em_neuron,
                radius=prune_radius, downsample_by=downsample_by)

            em_neuron_flipped = create_graph_from_swc(args.em_flipped_swc_file)
            em_swc_mask_flipped = visualize.rasterize_skeleton(
                np.zeros(raw[0].shape, dtype=np.bool), em_neuron_flipped,
                radius=radius, downsample_by=downsample_by)
            em_halo_mask_flipped = visualize.rasterize_skeleton(
                np.zeros(raw[0].shape, dtype=np.bool), em_neuron_flipped,
                radius=halo_radius, downsample_by=downsample_by)

            em_prune_mask_flipped = visualize.rasterize_skeleton(
                np.zeros(raw[0].shape, dtype=np.bool), em_neuron_flipped,
                radius=prune_radius, downsample_by=downsample_by)
            did_create_masks = True
            raw = None

        if mirrored:
            em_swc = args.em_flipped_swc_file
            em_mask = em_swc_mask_flipped
            em_halo = em_halo_mask_flipped
            em_prune = em_prune_mask_flipped
        else:
            em_swc = args.em_swc_file
            em_mask = em_swc_mask
            em_halo = em_halo_mask
            em_prune = em_prune_mask

        # save raw mip
        screenshot_folder = args.output_folder + "/screenshots"
        os.makedirs(screenshot_folder, exist_ok=True)
        outbase = screenshot_folder + "/%s_cr_%i_cscore_%3d_%s_" % (
        em, idx + 1, cov_score, lm_vol)

        outfns = [outbase + "1_raw.png", outbase + "2_masked_raw.png",
                  outbase + "4_masked_inst.png", outbase + "3_skel.png",
                  outbase + "5_ch.png", outbase + "6_ch_skel.png"]

        if not os.path.exists(outfns[-1]):
            visualize_frags(
                raw_file,
                inst_file,
                args.inst_key,
                skel_ids,
                scores,
                coverages,
                em_swc,
                outfns,
                pruned=pruned,
                skel_colors=colors,
                channel="auto",
                em_swc_mask=em_mask,
                em_halo_mask=em_halo,
                em_prune_mask=em_prune,
                verbose=False
            )


if __name__ == "__main__":
    main()
