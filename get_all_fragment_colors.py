import utils
import json
import os
import os.path
import argparse


def get_arguments():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=str, dest="raw_path",
                        default="/nrs/saalfeld/kainmuellerd/data/flylight/all_cat_2_3"
                        )
    parser.add_argument("--inst-path", type=str, dest="inst_path",
                        default="/nrs/saalfeld/kainmuellerd/ppp/setup22_200511_00/test/400000/instanced/"
                        )
    parser.add_argument("--inst-key", type=str, dest="inst_key",
                        default="vote_instances_rm_by_bbox_30"
                        )
    parser.add_argument("--lm-vol", type=str, dest="lm_vol",
                        default="BJD_124B11_AE_01-20171117_61_C4_REG_UNISEX_40x"
                        )
    parser.add_argument("--output-color-json", type=str,
                        dest="output_file"
                        )
    parser.add_argument("--verbose",
                        dest='verbose',
                        action='store_true'
                        )

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()

    raw_path = args.raw_path
    inst_path = args.inst_path
    inst_key = args.inst_key
    lm_vol = args.lm_vol
    verbose = args.verbose

    if os.path.isfile(args.output_file):
        color_dict = json.load(open(args.output_file))
    else:
        color_dict = {}

    lm_vols = [lm_vol]
    print("lm vols: %s" % lm_vols)

    for lm_vol in lm_vols:
        if lm_vol in list(color_dict.keys()):
            continue
        color_dict[lm_vol] = {}
        skel_ids = -1
        raw_file = raw_path + lm_vol + ".zarr"
        inst_file = inst_path + lm_vol + ".hdf"
        skel_colors, skel_ids = utils.get_skel_colors(raw_file, inst_file, inst_key, skel_ids=-1, bin_size=64,
                                                      verbose=verbose)
        for i, skel_id in enumerate(skel_ids):
            if verbose:
                print("id %i color %s" % (skel_id, skel_colors[i]))
            color_dict[lm_vol][str(skel_id)] = str(
                [int(skel_colors[i][0]), int(skel_colors[i][1]), int(skel_colors[i][2])])

        with open(args.output_file, 'w') as fp:
            json.dump(color_dict, fp, indent=4)


if __name__ == "__main__":
    main()
