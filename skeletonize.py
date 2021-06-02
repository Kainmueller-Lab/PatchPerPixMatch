import numpy as np
import h5py
import os
from glob import glob
import argparse
import kimimaro
from datetime import datetime


def get_arguments():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in-folder", type=str, dest="in_folder",
        required=True
    )
    parser.add_argument(
        "--in-format", type=str, dest="in_format", default="hdf"
    )
    parser.add_argument("--in-key", type=str, dest="in_key")
    parser.add_argument(
        "--min-cable-length", type=int, dest="min_cable_length"
    )
    parser.add_argument(
        "--out-dir", type=str, dest="out_dir"
    )
    parser.add_argument(
        "--num-worker", type=int, dest="num_worker",
        default=1
    )
    parser.add_argument(
        "--sample", type=str, dest="sample",
        default=None
    )

    args = parser.parse_args()
    return args


def create_skeleton(instances, num_worker=1):
    # teasar params:
    # scale and const determine radius of invalidation sphere, r(x,y,z) = scale * D(x,y,z) + const
    # do I want to detect soma? problem that soma and bundles might be same radius
    # r(x,y,z) = soma_invalidation_scale * DBF(x,y,z) + soma_invalidation_const

    skels = kimimaro.skeletonize(
        instances,
        teasar_params={
            'scale': 3,  # 1,
            'const': 4,  # physical units, should be large enough to ignore dendritic branches
            'pdrf_exponent': 4,
            'pdrf_scale': 100000,
            'soma_detection_threshold': 10,  # physical units, radius, soma detection and test for soma
            'soma_acceptance_threshold': 10,  # physical units, radius appr. 10 pixel
            'soma_invalidation_scale': 1.0,
            'soma_invalidation_const': 10,  # physical units
            'max_paths': 1000,
        },
        dust_threshold=0,
        # skip connected components with fewer than this many voxels; heads up: if components are skipped, returned list is shorter than #unique labels --> error; thus dust_threshold has to be set to 0
        # anisotropy=(0.44, 0.44, 0.44), # scale in um
        anisotropy=(1, 1, 1),  # heads up: using pixel here!
        fix_branching=True,
        fix_borders=True,
        progress=True,
        parallel=num_worker,  # <= 0 all cpu, 1 single process, 2+ multiprocess
        parallel_chunk_size=50,  # how many skeletons to process before updating progress bar
    )

    return skels


def skeleton_to_swc(skel, outfn):
    with open(outfn, "w") as f:
        f.write(skel.to_swc())


def main():
    args = get_arguments()
    print(args)

    # create output folder
    skeleton_base_folder = args.out_dir
    os.makedirs(skeleton_base_folder, exist_ok=True)

    # iterate through files
    if args.sample == None:
        in_files = glob(args.in_folder + "/*." + args.in_format)
    else:
        in_files = [args.in_folder + "/" + args.sample + "." + args.in_format]

    t1 = datetime.now()
    for in_file in in_files:

        sample_name = os.path.basename(in_file).split(".")[0]
        print("Processing sample %s..." % sample_name)

        line_name = os.path.basename(sample_name).split("-")[0]

        skeleton_folder = skeleton_base_folder + "/" + line_name
        os.makedirs(skeleton_folder, exist_ok=True)

        check_for_sample = glob(skeleton_folder + "/" + sample_name + "*")
        if len(check_for_sample) > 0:
            print("Skipping volume as skeleton folder already exists.")
            # continue

        # read segmentation volume
        if args.in_format == "hdf":
            inf = h5py.File(in_file, "r")
            instances = np.array(inf[args.in_key])
            inf.close()
        else:
            raise NotImplementedError

        instances = instances.astype(np.uint64)
        instances = np.transpose(instances)
        skeletons = create_skeleton(instances, args.num_worker)

        labels = np.unique(instances)

        for label in labels:
            if label == 0:
                continue
            skel = skeletons[label]
            if skel.cable_length() > args.min_cable_length:
                skel_swc_fn = os.path.join(skeleton_folder, sample_name + "_%i.swc" % label)
                skeleton_to_swc(skel, skel_swc_fn)
            else:
                print("Skipping skeleton %i with cable length %f" % (label, skel.cable_length()))
    print(datetime.now() - t1, " time for ", len(in_files), " samples")


if __name__ == "__main__":
    main()
