import numpy as np
import h5py
from glob import glob
import argparse
from datetime import datetime
from skimage import io
from scipy import ndimage


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
    parser.add_argument("--out-key", type=str, dest="out_key")
    parser.add_argument(
        "--mode", type=str, dest="mode",
        choices=['counts', 'bbox'],
        default='counts'
    )
    parser.add_argument(
        "--small-comps", type=int, dest="small_comps",
        required=True
    )
    parser.add_argument('--show-mip', dest='show_mip',
                        action='store_true'
                        )
    parser.add_argument(
        "--sample", type=str, dest="sample",
        default=None
    )
    parser.add_argument('--verbose', dest='verbose',
                        action='store_true'
                        )

    args = parser.parse_args()
    return args


def replace(array, old_values, new_values):
    values_map = np.arange(int(array.max() + 1), dtype=new_values.dtype)
    values_map[old_values] = new_values
    return values_map[array]


def get_bbox_diag(ob):
    if ob:
        return np.linalg.norm([int(ob[i].stop - ob[i].start) for i in range(len(ob))])
    else:
        return 0


def remove_small_components(array, compsize, mode='counts', verbose=False):
    if mode == 'counts':
        labels, counts = np.unique(array, return_counts=True)
        small_labels = labels[counts <= compsize]
    if mode == 'bbox':
        objs = ndimage.find_objects(array)
        bbox_diags = np.array([2 * compsize] + [get_bbox_diag(ob) for ob in objs])
        small_labels = np.where(bbox_diags <= compsize)[0]
    if verbose:
        print("num labels removed: %d of %d" % (len(small_labels), len(bbox_diags) - 1))

    array = replace(array, np.array(small_labels),
                    np.array([0] * len(small_labels)))

    return array


def color(src):
    labels = np.unique(src)
    colored = np.stack([
        np.zeros_like(src),
        np.zeros_like(src),
        np.zeros_like(src)
    ], axis=-1)

    for label in labels:
        if label == 0:
            continue
        label_color = np.random.randint(0, 255, 3)
        idx = src == label
        colored[idx, :] = label_color

    return colored


def main():
    args = get_arguments()
    print(args)

    # iterate through files
    if args.sample == None:
        in_files = glob(args.in_folder + "/*." + args.in_format)
    else:
        in_files = [args.in_folder + "/" + args.sample + "." + args.in_format]

    t1 = datetime.now()
    for in_file in in_files:
        if args.verbose:
            print("Processing sample %s" % in_file)
        if args.in_format == "hdf":
            inf = h5py.File(in_file, "a")
        else:
            raise NotImplementedError

        if args.out_key not in inf:
            instances = np.array(inf[args.in_key])
            dtype = instances.dtype
            if args.verbose:
                print(instances.shape, dtype)
            instances = remove_small_components(instances, args.small_comps, mode=args.mode, verbose=args.verbose)

            inf.create_dataset(
                args.out_key,
                data=instances.astype(dtype),
                dtype=dtype,
                compression="gzip"
            )
            if args.show_mip:
                colored = color(np.max(instances, axis=0))
                io.imsave(
                    in_file.replace(".hdf", "_%s.png" % args.out_key),
                    colored.astype(np.uint8)
                )

        else:
            print("Skipping, %s already exists..." % args.out_key)
            if args.show_mip:
                instances = np.array(inf[args.out_key])
                colored = color(np.max(instances, axis=0))
                io.imsave(
                    in_file.replace(".hdf", "_%s.png" % args.out_key),
                    colored.astype(np.uint8)
                )

        if args.in_format == "hdf":
            inf.close()

    if args.verbose:
        print(datetime.now() - t1, " time for ", len(in_files), " samples")


if __name__ == "__main__":
    main()
