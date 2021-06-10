import pickle
from numba import jit
import gc
import ctypes
import glob
import utils
import json
import numpy as np
import pandas as pd
import os
import argparse
from collections import OrderedDict
import time
from visualize import visualize_frags
from math import sqrt
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree


def get_arguments():
    if 'NRS' not in os.environ:
        os.environ['NRS'] = '/nrs/saalfeld/kainmuellerd'

    if 'DATASET' not in os.environ:
        os.environ['DATASET'] = 'ground_truth_set'

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-path", type=str, dest="raw_path",
                        default="/nrs/saalfeld/kainmuellerd/data/flylight/test/"
                        )
    parser.add_argument("--inst-path", type=str, dest="inst_path",
                        default="/nrs/saalfeld/kainmuellerd/ppp/setup22_200511_00/test/400000/instanced/"
                        )
    parser.add_argument("--inst-key", type=str, dest="inst_key",
                        default="vote_instances_rm_by_bbox_30"
                        )
    parser.add_argument("--em-name", type=str, dest="em",
                        default="947573616_RT_18U"
                        )
    parser.add_argument("--lm-vol-name", type=str, dest="test_lm_vol_name",
                        default=None
                        )
    parser.add_argument("--get-cov-version", type=str, dest="get_cov_version",
                        default=None
                        )
    parser.add_argument("--max-matches", type=int, dest="max_matches",
                        default=6)
    parser.add_argument("--em-base-id", type=str, dest="em_id",
                        default="40x_iter_3_len_30"
                        )
    parser.add_argument("--lm-base-id", type=str, dest="lm_id",
                        default="skeletons_min_length_200_cropped"
                        )
    parser.add_argument("--em-swc-base-folder", type=str,
                        dest="em_swc_base_folder",
                        default=os.environ['NRS'] + "/data/hemibrain/" +
                                os.environ['DATASET'] + "/"
                        )
    parser.add_argument("--nblast-json-path", type=str,
                        dest="nblast_json_path",
                        default=os.environ['NRS'] + "/flymatch/" + os.environ[
                            'DATASET'] + "/setup22_nblast/"
                        )
    parser.add_argument("--exp-id", type=str,
                        dest="exp_id",
                        default=""
                        )
    parser.add_argument('--base-lm-swc-path', type=str,
                        dest='base_lm_swc_path',
                        default="/nrs/saalfeld/kainmuellerd/ppp/setup22_200511_00/test/400000/"
                        )
    parser.add_argument('--skel-color-folder', type=str,
                        dest='skel_color_folder',
                        default=None
                        )
    parser.add_argument('--min-aggregate-coverage', type=float,
                        dest='min_aggregate_coverage',
                        default=40
                        )
    parser.add_argument('--min-lm-cable-length', type=int,
                        dest='min_lm_cable_length',
                        default=100
                        )
    parser.add_argument('--min-num-frag-points-hack', type=int,
                        dest='min_num_frag_points_hack',
                        default=0
                        )
    parser.add_argument('--min-nblast-score', type=float,
                        dest='min_nblast_score',
                        default=-1
                        )
    parser.add_argument('--max-nblast-score', type=float,
                        dest='max_nblast_score',
                        default=1
                        )
    parser.add_argument('--nblast-score-thresh', type=float,
                        dest='nblast_score_thresh',
                        default=-0.5
                        )
    parser.add_argument('--adaptive-score-thresh-factor', type=float,
                        dest='adaptive_score_thresh_factor',
                        default=0
                        # version "adaptive_score_thresh" had factor 200 at no re-sampling
                        )
    parser.add_argument('--adaptive-score-thresh-offset', type=float,
                        dest='adaptive_score_thresh_offset',
                        default=0
                        # version "adaptive_score_thresh" had offset 0 at no re-sampling
                        )
    parser.add_argument('--nblast-factor', type=float, dest='nblast_factor',
                        default=1
                        )
    parser.add_argument('--resmp-factor', type=float, dest='resmp_factor',
                        default=1
                        )
    parser.add_argument('--min-unary', type=int, dest='min_unary',
                        default=0
                        )
    parser.add_argument('--cov-score-thresh', type=int, dest='cov_score_thresh',
                        default=-60
                        )
    parser.add_argument('--max-coverage-dist', type=int, dest='max_cov_dist',
                        default=25
                        )
    parser.add_argument('--score-index', type=int, dest='score_index',
                        default=-1
                        )
    parser.add_argument("--both-sides",
                        dest='both_sides',
                        action='store_true'
                        )
    parser.add_argument("--verbose",
                        dest='verbose',
                        action='store_true'
                        )
    parser.add_argument('--show-mip',
                        dest='show_n_best',
                        type=int,
                        default=10
                        )
    parser.add_argument("--must-do-list", type=str, dest="must_do_list",
                        default=None
                        )

    parser.add_argument('--clustering-algo', type=str,
                        dest='clustering_algo',  # agglomerative
                        default="kmeans"
                        )

    args = parser.parse_args()
    return args


def load_em(args, em_swc_file, output_folder):
    # load em skel
    if args.verbose:
        print("loading em %s" % em_swc_file)
    em_points, parents = get_swc_points(em_swc_file)
    seg_length_per_em_point = get_seg_length_per_em_point(em_points, parents,
                                                          verbose=False)
    # debug seg length:
    with open(em_swc_file, "r") as neuron_file:
        lines = [[s for s in line.rstrip().split(" ")] for line in neuron_file
                 if not line[0] == "#"]
    lines = [
        " ".join(line[:-2] + [str(seg_length_per_em_point[i])] + [line[-1]]) for
        i, line in enumerate(lines)]
    with open(os.path.join(output_folder,
                           args.em + '_debug_seg_length.swc'),
              'w') as f:
        for l in lines:
            f.write("%s\n" % l)
    # end debug seg length
    em_num_points = len(em_points)
    seg_length_per_em_point *= em_num_points / np.sum(seg_length_per_em_point)
    if args.verbose:
        print("seg_length_per_em_point %s" % seg_length_per_em_point)
    return em_points, em_num_points, seg_length_per_em_point


mylib = None
solve1 = None
solve2 = None


def setup_ilp_solver(args):
    libfile = glob.glob(os.path.join(os.environ['HOME'],
                                     'PatchPerPixMatch/ilp_cpp/build/*/ilp_matching*.so'))

    if len(libfile) == 0:
        print(
            "cannot find ILP library, has to be compiled manually first, see ilp_cpp/setup.py")
        return None

    # 1. open the shared library
    global mylib
    mylib = ctypes.CDLL(libfile[0])
    mylib.make_solver.restype = None
    mylib.make_solver.argtypes = []
    mylib.delete_solver.restype = None
    mylib.delete_solver.argtypes = []

    global solve1
    mylib.solve.restype = ctypes.c_int
    mylib.solve.argtypes = [np.ctypeslib.ndpointer(dtype=np.float32),
                            ctypes.c_int, ctypes.c_int, ctypes.c_float,
                            np.ctypeslib.ndpointer(dtype=np.float64),
                            ctypes.c_int]
    solve1 = mylib.solve

    global solve2
    mylib.solve2.restype = ctypes.c_int
    mylib.solve2.argtypes = [ctypes.c_void_p,
                             ctypes.c_int, ctypes.c_int, ctypes.c_float,
                             ctypes.c_void_p,
                             ctypes.c_void_p,
                             ctypes.c_void_p,
                             ctypes.c_int]
    solve2 = mylib.solve2

    mylib.solve3.restype = ctypes.c_int
    mylib.solve3.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                             ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             np.ctypeslib.ndpointer(dtype=np.int32),
                             ctypes.c_float,
                             np.ctypeslib.ndpointer(dtype=np.int64),
                             np.ctypeslib.ndpointer(dtype=np.int32),
                             np.ctypeslib.ndpointer(dtype=np.float64),
                             ctypes.c_int]

    # if args.get_cov_version is not None and \
    # ("ilp" in args.get_cov_version or 'mixed' in args.get_cov_version):
    mylib.make_solver()


def maybe_em_tree(args, em_points):
    # TODO: rename, _default use tree now, too
    # if "ilp_v2" in args.get_cov_version or \
    #    "ilp_v3" in args.get_cov_version or \
    #    "mixed_v2" in args.get_cov_version:
    return cKDTree(em_points)
    # return sklearn.neighbors.KDTree(em_points, leaf_size=30)
    # else:
    #     return None


def get_swc_points(swc_file):
    with open(swc_file, "r") as neuron_file:
        lines = np.array([[float(s) for s in line.rstrip().split(" ")]
                          for line in neuron_file if
                          not line[0] == "#" and len(line) > 1])
        neuron_points = lines[:, 2:5]
        parents = lines[:, -1].astype(int)
    return neuron_points, parents


def get_dist_matrix_kd(em_points, em_tree, frag_points, dist_1):
    min_dist_per_fp, idcs = em_tree.query(frag_points,
                                          distance_upper_bound=dist_1)
    where = np.where(np.array(min_dist_per_fp) <= dist_1)[0]
    frag_points = frag_points[where]

    dist_matrix_2 = np.array(cdist(em_points, frag_points, 'sqeuclidean'))

    min_dist_per_fp = min_dist_per_fp[where]

    return dist_matrix_2, min_dist_per_fp


def get_dist_matrix(em_points, frag_points, dist2):
    dist_matrix_2 = np.array(cdist(em_points, frag_points, 'sqeuclidean'))

    # weed out frag points that are > dist2 away from em neuron:
    min_dist_per_fp = np.amin(dist_matrix_2, axis=0)
    where = np.where(min_dist_per_fp <= dist2)[0]
    dist_matrix_2 = dist_matrix_2[:, where]
    min_dist_per_fp = min_dist_per_fp[where]
    return dist_matrix_2, min_dist_per_fp


def get_single_frag_coverage_default(
    em_points, em_tree, seg_length_per_em_point, frag_points, dist2=625,
    resmp_factor=1, verbose=False):
    dist_1 = sqrt(dist2)

    dist_matrix_2, min_dist_per_fp = get_dist_matrix_kd(em_points, em_tree,
                                                        frag_points, dist_1)

    return get_single_frag_coverage_default_B(
        em_points, seg_length_per_em_point, frag_points, dist_matrix_2,
        min_dist_per_fp, dist2=dist2, resmp_factor=resmp_factor,
        verbose=verbose)


@jit(nopython=True)
def get_single_frag_coverage_default_B(
    em_points, seg_length_per_em_point, frag_points, dist_matrix_2,
    min_dist_per_fp, dist2=625, resmp_factor=1, verbose=False):
    dist_1 = sqrt(dist2)

    if dist_matrix_2.shape[1] == 0:
        return np.array([np.int64(x) for x in range(0)])  # = []

    # for each em point, the frag point index with minimum distance:
    # min_dist_fp_indices = np.argmin( dist_matrix_2, axis=1)
    min_dist_fp_indices = np.array([np.argmin(m) for m in dist_matrix_2])

    # copy the min distances per em point for returning them at the end:
    all_min_dist_2 = np.array([dist_matrix_2[i, min_dist_fp_indices[i]] for i in
                               range(len(min_dist_fp_indices))])

    # I think this is not needed:
    all_min_dist_2[all_min_dist_2 > dist2] = 100 * dist2

    # for each frag point, how many em points is it allowed to cover:
    # (default is 6, but some frag points might only cover less for reasons that made perfect sense to me when I coded this up)
    # max_num_covered_per_fp = np.array([ int(min(6,max(1,0.8*sqrt(dist2-mdpfp)))) for mdpfp in min_dist_per_fp ])
    max_num_covered_per_fp = 6  # max(1, int(6/resmp_factor))

    # the running copy of em points covered by the fragment, initialized to none (all 0):
    covered_running = np.zeros(all_min_dist_2.shape)

    for loopcount in range(int(dist_1 / resmp_factor + 1)):

        # which fragment points are left:
        fp_idcs = np.where(min_dist_per_fp < dist2)[0]
        if fp_idcs.shape[0] < 1:
            # if verbose: print("loop %i no more fp_idcs left, breaking" % loopcount)
            break

        # for each of the fragment points still in play, for which em point is it at minimum distance:
        eps_per_fp_idx = [np.where(min_dist_fp_indices == i)[0] for i in
                          fp_idcs]

        # whereall = np.where(np.isin(min_dist_fp_indices,fp_idcs))[0]
        # eps_per_fp_idx = [ whereall[min_dist_fp_indices[whereall]==i] for i in fp_idcs ]

        # unique, idx_groups = npi.group_by(min_dist_fp_indices, np.arange(len(min_dist_fp_indices)))
        # eps_per_fp_idx = idx_groups[fp_idcs]

        # and what are the resp. min distances:
        fp_dists_per_fp_idx = [all_min_dist_2[eps] for eps in eps_per_fp_idx]
        # weed out the em points for which the min distance is above the dist threshold:
        eps_per_fp_idx = [eps[fp_dists_per_fp_idx[i] < dist2] for i, eps in
                          enumerate(eps_per_fp_idx)]

        # how many em points is a frag point allowed to cover:
        # .. and how many does it in fact cover:
        eps_shape_per_fp_idx = np.array(
            [eps.shape[0] for eps in eps_per_fp_idx])

        # which frag points cover more em points than they're supposed to (True/False):
        do_restrict = np.where(eps_shape_per_fp_idx > max_num_covered_per_fp)[0]

        if do_restrict.shape[0] > 0:
            # for each frag point that covers more em points than it is supposed to, the resp. em points (i.e. for which it is at minimum distance):
            eps_per_fp_idx = [eps_per_fp_idx[r] for r in do_restrict]
            # which frag points cover more em points than they're supposed to (point indices):
            fp_idcs = fp_idcs[do_restrict]

            # and again the resp. min distances:
            fp_dists_per_fp_idx = [all_min_dist_2[eps] for eps in
                                   eps_per_fp_idx]

            # for each frag point, the excess em points, i.e. the ones that the frag point does not cover because it would be too many:
            not_covered_ep_idcs_per_fp_idx = [
                # np.argpartition(dists, max_num_covered_per_fp)[max_num_covered_per_fp:]
                np.argsort(dists)[max_num_covered_per_fp:]
                for dists in fp_dists_per_fp_idx]

            # ...concatenated into one array: (could do unique on this one, but not sure if it is worth the effort)
            # all_eps_not_covered = np.concatenate([ eps[not_covered_ep_idcs_per_fp_idx[e]] for e,eps in enumerate(eps_per_fp_idx) ])
            all_eps_not_covered = []
            for e, eps in enumerate(eps_per_fp_idx):
                all_eps_not_covered.extend(
                    eps[not_covered_ep_idcs_per_fp_idx[e]])
            all_eps_not_covered = np.array(all_eps_not_covered)

            # get the em points that are actually covered:
            all_min_dist_2[all_eps_not_covered] = 100 * dist2

        covered_dummy = np.where(all_min_dist_2 < dist2)[0]

        # add them to the list of em points that have been covered in previous iterations of the for loop:
        # (and break the for loop if nothing changes for a while)
        if np.amin(covered_running[covered_dummy]) == 1:
            break
        covered_running[covered_dummy] = 1

        if do_restrict.shape[0] < 1:
            break

        # prepare for the next iteration of the for loop:
        # "throw away" the frag points that were previously processed for covering too many em points:
        dist_matrix_2[:, fp_idcs] = 100 * dist2
        min_dist_per_fp[fp_idcs] = 100 * dist2
        # which em points had the thrown-away frag points as their minima:
        tmp = []
        for eps in eps_per_fp_idx:
            tmp.extend(eps)
        tmp = np.array(tmp, dtype=np.int32)
        all_eps_to_reiterate = np.unique(tmp)
        # compute new min frag point indices and distances for these em points:
        min_dist_fp_indices[all_eps_to_reiterate] = [
            np.argmin(dist_matrix_2[rit]) for rit in all_eps_to_reiterate]
        all_min_dist_2 = np.array(
            [dist_matrix_2[i, min_dist_fp_indices[i]] for i in
             range(len(min_dist_fp_indices))])

    covered = np.where(covered_running == 1)[0]

    return covered


def get_single_frag_coverage_mixed_v1(
    em_tree, em_points, seg_length_per_em_point, frag_points, dist2=625,
    resmp_factor=1, verbose=False, mylib=None, max_matches=6):
    if len(frag_points) > 250:
        return get_single_frag_coverage_ilp_v1(
            em_points, seg_length_per_em_point,
            frag_points, dist2=dist2, verbose=verbose,
            mylib=mylib, max_matches=max_matches)
    else:
        return get_single_frag_coverage_default(
            em_points, em_tree, seg_length_per_em_point,
            frag_points, dist2=dist2, resmp_factor=resmp_factor,
            verbose=verbose)


def get_single_frag_coverage_mixed_v2(
    em_tree, em_points, seg_length_per_em_point, frag_points, dist2=625,
    dist=25, resmp_factor=1, verbose=False, max_matches=6):
    if len(frag_points) > 150:
        cov = get_single_frag_coverage_ilp_v2(
            em_tree, len(em_points), seg_length_per_em_point, frag_points,
            dist=dist, verbose=verbose, max_matches=max_matches)
    else:
        cov = get_single_frag_coverage_default(
            em_points, em_tree, seg_length_per_em_point,
            frag_points, dist2=625, resmp_factor=resmp_factor,
            verbose=verbose)
    return cov


# v1 vs v2: v1 uses dist matrix, v2 computes kd tree
def get_single_frag_coverage_ilp_v1(em_points, seg_length_per_em_point,
                                    frag_points, dist2=625, verbose=False,
                                    mylib=None, max_matches=6):
    gc.disable()

    dist_matrix_2 = np.ascontiguousarray(
        np.array(cdist(em_points, frag_points,
                       'sqeuclidean'), dtype=np.float32))

    num_em = np.int32(len(em_points))
    num_frag = np.int32(len(frag_points))

    max_dist = np.float32(dist2)

    sol = np.zeros(num_em, dtype=np.float64)

    max_matches = np.int32(max_matches)
    success = mylib.solve(dist_matrix_2, num_em, num_frag,
                          max_dist, sol, max_matches)

    covered = [idx for idx in range(num_em) if sol[idx] > 0.5]

    gc.enable()
    return covered


def get_single_frag_coverage_ilp_v2(em_tree, num_em, seg_length_per_em_point,
                                    frag_points, dist=25, verbose=False,
                                    max_matches=6):
    max_dist = np.float32(dist)
    num_em = np.int32(num_em)
    max_neigh = np.int32(15)
    max_matches = np.int32(max_matches)

    (dists, inds) = em_tree.query(frag_points, k=max_neigh,
                                  distance_upper_bound=max_dist)

    covered = get_single_frag_coverage_ilp_v2_B(dists, inds,
                                                num_em, max_dist,
                                                max_matches)

    return covered


# @jit(nopython=True)
def get_single_frag_coverage_ilp_v2_B(dists, inds, num_em, max_dist,
                                      max_matches):
    distsT = []
    indsT = []
    num_neigh = []
    for ids, ds in zip(inds, dists):
        idx = -1
        for idx, (i, d) in enumerate(zip(ids, ds)):
            if d < max_dist:
                distsT.append(d)
                indsT.append(i)
            else:
                idx -= 1
                break
        if idx >= 0:
            num_neigh.append(idx + 1)
    dists = np.array(distsT, dtype=np.float64)
    inds = np.array(indsT, dtype=np.int64)
    num_neigh = np.array(num_neigh, dtype=np.int32)

    if num_neigh.shape[0] == 0:
        return [np.int32(x) for x in range(0)]  # = []
    if np.max(num_neigh) == 0:
        return [np.int32(x) for x in range(0)]  # = []

    num_frag = np.int32(len(num_neigh))

    sol = np.zeros(num_em, dtype=np.float64)

    success = solve2(dists.ctypes, num_em, num_frag, max_dist,
                     inds.ctypes, num_neigh.ctypes,
                     sol.ctypes, max_matches)

    covered = [idx for idx in range(num_em) if sol[idx] > 0.5]

    return covered


@jit(nopython=True)
def get_single_frag_coverage_ilp_v2_C(dists, inds, num_em, max_dist=25,
                                      max_matches=6):
    max_neigh = np.int32(25)
    num_neigh = np.array([min(len(l), max_neigh) for l in dists],
                         dtype=np.int32)
    if np.max(num_neigh) == 0:
        return [np.int32(x) for x in range(0)]  # = []

    dists = np.ascontiguousarray(
        np.array([d for ds in dists for d in ds[:max_neigh]]))
    inds = np.ascontiguousarray(
        np.array([i for ids in inds for i in ids[:max_neigh]], dtype=np.int64))

    num_frag = np.int32(np.count_nonzero(num_neigh))
    num_neigh = np.ascontiguousarray(num_neigh[num_neigh != 0], dtype=np.int32)

    sol = np.zeros(num_em, dtype=np.float64)

    max_matches = np.int32(max_matches)
    min_dists = np.zeros(num_em, dtype=np.float64)
    success = solve2(dists.ctypes, num_em, num_frag, max_dist,
                     inds.ctypes, num_neigh.ctypes,
                     sol.ctypes, min_dists.ctypes, max_matches)

    covered = [idx for idx in range(num_em) if sol[idx] > 0.5]

    return covered


def get_frag_coverage_ilp_v3(em_tree, num_em, seg_length_per_em_point,
                             frags_points, dist=25, verbose=False,
                             mylib=None, max_matches=6):
    print("solving all frags at once")
    gc.disable()

    max_dist = np.int32(dist)
    max_neigh = np.int32(50)
    num_frags_points = [fp.shape[0] for fp in frags_points]
    num_frags = np.int32(len(frags_points))
    frags_points = np.concatenate(frags_points, axis=0)
    (inds, dists) = em_tree.query_radius(frags_points, max_dist,
                                         return_distance=True,
                                         sort_results=True)

    num_neigh = np.ascontiguousarray(
        np.array([min(len(l), max_neigh) for l in dists], dtype=np.int32))
    dists = np.ascontiguousarray(
        np.array([d for ds in dists for d in ds[:max_neigh]]))
    inds = np.ascontiguousarray(
        np.array([i for ids in inds for i in ids[:max_neigh]], dtype=np.int64))

    num_em = np.int32(num_em)
    num_fp_total = np.int32(len(frags_points))
    num_frags_points = np.ascontiguousarray(
        np.array(num_frags_points, dtype=np.int32))
    max_dist = np.float32(dist)

    sol = np.zeros(num_em * num_frags, dtype=np.float64)

    max_matches = np.int32(max_matches)
    min_dists = np.zeros(num_em * num_frags, dtype=np.float64)

    success = mylib.solve3(dists, num_em, num_frags, num_fp_total,
                           num_frags_points, max_dist,
                           inds, num_neigh,
                           sol, min_dists, max_matches)

    if not success:
        print("-------------------INVALID SOLUTION-------------------")

    covered = [np.where(sol[num_em * frag:num_em * (frag + 1)] > 0.5)[0]
               for frag in range(num_frags)]

    gc.enable()
    return covered


def get_coverage(em_points, seg_length_per_em_point, frags_points, dist=25,
                 resmp_factor=1,
                 skel_ids=None, verbose=False, version=None, max_matches=6,
                 em_tree=None):
    em_num_points = len(em_points)
    dist2 = dist ** 2

    print("selecting version:", version, max_matches, dist, dist2)
    if not version or \
        not ('ilp' in version or 'mixed' in version):
        covered_points = [get_single_frag_coverage_default(
            em_points, em_tree, seg_length_per_em_point,
            frag_points, dist2=dist2, resmp_factor=resmp_factor,
            verbose=verbose)
            for frag_points in frags_points]
    elif version == "mixed_v1":
        covered_points = [get_single_frag_coverage_mixed_v1(
            em_tree, em_points, seg_length_per_em_point,
            frag_points, dist2=dist2, resmp_factor=1, verbose=verbose,
            mylib=mylib, max_matches=max_matches)
            for frag_points in frags_points]
    elif version == "mixed_v2":
        covered_points = [get_single_frag_coverage_mixed_v2(
            em_tree, em_points, seg_length_per_em_point,
            frag_points, dist2=dist2, dist=dist, resmp_factor=1,
            verbose=verbose, max_matches=max_matches)
            for frag_points in frags_points]
    elif version == "ilp_v1":
        covered_points = [get_single_frag_coverage_ilp_v1(
            em_points, seg_length_per_em_point,
            frag_points, dist2=dist2, verbose=verbose,
            mylib=mylib, max_matches=max_matches)
            for frag_points in frags_points]
    elif version == "ilp_v2":
        covered_points = [get_single_frag_coverage_ilp_v2(
            em_tree, len(em_points), seg_length_per_em_point,
            frag_points, dist=dist, verbose=verbose,
            max_matches=max_matches)
            for frag_points in frags_points]
    elif version == "ilp_v3":
        covered_points = get_frag_coverage_ilp_v3(
            em_tree, len(em_points), seg_length_per_em_point,
            frags_points, dist=dist, verbose=verbose,
            mylib=mylib, max_matches=max_matches)
        if skel_ids is None:
            skel_ids = len(frags_points)

        coverages = 100 * np.array(
            [np.sum(seg_length_per_em_point[closest_em_points])
             for closest_em_points in covered_points]) / em_num_points
        return coverages, covered_points
    else:
        raise RuntimeError("invalid version for get_single_coverage")

    if skel_ids is None:
        skel_ids = len(frags_points)

    coverages = np.array([100 * np.sum(
        seg_length_per_em_point[closest_em_points]) / em_num_points
                          for closest_em_points in covered_points])

    return coverages, covered_points


def compute_frag_coverage_scores(args, base_skel_path, em_points, em_tree,
                                 em_num_points, seg_length_per_em_point,
                                 lm_vols, v, lm_vol, lm_vol_skel_idcs,
                                 nblast_dict, score_thresh):
    # TODO fix len(lm_vols) for --both-sides
    print("processing %i-th of %i vols %s" % (v, len(lm_vols), lm_vol),
          flush=True)

    start = time.time()
    # prune fragments by nblast_score
    skel_ids = np.array(lm_vol_skel_idcs[lm_vol])
    skel_scores_all_variants = np.array(
        [nblast_dict[lm_vol + "_" + str(skel_id)] for skel_id in skel_ids])
    skel_scores_max = np.amax(skel_scores_all_variants, axis=1)

    # weed out low nblast frags:
    select = np.where(skel_scores_max > args.nblast_score_thresh)[0]
    print(
        "selected %i out of %i frags by nblast score (max across variants)" % (
        len(select), len(skel_ids)), flush=True)
    if len(select) == 0:
        return None
    skel_scores_all_variants = skel_scores_all_variants[select]
    skel_scores_max = skel_scores_max[select]
    skel_ids = skel_ids[select]

    # load lm skels
    lm_vol_base = lm_vol.split('-', 1)[0]
    frag_swcs = [os.path.join(
        base_skel_path,
        lm_vol_base,
        lm_vol + '_' + str(skel_id) + ".swc")
        for skel_id in skel_ids]

    print("frags_points")
    if "flipped" in frag_swcs[0]:
        base_skel_pathT = "/nrs/funke/hirschp/flymatch/ppp_all_cat_2_3/setup22_200511_00/test/400000/skeletons_vote_instances_rm_by_bbox_20_min_length_20_cropped_flipped"
    else:
        base_skel_pathT = "/nrs/funke/hirschp/flymatch/ppp_all_cat_2_3/setup22_200511_00/test/400000/skeletons_vote_instances_rm_by_bbox_20_min_length_20_cropped"
    with open(os.path.join(
        base_skel_pathT + "_pickled",
        lm_vol_base,
        lm_vol + ".pickle"), 'rb') as f:
        frags_pointsT = pickle.load(f)
    frags_points = [frags_pointsT[skel_id] for skel_id in skel_ids]
    frags_num_points = np.array([fp.shape[0] for fp in frags_points])
    print("frags_points[0][0] %s" % frags_points[0][0], flush=True)

    # weed out small low nblast frags: TODO: threshs via args
    if args.adaptive_score_thresh_factor > 0:
        select = np.where(
            skel_scores_max > args.adaptive_score_thresh_offset - frags_num_points / args.adaptive_score_thresh_factor)[
            0]
        print("selected %i out of %i frags by num points and nblast score" % (
        len(select), len(skel_ids)), flush=True)
        if len(select) == 0:
            return None
        skel_scores_all_variants = skel_scores_all_variants[select]
        skel_scores_max = skel_scores_max[select]
        skel_ids = skel_ids[select]
        frags_points = [frags_points[i] for i in select]
        frags_num_points = frags_num_points[select]

    # transform raw nblast values to scores:
    skel_scores_all_variants[
        skel_scores_all_variants > args.max_nblast_score] = args.max_nblast_score
    skel_scores_all_variants *= args.nblast_factor
    skel_scores_all_variants -= args.min_nblast_score
    skel_scores_max = np.amax(skel_scores_all_variants, axis=1)

    # weed out by small num frag points and low upper bound to unary:
    skel_rel_cable_lengths = 100 * 1.4 * frags_num_points / em_num_points
    skel_upper_bound_unaries = np.multiply(skel_scores_max,
                                           skel_rel_cable_lengths)
    if args.verbose:
        print("all skel_ids: %s" % skel_ids)
        print("skel_upper_bound_unaries: %s" % skel_upper_bound_unaries)
        print("frags_num_points: %s" % frags_num_points)
    select = np.where((skel_upper_bound_unaries > args.min_unary) & (
            frags_num_points >= args.min_num_frag_points_hack))[0]
    print(
        "selected %i out of %i frags by num points and upper bound to unary" % (
        len(select), len(skel_ids)), flush=True)
    if len(select) == 0:
        return None
    skel_scores_all_variants = skel_scores_all_variants[select]
    skel_scores_max = skel_scores_max[select]
    skel_ids = skel_ids[select]
    frags_points = [frags_points[i] for i in select]

    end = time.time()
    print("preliminaries, incl. loading swcs, took %s seconds" % (end - start),
          flush=True)

    lower_bound_cov_score = -1 * np.sum(skel_upper_bound_unaries)
    print("lower bound to cov score: %d" % lower_bound_cov_score)
    ######### TODO: one dict per score variant
    if lower_bound_cov_score > args.cov_score_thresh:
        # cov_score_dict[em][lm_vol] = utils.make_cov_score_entry(all_skel_ids=skel_ids,
        #                                                         all_skel_scores=skel_scores_max)
        print(
            "aborting %i-th of %i vols %s due to lower bound cov_score %d > %d" % (
                v, len(lm_vols), lm_vol, lower_bound_cov_score,
                args.cov_score_thresh), flush=True)
        print("individual upper bound unaries: %s" % skel_upper_bound_unaries)
        return None

    # get individual coverages
    start = time.time()

    print("skel_ids before get_coverage: %s" % skel_ids)
    skel_coverages, covered_points = get_coverage(em_points,
                                                  seg_length_per_em_point,
                                                  frags_points,
                                                  dist=args.max_cov_dist,
                                                  resmp_factor=args.resmp_factor,
                                                  skel_ids=skel_ids,
                                                  verbose=args.verbose,
                                                  version=args.get_cov_version,
                                                  max_matches=args.max_matches,
                                                  em_tree=em_tree)

    if args.verbose:
        for i, sid in enumerate(skel_ids):
            print("frag %i covers %i out of %i points, %f" % (
            sid, covered_points[i].shape[0], em_num_points, skel_coverages[i]))

    # weed out by 0 coverage:
    select = np.where(skel_coverages > 0.1)[0]
    print("selected %i out of %i frags >0 coverage" % (
    len(select), len(skel_ids)), flush=True)

    if len(select) == 0:
        return None
    skel_scores_all_variants = skel_scores_all_variants[select]
    skel_ids_saved = skel_ids[select]
    skel_coverages_saved = skel_coverages[select]
    frags_points_saved = [frags_points[i] for i in select]
    covered_points_saved = [covered_points[i] for i in select]

    end = time.time()
    print("single skel coverages took %s seconds" % (end - start), flush=True)
    if args.verbose:
        print("single skel coverages: %s" % skel_coverages, flush=True)

    return skel_scores_all_variants, skel_ids_saved, skel_coverages_saved, frags_points_saved, covered_points_saved


def filter_frags(args, score_variant_indcs, cov_score_dicts,
                 skel_scores_all_variants, skel_ids_saved,
                 skel_coverages_saved, frags_points_saved,
                 covered_points_saved, score_thresh, skel_color_folder,
                 em, seg_length_per_em_point, em_num_points,
                 lm_vols, v, lm_vol, dist_2, output_folder):
    filtered = []
    # loop over all score variants:
    for idx, svi in enumerate(score_variant_indcs):
        cov_score_dict = cov_score_dicts[idx]
        skel_scores = skel_scores_all_variants[:, svi]

        # select anew by threshold on specific score:
        select = np.where(skel_scores > score_thresh)[0]
        print("selected %i out of %i frags by specific score threshold" % (
        len(select), len(skel_ids_saved)), flush=True)
        if len(select) == 0:
            filtered.append(None)
            continue
        skel_scores = skel_scores[select]
        skel_ids = skel_ids_saved[select]
        skel_coverages = skel_coverages_saved[select]

        frags_points = [frags_points_saved[i] for i in select]
        covered_points = [covered_points_saved[i] for i in select]

        aggregate_coverage = 100 * np.sum(seg_length_per_em_point[
                                              pd.unique(
                                                  [i for cp in covered_points
                                                   for i in
                                                   cp])]) / em_num_points
        if aggregate_coverage < args.min_aggregate_coverage:
            if lm_vol not in cov_score_dict[em]:
                cov_score_dict[em][lm_vol] = utils.make_cov_score_entry(
                    all_skel_ids=skel_ids,
                    all_skel_scores=skel_scores,
                    all_skel_coverages=skel_coverages,
                    all_aggregate_coverage=aggregate_coverage)
            print(
                "aborting %i-th of %i vols %s due to aggregate coverage %d < %d" % (
                v, len(lm_vols), lm_vol, aggregate_coverage,
                args.min_aggregate_coverage), flush=True)
            print("individual coverages: %s" % skel_coverages)
            filtered.append(None)
            continue

        # weed out low unary frags:
        skel_unaries = np.multiply(skel_coverages, skel_scores)
        select = np.where(skel_unaries > args.min_unary)[0]
        print("selected %i out of %i frags by unary" % (
        len(select), len(skel_ids)), flush=True)
        if len(select) == 0:
            filtered.append(None)
            continue
        skel_scores = skel_scores[select]
        skel_ids = skel_ids[select]
        skel_coverages = skel_coverages[select]
        skel_unaries = skel_unaries[select]

        frags_points = [frags_points_saved[i] for i in select]
        covered_points = [covered_points_saved[i] for i in select]

        # more accurate lower bound to cov_score:
        lower_bound_cov_score = -1 * np.sum(skel_unaries)
        print(
            "more accurate lower bound to cov score: %d" % lower_bound_cov_score)
        if lower_bound_cov_score > args.cov_score_thresh:
            if lm_vol not in cov_score_dict[em]:
                cov_score_dict[em][lm_vol] = utils.make_cov_score_entry(
                    all_skel_ids=skel_ids,
                    all_skel_scores=skel_scores,
                    all_skel_coverages=skel_coverages)
            print(
                "aborting %i-th of %i vols %s due to lower bound cov_score %d > %d" % (
                v, len(lm_vols), lm_vol, lower_bound_cov_score,
                args.cov_score_thresh), flush=True)
            filtered.append(None)
            continue

        # transform skel scores according to coverages:
        t = np.vectorize(utils.transform_score_by_coverage)
        if args.verbose:
            print("skel scores before transform: %s" % skel_scores)
        skel_scores = t(skel_scores, skel_coverages)
        if args.verbose:
            print("skel scores after transform: %s" % skel_scores)

        # use *solely* for adjusted_coverage in bounds, *not* to get unaries in MRF:
        all_skel_scores_per_em_pt = np.zeros(
            (skel_scores.shape[0], em_num_points))
        for i, cps in enumerate(covered_points):
            all_skel_scores_per_em_pt[i][cps] = skel_scores[i]

        # use for adjusted_coverage in unaries:
        all_skel_coverages_per_em_pt = np.zeros(
            (skel_scores.shape[0], em_num_points))
        for i, cps in enumerate(covered_points):
            all_skel_coverages_per_em_pt[i][cps] = skel_coverages[i]

        filtered.append([skel_scores, skel_ids, skel_coverages, frags_points,
                         covered_points, all_skel_scores_per_em_pt,
                         all_skel_coverages_per_em_pt])

    return cov_score_dicts, filtered


def select_best_coverage_frags(args, score_variant_indcs, cov_score_dicts,
                               data_filtered_all_variants,
                               score_thresh, skel_color_folder,
                               em, seg_length_per_em_point, em_num_points,
                               em_swc_file,
                               lm_vols, v, lm_vol, dist_2, output_folder,
                               lower_bounds, upper_bounds,
                               lower_bounds_other, upper_bounds_other,
                               is_flipped):
    # loop over all score variants:
    for idx, svi in enumerate(score_variant_indcs):
        if data_filtered_all_variants[idx] is None:
            print("aborting, empty scores")
            continue
        cov_score_dict = cov_score_dicts[idx]
        skel_scores, skel_ids, skel_coverages, _, covered_points, _, all_skel_coverages_per_em_pt = \
        data_filtered_all_variants[idx]

        if lower_bounds is not None:
            lower_bound = lower_bounds[idx]
            lower_bound_other = lower_bounds_other[idx]
            upper_bound = upper_bounds[idx]
            upper_bound_other = upper_bounds_other[idx]

            if lower_bound > upper_bound_other:
                print(
                    "aborting %i-th of %i vols %s due to bounds lower %d > %d upper other side" % (
                    v, len(lm_vols), lm_vol, lower_bound, upper_bound_other),
                    flush=True)
                continue
        else:
            upper_bound_other = 0

        # get colors
        start = time.time()
        if skel_color_folder is not None:
            color_json = os.path.join(skel_color_folder,
                                      "fragment_colors_" + lm_vol + ".json")
            with open(color_json, 'r') as f:
                color_dict = json.load(f)
            skel_colors = np.array(
                [eval(color_dict[lm_vol][str(skel_id)]) for skel_id in
                 skel_ids])
        else:
            raw_file = os.path.join(args.raw_path, lm_vol + ".zarr")
            inst_file = os.path.join(args.inst_path, lm_vol + ".hdf")
            skel_colors = utils.get_skel_colors(raw_file, inst_file,
                                                args.inst_key, skel_ids,
                                                bin_size=64,
                                                verbose=args.verbose)
        end = time.time()
        print("get_skel_colors took %s seconds" % (end - start), flush=True)

        start = time.time()
        (min_energy, best_skel_cluster) = utils.cluster_fragments_by_color(
            em,
            lm_vol,
            covered_points, all_skel_coverages_per_em_pt,
            em_num_points, seg_length_per_em_point, skel_colors,
            skel_coverages, skel_scores, upper_bound_other,
            max_dist_2=dist_2, bin_size=32, skel_ids=skel_ids,
            verbose=args.verbose, clustering_algo=args.clustering_algo)
        end = time.time()
        print("cluster_fragments_by_color took %s seconds" % (end - start))

        start = time.time()
        best_skel_ids = [skel_ids[i] for i in best_skel_cluster]
        print(lm_vol, best_skel_cluster, best_skel_ids, min_energy)

        best_skel_scores = [float(skel_scores[i]) for i in best_skel_cluster]
        best_skel_coverages = [float(skel_coverages[i]) for i in
                               best_skel_cluster]
        best_skel_colors = [[int(skel_colors[i][0]), int(skel_colors[i][1]),
                             int(skel_colors[i][2])] for i in best_skel_cluster]
        print(best_skel_cluster)
        best_aggregate_coverage = 100 * np.sum(seg_length_per_em_point[
                                                   pd.unique([i
                                                              for cp in
                                                              [covered_points[c]
                                                               for c in
                                                               best_skel_cluster]
                                                              for i in
                                                              cp])]) / em_num_points

        if lm_vol in cov_score_dict[em]:
            prev_score = cov_score_dict[em][lm_vol]['cov_score']
        else:
            prev_score = 0
        if min_energy < prev_score:
            cov_score_dict[em][lm_vol] = utils.make_cov_score_entry(
                skel_ids=best_skel_ids, skel_scores=best_skel_scores,
                skel_coverages=best_skel_coverages,
                skel_colors=best_skel_colors,
                min_energy=min_energy,
                aggregate_coverage=best_aggregate_coverage,
                all_skel_ids=skel_ids, all_skel_scores=skel_scores,
                all_skel_coverages=skel_coverages, all_skel_colors=skel_colors,
                all_aggregate_coverage=-1)
            cov_score_dict[em][lm_vol]['mirrored'] = is_flipped

            # debug best_skel coverage:
            if args.verbose:
                utils.write_debug_coverage_swc(
                    args, all_skel_coverages_per_em_pt, best_skel_cluster,
                    em_num_points, em_swc_file, output_folder, lm_vol, svi)

        # end debug best_skel coverage
        end = time.time()
        print("all the rest took %s seconds" % (end - start), flush=True)

    return cov_score_dicts


def get_bounds(args, score_variant_indcs,
               data_filtered_all_variants,
               em_num_points, seg_length_per_em_point):
    lower_bounds = []
    upper_bounds = []
    for idx, svi in enumerate(score_variant_indcs):
        if data_filtered_all_variants[idx] is None:
            upper_bounds.append(np.inf)
            lower_bounds.append(-np.inf)
            continue
        skel_scores, _, skel_coverages, _, covered_points, all_skel_scores_per_em_pt, _ = \
        data_filtered_all_variants[idx]
        unaries = [float(utils.get_unary(i, skel_coverages, skel_scores))
                   for i in range(len(skel_scores))]
        if args.verbose:
            print("unaries:")
        if args.verbose:
            print(unaries)
        upper = -np.amax(unaries)
        upper_bounds.append(upper)
        skel_coverages_adjusted = utils.get_adjusted_coverages_by_score(
            covered_points, all_skel_scores_per_em_pt, em_num_points,
            seg_length_per_em_point, verbose=args.verbose)
        unaries_adjusted = [
            float(utils.get_unary(i, skel_coverages_adjusted, skel_scores))
            for i in range(len(skel_scores))]
        if args.verbose:
            print("unaries adjusted:")
        if args.verbose:
            print(unaries_adjusted)
        lower = -np.sum(unaries_adjusted)
        lower_bounds.append(lower)
        if upper < lower:
            # sanity checks:
            if len(skel_scores) != len(skel_coverages):
                print("error! lengths don't match")
            for i in range(len(skel_scores)):
                agg = np.sum(skel_coverages_adjusted[:i + 1])
                cov = skel_coverages[i]
                if agg < cov:
                    print("error! agg %f < cov %f at idx %i" % (agg, cov, i))
                u = -unaries[i]
                l = -np.sum(unaries_adjusted[:i + 1])
                if u < l:
                    print(
                        "error! u %f < l %f at idx %i. cov %.2f agg adj_cov %.2f score %.2f avg score %.2f" % (
                        u, l, i, skel_coverages[i],
                        np.sum(skel_coverages_adjusted[:i + 1]), skel_scores[i],
                        np.average(skel_scores[:i + 1])))
            raise Exception(
                "Error: upper bound %.2f < lower bound %.2f" % (upper, lower))
        if args.verbose:
            print(
                "bound comp variant %i: scores %s coverages %s adjusted_cov %s adjusted_cov_agg %f upper %f lower %f" % (
                idx, skel_scores, skel_coverages, skel_coverages_adjusted,
                np.sum(skel_coverages_adjusted), upper, lower))

    return lower_bounds, upper_bounds


def visualize_best(args, em, lm_vols, lm_vol, cov_score_dict, score_variant_idx,
                   em_swc_file, output_folder):
    if args.test_lm_vol_name is not None:
        lm_vols = [args.test_lm_vol_name]
        show_n_best = 1
    else:
        lm_vols = list(cov_score_dict[em].keys())
        show_n_best = args.show_n_best

    for idx in range(min(show_n_best, len(lm_vols))):
        lm_vol = lm_vols[idx]
        skel_ids = eval(cov_score_dict[em][lm_vol]['skel_ids'])
        scores = eval(cov_score_dict[em][lm_vol]['nblast_scores'])
        coverages = eval(cov_score_dict[em][lm_vol]['coverages'])
        colors = np.array(eval(cov_score_dict[em][lm_vol]['colors']))
        cov_score = -1 * cov_score_dict[em][lm_vol]['cov_score']
        raw_file = args.raw_path + lm_vol + ".zarr"
        inst_file = args.inst_path + lm_vol + ".hdf"
        # save raw mip
        base_filename = "%s_sv_%i_csmatch_%i_cscore_%03d_%s" % (
        em, score_variant_idx, idx, cov_score, lm_vol)
        outfn_raw = os.path.join(output_folder, "%s_1_raw.png" % base_filename)
        # save masked raw mip
        outfn_masked_raw = os.path.join(output_folder,
                                        "%s_2_masked_raw.png" % base_filename)
        # save masked inst mip with nblast scores
        outfn_masked_inst = os.path.join(output_folder,
                                         "%s_4_masked_inst.png" % base_filename)
        # save lm skel
        # save em skel on masked raw mip
        outfn_em_on_masked_raw = os.path.join(output_folder,
                                              "%s_3_em_skel_on_masked_raw.png" % base_filename)

        if not os.path.exists(outfn_em_on_masked_raw):
            visualize_frags(
                raw_file,
                inst_file,
                args.inst_key,
                skel_ids,
                scores,
                coverages,
                em_swc_file,
                [outfn_raw, outfn_masked_raw, outfn_masked_inst,
                 outfn_em_on_masked_raw],
                skel_colors=colors
            )


def get_seg_length_per_em_point(em_points, parents, power=1, verbose=False):
    # parents in swc start counting at 1, so:
    parents[parents != -1] = parents[parents != -1] - 1
    seg_length_per_em_point = np.zeros(parents.shape)
    em_point_idcs = np.array(list(range(0, parents.shape[0])))
    if verbose:
        print(em_point_idcs)
    # for debug:
    last_seg_length = 1
    # iteratively:
    while True:
        if parents.shape[0] == 0:
            break
        unique_parents, parent_counts = np.unique(parents, return_counts=True)
        branching_points = unique_parents[parent_counts > 1]
        if verbose:
            print("branching point orig idcs: %s" % branching_points)

        branch_start_idcs = sorted(
            np.unique(np.append(np.where(np.isin(parents, branching_points))[0],
                                np.where(parents == -1)[0])))
        if verbose:
            print("branch start idcs %s" % branch_start_idcs)
        seg_lengths = np.array([bp - branch_start_idcs[i]
                                for i, bp in enumerate(branch_start_idcs[1:])] +
                               [parents.shape[0] - branch_start_idcs[-1]]) + 1
        if verbose:
            print("seg lengths %s" % seg_lengths)
        if seg_lengths.shape[0] <= 5:
            # try chop off ends:
            total_chopped = 0
            chop_idcs_lengths = []
            for i in range(seg_lengths.shape[0]):
                start_pt_idx = branch_start_idcs[i]
                start_orig_pt_idx = em_point_idcs[start_pt_idx]
                end_pt_idx = branch_start_idcs[i + 1] - 1 if i + 1 < len(
                    branch_start_idcs) else len(parents) - 1
                end_orig_pt_idx = em_point_idcs[end_pt_idx]
                chop_length = min(seg_lengths[i], 50)
                total_chopped += chop_length
                chop_idcs = range(end_orig_pt_idx - chop_length,
                                  end_orig_pt_idx + 1)
                chop_idcs_lengths = chop_idcs_lengths + [
                    (chop_idcs, chop_length)]
            # three main branches get equal treatment:
            seg_length_per_em_point[em_point_idcs] = max(
                last_seg_length,
                ((em_point_idcs.shape[0] - total_chopped) / seg_lengths.shape[
                    0]) ** power)
            for (chop_idcs, chop_length) in chop_idcs_lengths:
                seg_length_per_em_point[chop_idcs] = chop_length ** power
            break

        seg_is_terminal = np.zeros(seg_lengths.shape)
        max_seg_length = np.amax(seg_lengths)

        for i in range(seg_lengths.shape[0]):
            # get parents of terminal points:
            start_pt_idx = branch_start_idcs[i]
            start_orig_pt_idx = em_point_idcs[start_pt_idx]
            end_pt_idx = branch_start_idcs[i + 1] - 1 if i + 1 < len(
                branch_start_idcs) else len(parents) - 1
            end_orig_pt_idx = em_point_idcs[end_pt_idx]
            # if parents[start_pt_idx]==-1 or end_orig_pt_idx not in list(branching_points):
            # not allowing delete root for now because awful branch re-organization necessary; TODO
            if end_orig_pt_idx not in list(branching_points):
                seg_is_terminal[i] = 1
                if verbose:
                    print("seg %i of length %i is terminal: %i" % (
                    i, seg_lengths[i], seg_is_terminal[i]))
            else:
                if verbose:
                    print("seg %i of length %i is terminal: %i" % (
                    i, seg_lengths[i], seg_is_terminal[i]))
                seg_lengths[i] = 2 * max_seg_length

        # remove shortest *terminal* seg and assign it it's length
        min_seg_idx = np.argmin(seg_lengths)
        start = branch_start_idcs[min_seg_idx]
        stop = branch_start_idcs[min_seg_idx + 1] - 1 if min_seg_idx + 1 < len(
            branch_start_idcs) else len(parents) - 1
        min_seg_pt_idcs = range(start, stop + 1)
        stop_orig_idx = em_point_idcs[stop]
        if parents[start] == -1:
            if verbose:
                print("deleting root branch, stop orig idx: %i" % stop_orig_idx)
            # then we're deleting the root branch, try to merge two offsprings: # need to set new roots:
            if stop_orig_idx in list(branching_points):
                # get end point of 1st offspring:
                nextstop = branch_start_idcs[
                               min_seg_idx + 2] - 1 if min_seg_idx + 2 < len(
                    branch_start_idcs) else len(parents) - 1
                maxparent = np.amax(parents)
                need_new_parents = np.where(parents == stop_orig_idx)[0]
                if verbose:
                    print("need new parents: %s" % need_new_parents)
                parents[need_new_parents] = range(maxparent + 1, maxparent + 1 +
                                                  need_new_parents[1:].shape[
                                                      0])  # -1

        if verbose:
            print(
                "shortest seg len %i, current idcs %s, start %i stop %i stop_orig %i, parent %i" % (
                    seg_lengths[min_seg_idx], min_seg_pt_idcs, start, stop,
                    stop_orig_idx, parents[start]))

        orig_point_idcs = em_point_idcs[min_seg_pt_idcs]
        if verbose:
            print("shortest seg len %i, orig idcs %s" % (
            seg_lengths[min_seg_idx], orig_point_idcs))
        seg_length_per_em_point[orig_point_idcs] = seg_lengths[
                                                       min_seg_idx] ** power
        last_seg_length = seg_lengths[min_seg_idx] ** power

        parents = np.delete(parents, min_seg_pt_idcs)
        em_point_idcs = np.delete(em_point_idcs, min_seg_pt_idcs)

        if start == stop + 1:
            print("error: 0-length segment")
            break

    print("seg length range %i .. %i" % (
    np.amin(seg_lengths), np.amax(seg_lengths)))
    return seg_length_per_em_point


def main():

    args = get_arguments()

    em = args.em
    lm_id = args.lm_id
    em_id = args.em_id
    base_skel_path = os.path.join(args.base_lm_swc_path, lm_id)
    min_aggregate_coverage = args.min_aggregate_coverage
    verbose = args.verbose
    nblast_json_path = args.nblast_json_path
    output_folder = args.nblast_json_path

    if args.both_sides:
        skel_color_folder_flipped = args.skel_color_folder
        if args.skel_color_folder is not None:
            skel_color_folder = skel_color_folder_flipped.replace("_flipped",
                                                                  "")
        else:
            skel_color_folder = None
    else:
        skel_color_folder = args.skel_color_folder

    if args.both_sides:
        em_swc_file_flipped = os.path.join(args.em_swc_base_folder,
                                           em_id,
                                           em + ".swc")
        em_swc_file = em_swc_file_flipped.replace("_flipped", "")
    else:
        em_swc_file = os.path.join(args.em_swc_base_folder,
                                   em_id,
                                   em + ".swc")
    nblast_json = os.path.join(nblast_json_path,
                               "nblastScores_" + em + "_flipped.json")
    score_thresh = args.nblast_score_thresh * args.nblast_factor - args.min_nblast_score

    # get skels and nblast scores from nblast dict
    with open(nblast_json.replace("_flipped", ""), 'r') as f:
        nblast_dict = json.load(f)[em]
        print(nblast_json.replace("_flipped", ""))
    if args.both_sides:
        with open(nblast_json, 'r') as f:
            nblast_dict_flipped = json.load(f)[em]
            print(nblast_json)

    # which lm to compute?
    lm_vol_skel_idcs = {}
    for lm_name in nblast_dict.keys():
        lm_name, skel_id = lm_name.rsplit('_', 1)
        lm_vol_skel_idcs.setdefault(lm_name, []).append(int(skel_id))
        lm_vols = lm_vol_skel_idcs.keys()

    if args.both_sides:
        lm_vol_skel_idcs_flipped = {}
        for lm_name in nblast_dict_flipped.keys():
            lm_name, skel_id = lm_name.rsplit('_', 1)
            lm_vol_skel_idcs_flipped.setdefault(lm_name, []).append(
                int(skel_id))
            lm_vols_flipped = lm_vol_skel_idcs_flipped.keys()

        # sorting not necessary, for debugging, I got non-deterministic order otherwise
        lm_vols_total = sorted(set().union(lm_vol_skel_idcs.keys(),
                                           lm_vol_skel_idcs_flipped.keys()))
    else:
        lm_vols_total = lm_vol_skel_idcs.keys()

    # load em
    em_points, em_num_points, seg_length_per_em_point = \
        load_em(args, em_swc_file, output_folder)
    if args.both_sides:
        em_points_flipped, em_num_points_flipped, seg_length_per_em_point_flipped = \
            load_em(args, em_swc_file_flipped, output_folder)

    # score_variant_indcs = [ args.score_index ] if args.score_index>-1 else range(len(eval(next(iter(nblast_dict.values())))))
    score_variant_indcs = [
        args.score_index] if args.score_index > -1 else range(
        len(next(iter(nblast_dict.values()))))

    cov_score_jsons = [os.path.join(nblast_json_path,
                                    "cov_scores_" + em + "_" + str(
                                        svi) + ".json")
                       for svi in score_variant_indcs]
    print("cov scores stored at %s" % cov_score_jsons)
    cov_score_dicts = []
    for cov_score_json in cov_score_jsons:
        if os.path.isfile(cov_score_json):
            with open(cov_score_json, 'r') as f:
                cov_score_dicts.append(json.load(f))
        else:
            cov_score_dicts.append({})
    # cov_score_dicts = [ {} for cov_score_json in cov_score_jsons  ]

    # TODO: compute intersection of lm_keys lists instead of just using list from last dict
    for cov_score_dict in cov_score_dicts:
        lm_keys = []
        em_keys = list(cov_score_dict.keys())
        if em in em_keys:
            lm_keys = list(cov_score_dict[em].keys())
        else:
            cov_score_dict[em] = {}

    must_do_vols = []
    if args.must_do_list is not None:
        if args.both_sides:
            raise RuntimeError("not tested with --both-sides")
        cmd = "from %s import %s" % (args.must_do_list, args.must_do_list)
        exec(cmd)
        must_do = eval(args.must_do_list)
        must_do_scs = [lm_sc_must for (em_must, lm_sc_must) in must_do if
                       em_must == em]
        must_do_vols = [lm_vol for lm_vol in lm_vols_total for sc in must_do_scs
                        if sc in lm_vol]
        lm_vols_total = lm_vols_total + must_do_vols
        print("em %s must do %s" % (em, must_do_vols))

    if args.test_lm_vol_name is not None:
        lm_vols = [args.test_lm_vol_name]
        lm_vols_flipped = [args.test_lm_vol_name]
        lm_vols_total = [args.test_lm_vol_name]

    # load ilp solver
    setup_ilp_solver(args)
    # maybe create em tree
    em_tree = maybe_em_tree(args, em_points)
    if args.both_sides:
        em_tree_flipped = maybe_em_tree(args, em_points_flipped)

    # iterate over all lm volumes
    for v, lm_vol in enumerate(lm_vols_total):
        # skip if already computed
        if lm_vol in lm_keys:
            print("lm %s already in dict for em %s" % (lm_vol, em), flush=True)
            if args.test_lm_vol_name is None:
                continue

        # hack to save on the fly:
        if (v % 1000 == 1):
            for i, svi in enumerate(score_variant_indcs):
                print("saving cov scores at %s" % cov_score_jsons[i])
                # cov_score_dicts[i] = eval(str(cov_score_dicts[i])) # to get rid of numpy values not dumpable to json
                cov_score_dicts[i][em] = OrderedDict(
                    sorted(cov_score_dicts[i][em].items(),
                           key=lambda i: i[1]['cov_score']))
                with open(cov_score_jsons[i], 'w') as fp:
                    json.dump(cov_score_dicts[i], fp, indent=4)

        if lm_vol not in lm_vols:
            cov_ret = None
        else:
            cov_ret = compute_frag_coverage_scores(
                args, base_skel_path.replace("_flipped", ""), em_points,
                em_tree,
                em_num_points, seg_length_per_em_point,
                lm_vols, v, lm_vol, lm_vol_skel_idcs,
                nblast_dict, score_thresh)
            if cov_ret is not None:
                skel_scores_all_variants, skel_ids_saved, skel_coverages_saved, frags_points_saved, covered_points_saved = cov_ret

        if args.both_sides:
            if lm_vol not in lm_vols_flipped:
                cov_ret_flipped = None
            else:
                cov_ret_flipped = compute_frag_coverage_scores(
                    args, base_skel_path, em_points_flipped, em_tree_flipped,
                    em_num_points_flipped, seg_length_per_em_point_flipped,
                    lm_vols_flipped, v, lm_vol, lm_vol_skel_idcs_flipped,
                    nblast_dict_flipped, score_thresh)
                if cov_ret_flipped is not None:
                    skel_scores_all_variants_flipped, skel_ids_saved_flipped, skel_coverages_saved_flipped, frags_points_saved_flipped, covered_points_saved_flipped = cov_ret_flipped
        else:
            cov_ret_flipped = None

        if cov_ret is not None:
            cov_score_dicts, data_filtered_all_variants = filter_frags(
                args, score_variant_indcs, cov_score_dicts,
                skel_scores_all_variants, skel_ids_saved,
                skel_coverages_saved, frags_points_saved,
                covered_points_saved, score_thresh, skel_color_folder,
                em, seg_length_per_em_point, em_num_points,
                lm_vols, v, lm_vol, args.max_cov_dist ** 2, output_folder)
            if all(v is None for v in data_filtered_all_variants):
                cov_ret = None

        if cov_ret_flipped is not None:
            cov_score_dicts, data_filtered_all_variants_flipped = filter_frags(
                args, score_variant_indcs, cov_score_dicts,
                skel_scores_all_variants_flipped, skel_ids_saved_flipped,
                skel_coverages_saved_flipped, frags_points_saved_flipped,
                covered_points_saved_flipped, score_thresh,
                skel_color_folder_flipped,
                em, seg_length_per_em_point_flipped, em_num_points_flipped,
                lm_vols_flipped, v, lm_vol, args.max_cov_dist ** 2,
                output_folder)
            if all(v is None for v in data_filtered_all_variants_flipped):
                cov_ret_flipped = None

        if cov_ret_flipped is None and cov_ret is None:
            pass
        elif cov_ret_flipped is not None and cov_ret is not None:
            lower_bounds, upper_bounds = get_bounds(
                args, score_variant_indcs, data_filtered_all_variants,
                em_num_points, seg_length_per_em_point)
            print(em, lm_vol, "upper, lower bounds", upper_bounds, lower_bounds)

            lower_bounds_flipped, upper_bounds_flipped = get_bounds(
                args, score_variant_indcs, data_filtered_all_variants_flipped,
                em_num_points_flipped, seg_length_per_em_point_flipped)
            print(em, lm_vol, "upper, lower bounds", upper_bounds_flipped,
                  lower_bounds_flipped)

            cov_score_dicts = select_best_coverage_frags(
                args, score_variant_indcs, cov_score_dicts,
                data_filtered_all_variants,
                score_thresh, skel_color_folder,
                em, seg_length_per_em_point, em_num_points, em_swc_file,
                lm_vols, v, lm_vol, args.max_cov_dist ** 2, output_folder,
                lower_bounds, upper_bounds,
                lower_bounds_flipped, upper_bounds_flipped, False)

            cov_score_dicts = select_best_coverage_frags(
                args, score_variant_indcs, cov_score_dicts,
                data_filtered_all_variants_flipped,
                score_thresh, skel_color_folder_flipped,
                em, seg_length_per_em_point_flipped, em_num_points_flipped,
                em_swc_file_flipped,
                lm_vols_flipped, v, lm_vol, args.max_cov_dist ** 2,
                output_folder,
                lower_bounds_flipped, upper_bounds_flipped,
                lower_bounds, upper_bounds, True)
        elif cov_ret_flipped is None:
            cov_score_dicts = select_best_coverage_frags(
                args, score_variant_indcs, cov_score_dicts,
                data_filtered_all_variants,
                score_thresh, skel_color_folder,
                em, seg_length_per_em_point, em_num_points, em_swc_file,
                lm_vols, v, lm_vol, args.max_cov_dist ** 2, output_folder,
                None, None, None, None, False)
        elif cov_ret is None:
            cov_score_dicts = select_best_coverage_frags(
                args, score_variant_indcs, cov_score_dicts,
                data_filtered_all_variants_flipped,
                score_thresh, skel_color_folder_flipped,
                em, seg_length_per_em_point_flipped, em_num_points_flipped,
                em_swc_file_flipped,
                lm_vols_flipped, v, lm_vol, args.max_cov_dist ** 2,
                output_folder,
                None, None, None, None, True)

        for i, cov_score_dict in enumerate(cov_score_dicts):
            if lm_vol not in cov_score_dict[em]:
                cov_score_dict[em][lm_vol] = utils.make_cov_score_entry()

    # delete solver (free C++ memory)
    if args.get_cov_version is not None and \
        ("ilp" in args.get_cov_version or 'mixed' in args.get_cov_version):
        mylib.delete_solver()

    # write results
    for i, cov_score_dict in enumerate(cov_score_dicts):
        # cov_score_dict = eval(str(cov_score_dict)) # to get rid of numpy values not dumpable to json
        cov_score_dict[em] = OrderedDict(
            sorted(cov_score_dict[em].items(), key=lambda i: i[1]['cov_score']))
        with open(cov_score_jsons[i], 'w') as fp:
            json.dump(cov_score_dict, fp, indent=4)

        visualize_best(args, em, lm_vols, lm_vol, cov_score_dict,
                       score_variant_indcs[i],
                       em_swc_file, output_folder)


if __name__ == "__main__":
    main()
