from numba import jit
import h5py
import zarr as z
import numpy as np
from math import exp
import os
# import time

import sklearn.metrics
import sklearn.cluster


def load_raw(raw_file, verbose=False):
    raw_format = raw_file.split(".")[-1]
    if raw_format == "hdf":
        hin = h5py.File(raw_file, "r")
        if "volumes/raw" in hin:
            raw = np.array(hin["volumes/raw"])
        else:
            sample = os.path.basename(raw_file).split(".")[0]
            print(sample)
            raw = np.array(hin[sample + "/raw"])
        hin.close()
    elif raw_format == "zarr":
        print("opening %s " % raw_file)
        rawg = z.open_group(raw_file)
        raw = np.array(rawg["volumes/raw"])

    if verbose:
        print(raw.shape)
    return raw


def load_inst(inst_file, inst_key, verbose=False):
    inst_format = inst_file.split(".")[-1]
    if inst_format == "hdf":
        hin = h5py.File(inst_file, "r")
        if inst_key in hin:
            print("opening %s %s" % (inst_file, inst_key))
            inst = np.array(hin[inst_key])
        else:
            print("hdf key for instances does not exist, returning")
        hin.close()
    elif inst_format == "zarr":
        print("opening %s " % inst_file)
        instg = z.open_group(inst_file)
        inst = np.array(instg["volumes"])
    if verbose:
        print(inst.shape)
    return inst


def get_skel_colors(raw_file, inst_file, inst_key, skel_ids, bin_size=64, verbose=False):
    raw = load_raw(raw_file)
    inst = load_inst(inst_file, inst_key)
    if skel_ids == -1:
        return_skel_ids = True
        skel_ids = np.unique(inst)
        skel_ids = skel_ids[skel_ids > 0]

    output = get_skel_colors_noload(raw, inst, skel_ids, bin_size=bin_size, verbose=verbose)
    if return_skel_ids:
        return output, skel_ids
    else:
        return output


def get_color_mode(col_list, col_weights=None, n_clusters=2, verbose=False):
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=0, n_jobs=1).fit(col_list,
                                                                                         sample_weight=col_weights)
    max_label_count = 0
    label_counts = [len(np.where(kmeans.labels_ == label)[0]) for label in range(n_clusters)]
    max_count_idx = np.argmax(label_counts)
    return kmeans.cluster_centers_[max_count_idx]


def get_skel_colors_noload(raw, inst, skel_ids, bin_size=64, verbose=False, mode='kmeans'):
    skel_colors = []
    where = np.isin(inst, skel_ids)
    num_voxels = len(where)
    if verbose:
        print("skel ids %s: num voxels %i" % (skel_ids, num_voxels))
    col_list = np.array([raw[0][where], raw[1][where], raw[2][where]]).T
    skel_list = inst[where]
    for skel_id in skel_ids:
        skel_col_list = col_list[skel_list == skel_id]
        assert len(skel_col_list) > 0
        if verbose:
            print("skel id %i: num voxels %i" % (skel_id, len(skel_col_list)))

        if mode == 'kmeans':
            col = get_color_mode(skel_col_list, verbose=verbose)
        elif mode == 'bin':
            col = max_color_bin(skel_col_list, bin_size=bin_size, verbose=verbose)
        else:
            col = get_color_mode(skel_col_list, verbose=verbose)
        if verbose:
            print("skel id: %i col: %s" % (skel_id, col), flush=True)
        skel_colors = skel_colors + [col]
    return skel_colors


def max_color_bin(a, bin_size=64, verbose=False):
    if verbose:
        print(a.shape)
    a2D = a.reshape(-1, a.shape[-1])
    a2D = np.floor_divide(a2D + int(bin_size / 2), bin_size)
    if verbose:
        print(a2D[0:2])
    col_range = a2D.max(0) + 1  # (256, 256, 256) # generically : a2D.max(0)+1
    if verbose:
        print(col_range)
    a1D = np.ravel_multi_index(a2D.T, col_range)
    bc = np.bincount(a1D)
    argsort = np.flip(np.argsort(bc))
    if verbose:
        for idx in range(0, min(10, len(argsort))):
            i = argsort[idx]
            col = np.array(np.unravel_index(i, col_range)) * bin_size
            count = bc[i]
            print("count %i col %s" % (count, col))
    return np.array(np.unravel_index(argsort[0], col_range)) * bin_size


def get_overlap_adjusted_coverages(frag_names, covered_points, num_points, verbose=False):
    coverage_count = np.zeros(num_points)
    for i, frag_name in enumerate(frag_names):
        cp = covered_points[frag_name]
        coverage_count[cp] += 1
    coverage_count[coverage_count == 0] = 1
    coverage_frac = 1 / coverage_count
    return np.array([100 * np.sum(coverage_frac[covered_points[frag_name]]) / num_points for frag_name in frag_names])


def get_adjusted_coverages(covered_points, all_point_dists_2, num_points, seg_length_per_em_point, max_dist_2,
                           verbose=False):
    covering_cp_idx_per_em_point = np.argmin(all_point_dists_2, axis=0)
    # if verbose: print("covering_cp_idx_per_em_point: %s" %covering_cp_idx_per_em_point[1050:1090])
    dists_2 = all_point_dists_2[covering_cp_idx_per_em_point, np.arange(num_points)]
    dist_filter = np.where(dists_2 < max_dist_2)
    covering_cp_idx_per_em_point = covering_cp_idx_per_em_point[dist_filter]

    seg_length_per_em_point_ = seg_length_per_em_point[dist_filter]

    return np.array(
        [100 * np.sum(seg_length_per_em_point_[covering_cp_idx_per_em_point == cpidx]) / num_points for cpidx in
         range(len(covered_points))])


# @jit(nopython=True)
def get_adjusted_coverages_by_score(covered_points, all_skel_scores, num_points, seg_length_per_em_point,
                                    verbose=False):
    covering_cp_idx_per_em_point = np.argmax(all_skel_scores, axis=0)
    if verbose:
        print("covering_cp_idx_per_em_point: %s" % covering_cp_idx_per_em_point[1050:1090])
    scores = all_skel_scores[covering_cp_idx_per_em_point, np.arange(num_points)]
    # covering_cp_idx_per_em_point = np.array([np.argmax(s) for s in all_skel_scores.T])
    # scores = np.array([all_skel_scores[covering_cp_idx_per_em_point[i], i] for i in range(num_points)])

    score_filter = np.where(scores > 0)
    covering_cp_idx_per_em_point = covering_cp_idx_per_em_point[score_filter]

    seg_length_per_em_point_ = seg_length_per_em_point[score_filter]

    return 100 * np.array([np.sum(seg_length_per_em_point_[covering_cp_idx_per_em_point == cpidx]) for cpidx in
                           range(len(covered_points))]) / num_points


def get_aggregate_coverage(frag_names, covered_points, num_points):
    all_covered_points = np.unique([cp for frag_name in frag_names for cp in covered_points[frag_name]])
    return 100 * len(all_covered_points) / num_points


def x_get_overlap_adjusted_coverages(em_neuron, frags, covered_points):
    # print(covered_points)
    num_points = em_neuron.nodes.shape[0]
    coverage_count = np.zeros(num_points)
    for i, frag in enumerate(frags):
        cp = covered_points[em_neuron.name][frag.name]
        coverage_count[cp] += 1
        if i == 0:
            print(len(cp))
            print(num_points)
            print(100 * len(cp) / num_points)
            print(cp, flush=True)
    coverage_count[coverage_count == 0] = 1
    coverage_frac = 1 / coverage_count
    # debug:
    cp = covered_points[em_neuron.name][frags[0].name]
    print(coverage_count[cp])
    print(coverage_frac[cp])
    return np.array(
        [100 * np.sum(coverage_frac[covered_points[em_neuron.name][frag.name]]) / num_points for frag in frags])


def get_pairwise(i, j, norm_colors, unaries, bin_size, skel_scores, skel_pair_overlaps, slack=32, verbose=False):
    p1 = get_pairwise_from_color(i, j, norm_colors, bin_size, slack, verbose)
    p2 = get_pairwise_from_overlap(i, j, skel_scores, skel_pair_overlaps, slack)
    return (p1, p2)


def get_pairwise_from_color(i, j, norm_colors, bin_size, slack=32, verbose=False):
    if (i == j):
        return 0
    if verbose:
        print("norm_colors: %s %s" % (norm_colors[i], norm_colors[j]))
    col_diff = norm_colors[i] - norm_colors[j]
    if (len(np.where(col_diff > -slack)[0])) == len(col_diff) or (len(np.where(col_diff < slack)[0])) == len(col_diff):
        pairwise = 0.1 * max(abs(col_diff)) / bin_size  # unaries[i]* # removed because it has to be adjusted_unaries[i]
    else:
        pairwise = abs(max(col_diff) - min(col_diff)) / bin_size  # unaries[i]*
    if verbose:
        print("%i, %i, pairwise %s" % (i, j, pairwise))
    return pairwise


def get_pairwise_from_overlap(i, j, skel_scores, skel_pair_overlaps, slack=32):
    if skel_pair_overlaps is None:
        return 0
    if (i == j):
        return 0
    pairwise_from_overlap = skel_pair_overlaps[i, j] * min(skel_scores[i], skel_scores[j])
    return pairwise_from_overlap


# @jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + exp(-x))


def transform_score_by_coverage(skel_score, skel_coverage):
    # v4:
    return (sigmoid(-skel_coverage / 5) + 1) * (skel_score - 2) + 2


# @jit(nopython=True)
def get_unary(i, skel_coverages_adjusted, skel_scores, verbose=False):
    # v2:
    # return skel_coverages[i]*min(1.5,skel_scores[i])
    # v3:
    # f= 1/(2*(sigmoid(skel_coverages[i]/25) - 0.5)+1)
    # v4:
    # f= (sigmoid(-skel_coverages[i]/5) +1)/2.25*2
    # v5:
    # f= sigmoid(-skel_coverages[i]/10) +1
    # if verbose: print("cov %.2f factor %.2f mapping score %.2f --> %.2f" % (skel_coverages[i], f, skel_scores[i], ( f*(skel_scores[i]-2) +2 )))
    # return skel_coverages_adjusted[i]*( f*(skel_scores[i]-2) +2 )
    return skel_coverages_adjusted[i] * skel_scores[i]


def cluster_fragments_by_color(
        em, lm_vol, covered_points, all_skel_scores,
        num_points, seg_length_per_em_point, skel_colors, skel_coverages,
        skel_scores, upper_bound_other,
        max_dist_2, skel_pair_overlaps=None, bin_size=64, skel_ids=None,
        verbose=False, clustering_algo="kmeans"):
    num_skels = range(len(skel_coverages))
    if skel_ids is None:
        skel_ids = range(len(covered_points))
    # hack to allow for as many clusters as skels:
    skel_colors = [skel_color + 0.001 * i for i, skel_color in enumerate(skel_colors)]
    norm_colors = np.array([skel_color * 255 / np.amax(skel_color) for skel_color in skel_colors])
    unaries = np.array([get_unary(i, skel_coverages, skel_scores) for i in num_skels])

    # pairwises = np.array([ [ get_pairwise_from_color(j,i, norm_colors, bin_size, slack=bin_size/2, verbose=verbose) for i in num_skels ] for j in num_skels ])
    pairwises_half = np.array([[get_pairwise_from_color(j, i, norm_colors, bin_size, slack=bin_size / 2,
                                                        verbose=verbose) if j < i else 0 for i in num_skels] for j in
                               num_skels])
    ## fill other half:
    pairwises = pairwises_half + pairwises_half.T
    if verbose:
        print("norm colors: %s" % np.unique(norm_colors, axis=0), flush=True)
    max_n_clusters = min(10, len(np.unique(norm_colors, axis=0)) + 1)

    # all_skel_scores = 0*all_point_dists_2
    # for i,cps in enumerate(covered_points): all_skel_scores[i][cps] = skel_scores[i]

    min_energy = -1 * np.amax(unaries)

    best_sel = np.array([np.argmax(unaries)])
    best_n = 0
    best_label = -1

    # np.save("norm_colors_{}.npy".format(lm_vol), norm_colors)
    # np.save("weights_{}.npy".format(lm_vol), unaries+np.min(unaries)+1)

    if clustering_algo == 'agglomerative':
        # start = time.time()
        norm_colors_weighted = []
        norm_colors_ids = []
        for nc, u in zip(norm_colors, unaries):
            norm_colors_ids.append(len(norm_colors_weighted))
            norm_colors_weighted += [nc] * max(1, int(np.round(u + 1)))

        kmeans = sklearn.cluster.AgglomerativeClustering(n_clusters=1, compute_full_tree=True).fit(norm_colors_weighted)
        # end = time.time()
        # print("agglo_cluster took %s seconds" % (end - start) )
        for n_clusters in range(max_n_clusters, 1, -1):
            labels_ = sklearn.cluster._agglomerative._hc_cut(n_clusters, kmeans.children_, kmeans.n_leaves_)
            labels_ = labels_[norm_colors_ids]
            # labels_ = np.array(kmeans.labels_, dtype=np.int32)
            best_nT, best_labelT, best_selT, min_energyT = cluster_internal_loop(
                n_clusters, unaries, min_energy, upper_bound_other,
                covered_points, all_skel_scores, num_points,
                seg_length_per_em_point, verbose, skel_scores,
                pairwises, labels_,
                best_sel, best_n, best_label)

            if min_energyT < min_energy:
                # np.save("labels_{}.npy".format(lm_vol), labels_)
                best_n, best_label, best_sel, min_energy = best_nT, best_labelT, best_selT, min_energyT
        # end2 = time.time()
        # print("cluster_internal took %s seconds" % (end2 - end) )

    elif clustering_algo == 'kmeans':
        for n_clusters in range(1, max_n_clusters):
            if verbose:
                print("clustering with %i clusters" % n_clusters)
            kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, tol=1e-2, n_init=5, random_state=0, n_jobs=1).fit(
                norm_colors, sample_weight=unaries)
            labels_ = np.array(kmeans.labels_, dtype=np.int32)
            best_nT, best_labelT, best_selT, min_energyT = cluster_internal_loop(
                n_clusters, unaries, min_energy, upper_bound_other,
                covered_points, all_skel_scores, num_points,
                seg_length_per_em_point, verbose, skel_scores,
                pairwises, labels_,
                best_sel, best_n, best_label)

            if min_energyT < min_energy:
                # np.save("labels_{}.npy".format(lm_vol), kmeans.labels_)
                best_n, best_label, best_sel, min_energy = best_nT, best_labelT, best_selT, min_energyT

    else:
        RuntimeError("invalid clustering algo!")

    # np.save("best_{}.npy".format(lm_vol), [best_n, best_label, best_sel])
    print("best_n %i best_label %i best_sel %s energy %f" % (best_n, best_label, best_sel, min_energy))
    return (min_energy, best_sel)


# @jit(nopython=True)
def cluster_internal_loop(n_clusters, unaries, min_energy, upper_bound_other,
                          covered_points, all_skel_scores, num_points,
                          seg_length_per_em_point, verbose, skel_scores,
                          pairwises, labels_,
                          best_sel, best_n, best_label):
    # print("kmeans iter %i" %kmeans.n_iter_)
    for label in range(n_clusters):
        selected_l = np.where(labels_ == label)[0]
        num_selected = len(selected_l)
        if num_selected == 0:
            continue

        prelim_energy = -np.sum(unaries[selected_l])
        if prelim_energy >= min_energy:
            if verbose:
                print("label %i lower bound energy %d > %d" % (label, prelim_energy, min_energy))
            continue
        if prelim_energy >= upper_bound_other:
            continue

        if verbose:
            print("  n_clusters %i, label %i, selected_l %s" % (n_clusters, label, selected_l))
        no_use = False
        while True:
            skel_coverages_adjusted = get_adjusted_coverages_by_score(
                [covered_points[i] for i in selected_l],
                all_skel_scores[selected_l], num_points,
                seg_length_per_em_point, verbose=verbose)
            if verbose:
                print("cov: %s" % skel_coverages[selected_l])
            if verbose:
                print("adjusted covs: %s" % skel_coverages_adjusted)
            # dump 0-coverage frags: don't do this, because they might be picked after removing others
            # selected_l = selected_l[skel_coverages_adjusted>0]
            # num_selected = len(selected_l)
            # skel_coverages_adjusted = skel_coverages_adjusted[skel_coverages_adjusted>0]

            adjusted_unaries = np.array(
                [get_unary(i, skel_coverages_adjusted, skel_scores[selected_l]) for i in range(num_selected)])
            if verbose:
                print("adjusted unaries: %s" % adjusted_unaries)

            prelim_energy = -np.sum(adjusted_unaries)
            if prelim_energy >= min_energy:
                no_use = True
                if verbose:
                    print("label %i lower bound energy %d > %d" % (label, prelim_energy, min_energy))
                break
            if prelim_energy >= upper_bound_other:
                no_use = True
                break

            weight = 0.5 / max(1, (len(selected_l) - 1))
            sumpairwises = np.sum(pairwises[selected_l][:, selected_l], axis=1)
            aggregated_pairwises = weight * np.multiply(adjusted_unaries, sumpairwises)
            # aggregated_pairwises = weight*np.array([ adjusted_unaries[ii]*np.sum(pairwises[i][selected_l]) for ii,i in enumerate(selected_l) ])

            if verbose:
                for i in range(num_selected):
                    if verbose:
                        print("skel idx %i score: %.2f, adjusted cov: %.2f, unary: %.2f agg_pairwise: %.2f" % (
                            skel_ids[selected_l[i]], skel_scores[selected_l[i]], skel_coverages_adjusted[i],
                            adjusted_unaries[i], aggregated_pairwises[i]))

            max_unaries = unaries[selected_l]
            if len(selected_l) > len(selected_l[aggregated_pairwises < max_unaries]):  # adjusted_unaries]):
                diff = aggregated_pairwises - max_unaries
                dump_idx = np.argmax(diff)
                if verbose:
                    print("dumped skels idcs %s by unary" % skel_ids[selected_l[dump_idx]])
                selected_l = np.delete(selected_l, dump_idx)
                num_selected = len(selected_l)
            elif len(selected_l) > len(selected_l[aggregated_pairwises < adjusted_unaries]):
                diff = aggregated_pairwises - adjusted_unaries
                dump_idx = np.argmax(diff)
                if verbose:
                    print("dumped skels idcs %s by adjusted unary" % skel_ids[selected_l[dump_idx]])
                selected_l = np.delete(selected_l, dump_idx)
                num_selected = len(selected_l)
            else:
                break

        if no_use:
            continue

        energy = -np.sum(adjusted_unaries) + np.sum(aggregated_pairwises)
        # if verbose: print("  n_clusters %i, label %i, selected_l %s" %(n_clusters,label,selected_l))
        # if verbose: print(" n_clusters %i, label %i, energy %f" %(n_clusters,label,energy))
        if (energy < min_energy):
            min_energy = energy
            best_sel = selected_l
            best_n = n_clusters
            best_label = label

    return best_n, best_label, best_sel, min_energy


def cluster_fragments_by_color_x(frags, covered_points, num_points, skel_colors, skel_coverages, skel_scores,
                                 skel_pair_overlaps=None, bin_size=64, verbose=False):
    num_skels = range(len(skel_coverages))
    print(len(skel_coverages), len(skel_scores))
    print(skel_coverages)
    print(skel_scores)
    # hack to allow for as many clusters as skels:
    skel_colors = [skel_color + 0.001 * i for i, skel_color in enumerate(skel_colors)]
    norm_colors = np.array([skel_color * 255 / np.amax(skel_color) for skel_color in skel_colors])
    unaries = np.array([get_unary(i, skel_coverages, skel_scores) for i in num_skels])
    pairwises = [[get_pairwise(j, i, norm_colors, unaries, bin_size, skel_scores, skel_pair_overlaps,
                               slack=bin_size / 2, verbose=verbose) for i in num_skels] for j in num_skels]
    if verbose:
        print("norm colors: %s" % np.unique(norm_colors, axis=0), flush=True)
    max_n_clusters = min(10, len(np.unique(norm_colors, axis=0)) + 1)

    min_energy = -1 * np.amax(unaries)
    best_sel = np.array([np.argmax(unaries)])
    best_n = 0
    best_label = -1

    for n_clusters in range(1, max_n_clusters):
        if verbose:
            print("clustering with %i clusters" % n_clusters)
        kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=0, n_jobs=1).fit(norm_colors,
                                                                                             sample_weight=unaries)
        for label in range(n_clusters):
            selected_l = np.where(kmeans.labels_ == label)[0]
            num_selected = len(selected_l)
            if num_selected == 0:
                continue

            prelim_energy = -np.sum(unaries[selected_l])
            if prelim_energy >= min_energy:
                # if verbose: print("label %i lower bound energy %d > %d" %(label, prelim_energy, min_energy))
                continue

            if verbose:
                print("  n_clusters %i, label %i, selected_l %s" % (n_clusters, label, selected_l))
            while True:
                frag_names_selected = [frags[i].name for i in selected_l]
                skel_coverages_adjusted = get_overlap_adjusted_coverages(frag_names_selected, covered_points,
                                                                         num_points, verbose=verbose)
                if verbose:
                    print("cov: %s" % skel_coverages[selected_l])
                if verbose:
                    print("adjusted covs: %s" % skel_coverages_adjusted)
                adjusted_unaries = np.array(
                    [get_unary(i, skel_coverages_adjusted, skel_scores[selected_l]) for i in range(num_selected)])
                prelim_energy = -np.sum(adjusted_unaries)

                weight = 1.5 / max(1, (len(selected_l) - 1))
                aggregated_pairwises = np.array(
                    [np.sum([weight * pairwises[i][j][0] + pairwises[i][j][1] for j in selected_l if j != i]) for i in
                     selected_l])

                if verbose:
                    for i in range(num_selected):
                        if verbose:
                            print("skel %s adjusted unary: %.1f agg_pairwise: %.1f" % (
                                frag_names_selected[i], adjusted_unaries[i], aggregated_pairwises[i]))

                if len(selected_l) > len(selected_l[aggregated_pairwises < adjusted_unaries]):
                    diff = aggregated_pairwises - adjusted_unaries
                    dump_idx = np.argmax(diff)
                    dumped = (frags[selected_l[dump_idx]].name).rsplit('_', 1)[1]
                    selected_l = np.delete(selected_l, dump_idx)
                    num_selected = len(selected_l)
                    if verbose:
                        print("dumped skels %s" % dumped)
                else:
                    break

            energy = -np.sum(adjusted_unaries) + np.sum(aggregated_pairwises)
            if verbose:
                print("  n_clusters %i, label %i, selected_l %s" % (n_clusters, label, selected_l))
            if verbose:
                print(" n_clusters %i, label %i, energy %f" % (n_clusters, label, energy))
            if (energy < min_energy):
                min_energy = energy
                best_sel = selected_l
                best_n = n_clusters
                best_label = label
    if verbose:
        print("best_n %i best_label %i best_sel %s energy %f" % (best_n, best_label, best_sel, min_energy))
    return (min_energy, best_sel)


def make_cov_score_entry(skel_ids=[], skel_scores=[],
                         skel_coverages=[], skel_colors=[],
                         min_energy=0.0, aggregate_coverage=0.0,
                         all_skel_ids=[], all_skel_scores=[], all_skel_coverages=[],
                         all_skel_colors=[], all_aggregate_coverage=0):
    entry = {}
    entry['skel_ids'] = str(skel_ids)  # hack for readable formatting of json; to be read with eval()
    entry['nblast_scores'] = str(skel_scores)
    entry['coverages'] = str(skel_coverages)
    entry['colors'] = str(skel_colors)
    entry['cov_score'] = float(min_energy)
    entry['aggregate_coverage'] = float(aggregate_coverage)

    # store all colors:
    # todo: colors don't match skel ids in coverage.json because of weeding out
    # all_colors = [ [int(c[0]), int(c[1]), int(c[2]) ] for c in skel_colors ]
    entry['all_skel_ids'] = str(all_skel_ids)
    entry['all_nblast_scores'] = str(all_skel_scores)
    entry['all_coverages'] = str(all_skel_coverages)
    print_colors = [[int(skel_color[0]), int(skel_color[1]), int(skel_color[2])] for skel_color in all_skel_colors]
    entry['all_colors'] = str(print_colors)
    entry['all_aggregate_coverage'] = float(all_aggregate_coverage)
    return entry


def write_debug_coverage_swc(args, all_coverages_per_em_point, best_skel_cluster,
                             em_num_points, em_swc_file, output_folder, lm_vol, svi):
    selected_covs = all_coverages_per_em_point[best_skel_cluster]
    covering_cp_idx_per_em_point = np.argmax(selected_covs, axis=0)
    covs = selected_covs[covering_cp_idx_per_em_point, np.arange(em_num_points)]
    covering_cp_idx_per_em_point[covs == 0] = -1
    covering_cp_idx_per_em_point = covering_cp_idx_per_em_point + 1

    with open(em_swc_file, "r") as neuron_file:
        lines = [[s for s in line.rstrip().split(" ")] for line in neuron_file if not line[0] == "#"]
    lines = [" ".join(line[:-2] + [str(covering_cp_idx_per_em_point[i])] + [line[-1]]) for i, line in enumerate(lines)]
    with open(os.path.join(
            output_folder,
            args.em + lm_vol + '_debug_coverage_' + str(svi) + '.swc'), 'w') as f:
        for l in lines:
            f.write("%s\n" % l)
