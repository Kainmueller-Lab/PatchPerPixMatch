import numpy as np
import h5py
import zarr as z
import argparse
import os
import os.path
from scipy import ndimage
from skimage import io
from prune_skeleton import create_graph_from_swc
import colorcet as cc
import utils
import matplotlib

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw


def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    return np.array([int(hex[i:i + 2], 16) for i in (0, 2, 4)])


def color(src, colormap=None):
    labels = np.unique(src)
    colored = np.stack(
        [np.zeros_like(src), np.zeros_like(src), np.zeros_like(src)],
        axis=-1)

    print(labels)

    for i, label in enumerate(labels):
        if label == 0:
            continue
        if colormap == 'glasbey':
            label_color = hex_to_rgb(cc.glasbey_light[i])
        else:
            label_color = np.random.randint(0, 255, 3)
        idx = src == label
        colored[idx, :] = label_color

    return colored


def get_arguments():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=str,
        dest="raw",
    )
    parser.add_argument(
        "--swc",
        type=str,
        dest="swc",
        nargs='+'
    )
    parser.add_argument(
        "--pred",
        type=str,
        dest="pred"
    )
    parser.add_argument(
        "--radius",
        type=int,
        dest="radius",
        default=1
    )
    parser.add_argument(
        "--output-folder",
        type=str,
        dest="output_folder"
    )

    args = parser.parse_args()
    return args


def bresenhamline_nslope(slope):
    scale = np.amax(np.abs(slope))
    slope_normalized = slope / float(scale) if scale != 0 else slope
    return slope_normalized


def bresenhamline(start_voxel, end_voxel, max_iter=5):
    if max_iter == -1:
        max_iter = np.amax(np.abs(end_voxel - start_voxel))
    dim = start_voxel.shape[0]
    nslope = bresenhamline_nslope(end_voxel - start_voxel)

    # steps to iterate on
    stepseq = np.arange(1, max_iter + 1)
    stepmat = np.tile(stepseq, (dim, 1)).T

    # some hacks for broadcasting properly
    bline = start_voxel[np.newaxis, :] + nslope[np.newaxis, :] * stepmat

    # approximate to nearest int
    return np.array(np.rint(bline), dtype=start_voxel.dtype)


def rasterize_line_segment(point, parent, skeletonized, downsample_by=1):
    # use Bresenham's line algorithm
    # based on http://code.activestate.com/recipes/
    # 578112-bresenhams-line-algorithm-in-n-dimensions/
    line_segment_points = bresenhamline(point, parent, max_iter=-1)
    line_segment_points = line_segment_points / downsample_by

    if line_segment_points.shape[0] > 0:
        idx = np.transpose(line_segment_points.astype(int))
        if np.max(idx[0]) >= skeletonized.shape[0] or \
            np.max(idx[1]) >= skeletonized.shape[1] or \
            np.max(idx[2]) >= skeletonized.shape[2]:
            print(np.max(idx, axis=0), skeletonized.shape)
        skeletonized[idx[0], idx[1], idx[2]] = True

    return skeletonized


def rasterize_skeleton(array, graph, radius=1, downsample_by=1):
    """
    Parameters
    ----------
    array : numpy.ndarray
        Destination array, where points are rasterized in
    graph : networkx graph of SwcPoints
        networkx graph, obtained by a swc file
    radius : int, optional
        Radius for rasterized skeleton
    """
    # iterate through swc points
    # swc point: node_id, x, y, z, parent_id, node_type, diameter
    visited = []
    for cnode in graph.nodes():
        for neighbor in graph.neighbors(cnode):
            if neighbor not in visited:
                # draw line
                p1 = np.around(np.array([
                    graph.nodes[cnode]['z'],
                    graph.nodes[cnode]['y'],
                    graph.nodes[cnode]['x']
                ])).astype(np.int32)
                p2 = np.around(np.array([
                    graph.nodes[neighbor]['z'],
                    graph.nodes[neighbor]['y'],
                    graph.nodes[neighbor]['x']
                ])).astype(np.int32)
                array = rasterize_line_segment(p1, p2, array,
                                               downsample_by=downsample_by)
        visited.append(cnode)

    if radius > 1:
        array = ndimage.binary_dilation(array, iterations=radius - 1)

    return array


def visualize_instances(inst_img, output_file,
                        max_axis=None, show_outline=False,
                        raw_img=None):
    label = inst_img
    raw = raw_img
    if show_outline:
        labels, locations = np.unique(label, return_index=True)
        print(labels)
        locations = np.delete(locations, np.where(labels == 0))
        labels = np.delete(labels, np.where(labels == 0))
        # for different colormaps, see https://colorcet.pyviz.org/
        colormap = cc.glasbey_light
        # uncomment to choose randomly from colormap
        # colormap = np.random.choice(colormap, size=len(labels),
        #                             replace=(len(labels)>len(colormap)))

        if raw is None:
            colored = np.zeros(label.shape[1:] + (3,), dtype=np.uint8)
        else:
            # convert raw to np.uint8
            raw = (raw / 510 * 255).astype(np.uint8)
            colored = np.stack([raw] * 3, axis=-1)
        for i, (lbl, loc) in enumerate(zip(labels, locations)):
            if lbl == 0:
                continue
            # heads up: assuming one instance per channel
            c = np.unravel_index(loc, label.shape)[0]
            outline = ndimage.distance_transform_cdt(label[c] == lbl) == 1
            # colored[outline, :] = np.random.randint(0, 255, 3)
            colored[outline, :] = hex_to_rgb(colormap[i])
    else:
        colored = color(label)
    io.imsave(output_file, colored.astype(np.uint8))


def visualize_frags(raw_file, inst_file, inst_key, skel_ids, nblast_scores,
                    coverages, em_swc_file, outfns, pruned=False,
                    skel_colors=None, radius=2, bin_size=64, channel=None,
                    em_swc_mask=None, em_halo_mask=None, em_prune_mask=None,
                    verbose=False):
    # read raw and prediction
    if verbose:
        print("loading raw: %s" % raw_file, flush=True)
    raw = utils.load_raw(raw_file)
    if verbose:
        print("loading inst: %s" % inst_file, flush=True)
    inst = utils.load_inst(inst_file, inst_key)
    if verbose:
        print("loading em: %s" % em_swc_file, flush=True)
    em_neuron = create_graph_from_swc(em_swc_file)

    visualize_frags_noload(raw, inst, skel_ids, nblast_scores, coverages,
                           em_neuron, outfns, pruned, skel_colors=skel_colors,
                           radius=radius, bin_size=bin_size, channel=channel,
                           em_swc_mask=em_swc_mask, em_halo_mask=em_halo_mask,
                           em_prune_mask=em_prune_mask, verbose=verbose)


def visualize_frags_noload(raw_img, inst_img, skel_ids, nblast_scores,
                           coverages, em_neuron, outfns, pruned=False,
                           skel_colors=None, radius=1, bin_size=64,
                           channel=None, em_swc_mask=None, em_halo_mask=None,
                           em_prune_mask=None, verbose=False):
    outfn_raw = outfns[0]
    outfn_masked_raw = outfns[1]
    outfn_masked_inst = outfns[2]
    outfn_em_on_masked_raw = outfns[3]
    if channel is not None:
        outfn_channel = outfns[4]
        outfn_em_on_channel = outfns[5]

    raw = raw_img
    inst = inst_img
    # inst_values = pd.unique(inst.flatten())
    # inst_values = np.unique(inst.ravel())
    skel_ids = np.asarray(skel_ids).astype(np.int)

    # get fragment colors:
    if skel_colors is None:
        skel_colors = utils.get_skel_colors_noload(raw, inst, skel_ids,
                                                   bin_size=bin_size,
                                                   verbose=verbose)

    if verbose:
        print("create raw mip: %s" % outfn_raw, flush=True)
    mip = np.max(raw, axis=1)
    mip = np.moveaxis(mip, 0, -1)
    clipval = np.amax(np.quantile(mip, 0.99, axis=(0, 1)))  # 510
    if verbose:
        print("clip raw: %s" % clipval, flush=True)
    mip = ((np.clip(mip, 0, clipval) / clipval) * 255).astype(np.uint8)

    io.imsave(outfn_raw, mip, check_contrast=False)

    if channel is not None:
        if (verbose):
            print("create channel mip: %s" % outfn_channel, flush=True)
        mip_channel = np.zeros(mip.shape)
        if channel == "auto":
            # determine best channel from colors
            avg_color = np.average(skel_colors, axis=0)
            channel = np.argmax(avg_color)
            print("auto channel, avg color: %s best channel: %s" % (
                avg_color, channel))
        if False:  # em_prune_mask is not None:
            swc_depth = np.argmax(em_swc_mask, axis=0)
            fromdepth = swc_depth[swc_depth > 0]
            fromdepth = np.min(fromdepth)
            todepth = swc_depth[swc_depth < em_swc_mask.shape[0] - 1]
            todepth = np.max(todepth)
            print("fromdepth %d todepth %d" % (fromdepth, todepth))
            raw[channel][:fromdepth] = 0
            raw[channel][todepth:] = 0
            mip = np.max(raw, axis=1)
            mip = np.moveaxis(mip, 0, -1)
        mip_depth = np.argmax(raw[channel], axis=0) / raw.shape[1]

        mip_channel_intensity = mip[:, :, channel]
        # wait for clipval from masked raw

    if (verbose):
        print("create masked raw mip: %s" % outfn_masked_raw, flush=True)
    where = np.isin(inst, skel_ids, invert=True)
    for i in range(raw.shape[0]):
        raw[i][where] = 0
    inst[where] = 0

    # clipval = int(min(510,np.amax(raw)))

    mip = np.max(raw, axis=1)
    if np.amax(mip) > 0:
        clipval = np.amax(np.array(
            [np.quantile(mip[i][mip[i] > 0], 0.90) if np.amax(mip[i]) > 0 else 0
             for i in range(mip.shape[0])]))  # 510
    else:
        clipval = 0
    if (verbose):
        print("clip masked raw: %s" % clipval, flush=True)
    mip = np.moveaxis(mip, 0, -1)
    mip = ((np.clip(mip, 0, clipval) / clipval) * 255).astype(np.uint8)

    # avoid pitch black:
    mip[mip == 0] = 100
    io.imsave(outfn_masked_raw, mip, check_contrast=False)

    if channel is not None:
        channel_clip = np.amax(
            np.quantile(mip_channel_intensity, 0.97, axis=(0, 1)))
        if (verbose):
            print("clip channel mip: min(%s,%s)>0" % (clipval, channel_clip),
                  flush=True)
        if clipval > 0:
            clipval = min(clipval, channel_clip)  # 510
        else:
            clipval = channel_clip
        mip_channel_intensity = ((np.clip(mip_channel_intensity, 0,
                                          clipval) / clipval) * 255).astype(
            np.uint8)

        # brand by colorbar:
        barwidth = 30
        barlength = 80
        depthimg = np.expand_dims(np.array(range(0, 256, 1)) / 255., axis=0)
        depthimg = np.multiply(np.ones((barwidth, depthimg.shape[1])), depthimg)
        mip_depth[:depthimg.shape[0], -10 - depthimg.shape[1]:-10] = depthimg
        mip_channel_intensity[:depthimg.shape[0],
        -10 - depthimg.shape[1]:-10] = 255

        cmap = matplotlib.cm.get_cmap('turbo')  # ('nipy_spectral')
        mip_channel = np.multiply(cmap(mip_depth)[:, :, :3],
                                  np.expand_dims(mip_channel_intensity,
                                                 axis=-1))
        # mip_channel = np.moveaxis(np.vstack(mip_channel), -1,0)

        # brand by channel:
        mip_channel[:barwidth, :barlength, channel] = 255

        # if channel==2:
        #    ## avoid blue on black, rather make it cyan:
        #    mip_channel[:,:,1] = mip[:,:,channel]
        io.imsave(outfn_channel, mip_channel, check_contrast=False)

    if (verbose):
        print("create masked inst mip: %s" % outfn_masked_inst, flush=True)
    # instmip = np.max(inst, axis=1)
    instmip = np.max(inst, axis=0)
    colormip = color(instmip)
    io.imsave(outfn_masked_inst, colormip.astype(np.uint8),
              check_contrast=False)

    if em_neuron or em_swc_mask:
        if (verbose):
            print("create em overlay raw mip: %s" % outfn_em_on_masked_raw,
                  flush=True)
        if em_swc_mask is None:
            rasterized = np.zeros(raw[0].shape, dtype=np.bool)
            em_swc_mask = rasterize_skeleton(rasterized, em_neuron, radius)
        swc_mip = np.max(em_swc_mask, axis=0) == 1
        if pruned:
            if em_prune_mask is None:
                em_prune_mask = rasterize_skeleton(rasterized, em_neuron,
                                                   radius=50)
            for i in range(raw.shape[0]):
                raw[i][em_prune_mask == 0] = 0
            mip = np.max(raw, axis=1)
            if np.amax(mip) > 0:
                clipval = np.amax(np.array([np.quantile(mip[i][mip[i] > 0],
                                                        0.90) if np.amax(
                    mip[i]) > 0 else 0 for i in range(mip.shape[0])]))  # 510
            else:
                clipval = 0
            if (verbose):
                print("clip masked raw: %s" % clipval, flush=True)
            mip = np.moveaxis(mip, 0, -1)
            mip = ((np.clip(mip, 0, clipval) / clipval) * 255).astype(np.uint8)
            mip[mip == 0] = 100
        mip[swc_mip] = [255, 0, 255]

        io.imsave(outfn_em_on_masked_raw, mip, check_contrast=False)

        if channel is not None:
            if (verbose):
                print("create em overlay channel mip: %s" % outfn_em_on_channel,
                      flush=True)
            swc_mip_channel = 255 * np.multiply(
                cmap(np.argmax(em_swc_mask, axis=0) / raw.shape[1])[:, :, :3],
                np.expand_dims(swc_mip, axis=-1))
            halo = np.where(np.max(em_halo_mask, axis=0) == 1)
            dim_factor = 2.5
            mip_channel /= dim_factor
            mip_channel[halo] = 0  # [255, 0, 255] #
            # mip_channel[swc_mip] = 0
            mip_channel[swc_mip_channel > 0] = swc_mip_channel[
                swc_mip_channel > 0]  # [255, 0, 255]
            # io.imsave(outfn_em_on_channel, mip_channel, check_contrast=False)
            # renew branding by channel:
            mip_channel[:barwidth, :barlength] *= dim_factor
            mip_channel[:depthimg.shape[0],
            -10 - depthimg.shape[1]:-10] *= dim_factor

            io.imsave(outfn_em_on_channel, mip_channel, check_contrast=False)

    if (verbose):
        print("drawing nblast scores and raw frag colors onto mips...",
              flush=True)
    # draw nblast scores and raw frag colors onto mips:
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 12,
        encoding="unic")
    for draw_img_path in [outfn_masked_raw, outfn_masked_inst,
                          outfn_em_on_masked_raw]:
        if not os.path.isfile(draw_img_path):
            continue
        if (verbose):
            print("drawing header row onto %s" % draw_img_path, flush=True)
        img = Image.open(draw_img_path)
        draw = ImageDraw.Draw(img)
        draw.text((5, 5), "frag", (255, 255, 255), font=font)
        draw.text((50, 5), "score", (255, 255, 255), font=font)
        draw.text((95, 5), "cover", (255, 255, 255), font=font)
        draw.text((140, 5), "color", (255, 255, 255), font=font)
        for i, s in enumerate(nblast_scores):
            hpos = 5 + (i + 1) * 14
            if hpos > img.size[1]:
                break
            if (verbose):
                print("drawing nblast score %i at pos %i, img height %i" % (
                    i, hpos, img.size[1]), flush=True)
            text = "%3i" % int(100 * s)
            cov = "%2i" % int(coverages[i])
            skel_id = skel_ids[i]
            col = (colormip[instmip == skel_id])[0] if len(
                colormip[instmip == skel_id]) > 0 else [255, 255, 255]
            raw_col = skel_colors[i].astype(np.int)
            max_col = np.amax(raw_col)
            raw_col_scaled = np.array(raw_col * 255 / max_col).astype(np.uint8)
            draw.text((5, hpos), "%i" % skel_id, tuple(col), font=font)
            draw.text((55, hpos), text, tuple(col), font=font)
            draw.text((100, hpos), cov, tuple(col), font=font)
            draw.text((130, hpos), "%s" % raw_col, tuple(raw_col_scaled),
                      font=font)
        img.save(draw_img_path)
    # draw colorbar legend onto colormips:
    font = ImageFont.truetype(
        "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf", 18,
        encoding="unic")
    for draw_img_path in [outfn_channel, outfn_em_on_channel]:
        if not os.path.isfile(draw_img_path):
            continue
        img = Image.open(draw_img_path)
        draw = ImageDraw.Draw(img)
        draw.text((5, barwidth + 5), "channel", (255, 255, 255), font=font)
        draw.text((mip_channel.shape[1] - 265, barwidth + 5), "0",
                  (255, 255, 255), font=font)
        draw.text((mip_channel.shape[1] - 45, barwidth + 5), str(raw.shape[1]),
                  (255, 255, 255), font=font)
        draw.text((mip_channel.shape[1] - 170, barwidth + 5), "depth",
                  (255, 255, 255), font=font)
        img.save(draw_img_path)

    if (verbose):
        print("done visualize_no_frags()", flush=True)


def visualize(raw_file, swc_file, output_folder, radius=1,
              split_channel=False, colors=None, outfns=None):
    # read raw and prediction
    if raw_file is None:
        raw = np.zeros((3, 394, 668, 1427), dtype=np.uint8)
    else:
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
        raw = ((np.clip(raw, 0, 510) / 510) * 255).astype(np.uint8)

    # read swc into networkx graph
    swc_masks = []
    if type(swc_file) != list:
        swc_files = [swc_file]
    else:
        swc_files = swc_file

    for swc_file in swc_files:
        skeleton = create_graph_from_swc(swc_file)
        rasterized = np.zeros(raw[0].shape, dtype=np.bool)
        swc_masks.append(rasterize_skeleton(rasterized, skeleton, radius))

    if output_folder is not None:
        output_base = output_folder
    else:
        output_base = ""

    mip = np.max(raw, axis=1)
    mip = np.moveaxis(mip, 0, -1)

    if raw_file is not None:
        raw_base = os.path.basename(raw_file).split(".")[0]
        if outfns is not None:
            output_base = outfns[-1]
        else:
            output_base = os.path.join(output_base, raw_base)
        io.imsave(output_base + ".png", mip, check_contrast=False)

    if colors is None:
        colors = [[0, 255, 255], [255, 0, 255], [255, 255, 0]]

    for i, (swc_file, rasterized, color) in enumerate(zip(
        swc_files, swc_masks, colors)):
        swc_base = os.path.basename(swc_file).split(".")[0]
        swc_mip = np.max(rasterized, axis=0) == 1

        if split_channel:
            red = mip[0]
            green = mip[1]
            blue = mip[2]
            w, h = red.shape
            dst = np.zeros((w * 3, h * 2, 3), dtype=np.uint8)
            compose = np.concatenate([red, green, blue], axis=0)
            compose = np.stack([compose] * 3, axis=-1)
            dst[:, :h] = compose
            compose[np.concatenate([swc_mip] * 3)] = [0, 255, 255]
            dst[:, h:] = compose
            io.imsave(output_base + ".png", dst, check_contrast=False)
        else:
            # if not os.path.exists(output_base + ".png"):
            #    io.imsave(output_base + ".png", mip, check_contrast=False)

            mip[swc_mip] = color
            if outfns is not None:
                output_base = outfns[i]
            else:
                output_base = output_base + "_" + swc_base + ".png"
            io.imsave(output_base, mip, check_contrast=False)


def main():
    args = get_arguments()
    print(args)

    visualize(args.raw, args.swc, args.output_folder, args.radius)


if __name__ == "__main__":
    main()
