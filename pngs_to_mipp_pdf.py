import glob
from fpdf import FPDF
from PyPDF2 import PdfFileWriter, PdfFileReader
import pandas as pd
import math
import json
import argparse


def get_arguments():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--png-path", type=str, dest="png_path",
        default="/nrs/saalfeld/kainmuellerd/flymatch/all_hemibrain_1.2_NB/setup22_nblast_20/results/11/541347811*/lm_cable_length_20_v4_adj_by_cov_numba_agglo_aT/screenshots/"
    )
    parser.add_argument(
        "--result-path", type=str, dest="result_path",
        default="/nrs/saalfeld/kainmuellerd/flymatch/all_hemibrain_1.2_NB/setup22_nblast_20/results/11/541347811*/lm_cable_length_20_v4_adj_by_cov_numba_agglo_aT"
    )
    parser.add_argument(
        "--result-filename", type=str, dest="result_filename",
        default="PatchPerPixMatch.pdf"
    )
    parser.add_argument(
        "--num-pages", type=int, dest="howmany",
        default=10
    )

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    print(args)

    do_cscore = True
    howmany = args.howmany

    do_pdf = True
    do_spreadsheet = True

    def sort_keys(x):
        em = int(x.split('/')[-1].split('-')[0])
        rank = int(x.split('/')[-1].split('_cr_')[1].split('_')[0])
        return (em, rank)

    gen1expr_path = "/nrs/saalfeld/kainmuellerd/data/flylight/Gen1Gal4ExprPatterns/"

    for nblast_setup in ["_both"]:
        if do_cscore:
            basepath = args.png_path
            all_imgs = glob.glob(basepath + "*.png")
            imagelist1 = sorted(
                [f for f in all_imgs for i in range(1, howmany + 1) if "cr_%i_" % i in f and "1_raw.png" in f],
                key=lambda x: sort_keys(x))
            imagelist2 = sorted(
                [f for f in all_imgs for i in range(1, howmany + 1) if "cr_%i_" % i in f and "5_ch.png" in f],
                key=lambda x: sort_keys(x))
            imagelist3 = sorted(
                [f for f in all_imgs for i in range(1, howmany + 1) if "cr_%i_" % i in f and "6_ch_skel.png" in f],
                key=lambda x: sort_keys(x))
            imagelist4 = sorted(
                [f for f in all_imgs for i in range(1, howmany + 1) if "cr_%i_" % i in f and "2_masked_raw.png" in f],
                key=lambda x: sort_keys(x))
            imagelist5 = sorted(
                [f for f in all_imgs for i in range(1, howmany + 1) if "cr_%i_" % i in f and "3_skel.png" in f],
                key=lambda x: sort_keys(x))

        print(len(imagelist1))
        print(len(imagelist2))
        print(len(imagelist3))
        print(len(imagelist4))
        print(len(imagelist5))

        max_num_pages = 3750
        num_parts = math.ceil(len(imagelist1) / max_num_pages)

        if do_pdf:
            for part in range(num_parts):
                pdf = FPDF('L', 'pt', (2004, 2854))
                pdf.set_font('Arial', 'B', 32)
                pdf.set_text_color(0, 0, 0)

                start = part * max_num_pages
                stop = min((part + 1) * max_num_pages, len(imagelist1))
                # imagelist is the list with all image filenames
                for i, image in enumerate(imagelist1[start:stop]):
                    text = image.split('/')[-1]
                    textsplit = text.split('_cr_')[1].split('_')
                    em = "EM: " + text.split('_cr_')[0]
                    crankscore = "PatchPerPixMatch rank: " + textsplit[0] + \
                                 ", score: " + textsplit[2]
                    line = '_'.join(textsplit[3:7]).split('-')[0]
                    oldline = line
                    line = "R" + line.split('_')[1]
                    sc = '_'.join(textsplit[6:9]).split('-')[1]
                    lm = 'LM: Line Name ' + line + ', Slide Code ' + sc
                    pdf.add_page()
                    pdf.image(image, 0, 0, 1427, 668)
                    pdf.image(imagelist2[start + i], 0, 668, 1427, 668)
                    pdf.image(imagelist3[start + i], 0, 1336, 1427, 668)
                    pdf.image(imagelist4[start + i], 1427, 668, 1427, 668)
                    pdf.image(imagelist5[start + i], 1427, 1336, 1427, 668)
                    pdf.set_xy(1500, 50)
                    pdf.cell(0, txt=em, align='L')
                    pdf.set_xy(1500, 100)
                    pdf.cell(0, txt=lm, align='L')
                    pdf.set_xy(1500, 150)
                    pdf.cell(0, txt=crankscore, align='L')

                    full_patterns = glob.glob(gen1expr_path + oldline + "/*_bw.png")
                    if len(full_patterns) > 0:
                        pdf.image(full_patterns[0], 1500, 250, 714, 334)

                    if (i % 20 == 0):
                        print("adding page %i of %i" % (i + start, len(imagelist1)))

                if do_cscore:
                    if num_parts > 1:
                        filename = "PatchPerPixMatch_top_" + str(
                            howmany) + "_ranks_overview_nblast" + nblast_setup + "_part_" + str(part) + "_of_" + str(
                            num_parts) + ".pdf"
                    else:
                        filename = args.result_path + args.result_filename

                dummyfilename = args.result_path + "dummy" + nblast_setup + ".pdf"
                print("printing %s" % dummyfilename)
                pdf.output(dummyfilename, "F")

                # add bookmarks:
                output = PdfFileWriter()  # open output
                output.setPageMode("/UseOutlines")  # This is what tells the PDF to open to bookmarks

                print("opening %s" % dummyfilename)
                inp = PdfFileReader(open(dummyfilename, 'rb'))  # open input

                for i, image in enumerate(imagelist1[start:stop]):

                    if i % howmany == 0:
                        line_dict = {}

                    text = image.split('/')[-1]
                    textsplit = text.split('_cr_')[1].split('_')
                    em = text.split('_cr_')[0]
                    line_name = '_'.join(textsplit[3:7]).split('-')[0]
                    line_name = "R" + line_name.split('_')[1]
                    if line_name in list(line_dict.keys()):
                        line_dict[line_name]['insert_at'] += 1
                    else:
                        line_dict[line_name] = {}
                        line_dict[line_name]['insert_at'] = i
                        line_dict[line_name]['parent'] = None
                    insert_at = line_dict[line_name]['insert_at']
                    for l in list(line_dict.keys()):
                        if l != line_name:
                            ia = line_dict[l]['insert_at']
                            if ia >= insert_at:
                                line_dict[l]['insert_at'] += 1
                    scs = '_'.join(textsplit[6:9])
                    line = line_name + "." + scs
                    crank = textsplit[0]
                    # insert page
                    output.insertPage(inp.getPage(i), index=insert_at)
                    if i % howmany == 0:
                        em_parent = output.addBookmark(em, insert_at, parent=None)  # add bookmark
                    ranktext = crank + ": " + line
                    if line_dict[line_name]['parent'] is None:
                        # add bookmark
                        line_dict[line_name]['parent'] = output.addBookmark(
                            ranktext, insert_at, parent=em_parent)
                    else:
                        lm_parent = line_dict[line_name]['parent']
                        output.addBookmark(ranktext, insert_at, parent=lm_parent)

                    if (i % 20 == 0):
                        print("adding bookmark for page %i of %i" % (i + start, len(imagelist1)))

                outputStream = open(filename, 'wb')  # creating result pdf JCT
                output.write(outputStream)  # writing to result pdf JCT
                outputStream.close()  # closing result JCT

        if do_spreadsheet:
            # create spreadsheet:
            spreadsheet_basename = args.result_path + (args.result_filename).replace(".pdf", "")
            spreadsheetname = spreadsheet_basename + ".csv"
            for i, image in enumerate(imagelist1):
                text = image.split('/')[-1]
                textsplit = text.split('_cr_')[1].split('_')
                line = '_'.join(textsplit[3:7]).split('-')[0]
                line = "R" + line.split('_')[1]
                sc = '_'.join(textsplit[6:9]).split('-')[1]
                crank = textsplit[0]

                if i % howmany == 0:
                    fs = open(spreadsheetname, "w")
                    text = "Line Name,Slide Code,rank\n"
                    fs.write(text)

                text = line + "," + sc + "," + crank + "\n"
                fs.write(text)
                if i % howmany == howmany - 1:
                    fs.close()

            xlsx_filename = spreadsheet_basename + ".xlsx"
            writer = pd.ExcelWriter(
                xlsx_filename)
            for i, image in enumerate(imagelist1):
                if i % howmany == 0:
                    text = image.split('/')[-1]
                    textsplit = text.split('_')
                    em = textsplit[0]

                    df = pd.read_csv(spreadsheetname, index_col=0)
                    df.to_excel(writer, sheet_name=em)
                    writer.save()


if __name__ == "__main__":
    main()
