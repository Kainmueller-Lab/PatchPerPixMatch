import argparse
import glob
import json

from deepmerge import always_merger


def get_arguments():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-folder", type=str, dest="input_folder",
        required=True
    )
    parser.add_argument("--output-folder",
                        type=str,
                        dest="output_folder"
                        )
    parser.add_argument("--output-suffix",
                        type=str,
                        dest="output_suffix"
                        )

    args = parser.parse_args()
    return args


def main():
    args = get_arguments()
    print(args)

    input_folder = args.input_folder

    substitute_files = glob.glob(input_folder + "/*.json")
    assert (len(substitute_files))

    print("loading %s" % substitute_files[0])
    score_dict = json.load(open(substitute_files[0]))
    for substitute_file in substitute_files[1:]:
        # print("loading %s" % substitute_file)
        score_dict_2 = json.load(open(substitute_file))
        score_dict = always_merger.merge(score_dict, score_dict_2)
    # sort:
    for em in list(score_dict.keys()):
        score_dict[em] = {k: v for k, v in sorted((score_dict[em]).items(),
                                                  key=lambda item: item[1][0],
                                                  reverse=True)}

        score_file = args.output_folder + "/nblastScores_" + em + args.output_suffix + ".json"
        with open(score_file, 'w') as fp:
            dummy = {em: score_dict[em]}
            json.dump(dummy, fp, indent=4)


if __name__ == "__main__":
    main()
