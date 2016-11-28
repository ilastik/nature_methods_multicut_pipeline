import subprocess
import argparse
import os

# evaluate proposals against labels with the metric from isbi12 challenge

def eval_lazy(proposal_edges, rag):
    import vigra
    import numpy as np
    from Tools import edges_to_binary

    labels_path = "/home/consti/Work/nature_experiments/isbi12_data/groundtruth/train-labels.tif"
    temp_proposal = "/tmp/proposal_isbi12.tif"

    proposal_vol = edges_to_binary(rag, proposal_edges)

    vigra.impex.writeVolume(proposal_vol, temp_proposal, '', dtype = np.uint8 )
    ri, vi = eval_with_isbi_metrics(labels_path, temp_proposal)

    os.remove(temp_proposal)

    return ri, vi


def eval_with_isbi_metrics(labels_path, proposal_path,
        fiji_exe = "/home/consti/Desktop/FIJI",
        fiji_script = "/home/consti/Work/multicut_pipeline/software/multicut_exp/isbi12/isbi12_eval_script.bsh"):

    tmp_file = "/tmp/isbi_metrics_tmp.txt"
    #curr_dir = os.getcwd()
    #os.chdir("/home/consti/Desktop")
    subprocess.call([fiji_exe,
        fiji_script, labels_path, proposal_path, tmp_file])
    #os.chdir(curr_dir)
    # read the temp file
    with open(tmp_file, 'r') as f:
        ri_score = float(f.readline())
        vi_score = float(f.readline())
    # delete the tmp file
    os.remove(tmp_file)
    return ri_score, vi_score


def process_command_line():
    parser = argparse.ArgumentParser(
            description='Input for isbiscript')

    parser.add_argument('labels_path', type=str, help = 'Path to groundtruth labels')
    parser.add_argument('proposal_path',  type=str, help = 'Path to proposed labels')
    parser.add_argument('--fiji_exe', type = str,
            default = "/home/consti/Desktop/FIJI",
            help = 'Path to fiji executable')
    parser.add_argument('--fiji_script', type = str,
            default = "/home/consti/Work/multicut_pipeline/software/multicut_exp/isbi12/isbi12_eval_script.bsh",
            help = 'Path to beanshell script with metric')

    args = parser.parse_args()

    return args


def main():
    args = process_command_line()
    ri, vi = eval_with_isbi_metrics(args.labels_path, args.proposal_path,
            args.fiji_exe, args.fiji_script)

    print "RandIndex:", ri
    print "VoI:", vi


if __name__ == '__main__':
    main()
