import multiprocessing

import numpy as np
from Bio import SeqIO
from RNAdist.dp.pmcomp import pmcomp_distance
import pandas as pd
from RNAdist.dp.cpedistance import cp_expected_distance, binding_site_distance
from RNAdist.sampling.ed_sampling import sample
from multiprocessing import Pool
from RNAdist.dp.viennarna_helpers import set_md_from_config
import RNA
from typing import Dict, Any, Callable, List, Tuple, Union
import pickle
import os


BSCSVHEADER = ["sequence_name", "name1", "start1", "stop1", "name2", "start2", "stop2", "expected_distance"]


def _fasta_wrapper(func: Callable, fasta: str, md_config: Dict[str, Any], num_threads: int = 1, sample: int = None):
    calls = []
    desc = []
    if sample:
        for sr in SeqIO.parse(fasta, "fasta"):
            calls.append((str(sr.seq), md_config, sample))
            desc.append(sr.description)
    else:
        for sr in SeqIO.parse(fasta, "fasta"):
            calls.append((str(sr.seq), md_config))
            desc.append(sr.description)
    if num_threads <= 1:
        data = [func(*call) for call in calls]
    else:
        with Pool(num_threads) as pool:
            data = pool.starmap(func, calls)
    data = {desc[x]: d for x, d in enumerate(data)}
    return data


def _pmcomp_mp_wrapper(seq, md_config):
    md = RNA.md()
    md = set_md_from_config(md, config=md_config)
    return pmcomp_distance(sequence=seq, md=md)


def _cp_mp_wrapper(seq, md_config):
    md = RNA.md()
    md = set_md_from_config(md, config=md_config)
    return cp_expected_distance(sequence=seq, md=md)


def _sampling_mp_wrapper(seq, md_config, nr_samples):
    md = RNA.md()
    md = set_md_from_config(md, config=md_config)
    return sample(seq, nr_samples, md)


def pmcomp_from_fasta(fasta: str, md_config: Dict[str, Any], num_threads: int = 1):
    """Calculates the pmcomp matrix for every sequence in a fasta file

    Args:
        fasta (str): Path to a fasta file
        md_config (Dict[str, Any]): A dictionary containing keys and values to set up the ViennaRNA Model details
        num_threads (int): number of parallel processes to use
    Returns:
        Dict[str, np.ndarray]: Dictionary of Numpy arrays of shape :code:`|S| x |S|` with S being the nucleotide sequence.
        The sequence identifier is the dict key and the expected distance matrices are the values
    """
    return _fasta_wrapper(_pmcomp_mp_wrapper, fasta, md_config, num_threads)


def clote_ponty_from_fasta(fasta: str, md_config: Dict[str, Any], num_threads: int = 1):
    """Calculates the clote-ponty matrix for every sequence in a fasta file

       Args:
           fasta (str): Path to a fasta file
           md_config (Dict[str, Any]): A dictionary containing keys and values to set up the ViennaRNA Model details
           num_threads (int): number of parallel processes to use
       Returns:
           Dict[str, np.ndarray]: Dictionary of Numpy arrays of shape :code:`|S| x |S|` with S being the nucleotide sequence.
           The sequence identifier is the dict key and the expected distance matrices are the values
       """
    return _fasta_wrapper(_cp_mp_wrapper, fasta, md_config, num_threads)


def sampled_distance_from_fasta(fasta: str, md_config: Dict[str, Any], num_threads: int = 1, nr_samples: int = 1000):
    """Calculates the averaged distance matrix for every sequence in a fasta file using probabilistic backtracking

       Args:
           fasta (str): Path to a fasta file
           md_config (Dict[str, Any]): A dictionary containing keys and values to set up the ViennaRNA Model details
           num_threads (int): number of parallel processes to use
           nr_samples (int): How many samples to average the distance
       Returns:
           Dict[str, np.ndarray]: Dictionary of Numpy arrays of shape :code:`|S| x |S|` with S being the nucleotide sequence.
           The sequence identifier is the dict key and the expected distance matrices are the values
       """
    return _fasta_wrapper(_sampling_mp_wrapper, fasta, md_config, num_threads, nr_samples)


def _read_beds(beds, names):
    sites = {}
    for idx, bed in enumerate(beds):
        if names is None:
            filename = os.path.basename(bed)
        else:
            filename = names[idx]
        with open(bed, "r") as handle:
            for line in handle:
                line = line.split("\t")
                start = int(line[1])
                end = int(line[2])
                name = line[0]
                if name not in sites:
                    sites[name] = []
                sites[name].append((start, end, filename))
    return sites


def _output_iterator(results):
    for entry in results:
        s1 = entry[0][0]
        s2 = entry[0][1]
        yield entry[2], s1[2], s1[0], s1[1], s2[2], s2[0], s2[1], entry[1]


def bed_distance_wrapper(
        fasta: str,
        beds: List[str],
        md_config: Dict[str, Any],
        names: List[str] = None,
        num_threads: int = 1
):
    """ Command Line Wrapper for binding site distance calculation.

    This will use fasta files and bed files of binding sites to calculate expected distances
    of pairwise binding sites. The output will be stored in a TSV file specified via outfile.

    Args:
        fasta (str): Path to a fasta file
        beds (List[str]): List of multiple bed file paths.
            The bed files must contain first columns matching entries from the fasta headers
        md_config (Dict[str, Any]): A dictionary containing keys and values to set up the ViennaRNA Model details
        names: The names for bed files displayed in the output file if not provided (default) it will use the filenames
        num_threads (int): number of parallel processes to use
    Returns: pd.Dataframe
        pandas dataframe with the following columns:

            "seq_name", "name1", "start1", "stop1", "name2", "start2", "stop2", "expected_distance"

        seq_name is the header within the fasta sequence or the first column in the bed files
        name is the name for the bed file specified via names
        start and stop correspong to the binding sites

    """
    sites = _read_beds(beds, names)
    seqs = {}
    for sr in SeqIO.parse(fasta, "fasta"):
        seqs[sr.description] = str(sr.seq)
    calls = []
    for seq_desc, binding_sites in sites.items():
        if seq_desc not in seqs:
            raise KeyError("Missing sequence in Fasta file")
        seq = seqs[seq_desc]
        for x, bs1 in enumerate(binding_sites):
            for bs2 in binding_sites[x+1:]:
                bs_list = (bs1, bs2)
                bs_list = sorted(bs_list)
                if bs_list[0][1] < bs_list[1][0]:
                    calls.append((seq, md_config, bs_list, seq_desc))
                else:
                    print(f"Skipping sites {bs_list} on {seq_desc} due to overlapping intervals")
    if num_threads > 1:
        with multiprocessing.Pool(num_threads) as pool:
            results = pool.starmap(_bs_wrapper, calls)
    else:
        results = [_bs_wrapper(*call) for call in calls]
    df = pd.DataFrame([x for x in _output_iterator(results)], columns=BSCSVHEADER)
    return df


def _bs_wrapper(seq, md_config, binding_sites, seq_name) -> Tuple[List[Tuple[int, int]], float, str]:
    md = RNA.md()
    md = set_md_from_config(md, config=md_config)
    fc = RNA.fold_compound(seq, md)
    sites = (binding_sites[0][:-1], binding_sites[1][:-1])
    distance = binding_site_distance(fc, sites)
    return binding_sites, distance, seq_name


def _pickle_data(data, outfile):
    with open(outfile, "wb") as handle:
        pickle.dump(data, handle)


def _sampled_distance_executable_wrapper(args):
    md_config = md_config_from_args(args)
    data = sampled_distance_from_fasta(
        fasta=args.input,
        md_config=md_config,
        num_threads=args.num_threads,
        nr_samples=args.nr_samples
    )
    _pickle_data(data, args.output)


def _cp_executable_wrapper(args):
    md_config = md_config_from_args(args)
    data = clote_ponty_from_fasta(
        fasta=args.input,
        md_config=md_config,
        num_threads=args.num_threads,
    )
    _pickle_data(data, args.output)


def _pmcomp_executable_wrapper(args):
    md_config = md_config_from_args(args)
    data = pmcomp_from_fasta(
        fasta=args.input,
        md_config=md_config,
        num_threads=args.num_threads,
    )
    _pickle_data(data, args.output)


def _bs_bed_executable_wrapper(args):
    md_config = md_config_from_args(args)
    df = bed_distance_wrapper(
        fasta=args.input,
        md_config=md_config,
        names=args.names,
        beds=args.bed_files,
        num_threads=args.num_threads
    )
    df.to_csv(args.output, sep="\t", index=False)


def md_config_from_args(args):
    md_config = {
        "temperature": args.temperature,
        "min_loop_size": args.min_loop_size,
        "noGU": args.noGU,
    }
    return md_config


def export_array(array: np.ndarray, file: str):
    df = pd.DataFrame(array)
    df.to_csv(file, sep="\t")


def export_all(output_data: Dict[str, np.ndarray], outdir: Union[str, os.PathLike]):
    """Extracts expected distances from a pickled output file and stores them in TSV files.
    Therefore, it uses the original FASTA identifiers as filenames.

    Args:
        output_data (Dict): The Data from which the output should be extracted.
        outdir (str, PathLike): The directory where TSV output files will be stored
    """
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    if not os.path.isdir(outdir):
        raise FileExistsError("The specified path does not look like a directory")
    for key, value in output_data.items():
        key = key.replace(" ", "_")
        out_path = os.path.join(outdir, f"{key}.tsv")
        export_array(value, out_path)


def _export_all_cmd(args):
    with open(args.data_file, "rb") as handle:
        output_data = pickle.load(handle)
    export_all(output_data, args.outdir)
