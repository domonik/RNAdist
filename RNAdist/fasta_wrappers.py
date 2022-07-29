from Bio import SeqIO
from RNAdist.DPModels.pmcomp import pmcomp_distance
from RNAdist.DPModels.clote import cp_expected_distance
from RNAdist.sampling.expected_length_sampling import sample_distance
from multiprocessing import Pool
from RNAdist.DPModels.viennarna_helpers import set_md_from_config
import RNA
from typing import Dict, Any, Callable
import pickle


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
    return sample_distance(seq, nr_samples, md)[0]


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


def md_config_from_args(args):
    md_config = {
        "temperature": args.temperature,
        "min_loop_size": args.min_loop_size,
        "noGU": args.noGU,
    }
    return md_config