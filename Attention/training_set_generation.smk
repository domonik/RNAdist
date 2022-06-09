from training_set_generation import create_random_fasta, training_set_from_fasta

rule all:
    input:
        index_dir = expand("Datasets/generation/labels/new_random_s{start}_e{end}_tmp{temp}/", start=[40], end=[200, 400], temp=[37, 35])



rule generate_random_fasta:
    output:
        fasta = "Datasets/generation/random_{start}_{end}.fasta"
    run:
        create_random_fasta(output.fasta, 200000, (int(wildcards.start), int(wildcards.end)), seed=0)


rule generate_labels:
    input:
        fasta = rules.generate_random_fasta.output.fasta
    output:
        dir = directory("Datasets/generation/labels/new_random_s{start}_e{end}_tmp{temp}/"),
    threads: 100
    run:
        config = {"temperature": float(wildcards.temp)}
        training_set_from_fasta(input.fasta, output.dir, config, num_threads=threads, nr_samples=1000)

