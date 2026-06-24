#!/usr/bin/env python3
"""
Gene Regions Preparation Script

Extracts gene coordinates from a GENCODE GTF and produces the tab-separated
files expected by pretrain_data_prep.py:

    {output_dir}/{pop}/{gene_ds}_chr{chr}_{pop}_{split}_gene_exp_peer_adjusted.txt

Each file has three columns: TargetID, GeneStart, GeneEnd.
Since only coordinates are needed (not expression values), all population/split
combinations for the same chromosome share the same gene list.

Usage:
    python gene_regions_prep.py \\
        --gtf /mnt/storage4/rallendes/snp_data/genecode/gencode.v50.annotation.gtf.gz \\
        --chr 22 \\
        --output_dir ./dataset/GEUVADIS/split
"""

import argparse
import gzip
import os
import re


POPULATIONS = ["EAS", "EUR", "AFR", "AMR", "SAS", "ALL"]
SPLITS = ["train", "val", "test"]


def parse_attribute(attr_string, key):
    """Extract a value from a GTF attribute string."""
    match = re.search(rf'{key} "([^"]+)"', attr_string)
    return match.group(1) if match else None


def load_gene_regions(gtf_path, chr_num, gene_types=None):
    """
    Parse GENCODE GTF and return gene records for the given chromosome.

    Args:
        gtf_path:   Path to GTF file (plain or .gz)
        chr_num:    Chromosome number/letter (e.g. '22', 'X')
        gene_types: Set of gene_type values to keep; None means keep all

    Returns:
        List of (gene_id, start, end) tuples, sorted by start position
    """
    chrom = f"chr{chr_num}"
    genes = []

    opener = gzip.open if gtf_path.endswith(".gz") else open

    with opener(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 9:
                continue
            if fields[0] != chrom or fields[2] != "gene":
                continue

            attrs = fields[8]
            gene_id = parse_attribute(attrs, "gene_id")
            gene_type = parse_attribute(attrs, "gene_type")

            if gene_id is None:
                continue
            if gene_types and gene_type not in gene_types:
                continue

            # Strip version suffix from Ensembl ID (e.g. ENSG00000123.4 → ENSG00000123)
            gene_id = gene_id.split(".")[0]
            start = int(fields[3])
            end = int(fields[4])
            genes.append((gene_id, start, end))

    genes.sort(key=lambda x: x[1])
    return genes


def write_gene_file(genes, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("TargetID\tGeneStart\tGeneEnd\n")
        for gene_id, start, end in genes:
            fh.write(f"{gene_id}\t{start}\t{end}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Build gene region files for pretrain_data_prep.py from a GENCODE GTF."
    )
    parser.add_argument("--gtf", required=True,
                        help="Path to GENCODE GTF file (.gtf or .gtf.gz)")
    parser.add_argument("--chr", required=True,
                        help="Chromosome to process (e.g. 22, 6, X)")
    parser.add_argument("--output_dir", default="./dataset/GEUVADIS/split",
                        help="Root output directory (default: ./dataset/GEUVADIS/split)")
    parser.add_argument("--gene_ds", default="GEUVADIS",
                        help="Gene dataset name used in output filenames (default: GEUVADIS)")
    parser.add_argument("--populations", nargs="+", default=POPULATIONS,
                        help="Populations to generate files for (default: all)")
    parser.add_argument("--splits", nargs="+", default=SPLITS,
                        help="Splits to generate files for (default: train val test)")
    parser.add_argument("--gene_types", nargs="+", default=["protein_coding"],
                        help="GENCODE gene_type values to include (default: protein_coding)")
    parser.add_argument("--all_gene_types", action="store_true",
                        help="Include all gene types (overrides --gene_types)")

    args = parser.parse_args()
    chr_str = str(args.chr)
    gene_types = None if args.all_gene_types else set(args.gene_types)

    print(f"Parsing GTF: {args.gtf}")
    print(f"Chromosome : chr{chr_str}")
    print(f"Gene types : {'all' if gene_types is None else sorted(gene_types)}")

    genes = load_gene_regions(args.gtf, chr_str, gene_types)
    print(f"Genes found: {len(genes)}")

    if not genes:
        raise ValueError(f"No genes found for chr{chr_str} with the given filters.")

    # Write identical file for every population/split combination
    written = 0
    for pop in args.populations:
        for split in args.splits:
            path = os.path.join(
                args.output_dir, pop,
                f"{args.gene_ds}_chr{chr_str}_{pop}_{split}_gene_exp_peer_adjusted.txt"
            )
            write_gene_file(genes, path)
            written += 1

    print(f"Written {written} files under {args.output_dir}")
    print(f"Example: {args.output_dir}/{args.populations[0]}/"
          f"{args.gene_ds}_chr{chr_str}_{args.populations[0]}_{args.splits[0]}"
          f"_gene_exp_peer_adjusted.txt")


if __name__ == "__main__":
    main()
