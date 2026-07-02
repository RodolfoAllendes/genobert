#!/usr/bin/env python3
"""
VCF Preparation Script

Produces the split VCFs required by pretrain_data_prep.py as a two-step pipeline:

  Step 'clean': Deduplicates variants, filters to biallelic SNPs, normalizes.
                Chr6 also extracts the HLA region as a separate file.
                Input:  {raw_vcf}
                Output: {output_dir}/ref/{raw_vcf_stem}_ref.vcf.gz

  Step 'split': Assigns superpopulation labels from PED metadata, creates
                stratified 8:1:1 train/val/test splits (stratified by
                superpopulation x sex), applies HWE+MAF filters, and
                restricts all splits to the common post-QC SNP set.
                Input:  {output_dir}/ref/{raw_vcf_stem}_ref.vcf.gz
                Output: {output_dir}/split/{raw_vcf_stem}_{pop}_{split}.vcf.gz

Usage:
    python vcf_prep.py --step clean --chr 22
    python vcf_prep.py --step split --chr 22
    python vcf_prep.py --step all   --chr 22
"""

import argparse
import pandas as pd
import os
import random
import subprocess
import time
from collections import Counter, defaultdict
from shlex import quote


POPULATION_TO_SUPERPOP = {
    "CHB": "EAS", "JPT": "EAS", "CHS": "EAS", "CDX": "EAS", "KHV": "EAS", "CHD": "EAS",
    "CEU": "EUR", "TSI": "EUR", "GBR": "EUR", "FIN": "EUR", "IBS": "EUR",
    "YRI": "AFR", "LWK": "AFR", "GWD": "AFR", "MSL": "AFR", "ESN": "AFR",
    "ASW": "AMR", "ACB": "AMR", "MXL": "AMR", "PUR": "AMR", "CLM": "AMR", "PEL": "AMR",
    "GIH": "SAS", "PJL": "SAS", "BEB": "SAS", "STU": "SAS", "ITU": "SAS",
}

ALL_SUPERPOPS = ["EAS", "EUR", "AFR", "AMR", "SAS", "ALL"]
SPLITS = ["train", "val", "test"]


# ── helpers ───────────────────────────────────────────────────────────────────

def run(cmd, *, shell=False, check=True):
    display = cmd if isinstance(cmd, str) else " ".join(cmd)
    print(f"  + {display}")
    result = subprocess.run(cmd, shell=shell, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"STDERR:\n{result.stderr.strip()}")
        raise RuntimeError(f"Command failed (exit {result.returncode}): {display}")
    return result


def tabix(path):
    run(["tabix", "-p", "vcf", path])


def _chrom_sort_key(line):
    """Sort key for CHROM\tPOS lines that handles numeric and XY chromosomes."""
    chrom, pos = line.split("\t")
    c = chrom.replace("chr", "")
    try:
        return (int(c), int(pos))
    except ValueError:
        return ({"X": 23, "Y": 24, "MT": 25, "M": 25}.get(c, 99), int(pos))


def _vcf_stem(path):
    """Basename of a VCF path with .vcf.gz (or .vcf) stripped."""
    base = os.path.basename(path)
    if base.endswith(".vcf.gz"):
        return base[:-7]
    if base.endswith(".vcf"):
        return base[:-4]
    return base


# ── step: clean ───────────────────────────────────────────────────────────────

def step_clean(args, chr_str):
    """Deduplicate, filter biallelic SNPs, normalize; handle chr6 HLA region."""
    tmp = os.path.join(args.qc_dir, "temp")
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(args.ref_dir, exist_ok=True)

    stem = _vcf_stem(args.raw_vcf)
    print(f"\n[clean] chr{chr_str}")
    print(f"  Input: {args.raw_vcf}")

    no_dup    = os.path.join(tmp, f"{stem}_no_dup.vcf.gz")
    biallelic = os.path.join(tmp, f"{stem}_biallelic.vcf.gz")
    ref_vcf   = os.path.join(args.ref_dir, f"{stem}_ref.vcf.gz")

    # 1. Remove duplicate variants
    run(["bcftools", "norm", "-d", "all",
         "--threads", str(args.threads), "-Oz", "-o", no_dup, args.raw_vcf])

    # 2. Keep only biallelic SNPs
    run(["bcftools", "view", "--min-alleles", "2", "--max-alleles", "2", "-v", "snps",
         "--threads", str(args.threads), "-Oz", "-o", biallelic, no_dup])

    # 3. Split multi-allelic sites; chr6 also extracts the HLA region
    run(["bcftools", "norm", "-m", "-any",
         "--threads", str(args.threads), "-Oz", "-o", ref_vcf, biallelic])
    tabix(ref_vcf)

    if chr_str == "6":
        hla_vcf = os.path.join(args.ref_dir, f"{stem}_HLA.vcf.gz")
        run(["bcftools", "view",
             "-r", f"6:{args.hla_start}-{args.hla_end}",
             "--threads", str(args.threads), "-Oz", "-o", hla_vcf, ref_vcf])
        tabix(hla_vcf)
        print(f"  HLA region → {hla_vcf}")

    if not args.keep_intermediates:
        for f in [no_dup, biallelic]:
            if os.path.exists(f):
                os.remove(f)

    print(f"  → {ref_vcf}")
    return ref_vcf


# ── step: split ───────────────────────────────────────────────────────────────

def assign_populations(args, ref_vcf):
    """
    Cross-reference VCF samples with PED metadata to assign superpopulation
    labels. Writes per-superpopulation sample ID files and a combined
    population-info file used for stratified splitting.

    Returns path to the population_info file.
    """
    stem = _vcf_stem(args.raw_vcf)

    result = run(["bcftools", "query", "-l", ref_vcf])
    raw_ids = result.stdout.strip().splitlines()
    # Strip leading "0_" prefix sometimes present in 1KGP VCFs
    vcf_samples = {(sid[2:] if sid.startswith("0_") else sid) for sid in raw_ids}

    meta = pd.read_csv(args.metadata_file, sep="\t", usecols=["sample", "super_pop", "gender"])
    meta = meta[meta["super_pop"].isin(set(POPULATION_TO_SUPERPOP.values()))]
    meta = meta[meta["sample"].isin(vcf_samples)]

    pop_samples = dict(zip(meta["sample"], zip(meta["super_pop"], meta["gender"])))

    # Include VCF samples not in the panel under a dummy superpop so they
    # appear in the ALL split even without metadata.
    unmatched = vcf_samples - set(pop_samples)
    for sid in sorted(unmatched):
        pop_samples[sid] = ("UNK", "unknown")

    print(f"  Metadata samples : {len(meta)}")
    print(f"  VCF samples      : {len(vcf_samples)}")
    print(f"  Common           : {len(pop_samples) - len(unmatched)}")
    print(f"  Unmatched (→ ALL): {len(unmatched)}")
    print(f"  Distribution     : {dict(Counter(sp for sp, _ in pop_samples.values()))}")

    tmp = os.path.join(args.qc_dir, "temp")
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(args.subset_dir, exist_ok=True)

    pop_info_path = os.path.join(tmp, f"{stem}_population_info.txt")
    with open(pop_info_path, "w") as fh:
        for sample, (sp, sex) in pop_samples.items():
            fh.write(f"{sample}\t{sp}\t{sex}\n")

    for sp in args.populations:
        ids_path = os.path.join(args.subset_dir, f"{stem}_{sp}_ids.txt")
        with open(ids_path, "w") as fh:
            for sample, (race, _) in pop_samples.items():
                if sp == "ALL" or race == sp:
                    fh.write(f"{sample}\n")

    return pop_info_path


def stratified_split(pop_info_path, superpop, train_ratio, val_ratio, seed):
    """
    8:1:1 stratified split by (superpopulation, sex) bins.
    Returns {"train": [...], "val": [...], "test": [...]}.
    """
    bins = defaultdict(list)
    with open(pop_info_path) as fh:
        for line in fh:
            sample, race, sex = line.strip().split("\t")
            bins[f"{race}_{sex}"].append(sample)

    result = {"train": [], "val": [], "test": []}
    for key, samples in bins.items():
        if superpop != "ALL" and not key.startswith(superpop):
            continue
        rng = random.Random(seed)
        shuffled = samples.copy()
        rng.shuffle(shuffled)
        n = len(shuffled)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        result["train"].extend(shuffled[:n_train])
        result["val"].extend(shuffled[n_train:n_train + n_val])
        result["test"].extend(shuffled[n_train + n_val:])

    return result


def get_vcf_positions(vcf_path):
    """Return the set of 'CHROM\\tPOS' strings present in a VCF."""
    out = run(["bcftools", "query", "-f", "%CHROM\t%POS\n", vcf_path])
    return set(out.stdout.strip().splitlines())


def step_split(args, chr_str):
    """Population filtering + stratified splitting + HWE/MAF QC."""
    tmp = os.path.join(args.qc_dir, "temp")
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(args.split_dir, exist_ok=True)

    stem = _vcf_stem(args.raw_vcf)
    ref_vcf = os.path.join(args.ref_dir, f"{stem}_ref.vcf.gz")
    if not os.path.exists(ref_vcf):
        raise FileNotFoundError(
            f"Reference VCF not found: {ref_vcf}\n"
            "Run --step clean first."
        )

    print(f"\n[split] chr{chr_str}")
    print(f"  Input: {ref_vcf}")

    pop_info_path = assign_populations(args, ref_vcf)

    for pop in args.populations:
        print(f"\n  Population: {pop}")

        # 1. Extract population-specific VCF (ALL reuses ref_vcf directly)
        if pop == "ALL":
            pop_vcf = ref_vcf
        else:
            pop_vcf = os.path.join(tmp, f"{stem}_{pop}.vcf.gz")
            ids_file = os.path.join(args.subset_dir, f"{stem}_{pop}_ids.txt")
            run(["bcftools", "view", "-S", ids_file,
                 "--threads", str(args.threads), "-Oz", "-o", pop_vcf, ref_vcf])
            tabix(pop_vcf)

        # 2. Stratified 8:1:1 split
        split_samples = stratified_split(
            pop_info_path, pop, args.train_ratio, args.val_ratio, args.seed
        )
        for sp, samples in split_samples.items():
            ids_path = os.path.join(args.subset_dir, f"{stem}_{pop}_{sp}_ids.txt")
            with open(ids_path, "w") as fh:
                fh.write("\n".join(sorted(samples)))
            print(f"    {sp}: {len(samples)} samples")

        # 3. Create initial split VCFs
        for split in SPLITS:
            ids_file = os.path.join(args.subset_dir, f"{stem}_{pop}_{split}_ids.txt")
            initial = os.path.join(tmp, f"{stem}_{pop}_{split}_initial.vcf.gz")
            run(["bcftools", "view", "-S", ids_file,
                 "--threads", str(args.threads), "-Oz", "-o", initial, pop_vcf])
            tabix(initial)

        # 4. HWE + MAF filtering per split
        for split in SPLITS:
            initial = os.path.join(tmp, f"{stem}_{pop}_{split}_initial.vcf.gz")
            tagged = os.path.join(tmp, f"{stem}_{pop}_{split}_hwe.vcf.gz")
            filtered = os.path.join(tmp, f"{stem}_{pop}_{split}.vcf.gz")

            run(["bcftools", "+fill-tags", initial, "-Oz", "-o", tagged, "--", "-t", "HWE"])
            run(["bcftools", "filter",
                 "--include", f"MAF>={args.maf_min} & MAF<={args.maf_max} & HWE>{args.hwe_threshold}",
                 "--threads", str(args.threads), "-Oz", "-o", filtered, tagged])
            tabix(filtered)

            n_in = run(["bcftools", "query", "-f", "%ID\n", initial]).stdout.count("\n")
            n_out = run(["bcftools", "query", "-f", "%ID\n", filtered]).stdout.count("\n")
            print(f"    {split}: {n_in:,} → {n_out:,} SNPs after HWE+MAF")

        # 5. Intersect post-QC SNP sets across all three splits
        positions = {
            split: get_vcf_positions(
                os.path.join(tmp, f"{stem}_{pop}_{split}.vcf.gz")
            )
            for split in SPLITS
        }
        common_pos = positions["train"] & positions["val"] & positions["test"]
        print(f"    Common SNPs: {len(common_pos):,}")

        pos_file = os.path.join(tmp, f"{stem}_{pop}_common_positions.txt")
        with open(pos_file, "w") as fh:
            for line in sorted(common_pos, key=_chrom_sort_key):
                fh.write(line + "\n")

        # 6. Final VCFs: common SNPs only, GT format field only
        for split in SPLITS:
            filtered = os.path.join(tmp, f"{stem}_{pop}_{split}.vcf.gz")
            final = os.path.join(args.split_dir, f"{stem}_{pop}_{split}.vcf.gz")
            cmd = (
                f"bcftools view -R {quote(pos_file)} {quote(filtered)} | "
                f"bcftools annotate -x FORMAT,^FORMAT/GT "
                f"--threads {args.threads} -Oz -o {quote(final)}"
            )
            run(cmd, shell=True)
            tabix(final)
            print(f"    → {final}")

        if not args.keep_intermediates:
            for split in SPLITS:
                for suffix in [
                    "_initial.vcf.gz", "_initial.vcf.gz.tbi",
                    "_hwe.vcf.gz", "_hwe.vcf.gz.tbi",
                    ".vcf.gz", ".vcf.gz.tbi",
                ]:
                    p = os.path.join(tmp, f"{stem}_{pop}_{split}{suffix}")
                    if os.path.exists(p):
                        os.remove(p)
            to_remove = [pos_file]
            if pop_vcf != ref_vcf:
                to_remove += [pop_vcf, pop_vcf + ".tbi"]
            for p in to_remove:
                if os.path.exists(p):
                    os.remove(p)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clean raw 1KGP VCFs and create stratified train/val/test splits."
    )

    parser.add_argument("--step", required=True, choices=["clean", "split", "all"],
                        help="Pipeline step: clean raw VCFs, split into train/val/test, or both")
    parser.add_argument("--chr", required=True,
                        help="Chromosome to process (e.g. 22, 6, X)")
    # Input file naming
    parser.add_argument(
        "--raw_vcf",
        required=True,
        help="Path to the raw input VCF file for this chromosome",
    )
    parser.add_argument(
        "--metadata_file",
        default="/mnt/storage4/rallendes/snp_data/1KGP/integrated_call_samples_v3.20130502.ALL.panel",
        help="Path to 1KGP PED metadata file",
    )

    # Directories
    parser.add_argument("--output_dir", default=".",
                        help="Root output directory; ref/, split/, subsets/, and qc/ are created inside it (default: .)")

    # Split parameters
    parser.add_argument("--populations", nargs="+", default=ALL_SUPERPOPS,
                        help="Superpopulations to process (default: EAS EUR AFR AMR SAS ALL)")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="Training fraction (default: 0.8)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Validation fraction (default: 0.1; remainder goes to test)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducible splits (default: 42)")

    # QC thresholds
    parser.add_argument("--hwe_threshold", type=float, default=1e-6,
                        help="Minimum HWE p-value to retain a SNP (default: 1e-6)")
    parser.add_argument("--maf_min", type=float, default=0.001,
                        help="Minimum MAF (default: 0.001)")
    parser.add_argument("--maf_max", type=float, default=0.999,
                        help="Maximum MAF (default: 0.999)")

    # chr6 HLA coordinates
    parser.add_argument("--hla_start", type=int, default=28477797,
                        help="HLA region start position on chr6 (default: 28477797)")
    parser.add_argument("--hla_end", type=int, default=33448354,
                        help="HLA region end position on chr6 (default: 33448354)")

    # Misc
    parser.add_argument("--threads", type=int, default=4,
                        help="bcftools thread count (default: 4)")
    parser.add_argument("--keep_intermediates", action="store_true",
                        help="Retain intermediate files in qc/temp/ for debugging")

    args = parser.parse_args()
    chr_str = str(args.chr)

    args.ref_dir    = os.path.join(args.output_dir, "ref")
    args.split_dir  = os.path.join(args.output_dir, "split")
    args.subset_dir = os.path.join(args.output_dir, "subsets")
    args.qc_dir     = os.path.join(args.output_dir, "qc")

    if args.step in ("split", "all") and not os.path.exists(args.metadata_file):
        parser.error(f"--metadata_file not found: {args.metadata_file}")

    start = time.time()

    if args.step in ("clean", "all"):
        step_clean(args, chr_str)

    if args.step in ("split", "all"):
        step_split(args, chr_str)

    elapsed = time.time() - start
    h, m = divmod(int(elapsed), 3600)
    m //= 60
    print(f"\nCompleted chr{chr_str} in {h}h {m}m ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
