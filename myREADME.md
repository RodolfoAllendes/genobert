# GenoBERT — Data Preparation Pipeline

This document tracks the complete data preparation pipeline for training GenoBERT
on the 1000 Genomes Project (1KGP) dataset, including scripts added beyond the
original repository.

---

## Data Sources

| Data | Location |
|------|----------|
| Raw 1KGP VCFs (GRCh38) | `/mnt/storage4/rallendes/snp_data/1KGP/20190312_biallelic_SNV_and_INDEL/` |
| Population metadata | `/mnt/storage4/rallendes/snp_data/1KGP/integrated_call_samples_v3.20130502.ALL.panel` |
| GENCODE v50 GTF (GRCh38) | `/mnt/storage4/rallendes/snp_data/genecode/gencode.v50.annotation.gtf.gz` |
| Split VCFs (output) | `dataset/1KGP/split/` |
| Gene region files (output) | `dataset/genecode/split/` |

---

## Full Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 0 | `vcf_prep.py` | Clean raw VCFs; create population-stratified train/val/test splits |
| 0b | `gene_regions_prep.py` | Build gene coordinate files from GENCODE GTF |
| 1 | `pretrain_data_prep.py` | Create per-gene HDF5 files from split VCFs |
| 2 | `merge_genes.py` | Merge per-gene HDF5 files with deduplication |
| 3 | `pretrain.py` | Train the model |
| 4 | `test_pretrain.py` | Evaluate the trained model |

---

## Step 0 — VCF Preparation (`vcf_prep.py`)

Cleans raw 1KGP VCFs and produces population-stratified train/val/test splits.

**Two sub-steps:**
- `clean`: deduplicate variants, keep biallelic SNPs only, normalize multi-allelic sites.
  Chr6 also extracts the HLA region as a separate file.
- `split`: assigns superpopulation labels from the panel metadata, creates an **8:1:1
  stratified split by superpopulation × sex** (`random.seed(42)`), applies HWE > 1e-6
  and 0.001 ≤ MAF ≤ 0.999 filters per split, then restricts all three splits to the
  intersection of variants passing QC.

**Output structure:**
```
dataset/1KGP/
├── ref/      # cleaned reference VCF (all populations)
├── split/    # final VCFs: {stem}_{pop}_{split}.vcf.gz
├── subsets/  # sample ID lists per population and split
└── qc/       # intermediate files
```

**Command (chr22):**
```bash
python vcf_prep.py \
    --step all \
    --chr 22 \
    --raw_vcf /mnt/storage4/rallendes/snp_data/1KGP/20190312_biallelic_SNV_and_INDEL/ALL.chr22.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz \
    --metadata_file /mnt/storage4/rallendes/snp_data/1KGP/integrated_call_samples_v3.20130502.ALL.panel \
    --output_dir ./dataset/1KGP \
    --threads 8
```

**SLURM (all chromosomes):**
```bash
sbatch --array=0-23%4 job/00_vcf_prep.batch
```

---

## Step 0b — Gene Region Files (`gene_regions_prep.py`)

`pretrain_data_prep.py` uses gene coordinates to define genomic windows for segmentation.
Only three columns are needed (`TargetID`, `GeneStart`, `GeneEnd`) — no expression values.
All population/split combinations for the same chromosome share the same gene list.

**Command (chr22, all populations):**
```bash
python gene_regions_prep.py \
    --gtf /mnt/storage4/rallendes/snp_data/genecode/gencode.v50.annotation.gtf.gz \
    --chr 22 \
    --output_dir ./dataset/genecode/split \
    --gene_ds genecode \
    --populations EAS EUR AFR AMR SAS ALL
```

Produces 15 files (6 populations × 3 splits) under `dataset/genecode/split/{pop}/`.
Add `--all_gene_types` to include lncRNA and other biotypes beyond protein-coding.

---

## Step 1 — Per-Gene HDF5 Files (`pretrain_data_prep.py`)

Creates per-gene HDF5 files from split VCFs using global SNP-index segmentation.
The VCF loading phase (~1000s for chr22/ALL) is single-threaded; parallelism starts
only after the full chromosome is loaded into memory.

**Parallelism notes:**
- This is a CPU-only step — set `OMP_NUM_THREADS=1` etc. to prevent numpy from
  spawning internal threads that compete with the worker processes.
- With 30 workers writing to NFS simultaneously some files may be silently corrupted.
  Use `--skip_existing` on re-runs so valid files from previous runs are preserved.
- Reduce `--num_workers` (8–10) when retrying failures to lower NFS contention.

**Command (chr22, ALL population, train split):**
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python pretrain_data_prep.py \
    --genotype_ds 1KGP \
    --gene_ds genecode \
    --chr 22 \
    --race ALL \
    --gene_pop ALL \
    --split train \
    --node_id 0 \
    --total_nodes 1 \
    --pretrain_vcf dataset/1KGP/split/ALL.chr22.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased_ALL_train.vcf.gz \
    --gene_exp_path dataset/genecode/split \
    --output_dir ./res_pt/1KGP \
    --model_input_width 258 \
    --overlap_size 8 \
    --num_workers 30 \
    --verbose
```

Repeat with `--split val` and `--split test`.

**Handling NFS write failures:**

Some genes may fail on the first run due to NFS contention. The summary shows
`Saved / Skipped (existing) / No segments / Failed`. To retry only the failed genes:

```bash
# 1. Remove corrupt (partially written) HDF5 files
python - <<'EOF'
import h5py, os, glob
for f in sorted(glob.glob("res_pt/1KGP/train/*.hdf5")):
    try:
        with h5py.File(f, 'r'): pass
    except OSError:
        print(f"Removing: {f}")
        os.remove(f)
EOF

# 2. Re-run with --skip_existing and fewer workers to reduce contention
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python pretrain_data_prep.py \
    ... \
    --num_workers 8 \
    --skip_existing \
    --verbose
```

Repeat the clean+retry cycle until `Failed: 0`.

**SLURM:**
```bash
sbatch --array=0-199%50 --cpus-per-task=4 --mem=64G job/01_data_prep_array.batch
```

---

## Step 2 — Merge HDF5 Files (`merge_genes.py`)

Merges per-gene HDF5 files into a single file with optional deduplication.

```bash
python merge_genes.py \
    --input_dir ./res_pt/1KGP/train \
    --prefix 1KGP_chr22_ALL_seg258_overlap8_train \
    --apply_dedup

python merge_genes.py \
    --input_dir ./res_pt/1KGP/val \
    --prefix 1KGP_chr22_ALL_seg258_overlap8_val \
    --apply_dedup
```

---

## Step 3 — Train (`pretrain.py`)

```bash
python pretrain.py --configFile configs/1KGP_chr22_ALL.yaml
```

See `configs/example_pretrain.yaml` for configuration options. Key parameters must
match the data prep settings (`segLen`, `overlap`, `population`, `chromosome`).

---

## Step 4 — Test (`test_pretrain.py`)

Prepare test HDF5 files first (same as Step 1 with `--split test`), then:

```bash
python test_pretrain.py \
    --configFile configs/1KGP_chr22_ALL.yaml \
    --checkpoint checkpoints_pt/1KGP_ALL_chr22/checkpoint_epoch_100.pth \
    --maskProb 0.05 0.15 0.5
```
