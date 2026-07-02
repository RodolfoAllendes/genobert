# GenoBERT — Data Preparation & Training Pipeline

This document tracks the complete pipeline for training and evaluating GenoBERT,
including scripts added beyond the original repository.

---

## Data Sources

### 1KGP pipeline

| Data | Location |
|------|----------|
| Raw 1KGP VCFs (GRCh38) | `/mnt/storage4/rallendes/snp_data/1KGP/20190312_biallelic_SNV_and_INDEL/` |
| Population metadata | `/mnt/storage4/rallendes/snp_data/1KGP/integrated_call_samples_v3.20130502.ALL.panel` |
| GENCODE v50 GTF (GRCh38) | `/mnt/storage4/rallendes/snp_data/genecode/gencode.v50.annotation.gtf.gz` |
| Split VCFs (output) | `dataset/1KGP/split/` |
| Gene region files (output) | `dataset/genecode/split/` |

### 24donor pipeline

| Data | Location |
|------|----------|
| Reference VCF (3202 samples, 13k SNPs, chr22) | `/mnt/storage2/rallendes/data/scRNAseq/SNP_data/SNP_data_isec_complete/reference_version/chr22.phased_24donor_reference.vcf.gz` |
| Population metadata (same as 1KGP) | `/mnt/storage4/rallendes/snp_data/1KGP/integrated_call_samples_v3.20130502.ALL.panel` |
| Gene region files (reuse 1KGP) | `dataset/genecode/split/` |
| Split VCFs (output) | `dataset/24donor/split/` |
| scRNA-seq test VCFs (18 samples) | `/mnt/storage2/rallendes/data/scRNAseq/Lipid_High/variants/per_donor/possorted_genome_bam/mapphased/chr22/` |
| Ground truth genotypes (18 samples) | `/mnt/storage2/rallendes/data/scRNAseq/SNP_data/SNP_data_isec_complete/chr22.phased_24donor_GTC_all_merged.vcf.gz` |

---

## Full Pipeline Overview

| Step | Script | Description |
|------|--------|-------------|
| 0 | `vcf_prep.py` | Clean raw VCFs; create population-stratified train/val/test splits |
| 0b | `gene_regions_prep.py` | Build gene coordinate files from GENCODE GTF |
| 1 | `pretrain_data_prep.py` | Create per-gene HDF5 files from split VCFs |
| 2 | `merge_genes.py` | Merge per-gene HDF5 files with deduplication |
| 3 | `pretrain.py` | Train the model |
| 4a | `test_pretrain.py` | Internal evaluation (random masking on reference test split) |
| 4b | `eval_pretrain.py` | External evaluation on scRNA-seq samples vs ground truth |

---

## Step 0 — VCF Preparation (`vcf_prep.py`)

Cleans raw VCFs and produces population-stratified train/val/test splits.

**Two sub-steps:**
- `clean`: deduplicate variants, keep biallelic SNPs only, normalize multi-allelic sites.
  Chr6 also extracts the HLA region as a separate file.
- `split`: assigns superpopulation labels from the panel metadata, creates an **8:1:1
  stratified split by superpopulation × sex** (`random.seed(42)`), applies HWE > 1e-6
  and 0.001 ≤ MAF ≤ 0.999 filters per split, then restricts all three splits to the
  intersection of variants passing QC.

**Note:** VCF samples not present in the panel file are assigned to `UNK` superpopulation
and included in the `ALL` split automatically. This handles the 24donor VCF where 698 of
3202 samples have no panel metadata.

**Output structure:**
```
dataset/{name}/
├── ref/      # cleaned reference VCF (all populations)
├── split/    # final VCFs: {stem}_{pop}_{split}.vcf.gz
├── subsets/  # sample ID lists per population and split
└── qc/       # intermediate files
```

### 1KGP (chr22)
```bash
python vcf_prep.py \
    --step all \
    --chr 22 \
    --raw_vcf /mnt/storage4/rallendes/snp_data/1KGP/20190312_biallelic_SNV_and_INDEL/ALL.chr22.shapeit2_integrated_snvindels_v2a_27022019.GRCh38.phased.vcf.gz \
    --metadata_file /mnt/storage4/rallendes/snp_data/1KGP/integrated_call_samples_v3.20130502.ALL.panel \
    --output_dir ./dataset/1KGP \
    --threads 8
```

### 24donor (chr22, ALL population only)
```bash
python vcf_prep.py \
    --step all \
    --chr 22 \
    --raw_vcf /mnt/storage2/rallendes/data/scRNAseq/SNP_data/SNP_data_isec_complete/reference_version/chr22.phased_24donor_reference.vcf.gz \
    --metadata_file /mnt/storage4/rallendes/snp_data/1KGP/integrated_call_samples_v3.20130502.ALL.panel \
    --output_dir ./dataset/24donor \
    --populations ALL \
    --threads 8
```

Result: 3202 samples (2504 matched + 698 UNK) → 7,662 SNPs after HWE+MAF intersection.
Train: 2559 samples | Val: 316 | Test: 327.

**SLURM (1KGP, all chromosomes):**
```bash
sbatch --array=0-23%4 job/00_vcf_prep.batch
```

---

## Step 0b — Gene Region Files (`gene_regions_prep.py`)

`pretrain_data_prep.py` uses gene coordinates to define genomic windows for segmentation.
Only three columns are needed (`TargetID`, `GeneStart`, `GeneEnd`) — no expression values.
All population/split combinations for the same chromosome share the same gene list.

The 24donor pipeline reuses the same gene region files generated for 1KGP.

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
Found 447 protein-coding genes on chr22.

---

## Step 1 — Per-Gene HDF5 Files (`pretrain_data_prep.py`)

Creates per-gene HDF5 files from split VCFs using global SNP-index segmentation.
The VCF loading phase is single-threaded; parallelism starts only after the full
chromosome is loaded into memory.

**Parallelism notes:**
- This is a CPU-only step — set `OMP_NUM_THREADS=1` etc. to prevent numpy from
  spawning internal threads that compete with the worker processes.
- With 30 workers writing to NFS simultaneously some files may be silently corrupted.
  Use `--skip_existing` on re-runs so valid files from previous runs are preserved.
- Reduce `--num_workers` (8–10) when retrying failures to lower NFS contention.

### 1KGP (chr22, ALL population)
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

### 24donor (chr22, ALL population)
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python pretrain_data_prep.py \
    --genotype_ds 24donor \
    --gene_ds genecode \
    --chr 22 \
    --race ALL \
    --gene_pop ALL \
    --split train \
    --node_id 0 \
    --total_nodes 1 \
    --pretrain_vcf dataset/24donor/split/chr22.phased_24donor_reference_ALL_train.vcf.gz \
    --gene_exp_path dataset/genecode/split \
    --output_dir ./res_pt/24donor \
    --model_input_width 258 \
    --overlap_size 8 \
    --min_snps 10 \
    --num_workers 30 \
    --verbose
```

Repeat with `--split val` and `--split test`. The `--min_snps 10` threshold is used
because the 24donor VCF has only 7,662 SNPs on chr22 (vs ~245k for 1KGP), so the
default of 32 would exclude too many genes. Even with 10, 213/447 genes have no SNPs
in their region and produce no output.

**Handling NFS write failures:**

Some genes may fail on the first run due to NFS contention. The summary shows
`Saved / Skipped (existing) / No segments / Failed`. To retry only the failed genes:

```bash
# 1. Remove corrupt (partially written) HDF5 files
python - <<'EOF'
import h5py, os, glob
files = sorted(glob.glob("res_pt/1KGP/train/*.hdf5"))
removed, valid = 0, 0
for f in files:
    try:
        with h5py.File(f, 'r') as fh:
            _ = fh['snps'][:]
            _ = fh['snpsIndex'][:]
        valid += 1
    except OSError:
        print(f"Removing: {f}")
        os.remove(f)
        removed += 1
print(f"Total: {len(files)} | Valid: {valid} | Removed: {removed}")
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

**NFS issue:** Both reads and writes of large HDF5 files over NFS fail intermittently
with `errno = 5` (Input/output error). Two workarounds:
1. **Run on the NFS host directly** — if you SSH into the machine that owns `/mnt/storage4/`,
   operations are local and the problem disappears.
2. **Use local disk as staging** — write to `/tmp`, merge there, move result to NFS.

```bash
# Check and remove corrupt per-gene files before merging (repeat for val/test)
python - <<'EOF'
import h5py, os, glob
files = sorted(glob.glob("res_pt/1KGP/train/*.hdf5"))
removed, valid = 0, 0
for f in files:
    try:
        with h5py.File(f, 'r') as fh:
            _ = fh['snps'][:]
            _ = fh['snpsIndex'][:]
        valid += 1
    except OSError:
        print(f"Removing: {f}")
        os.remove(f)
        removed += 1
print(f"Total: {len(files)} | Valid: {valid} | Removed: {removed}")
EOF

# Train
python merge_genes.py \
    --input_dir res_pt/1KGP/train \
    --output_dir /tmp/merge_train \
    --prefix 1KGP_chr22_ALL_seg258_overlap8_train \
    --apply_dedup --shuffle_seed 42
mv /tmp/merge_train/*.hdf5 res_pt/1KGP/train/

# Val
python merge_genes.py \
    --input_dir res_pt/1KGP/val \
    --output_dir /tmp/merge_val \
    --prefix 1KGP_chr22_ALL_seg258_overlap8_val \
    --apply_dedup --shuffle_seed 42
mv /tmp/merge_val/*.hdf5 res_pt/1KGP/val/

# Test
python merge_genes.py \
    --input_dir res_pt/1KGP/test \
    --output_dir /tmp/merge_test \
    --prefix 1KGP_chr22_ALL_seg258_overlap8_test \
    --apply_dedup --shuffle_seed 42
mv /tmp/merge_test/*.hdf5 res_pt/1KGP/test/
```

Replace `1KGP` with `24donor` and update prefixes for the 24donor pipeline.

---

## Step 3 — Train (`pretrain.py`)

### Single GPU
```bash
MASTER_ADDR=localhost MASTER_PORT=12341 python pretrain.py \
    --configFile configs/24donor_chr22_ALL.yaml
```

### Multi-GPU (torchrun)
```bash
torchrun --nproc_per_node=2 --master_port=12341 pretrain.py \
    --configFile configs/24donor_chr22_ALL.yaml
```

`pretrain.py` supports both SLURM (`srun`) and `torchrun` launchers.
Stage merged HDF5 files to local `/tmp` before training to avoid NFS read latency —
set `resPtDir: /tmp` in the config.

### 24donor results (chr22, ALL, 100 epochs, 2× RTX PRO 6000)
- Training time: ~1h54m
- Checkpoints saved every 10 epochs to `checkpoints_pt/24donor_ALL_chr22/`

See `configs/24donor_chr22_ALL.yaml` and `configs/1KGP_chr22_ALL.yaml` for full settings.

---

## Step 4a — Internal Test (`test_pretrain.py`)

Evaluates the model on the reference test split with random masking.
Run after building the test HDF5 (Step 1 + Step 2 with `--split test`).

```bash
# Stage test data to local disk
mkdir -p /tmp/24donor/test
cp res_pt/24donor/test/24donor_chr22_ALL_seg258_overlap8_test_all.hdf5 /tmp/24donor/test/

MASTER_ADDR=localhost MASTER_PORT=12341 python test_pretrain.py \
    --configFile configs/24donor_chr22_ALL.yaml \
    --checkpoint checkpoints_pt/24donor_ALL_chr22/pt_24donor_ALL_chr22_PT_epoch_100.pth \
    --maskProb 0.05 0.15 0.5
```

### 24donor internal test results (epoch 100)

| Mask % | Loss | All Acc | Masked Acc |
|--------|------|---------|------------|
| 5% | 0.0069 | 0.9939 | 0.9206 |
| 15% | 0.0102 | 0.9879 | 0.9204 |
| 50% | 0.0470 | 0.9399 | 0.8800 |

Per-class accuracy at 50% masking: `0|0`: 97.86% · `0|1`: 85.06% · `1|0`: 84.88% · `1|1`: 93.70%

---

## Step 4b — External Evaluation on scRNA-seq Samples (`eval_pretrain.py`)

Evaluates imputation accuracy on 18 external scRNA-seq test samples.
Each sample's observed scRNA-seq variant calls are kept; all other positions in the
training SNP set are masked for the model to impute. Predictions are compared against
microarray ground truth. Reports concordance (genotype match rate) and R²
(Pearson r² between imputed and true dosage, using soft model probabilities).

```bash
python eval_pretrain.py \
    --configFile configs/24donor_chr22_ALL.yaml \
    --checkpoint checkpoints_pt/24donor_ALL_chr22/pt_24donor_ALL_chr22_PT_epoch_100.pth \
    --snp_vcf dataset/24donor/split/chr22.phased_24donor_reference_ALL_train.vcf.gz \
    --scrna_dir /mnt/storage2/rallendes/data/scRNAseq/Lipid_High/variants/per_donor/possorted_genome_bam/mapphased/chr22 \
    --ground_truth /mnt/storage2/rallendes/data/scRNAseq/SNP_data/SNP_data_isec_complete/chr22.phased_24donor_GTC_all_merged.vcf.gz
```

**Arguments:**
- `--snp_vcf`: defines the 7,662-SNP universe the model was trained on
- `--scrna_dir`: per-sample VCFs with sparse scRNA-seq variant calls (one file per sample)
- `--ground_truth`: multi-sample microarray VCF used as truth for scoring

**Note on context sparsity:** scRNA-seq samples have only 13–114 observed positions out of
7,662 training SNPs (<1.5% context). At this extreme masking level the model defaults to
predicting 0|0 (majority class), achieving high 0|0 accuracy but poor het/1|1 accuracy.
This is a fundamental limitation compared to reference-panel-based tools (BEAGLE, STICI)
which explicitly match sparse observations against haplotype panels.
