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

### 1KGP_hc pipeline

| Data | Location |
|------|----------|
| 1KGP high-coverage Illumina VCF (chr22, phased) | `dataset/1KGP_hc/split/1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel_maf05.vcf.gz` |
| Population metadata (same as 1KGP) | `/mnt/storage4/rallendes/snp_data/1KGP/integrated_call_samples_v3.20130502.ALL.panel` |
| Gene region files (reuse 1KGP) | `dataset/genecode/split/` |
| Split VCFs (output) | `dataset/1KGP_hc/split/` |
| Merged HDF5 train (167 MB) | `res_pt/1KGP_hc/train/1KGP_hc_chr22_ALL_seg258_overlap8_train_all.hdf5` |
| Merged HDF5 val (20 MB) | `res_pt/1KGP_hc/val/1KGP_hc_chr22_ALL_seg258_overlap8_val_all.hdf5` |
| Merged HDF5 test | `res_pt/1KGP_hc/test/1KGP_hc_chr22_ALL_seg258_overlap8_test_all.hdf5` |

62,422 SNPs after MAF ≥ 5% filter. Split: 2559 train / 316 val / 327 test samples.
Training genotype distribution: **0\|0: 56.1%** · **het: 28.1%** · **1\|1: 15.8%**

### 24donor pipeline

| Data | Location |
|------|----------|
| Reference VCF (3202 samples, 13k SNPs, chr22) | `/mnt/storage2/rallendes/data/scRNAseq/SNP_data/SNP_data_isec_complete/reference_version/chr22.phased_24donor_reference.vcf.gz` |
| Population metadata (same as 1KGP) | `/mnt/storage4/rallendes/snp_data/1KGP/integrated_call_samples_v3.20130502.ALL.panel` |
| Gene region files (reuse 1KGP) | `dataset/genecode/split/` |
| Split VCFs (output) | `dataset/24donor/split/` |
| scRNA-seq test VCFs — mapphased (18 samples) | `/mnt/storage2/rallendes/data/scRNAseq/Lipid_High/variants/per_donor/possorted_genome_bam/mapphased/chr22/` |
| scRNA-seq test VCFs — targeted (18 samples) | `/mnt/storage2/rallendes/data/scRNAseq/Lipid_High/variants/per_donor/possorted_genome_bam_targetted/` |
| Ground truth genotypes (18 samples) | `/mnt/storage2/rallendes/data/scRNAseq/SNP_data/SNP_data_isec_complete/chr22.phased_24donor_GTC_all_merged.vcf.gz` |

**Two scRNA-seq VCF sources:**
- **mapphased**: BEAGLE-phased calls against the full 1KGP WGS (1,740,012 SNPs). Only ~13% of
  positions fall within the 13,175-SNP reference, so this source provides minimal usable context.
- **targeted**: Genotyped directly at reference positions using pileup. Includes explicit 0\|0 calls
  (~92% of non-missing positions), giving a realistic genotype distribution. **Use this source.**

### 24donor_ref pipeline

Dataset created from the same source VCF as 24donor but skipping the MAF/HWE filtering step,
retaining all 13,175 positions in the reference intersection.

| Data | Location |
|------|----------|
| Source VCF (3202 samples, 13,175 SNPs, chr22) | `/mnt/storage2/rallendes/data/scRNAseq/SNP_data/SNP_data_isec_complete/reference_version/chr22.phased_24donor_reference.vcf.gz` |
| Reference VCF (output) | `dataset/24donor_ref/ref/chr22.phased_24donor_reference_full_ref.vcf.gz` |
| Split VCFs (output) | `dataset/24donor_ref/split/` |
| Merged HDF5 train (129,915 samples) | `res_pt/24donor_ref/train/24donor_ref_chr22_ALL_seg258_overlap8_train_all.hdf5` |
| Merged HDF5 val (16,100 samples) | `res_pt/24donor_ref/val/24donor_ref_chr22_ALL_seg258_overlap8_val_all.hdf5` |
| Merged HDF5 test | `res_pt/24donor_ref/test/24donor_ref_chr22_ALL_seg258_overlap8_test_all.hdf5` |

13,175 SNPs (no MAF/HWE filter). Split: 2,559 train / 316 val / 327 test samples.

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
| 4b | `eval_pretrain.py` | External evaluation on scRNA-seq samples vs ground truth (single-pass) |
| 4c | `eval_pretrain_iterative.py` | External evaluation with iterative pseudo-observed unlocking |

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

### 24donor_ref (chr22, ALL — no MAF/HWE filter)

The `24donor_ref` dataset bypasses `vcf_prep.py` filtering to retain all 13,175 reference positions.
The source VCF is already clean and phased; only the train/val/test split step is needed.

```bash
# Split samples (same proportions as 24donor)
bcftools view -S dataset/24donor/subsets/ALL_train_samples.txt \
    /mnt/storage2/rallendes/data/scRNAseq/SNP_data/SNP_data_isec_complete/reference_version/chr22.phased_24donor_reference.vcf.gz \
    -O z -o dataset/24donor_ref/split/chr22.phased_24donor_reference_full_ALL_train.vcf.gz
bcftools index -t dataset/24donor_ref/split/chr22.phased_24donor_reference_full_ALL_train.vcf.gz
# Repeat for val and test
```

Result: 13,175 SNPs (no filter). Train: 2,559 / Val: 316 / Test: 327 samples (same split as 24donor).

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

### 24donor_ref (chr22, ALL population)
```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python pretrain_data_prep.py \
    --genotype_ds 24donor_ref \
    --gene_ds genecode \
    --chr 22 \
    --race ALL \
    --gene_pop ALL \
    --split train \
    --node_id 0 \
    --total_nodes 1 \
    --pretrain_vcf dataset/24donor_ref/split/chr22.phased_24donor_reference_full_ALL_train.vcf.gz \
    --gene_exp_path dataset/genecode/split \
    --output_dir ./res_pt/24donor_ref \
    --model_input_width 258 \
    --overlap_size 8 \
    --min_snps 10 \
    --num_workers 30 \
    --verbose
```

Repeat with `--split val` and `--split test`. The gene region files are the same as 24donor
(reused from 1KGP). Run `merge_genes.py` with prefix `24donor_ref_chr22_ALL_seg258_overlap8_{split}`
and output to `/tmp` (NFS staging required — see Step 2).

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

### 24donor_ref results (chr22, ALL, 100 epochs, 2× RTX PRO 6000)

Two model variants trained on the 13,175-SNP reference:

| Config | maskProb | Final val acc | Best val acc | Checkpoint dir |
|--------|----------|--------------|--------------|----------------|
| `configs/24donor_ref_chr22_ALL_mask95.yaml` | 0.95 | 0.7734 (ep 100) | 0.7761 (ep 88) | `checkpoints_pt/24donor_ref_ALL_chr22/` |
| `configs/24donor_ref_chr22_ALL_mask50.yaml` | 0.50 | in progress | — | `checkpoints_pt/24donor_ref_ALL_chr22/` |

```bash
# mask95 (completed)
torchrun --nproc_per_node=2 pretrain.py --configFile configs/24donor_ref_chr22_ALL_mask95.yaml

# mask50 (in progress)
torchrun --nproc_per_node=2 pretrain.py --configFile configs/24donor_ref_chr22_ALL_mask50.yaml
```

Training uses 129,915 train / 16,100 val samples (~2 min/epoch on 2× RTX PRO 6000).

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

### Targeted VCF evaluation — 24donor models

The targeted VCFs (`possorted_genome_bam_targetted/`) genotype directly at reference
positions and include explicit 0\|0 calls, giving a realistic context distribution.
Results for 18 samples, ground truth = `chr22.phased_24donor_GTC_all_merged.vcf.gz`:

```bash
# 24donor mask50 (7,662 SNPs)
python eval_pretrain.py \
    --configFile configs/24donor_chr22_ALL.yaml \
    --checkpoint checkpoints_pt/24donor_ALL_chr22/pt_24donor_ALL_chr22_PT_epoch_100.pth \
    --snp_vcf dataset/24donor/split/chr22.phased_24donor_reference_ALL_train.vcf.gz \
    --scrna_dir /mnt/storage2/rallendes/data/scRNAseq/Lipid_High/variants/per_donor/possorted_genome_bam_targetted/ \
    --ground_truth /mnt/storage2/rallendes/data/scRNAseq/SNP_data/SNP_data_isec_complete/chr22.phased_24donor_GTC_all_merged.vcf.gz \
    --output eval_results_24donor_targettedscRNAcall.json

# 24donor_ref mask95 (13,178 SNPs)
python eval_pretrain.py \
    --configFile configs/24donor_ref_chr22_ALL_mask95.yaml \
    --checkpoint checkpoints_pt/24donor_ref_ALL_chr22/pt_24donor_ref_ALL_chr22_PT_mask95_epoch_100.pth \
    --snp_vcf dataset/24donor_ref/split/chr22.phased_24donor_reference_full_ALL_train.vcf.gz \
    --scrna_dir /mnt/storage2/rallendes/data/scRNAseq/Lipid_High/variants/per_donor/possorted_genome_bam_targetted/ \
    --ground_truth /mnt/storage2/rallendes/data/scRNAseq/SNP_data/SNP_data_isec_complete/chr22.phased_24donor_GTC_all_merged.vcf.gz \
    --output eval_results_24donor_ref_mask95_targetted.json
```

| Model | SNPs | Concordance | Unphased | 0\|0 acc | het (unphased) | 1\|1 acc | R² |
|-------|------|------------|---------|---------|----------------|---------|-----|
| 24donor mask50 | 7,662 | 45.52% | 49.17% | 70.1% | 27.6% | 6.8% | **0.0124** |
| 24donor_ref mask95 | 13,178 | 58.42% | 58.45% | **92.1%** | 0.28% | 10.7% | 0.0049 |

The mask95 model achieved higher nominal concordance solely by collapsing to 0\|0 prediction
(92.1% 0\|0 accuracy vs 70.1% for mask50). Het recovery is essentially zero (0.28% vs 27.6%),
and R² is 2.5× lower. The mask50 model remains the better imputer: it captures
individual-level variation (R²=0.0124) and can recover heterozygous sites.

**Root cause:** at 95% masking a 256-token window has only ~13 visible tokens. The model never
learned to use LD context — it defaulted to the population prior (predict 0\|0 everywhere).
A larger window would give ~26 visible tokens at 95% masking, which is still insufficient;
the fix is lower masking rate, not window size.

---

## 1KGP_hc Pipeline

The 1KGP high-coverage (1KGP_hc) dataset uses the same pipeline steps as above.
Two model variants were trained, differing only in masking probability during training.

### Configs

| Config | runId | maskProb | Notes |
|--------|-------|----------|-------|
| `configs/1KGP_hc_chr22_ALL.yaml` | `1KGP_hc_ALL_chr22_PT` | 0.50 | Standard training |
| `configs/1KGP_hc_chr22_ALL_mask95.yaml` | `1KGP_hc_ALL_chr22_PT_mask95` | 0.95 | High-masking variant; `checkpointPrefix: "mask95_"` |

### Training

```bash
# mask50 (standard)
torchrun --nproc_per_node=2 --master_port=12341 pretrain.py \
    --configFile configs/1KGP_hc_chr22_ALL.yaml

# mask95 (retrain from scratch at 95% masking)
torchrun --nproc_per_node=2 --master_port=12341 pretrain.py \
    --configFile configs/1KGP_hc_chr22_ALL_mask95.yaml
```

Both trained for 100 epochs on 2× RTX PRO 6000. Checkpoints saved every 10 epochs to
`checkpoints_pt/1KGP_hc_ALL_chr22/`.

---

## Step 4a — Internal Test Results: 1KGP_hc

```bash
# mask50
python test_pretrain.py \
    --configFile configs/1KGP_hc_chr22_ALL.yaml \
    --checkpoint checkpoints_pt/1KGP_hc_ALL_chr22/pt_1KGP_hc_ALL_chr22_PT_epoch_100.pth \
    --maskProb 0.05 0.15 0.5 0.95

# mask95
python test_pretrain.py \
    --configFile configs/1KGP_hc_chr22_ALL_mask95.yaml \
    --checkpoint checkpoints_pt/1KGP_hc_ALL_chr22/pt_1KGP_hc_ALL_chr22_PT_mask95_epoch_100.pth \
    --maskProb 0.05 0.15 0.5 0.95
```

### mask50 model (`test_results_1KGP_hc_ALL_chr22_PT.json`)

| Mask % | Masked acc | 0\|0 | het | 1\|1 |
|--------|-----------|------|-----|------|
| 5% | 95.4% | 99.8% | 99.7% | 99.8% |
| 15% | 98.2% | 99.9% | 99.5% | 99.7% |
| 50% | 97.4% | 99.4% | 97.3% | 98.7% |
| 95% | 55.3% | 91.7% | 7.3% | 22.1% |

### mask95 model (`test_results_1KGP_hc_ALL_chr22_PT_mask95.json`)

| Mask % | Masked acc | 0\|0 | het | 1\|1 |
|--------|-----------|------|-----|------|
| 5% | 58.8% | 85.3% | 67.8% | 54.4% |
| 15% | 59.1% | 87.1% | 71.7% | 57.5% |
| 50% | 57.5% | 83.2% | 62.2% | 57.5% |
| 95% | 53.6% | 76.9% | 27.3% | 30.1% |

The mask95 model is consistently worse at all masking rates compared to mask50, including
at its own training rate of 95%. Training at extreme masking forces the model to rely on
the population prior rather than local LD context, causing accuracy to collapse at all
masking densities.

---

## Step 4b — External Evaluation: 1KGP_hc on scRNA-seq Samples

### Single-pass eval (`eval_pretrain.py`)

```bash
python eval_pretrain.py \
    --configFile configs/1KGP_hc_chr22_ALL_mask95.yaml \
    --checkpoint checkpoints_pt/1KGP_hc_ALL_chr22/pt_1KGP_hc_ALL_chr22_PT_mask95_epoch_100.pth \
    --snp_vcf dataset/1KGP_hc/split/1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel_maf05_ALL_train.vcf.gz \
    --scrna_dir /mnt/storage2/rallendes/data/scRNAseq/Lipid_High/variants/per_donor/possorted_genome_bam/mapphased/chr22 \
    --ground_truth /mnt/storage2/rallendes/data/scRNAseq/SNP_data/SNP_data_isec_complete/chr22.phased_24donor_GTC_all_merged.vcf.gz
```

**Context:** 18 scRNA-seq samples, 57–765 observed positions out of 62,422 SNPs (~1.2%).
Ground truth available at ~6,100 positions (intersection of 24donor panel ∩ 1KGP_hc SNPs).

| Model | Concordance | 0\|0 acc | het acc | 1\|1 acc | R² |
|-------|------------|---------|---------|---------|-----|
| mask50 | ~18% | low | ~0% | ~100% | ~0 |
| mask95 | ~45.3% | 84.7% | 1.7% | 16.3% | ~0.002 |

- mask50 collapses to predicting all 1\|1 (following the skewed scRNA-seq context).
- mask95 collapses to predicting all 0\|0 (strong majority-class prior overrides context).

### Clean-context control: GT 5% per-sample VCFs

To isolate context quality from model quality, a random 5% sample of the 6,143 ground
truth intersection positions (307 SNPs, seed=42) was used as observed context instead of
scRNA-seq calls. These per-sample VCFs contain real, phased genotypes with a realistic
class distribution (~51% 0\|0, 30% het, 19% 1\|1 — matching the training distribution).

```bash
# Generate the 5% subset VCF and split into per-sample files
# (regions file: /tmp/gt_5pct_regions.txt, 307 positions, seed=42)
bcftools view -R /tmp/gt_5pct_regions.txt \
    /mnt/storage2/rallendes/data/scRNAseq/SNP_data/SNP_data_isec_complete/chr22.phased_24donor_GTC_all_merged.vcf.gz \
    -O z -o /mnt/storage4/rallendes/snp_data/chr22.phased_24donor_GTC_all_merged_5pct.vcf.gz
bcftools index -t /mnt/storage4/rallendes/snp_data/chr22.phased_24donor_GTC_all_merged_5pct.vcf.gz

for SAMPLE in $(bcftools query -l .../chr22.phased_24donor_GTC_all_merged_5pct.vcf.gz); do
    bcftools view -s "$SAMPLE" ...5pct.vcf.gz -O z -o gt_5pct_per_sample/${SAMPLE}.vcf.gz
    bcftools index -t gt_5pct_per_sample/${SAMPLE}.vcf.gz
done
```

Results (24 samples, 5,836 imputed positions each, ground truth = full GT VCF):

| Model | Context | Concordance | Unphased | 0\|0 acc | het unphased | 1\|1 acc | R² |
|-------|---------|------------|---------|---------|-------------|---------|-----|
| mask95 | GT 5% | 45.5% | — | ~85% | ~2% | ~16% | 0.004 |
| mask50 | GT 5% | 33.2% | 33.9% | 45.5% | 5.0% | 55.1% | 0.003 |

**Key observations:**
- Unphased concordance is only +0.7% above phased for mask50, confirming the model is not
  confusing phase — it is simply failing to predict het altogether.
- mask50 behaves bimodally: predictions split between 0\|0 (45.5% of true 0\|0 correct) and
  1\|1 (55.1% of true 1\|1 correct), with negligible het predictions (~5% unphased).
  With ~1.2 observed tokens per segment, conflicting 0\|0 and 1\|1 context signals prevent
  reliable LD-based inference.
- mask95 is robust to context distribution quality (45.3% scRNA-seq vs 45.5% GT 5%) because
  its strong 0\|0 prior dominates sparse context regardless of composition.

---

## Step 4c — Iterative Evaluation (`eval_pretrain_iterative.py`)

Extends single-pass eval with multi-round pseudo-observed unlocking. In each round,
the top-confidence masked predictions are added to the observed context and inference
is re-run. A `committed_preds` dict tracks predictions at unlock time so all originally
masked positions are included in the final score.

```bash
python eval_pretrain_iterative.py \
    --configFile configs/1KGP_hc_chr22_ALL_mask95.yaml \
    --checkpoint checkpoints_pt/1KGP_hc_ALL_chr22/pt_1KGP_hc_ALL_chr22_PT_mask95_epoch_100.pth \
    --snp_vcf dataset/1KGP_hc/split/1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel_maf05_ALL_train.vcf.gz \
    --scrna_dir /mnt/storage2/rallendes/data/scRNAseq/Lipid_High/variants/per_donor/possorted_genome_bam/mapphased/chr22 \
    --ground_truth /mnt/storage2/rallendes/data/scRNAseq/SNP_data/SNP_data_isec_complete/chr22.phased_24donor_GTC_all_merged.vcf.gz \
    --n_rounds 4 --unlock_frac 0.25 --unlock_strategy confidence
```

**`--unlock_strategy` options:**
- `confidence` *(default)*: all tokens ranked by softmax confidence
- `het`: only 0\|1 and 1\|0 predictions eligible for unlocking
- `nonref`: het + 1\|1 predictions eligible

### Results summary (mask95 model, 18 samples)

| Strategy | Rounds | unlock_frac | Final concordance | 0\|0 acc | het acc | 1\|1 acc | Output JSON |
|----------|--------|-------------|------------------|---------|---------|---------|-------------|
| confidence | 4 | 0.25 | ~48.1% | high | ~0.8% | low | `eval_results_iterative_1KGP_hc_ALL_chr22_PT_mask95.json` |
| het | 4 | 0.25 | 44.5% | 82.0% | 4.0% | 15.4% | `eval_results_iterative_1KGP_hc_ALL_chr22_PT_mask95_het.json` |
| confidence | 8 | 0.10 | 47.9% | 92.8% | 0.8% | 10.1% | `eval_results_iterative_1KGP_hc_ALL_chr22_PT_mask95_conf-8-.1.json` |

Note: "round 1 concordance" (before any unlocking) = ~45.3% for all strategies —
this is the true single-pass baseline.

---

## Findings and Conclusions

### scRNA-seq genotype distribution
The scRNA-seq variant caller reports only positions where ALT alleles were observed.
This is by design — 0\|0 positions produce no VCF record. Across all 18 test samples:

| Genotype | Count | % |
|----------|-------|---|
| 1\|1 | 7,239 | 78.4% |
| 0\|1 + 1\|0 | 1,994 | 21.6% |
| 0\|0 | 0 | 0% |

This is inverted relative to the training distribution (56% 0\|0, 28% het, 16% 1\|1).
The extreme 1\|1 enrichment is a combination of:
(a) variant callers only reporting ALT positions; and
(b) allele dropout — low per-site coverage in scRNA-seq causes het sites to appear 1\|1.

### Masking rate mismatch
- scRNA-seq provides ~1.2% observed context (98.8% effectively masked).
- mask50 model: trained at 50% masking → collapses to 1\|1 at 98.8% (follows skewed context).
- mask95 model: trained at 95% masking → collapses to 0\|0 at 98.8% (strong prior overrides skewed context).
- Neither model performs LD-aware imputation at this density; both fall back to class-prior prediction.

### Iterative unlocking trade-offs
| Strategy | Effect |
|----------|--------|
| `confidence` | Unlocks mostly 0\|0 (high confidence); overall concordance improves slightly but het accuracy drops further as context fills with 0\|0. Plateau reached quickly (~10% pseudo-observed). |
| `het` | Unlocks het predictions; het accuracy improves (1.7% → 4.0%) but is still near-random. Adding het-heavy context suppresses 0\|0 accuracy; overall concordance decreases. |

In all cases R² ≈ 0.003, indicating near-zero dosage correlation with ground truth.
The segment-local window (256 SNPs, ~3 scRNA observations per window) provides
insufficient haplotype context for reliable imputation at these densities.

### Majority-class baseline
A naive imputer that always predicts 0\|0 achieves 49.0% concordance on the evaluation
positions (68,635 / 140,064 positions across 24 samples). This is the trivial floor:
any method below it is performing worse than ignoring the input entirely.

Both 1KGP_hc models fall **below** this baseline on the scRNA-seq task:

| Model | Context | Concordance | vs baseline |
|-------|---------|------------|-------------|
| Majority-class (0\|0 always) | — | 49.0% | reference |
| mask95 | scRNA-seq VCF | 45.3% | −3.7% |
| mask95 | GT 5% clean | 45.5% | −3.5% |
| mask50 | GT 5% clean | 33.2% | −15.8% |
| mask50 | scRNA-seq VCF | ~18% | −31% |

The negligible difference between scRNA-seq context (45.3%) and clean GT context (45.5%)
for mask95 shows that **context distribution quality is not the binding constraint** —
the structural signal bottleneck (1.2 observations per 256-SNP segment) would need to
be addressed first.

### Unphased concordance
Phased concordance counts 0\|1 vs 1\|0 as a mismatch; unphased concordance treats any
het prediction as correct for a true-het position. For mask50 on GT 5% context:

| Metric | Value |
|--------|-------|
| Phased concordance | 33.2% |
| Unphased concordance | 33.9% |
| Difference | +0.7% |

The near-zero difference confirms the model is **not confusing phase** — it is simply
failing to predict het at all. Per-class breakdown for mask50 (GT 5% context):

| True class | Acc (phased) | Acc (unphased) |
|------------|-------------|----------------|
| 0\|0 | 45.5% | 45.5% |
| het | ~3% (split 0\|1 / 1\|0) | 5.0% |
| 1\|1 | 55.1% | 55.1% |

mask50 behaves bimodally — predictions are split between 0\|0 and 1\|1 with only ~5%
het predictions (unphased). With ~1.2 observed tokens per segment the conflicting 0\|0
and 1\|1 context signals do not resolve to a confident het call; the model defaults to
whichever homozygote class is more prevalent in the segment window.

### Comparison vs reference-panel methods
BEAGLE and STICI use chromosome-scale HMM over thousands of reference haplotypes —
a qualitatively different approach that is not context-density-limited in the same way.
At <5% observation density the comparison inherently favours reference-panel HMM methods.
The structural bottleneck (segment-local LD with insufficient observations) must be
addressed before GenoBERT can be meaningfully compared to these methods at scRNA-seq
observation densities.

---

## Future Work

### 1. Pileup-based 0|0 context recovery
The scRNA-seq BAM files contain coverage information at positions not reported in the VCF.
A position with ≥N reads that is absent from the called VCF is high-confidence 0\|0.
Adding these positions to the initial observed context would:
- Restore a realistic 0\|0 fraction, reducing the initial context mismatch
- Give the mask50 model (the better imputer) a usable starting distribution

**Implementation plan:**
1. For each sample, run `samtools depth -a -b snp_positions.bed sample.bam`
2. Positions with depth ≥ threshold (e.g., 3 reads) and absent from the per-sample VCF → add as 0\|0
3. Pass the augmented `observed` dict to `eval_pretrain.py` / `eval_pretrain_iterative.py`

Key question: how many 0\|0 positions can be recovered per sample? This determines
whether the pileup augmentation materially changes the context distribution.

### 2. Class-conditional masking training
The mismatch between training and scRNA-seq observation distributions is not only about
density but also about which class of tokens is observed. Training the model to impute
from a context that mirrors scRNA-seq observations would require class-conditional masking:

| Class | P(observed in scRNA) | Required mask prob |
|-------|---------------------|-------------------|
| 0\|0 | ~0% | ~100% |
| het | ~0.9% | ~99.1% |
| 1\|1 | ~6.0% | ~94.0% |

These probabilities should be derived empirically from pileup-augmented context
(after step 1 above) to ensure training and inference distributions are aligned.
The two interventions must be designed jointly: the training masking distribution
should match the pileup-augmented inference context, not the raw scRNA-seq VCF distribution.

**Implementation:** add a `scrna` sampling mode to `data/utils.py` with configurable
per-class masking probabilities, controlled by new config keys
(e.g., `maskProbByClass: {ref: 1.0, het: 0.99, alt: 0.94}`).
