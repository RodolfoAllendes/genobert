#!/usr/bin/env python3
"""
GenoBERT Imputation Evaluation on scRNA-seq Test Samples.

For each test sample, positions observed in the scRNA-seq VCF are kept as-is;
all other positions in the training SNP set are masked. The model imputes the
masked positions, and predictions are compared against microarray ground truth.

Usage:
    python eval_pretrain.py \
        --configFile configs/24donor_chr22_ALL.yaml \
        --checkpoint checkpoints_pt/24donor_ALL_chr22/pt_24donor_ALL_chr22_PT_epoch_100.pth \
        --snp_vcf dataset/24donor/split/chr22.phased_24donor_reference_ALL_train.vcf.gz \
        --scrna_dir /mnt/storage2/rallendes/data/scRNAseq/Lipid_High/variants/per_donor/possorted_genome_bam/mapphased/chr22 \
        --ground_truth /mnt/storage2/rallendes/data/scRNAseq/SNP_data/SNP_data_isec_complete/chr22.phased_24donor_GTC_all_merged.vcf.gz
"""

import argparse
import json
import numpy as np
import os
import random
import subprocess
import torch
import torch.distributed as dist
import torch.nn as nn
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from config.modelconfig import ModelConfig
from model.genobert import GenoBERTMLM


# ── genotype encoding (must match training) ────────────────────────────────────

GT_TO_TOKEN    = {'0|0': 1, '0|1': 2, '1|0': 3, '1|1': 4}
TOKEN_TO_GT    = {v: k for k, v in GT_TO_TOKEN.items()}
TOKEN_TO_DOSAGE = {1: 0, 2: 1, 3: 1, 4: 2}   # alt allele count: 0|0→0, het→1, 1|1→2
HET_TOKENS     = {2, 3}                        # 0|1 and 1|0 — equivalent unphased

MASK_ID     = 0
CLS_ID      = 5
SEP_ID      = 6
PAD_ID      = 7
MODEL_WIDTH = 258    # 1 CLS + 256 SNP tokens + 1 SEP
TOKEN_SPAN  = 256
STRIDE      = 248


# ── bcftools helpers ───────────────────────────────────────────────────────────

def run_bcftools(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=True)
    return result.stdout


def load_snp_list(vcf_path):
    """Return ordered list of (chrom, pos, ref, alt) from a VCF."""
    out = run_bcftools(f"bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\n' {vcf_path}")
    snps = []
    for line in out.strip().splitlines():
        ch, pos, ref, alt = line.split('\t')
        snps.append((ch, int(pos), ref, alt))
    return snps


def load_vcf_genotypes(vcf_path, positions):
    """Load genotypes from a single-sample VCF. Returns {pos: token}."""
    out = run_bcftools(f"bcftools query -f '%POS\t[%GT]\n' {vcf_path}")
    gt = {}
    for line in out.strip().splitlines():
        if not line:
            continue
        pos_s, gt_s = line.split('\t')
        pos = int(pos_s)
        if pos in positions:
            tok = GT_TO_TOKEN.get(gt_s.replace('/', '|'))
            if tok is not None:
                gt[pos] = tok
    return gt


def load_ground_truth(gt_vcf, sample_id, positions):
    """Load ground truth genotypes for one sample from a multi-sample VCF."""
    out = run_bcftools(f"bcftools query -s {sample_id} -f '%POS\t[%GT]\n' {gt_vcf}")
    gt = {}
    for line in out.strip().splitlines():
        if not line:
            continue
        pos_s, gt_s = line.split('\t')
        pos = int(pos_s)
        if pos in positions:
            tok = GT_TO_TOKEN.get(gt_s.replace('/', '|'))
            if tok is not None:
                gt[pos] = tok
    return gt


# ── segment construction ───────────────────────────────────────────────────────

def build_segments(snps, token_span=TOKEN_SPAN, stride=STRIDE):
    """Partition SNPs into overlapping windows (same logic as pretrain_data_prep.py)."""
    n = len(snps)
    segments, start = [], 0
    while start < n:
        end = min(start + token_span, n)
        segments.append([(i, snps[i][1]) for i in range(start, end)])
        if end == n:
            break
        start += stride
    return segments


def build_segment_arrays(segments, observed_gt):
    """
    Build (snps_arr, idx_arr) for all segments of one sample.

    observed_gt: {pos: token} for positions seen in scRNA-seq.
    Unobserved positions → MASK_ID.

    Returns:
        snps_arr : (n_segs, MODEL_WIDTH)  int64
        idx_arr  : (n_segs, MODEL_WIDTH, 2) int64
    """
    n = len(segments)
    snps_arr = np.full((n, MODEL_WIDTH), PAD_ID, dtype=np.int64)
    idx_arr  = np.full((n, MODEL_WIDTH, 2), -3, dtype=np.int64)

    for si, seg in enumerate(segments):
        snps_arr[si, 0]  = CLS_ID
        idx_arr[si, 0]   = [-1, -1]

        for ti, (g_idx, pos) in enumerate(seg):
            t = ti + 1
            snps_arr[si, t] = observed_gt.get(pos, MASK_ID)
            idx_arr[si, t]  = [g_idx, pos]

        sep = len(seg) + 1
        snps_arr[si, sep] = SEP_ID
        idx_arr[si, sep]  = [-2, -2]

    return snps_arr, idx_arr


# ── genomic bias (same as test_pretrain.py) ────────────────────────────────────

def compute_genomic_bias(snps_index, config, device):
    batch_size = snps_index.shape[0]
    batch_bias = torch.zeros(batch_size, snps_index.shape[1], dtype=torch.float32, device=device)
    for b in range(batch_size):
        valid = snps_index[b, :, 1] > 0
        if valid.any():
            gpos = snps_index[b, valid, 1].float()
            mn, mx = gpos.min(), gpos.max()
            batch_bias[b, valid] = 0.5 if mx == mn else 0.1 + 0.8 * (gpos - mn) / (mx - mn)
        batch_bias[b, snps_index[b, :, 1] == -1] = 0.0
        batch_bias[b, snps_index[b, :, 1] == -2] = 1.0
    return batch_bias


# ── per-sample evaluation ─────────────────────────────────────────────────────

def evaluate_sample(model, sample_id, scrna_vcf, gt_vcf, snps, segments, config, device):
    positions = {s[1] for s in snps}

    observed = load_vcf_genotypes(scrna_vcf, positions)
    truth    = load_ground_truth(gt_vcf, sample_id, positions)

    if not truth:
        return None

    snps_np, idx_np = build_segment_arrays(segments, observed)
    snps_t = torch.tensor(snps_np, dtype=torch.long).to(device)
    idx_t  = torch.tensor(idx_np,  dtype=torch.long).to(device)
    pad_mask = (snps_t != PAD_ID)

    bias = compute_genomic_bias(idx_t, config, device) if config.enableBias else None

    with torch.no_grad():
        logits = model(snps_t, bias, pad_mask)    # (n_segs, MODEL_WIDTH, vocab)

    # Soft dosage: E[alt alleles] = P(0|1) + P(1|0) + 2*P(1|1)  (tokens 2, 3, 4)
    probs       = torch.softmax(logits, dim=-1).cpu().numpy()   # (n_segs, WIDTH, vocab)
    preds       = np.argmax(probs, axis=-1)                     # (n_segs, WIDTH)
    snps_np_cpu = snps_t.cpu().numpy()

    # Evaluate masked positions only; skip overlapping SNPs after first occurrence
    seen = set()
    correct = unphased_correct = total = 0
    pc_correct          = defaultdict(int)
    pc_total            = defaultdict(int)
    unphased_pc_correct = defaultdict(int)  # 'het' aggregates 0|1 + 1|0
    unphased_pc_total   = defaultdict(int)
    true_dosages = []
    pred_dosages = []

    for si, seg in enumerate(segments):
        for ti, (_, pos) in enumerate(seg):
            if pos in seen:
                continue
            t = ti + 1
            if snps_np_cpu[si, t] != MASK_ID:
                continue    # position was observed — skip
            true_tok = truth.get(pos)
            if true_tok is None:
                continue
            pred_tok = preds[si, t]
            gt_name  = TOKEN_TO_GT[true_tok]
            gt_unphased = 'het' if true_tok in HET_TOKENS else gt_name

            pc_total[gt_name] += 1
            unphased_pc_total[gt_unphased] += 1

            if pred_tok == true_tok:
                correct += 1
                pc_correct[gt_name] += 1
            # unphased: het→het is correct regardless of phase direction
            if (true_tok in HET_TOKENS and pred_tok in HET_TOKENS) or pred_tok == true_tok:
                unphased_correct += 1
                unphased_pc_correct[gt_unphased] += 1

            total += 1
            seen.add(pos)

            # Dosage for R²
            true_dosages.append(TOKEN_TO_DOSAGE[true_tok])
            pred_dosages.append(float(probs[si, t, 2] + probs[si, t, 3] + 2 * probs[si, t, 4]))

    acc             = correct          / total if total else 0.0
    unphased_acc    = unphased_correct / total if total else 0.0
    pc_acc          = {gt: pc_correct[gt]          / pc_total[gt]          for gt in pc_total}
    unphased_pc_acc = {gt: unphased_pc_correct[gt] / unphased_pc_total[gt] for gt in unphased_pc_total}

    # Pearson R² between imputed dosage and true dosage
    r2 = 0.0
    if len(true_dosages) >= 2:
        td = np.array(true_dosages)
        pd = np.array(pred_dosages)
        if td.std() > 0 and pd.std() > 0:
            r2 = float(np.corrcoef(td, pd)[0, 1] ** 2)

    return {
        'sample_id':                    sample_id,
        'n_snps':                       len(snps),
        'n_observed':                   len(observed),
        'n_imputed':                    total,
        'n_correct':                    correct,
        'concordance':                  acc,
        'unphased_concordance':         unphased_acc,
        'r2':                           r2,
        'per_class_accuracy':           pc_acc,
        'per_class_total':              dict(pc_total),
        'unphased_per_class_accuracy':  unphased_pc_acc,
        'unphased_per_class_total':     dict(unphased_pc_total),
        'accuracy':                     acc,   # backward compat
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GenoBERT imputation accuracy on scRNA-seq test samples."
    )
    parser.add_argument('--configFile',   required=True,
                        help="Path to config YAML file")
    parser.add_argument('--checkpoint',   required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument('--snp_vcf',      required=True,
                        help="VCF defining the training SNP set (e.g. *_ALL_train.vcf.gz)")
    parser.add_argument('--scrna_dir',    required=True,
                        help="Directory with per-sample scRNA-seq VCFs (*.vcf.gz)")
    parser.add_argument('--ground_truth', required=True,
                        help="Multi-sample ground truth VCF")
    parser.add_argument('--output',       default=None,
                        help="Output JSON file (default: eval_results_{runId}.json)")
    args = parser.parse_args()

    # Single-GPU / CPU — no DDP needed for 18 samples
    use_gpu = torch.cuda.is_available()
    device  = torch.device('cuda:0' if use_gpu else 'cpu')
    if use_gpu:
        torch.cuda.set_device(device)

    # Load config and model
    config = ModelConfig.from_yaml(args.configFile)
    model  = GenoBERTMLM(config).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd   = ckpt.get('state_dict', ckpt)
    sd   = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    # Build global segment structure from training SNP set
    snps     = load_snp_list(args.snp_vcf)
    segments = build_segments(snps)

    # Ground truth sample list
    gt_samples = set(
        run_bcftools(f"bcftools query -l {args.ground_truth}").strip().splitlines()
    )

    scrna_vcfs = sorted(Path(args.scrna_dir).glob('*.vcf.gz'))

    print(f"\n{'='*60}")
    print(f"GenoBERT Imputation Evaluation")
    print(f"{'='*60}")
    print(f"Config     : {args.configFile}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"SNPs       : {len(snps)}  |  Segments: {len(segments)}")
    print(f"Samples    : {len(scrna_vcfs)} scRNA-seq VCFs found")
    print(f"Device     : {device}")
    print(f"{'='*60}\n")

    results = []
    for vcf_path in tqdm(scrna_vcfs, desc="Evaluating samples", unit="sample"):
        sample_id = vcf_path.name[:-7]    # strip .vcf.gz

        if sample_id not in gt_samples:
            tqdm.write(f"  SKIP {sample_id} (not in ground truth)")
            continue

        r = evaluate_sample(
            model, sample_id, str(vcf_path), args.ground_truth,
            snps, segments, config, device
        )
        if r is None:
            tqdm.write(f"  SKIP {sample_id} (no ground truth genotypes found)")
            continue
        results.append(r)

    # Aggregate across samples
    tot_correct          = sum(r['n_correct']          for r in results)
    tot_unphased_correct = sum(r['unphased_concordance'] * r['n_imputed'] for r in results)
    tot_imputed          = sum(r['n_imputed']           for r in results)
    overall_concordance          = tot_correct          / tot_imputed if tot_imputed else 0.0
    overall_unphased_concordance = tot_unphased_correct / tot_imputed if tot_imputed else 0.0
    mean_r2 = float(np.mean([r['r2'] for r in results])) if results else 0.0

    agg_c = defaultdict(int)
    agg_t = defaultdict(int)
    for r in results:
        for gt, n in r['per_class_total'].items():
            agg_t[gt] += n
            agg_c[gt] += round(r['per_class_accuracy'][gt] * n)
    agg_acc = {gt: agg_c[gt] / agg_t[gt] for gt in agg_t}

    uagg_c = defaultdict(int)
    uagg_t = defaultdict(int)
    for r in results:
        for gt, n in r['unphased_per_class_total'].items():
            uagg_t[gt] += n
            uagg_c[gt] += round(r['unphased_per_class_accuracy'][gt] * n)
    uagg_acc = {gt: uagg_c[gt] / uagg_t[gt] for gt in uagg_t}

    # Summary table
    print(f"\n{'='*80}")
    print(f"Summary ({len(results)} samples evaluated)")
    print(f"{'Sample':<30} {'Observed':>10} {'Imputed':>10} {'Concordance':>13} {'Unphased':>10} {'R²':>8}")
    print('-' * 81)
    for r in results:
        obs_str = f"{r['n_observed']}/{r['n_snps']}"
        print(f"  {r['sample_id']:<28} {obs_str:>10} {r['n_imputed']:>10} "
              f"{r['concordance']:>13.4f} {r['unphased_concordance']:>10.4f} {r['r2']:>8.4f}")
    print('-' * 81)
    print(f"  {'Overall':<28} {'':>10} {tot_imputed:>10} "
          f"{overall_concordance:>13.4f} {overall_unphased_concordance:>10.4f} {mean_r2:>8.4f}")

    print(f"\nPer-class accuracy (phased / unphased):")
    for gt in ['0|0', '0|1', '1|0', '1|1']:
        if gt in agg_acc:
            print(f"  {gt}: {agg_acc[gt]:.4f}  (n={agg_t[gt]:,})")
    print(f"  het (unphased): {uagg_acc.get('het', 0):.4f}  (n={uagg_t.get('het', 0):,})")
    print(f"{'='*80}\n")

    # Save results
    output_file = args.output or f"eval_results_{config.runId}.json"
    output_data = {
        'config_file':                       args.configFile,
        'checkpoint':                        args.checkpoint,
        'snp_vcf':                           args.snp_vcf,
        'run_id':                            config.runId,
        'dataset':                           config.dataset,
        'chromosome':                        config.chromosome,
        'population':                        config.population,
        'n_samples':                         len(results),
        'n_snps':                            len(snps),
        'n_segments':                        len(segments),
        'timestamp':                         datetime.now().isoformat(),
        'overall_concordance':               overall_concordance,
        'overall_unphased_concordance':      overall_unphased_concordance,
        'mean_r2':                           mean_r2,
        'total_imputed':                     tot_imputed,
        'per_class_accuracy':                agg_acc,
        'per_class_total':                   dict(agg_t),
        'unphased_per_class_accuracy':       uagg_acc,
        'unphased_per_class_total':          dict(uagg_t),
        'per_sample':                        results,
    }
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
