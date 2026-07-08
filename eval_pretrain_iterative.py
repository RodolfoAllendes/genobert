#!/usr/bin/env python3
"""
GenoBERT Iterative Imputation Evaluation on scRNA-seq Test Samples.

Extends eval_pretrain.py with iterative masked prediction:
  Round 0 : only scRNA-seq observed positions are unmasked
  Round k : unlock the top (unlock_frac) most-confident masked predictions,
            add them as pseudo-observed context, re-run inference
  Final   : evaluate at originally-masked positions vs ground truth

Averaging probabilities across overlapping segments gives better predictions
than the single-occurrence approach in eval_pretrain.py.

Usage:
    python eval_pretrain_iterative.py \
        --configFile configs/1KGP_hc_chr22_ALL_mask95.yaml \
        --checkpoint checkpoints_pt/1KGP_hc_ALL_chr22/pt_1KGP_hc_ALL_chr22_PT_mask95_epoch_100.pth \
        --snp_vcf dataset/1KGP_hc/split/1kGP_high_coverage_Illumina.chr22.filtered.SNV_INDEL_SV_phased_panel_maf05_ALL_train.vcf.gz \
        --scrna_dir /mnt/storage2/rallendes/data/scRNAseq/Lipid_High/variants/per_donor/possorted_genome_bam/mapphased/chr22 \
        --ground_truth /mnt/storage2/rallendes/data/scRNAseq/SNP_data/SNP_data_isec_complete/chr22.phased_24donor_GTC_all_merged.vcf.gz \
        --n_rounds 4 --unlock_frac 0.25
"""

import argparse
import json
import numpy as np
import os
import torch
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from config.modelconfig import ModelConfig
from model.genobert import GenoBERTMLM

# Re-use helpers and constants from eval_pretrain
from eval_pretrain import (
    GT_TO_TOKEN, TOKEN_TO_GT, TOKEN_TO_DOSAGE, HET_TOKENS,
    MASK_ID, PAD_ID, MODEL_WIDTH, TOKEN_SPAN, STRIDE,
    load_snp_list, load_vcf_genotypes, load_ground_truth,
    build_segments, build_segment_arrays, compute_genomic_bias,
    run_bcftools,
)


# ── inference pass ────────────────────────────────────────────────────────────

def infer_all_masked(model, segments, current_observed, config, device):
    """
    One full inference pass.

    For every position that is currently masked (not in current_observed),
    averages softmax probabilities across all overlapping segments.

    Returns:
        dict[pos -> (pred_token: int, confidence: float, avg_probs: ndarray)]
    """
    snps_np, idx_np = build_segment_arrays(segments, current_observed)
    snps_t   = torch.tensor(snps_np, dtype=torch.long, device=device)
    idx_t    = torch.tensor(idx_np,  dtype=torch.long, device=device)
    pad_mask = (snps_t != PAD_ID)
    bias     = compute_genomic_bias(idx_t, config, device) if config.enableBias else None

    with torch.no_grad():
        logits = model(snps_t, bias, pad_mask)          # (n_segs, WIDTH, vocab)

    probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (n_segs, WIDTH, vocab)

    # Accumulate probabilities per position across overlapping segments
    pos_probs: dict[int, list] = defaultdict(list)
    for si, seg in enumerate(segments):
        for ti, (_, pos) in enumerate(seg):
            t = ti + 1
            if snps_np[si, t] == MASK_ID:               # only masked positions
                pos_probs[pos].append(probs[si, t])

    predictions = {}
    for pos, prob_list in pos_probs.items():
        avg  = np.mean(prob_list, axis=0)               # (vocab,)
        tok  = int(np.argmax(avg))
        predictions[pos] = (tok, float(avg[tok]), avg)

    return predictions


# ── per-sample iterative evaluation ──────────────────────────────────────────

def iterative_evaluate_sample(
    model, sample_id, scrna_vcf, gt_vcf,
    snps, segments, config, device,
    n_rounds, unlock_frac, unlock_strategy='confidence',
):
    """
    Iteratively impute one sample.

    n_rounds   : number of unlock rounds before the final evaluation pass
                 (total inference passes = n_rounds + 1)
    unlock_frac: fraction of remaining masked positions to unlock per round
    """
    positions = {s[1] for s in snps}
    observed  = load_vcf_genotypes(scrna_vcf, positions)
    truth     = load_ground_truth(gt_vcf, sample_id, positions)
    if not truth:
        return None

    current_observed = dict(observed)   # grows with pseudo-observed each round
    n_originally_masked = sum(
        1 for s in snps if s[1] not in observed
    )

    # committed_preds records the prediction for every originally-masked position
    # at the moment it is either (a) unlocked or (b) evaluated in the final pass.
    committed_preds: dict[int, tuple] = {}
    round_stats = []

    for rnd in range(n_rounds + 1):     # n_rounds unlock rounds + 1 final pass
        preds = infer_all_masked(model, segments, current_observed, config, device)

        # ── snapshot concordance at this round ────────────────────────────────
        rnd_correct = rnd_unphased_correct = rnd_total = 0
        rnd_pc_correct: dict[str, int] = defaultdict(int)
        rnd_pc_total:   dict[str, int] = defaultdict(int)
        for pos, true_tok in truth.items():
            if pos in observed:
                continue
            if pos in committed_preds:
                pred_tok, _, avg_probs = committed_preds[pos]
            elif pos in preds:
                pred_tok, _, avg_probs = preds[pos]
            else:
                continue
            gt_name = TOKEN_TO_GT[true_tok]
            rnd_pc_total[gt_name] += 1
            if pred_tok == true_tok:
                rnd_correct += 1
                rnd_pc_correct[gt_name] += 1
            if (true_tok in HET_TOKENS and pred_tok in HET_TOKENS) or pred_tok == true_tok:
                rnd_unphased_correct += 1
            rnd_total += 1
        rnd_concordance          = rnd_correct          / rnd_total if rnd_total else 0.0
        rnd_unphased_concordance = rnd_unphased_correct / rnd_total if rnd_total else 0.0
        rnd_pc_acc = {gt: rnd_pc_correct[gt] / rnd_pc_total[gt] for gt in rnd_pc_total}

        n_pseudo = len(current_observed) - len(observed)
        n_unlocked_this_round = 0

        if rnd < n_rounds:
            if unlock_strategy == 'het':
                eligible = {2, 3}       # 0|1, 1|0
            elif unlock_strategy == 'nonref':
                eligible = {2, 3, 4}    # het + 1|1
            else:
                eligible = {1, 2, 3, 4} # all (confidence)
            candidates = sorted(
                [(conf, pos, tok) for pos, (tok, conf, _) in preds.items()
                 if tok in eligible],
                key=lambda x: x[0], reverse=True,
            )
            n_unlocked_this_round = max(1, int(len(candidates) * unlock_frac))
            for _, pos, tok in candidates[:n_unlocked_this_round]:
                current_observed[pos] = tok
                committed_preds[pos] = preds[pos]
        else:
            for pos, pred in preds.items():
                if pos not in committed_preds:
                    committed_preds[pos] = pred

        round_stats.append({
            'round':                rnd,
            'label':                f'unlock {rnd+1}' if rnd < n_rounds else 'final eval',
            'n_unlocked':           n_unlocked_this_round,
            'n_pseudo_observed':    n_pseudo + n_unlocked_this_round if rnd < n_rounds else n_pseudo,
            'pct_pseudo_observed':  round(100 * (n_pseudo + n_unlocked_this_round) / max(1, n_originally_masked), 1)
                                    if rnd < n_rounds else
                                    round(100 * n_pseudo / max(1, n_originally_masked), 1),
            'concordance':          rnd_concordance,
            'unphased_concordance': rnd_unphased_concordance,
            'per_class_accuracy':   rnd_pc_acc,
        })

    # ── evaluation at originally-masked positions ─────────────────────────────
    correct = unphased_correct = total = 0
    pc_correct:          dict[str, int] = defaultdict(int)
    pc_total:            dict[str, int] = defaultdict(int)
    unphased_pc_correct: dict[str, int] = defaultdict(int)
    unphased_pc_total:   dict[str, int] = defaultdict(int)
    true_dosages, pred_dosages = [], []

    for pos, true_tok in truth.items():
        if pos in observed:                 # was directly observed — skip
            continue
        if pos not in committed_preds:      # not in any segment — skip
            continue
        pred_tok, _, avg_probs = committed_preds[pos]
        gt_name     = TOKEN_TO_GT[true_tok]
        gt_unphased = 'het' if true_tok in HET_TOKENS else gt_name

        pc_total[gt_name]         += 1
        unphased_pc_total[gt_unphased] += 1

        if pred_tok == true_tok:
            correct               += 1
            pc_correct[gt_name]   += 1
        if (true_tok in HET_TOKENS and pred_tok in HET_TOKENS) or pred_tok == true_tok:
            unphased_correct                    += 1
            unphased_pc_correct[gt_unphased]    += 1
        total += 1

        true_dosages.append(TOKEN_TO_DOSAGE[true_tok])
        pred_dosages.append(float(avg_probs[2] + avg_probs[3] + 2 * avg_probs[4]))

    acc          = correct          / total if total else 0.0
    unphased_acc = unphased_correct / total if total else 0.0
    pc_acc          = {gt: pc_correct[gt]          / pc_total[gt]          for gt in pc_total}
    unphased_pc_acc = {gt: unphased_pc_correct[gt] / unphased_pc_total[gt] for gt in unphased_pc_total}

    r2 = 0.0
    if len(true_dosages) >= 2:
        td, pd = np.array(true_dosages), np.array(pred_dosages)
        if td.std() > 0 and pd.std() > 0:
            r2 = float(np.corrcoef(td, pd)[0, 1] ** 2)

    return {
        'sample_id':                    sample_id,
        'n_snps':                       len(snps),
        'n_observed':                   len(observed),
        'n_pseudo_observed':            len(current_observed) - len(observed),
        'n_imputed':                    total,
        'n_correct':                    correct,
        'concordance':                  acc,
        'unphased_concordance':         unphased_acc,
        'accuracy':                     acc,
        'r2':                           r2,
        'per_class_accuracy':           pc_acc,
        'per_class_total':              dict(pc_total),
        'unphased_per_class_accuracy':  unphased_pc_acc,
        'unphased_per_class_total':     dict(unphased_pc_total),
        'round_stats':                  round_stats,
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Iterative GenoBERT imputation evaluation on scRNA-seq samples."
    )
    parser.add_argument('--configFile',   required=True)
    parser.add_argument('--checkpoint',   required=True)
    parser.add_argument('--snp_vcf',      required=True,
                        help="VCF defining the training SNP set")
    parser.add_argument('--scrna_dir',    required=True,
                        help="Directory with per-sample scRNA-seq VCFs (*.vcf.gz)")
    parser.add_argument('--ground_truth', required=True,
                        help="Multi-sample ground truth VCF")
    parser.add_argument('--n_rounds',     type=int,   default=4,
                        help="Unlock rounds before final eval pass (default: 4)")
    parser.add_argument('--unlock_frac',     type=float, default=0.25,
                        help="Fraction of candidate positions to unlock per round (default: 0.25)")
    parser.add_argument('--unlock_strategy', default='confidence',
                        choices=['confidence', 'het', 'nonref'],
                        help="Which predicted tokens to consider for unlocking: "
                             "'confidence' = all tokens ranked by confidence (default); "
                             "'het' = only 0|1 and 1|0 predictions; "
                             "'nonref' = all non-0|0 predictions (het + 1|1)")
    parser.add_argument('--output',          default=None)
    args = parser.parse_args()

    use_gpu = torch.cuda.is_available()
    device  = torch.device('cuda:0' if use_gpu else 'cpu')
    if use_gpu:
        torch.cuda.set_device(device)

    config = ModelConfig.from_yaml(args.configFile)
    model  = GenoBERTMLM(config).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    sd   = ckpt.get('state_dict', ckpt)
    sd   = {(k[7:] if k.startswith('module.') else k): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()

    snps     = load_snp_list(args.snp_vcf)
    segments = build_segments(snps)

    gt_samples = set(
        run_bcftools(f"bcftools query -l {args.ground_truth}").strip().splitlines()
    )
    scrna_vcfs = sorted(Path(args.scrna_dir).glob('*.vcf.gz'))

    print(f"\n{'='*64}")
    print(f"GenoBERT Iterative Imputation Evaluation")
    print(f"{'='*64}")
    print(f"Config      : {args.configFile}")
    print(f"Checkpoint  : {args.checkpoint}")
    print(f"SNPs        : {len(snps)}  |  Segments: {len(segments)}")
    print(f"Samples     : {len(scrna_vcfs)} scRNA-seq VCFs found")
    print(f"Rounds      : {args.n_rounds} unlock + 1 eval  |  unlock_frac: {args.unlock_frac}  |  strategy: {args.unlock_strategy}")
    print(f"Device      : {device}")
    print(f"{'='*64}\n")

    results = []
    for vcf_path in tqdm(scrna_vcfs, desc="Evaluating samples", unit="sample"):
        sample_id = vcf_path.name[:-7]
        if sample_id not in gt_samples:
            tqdm.write(f"  SKIP {sample_id} (not in ground truth)")
            continue
        r = iterative_evaluate_sample(
            model, sample_id, str(vcf_path), args.ground_truth,
            snps, segments, config, device, args.n_rounds, args.unlock_frac,
            args.unlock_strategy,
        )
        if r is None:
            tqdm.write(f"  SKIP {sample_id} (no ground truth genotypes)")
            continue
        results.append(r)

    tot_correct          = sum(r['n_correct'] for r in results)
    tot_unphased_correct = sum(r['unphased_concordance'] * r['n_imputed'] for r in results)
    tot_imputed          = sum(r['n_imputed'] for r in results)
    overall_concordance          = tot_correct          / tot_imputed if tot_imputed else 0.0
    overall_unphased_concordance = tot_unphased_correct / tot_imputed if tot_imputed else 0.0
    mean_r2 = float(np.mean([r['r2'] for r in results])) if results else 0.0

    agg_c: dict[str, int] = defaultdict(int)
    agg_t: dict[str, int] = defaultdict(int)
    for r in results:
        for gt, n in r['per_class_total'].items():
            agg_t[gt] += n
            agg_c[gt] += round(r['per_class_accuracy'][gt] * n)
    agg_acc = {gt: agg_c[gt] / agg_t[gt] for gt in agg_t}

    uagg_c: dict[str, int] = defaultdict(int)
    uagg_t: dict[str, int] = defaultdict(int)
    for r in results:
        for gt, n in r['unphased_per_class_total'].items():
            uagg_t[gt] += n
            uagg_c[gt] += round(r['unphased_per_class_accuracy'][gt] * n)
    uagg_acc = {gt: uagg_c[gt] / uagg_t[gt] for gt in uagg_t}

    # ── per-round aggregate summary ───────────────────────────────────────────
    n_passes = args.n_rounds + 1
    print(f"\n{'='*86}")
    print(f"Per-round progress (averaged over {len(results)} samples)")
    print(f"{'Round':<14} {'Unlocked':>10} {'% Pseudo-obs':>14} {'Concordance':>13} {'Unphased':>10}  0|0 / het / 1|1")
    print('-' * 86)
    for rnd in range(n_passes):
        rs_list = [r['round_stats'][rnd] for r in results if len(r['round_stats']) > rnd]
        if not rs_list:
            continue
        label        = rs_list[0]['label']
        avg_unlock   = np.mean([rs['n_unlocked'] for rs in rs_list])
        avg_pct      = np.mean([rs['pct_pseudo_observed'] for rs in rs_list])
        avg_conc     = np.mean([rs['concordance'] for rs in rs_list])
        avg_unphased = np.mean([rs['unphased_concordance'] for rs in rs_list])
        hom_ref = np.mean([rs['per_class_accuracy'].get('0|0', 0) for rs in rs_list])
        het     = np.mean([(rs['per_class_accuracy'].get('0|1', 0) +
                            rs['per_class_accuracy'].get('1|0', 0)) / 2 for rs in rs_list])
        hom_alt = np.mean([rs['per_class_accuracy'].get('1|1', 0) for rs in rs_list])
        print(f"  {label:<12} {avg_unlock:>10.0f} {avg_pct:>13.1f}% {avg_conc:>13.4f} {avg_unphased:>10.4f}  "
              f"{hom_ref:.3f} / {het:.3f} / {hom_alt:.3f}")
    print(f"{'='*86}")

    print(f"\n{'='*86}")
    print(f"Summary ({len(results)} samples, {args.n_rounds} unlock rounds, unlock_frac={args.unlock_frac}, strategy={args.unlock_strategy})")
    print(f"{'Sample':<30} {'Obs/Total':>12} {'Imputed':>10} {'Concordance':>13} {'Unphased':>10} {'R²':>8}")
    print('-' * 86)
    for r in results:
        obs_str = f"{r['n_observed']}/{r['n_snps']}"
        print(f"  {r['sample_id']:<28} {obs_str:>12} {r['n_imputed']:>10} "
              f"{r['concordance']:>13.4f} {r['unphased_concordance']:>10.4f} {r['r2']:>8.4f}")
    print('-' * 86)
    print(f"  {'Overall':<28} {'':>12} {tot_imputed:>10} "
          f"{overall_concordance:>13.4f} {overall_unphased_concordance:>10.4f} {mean_r2:>8.4f}")

    print(f"\nPer-class accuracy (phased / unphased):")
    for gt in ['0|0', '0|1', '1|0', '1|1']:
        if gt in agg_acc:
            print(f"  {gt}: {agg_acc[gt]:.4f}  (n={agg_t[gt]:,})")
    print(f"  het (unphased): {uagg_acc.get('het', 0):.4f}  (n={uagg_t.get('het', 0):,})")
    print(f"{'='*86}\n")

    # aggregate round stats across samples for JSON
    n_passes = args.n_rounds + 1
    agg_round_stats = []
    for rnd in range(n_passes):
        rs_list = [r['round_stats'][rnd] for r in results if len(r['round_stats']) > rnd]
        if rs_list:
            agg_round_stats.append({
                'round':            rnd,
                'label':            rs_list[0]['label'],
                'mean_n_unlocked':  float(np.mean([rs['n_unlocked'] for rs in rs_list])),
                'mean_pct_pseudo':  float(np.mean([rs['pct_pseudo_observed'] for rs in rs_list])),
                'mean_concordance': float(np.mean([rs['concordance'] for rs in rs_list])),
            })

    output_file = args.output or f"eval_results_iterative_{config.runId}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'config_file':         args.configFile,
            'checkpoint':          args.checkpoint,
            'snp_vcf':             args.snp_vcf,
            'run_id':              config.runId,
            'n_rounds':                          args.n_rounds,
            'unlock_frac':                       args.unlock_frac,
            'unlock_strategy':                   args.unlock_strategy,
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
            'round_stats':                       agg_round_stats,
            'per_sample':                        results,
        }, f, indent=2)
    print(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
