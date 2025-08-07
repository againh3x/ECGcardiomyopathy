import wfdb
import neurokit2 as nk
import numpy as np, pandas as pd
from scipy.signal import welch
from scipy.stats  import skew, kurtosis
import pandas as pd

SR = 500            # Hz
ECG_LEN = 10 * SR   # 5 000 samples / lead (10-s strip)
df = pd.read_csv('full_cohort.csv')

ms  = lambda s: s * 1000 / SR        # samples → ms
amp = lambda sig, i: sig[i]          # amplitude helper
safe = lambda d, k: d.get(k, [])     # missing-key helper
import numpy as np




def ecg_features_one_record(rec_path: str) -> pd.DataFrame:
    """Return a single-row DataFrame of 281 ECG features for one WFDB record."""
    print(f"Processing ECG record: {rec_path}")
    rec   = wfdb.rdrecord(rec_path)
    sigs  = rec.p_signal.T          # shape (12, 5000)
    leads = rec.sig_name
    rpeaks_per_lead = {}
    for sig, lead in zip(sigs, leads):
        clean = nk.ecg_clean(sig, sampling_rate=SR)
        _, info = nk.ecg_peaks(clean, sampling_rate=SR)
        rpeaks_per_lead[lead] = info["ECG_R_Peaks"]
    if all(len(peaks) == 0 for peaks in rpeaks_per_lead.values()):
        print(f"  → skipped {rec_path}: no R‑peaks in any lead")
        return None   
    # choose the reference vector (most R‑peaks)
    ref_lead  = max(rpeaks_per_lead, key=lambda L: len(rpeaks_per_lead[L]))
    ref_rpk   = rpeaks_per_lead[ref_lead]
    print(f"  → reference lead = {ref_lead}  ({len(ref_rpk)} R‑peaks)")
    if (len(ref_rpk) == 1):
        print(f"  → skipped {rec_path}: only 1 R‑peaks in any lead")
        return None 
    per_lead_rows = []
    all_u_peaks = []
    # ───────── Loop over 12 leads ─────────
    for sig, lead in zip(sigs, leads):
        if len(sig) != ECG_LEN:
            raise ValueError(f"{lead}: expected {ECG_LEN} samples, got {len(sig)}")

        clean = nk.ecg_clean(sig, sampling_rate=SR)
        if np.all(clean == 0):
            clean[:] = 1e-12
        freqs, psd = welch(clean, fs=SR, nperseg=SR*2)

        #get the mean frequency
        mean_freq = (freqs * psd).sum() / psd.sum()

        #get the fourier transform median frequency
        cum_power = np.cumsum(psd)
        total_power = cum_power[-1]
        median_freq = freqs[np.searchsorted(cum_power, total_power/2)]

        #skenwess and kurtosis (for the 3 and 4 powers of sample sums - mean over std)
        skw = skew(clean)
        krt = kurtosis(clean) 

        #find r peaks and dilineate the waves with prominence instead of wavelet transform
        r_pk = rpeaks_per_lead[lead]
        if len(r_pk) < 7:
            print(f"    {lead}: only {len(r_pk)} peaks → reusing {ref_lead}")
            r_pk = ref_rpk

        _, waves = nk.ecg_delineate(
            clean, r_pk, sampling_rate=SR,
            method="prominence", show=False, show_type="all"
        )

        # landmark vectors (with fall-backs)
        r_on,  r_off  = safe(waves, "ECG_R_Onsets"),  safe(waves, "ECG_R_Offsets")
        q_pk,  s_pk   = safe(waves, "ECG_Q_Peaks"),   safe(waves, "ECG_S_Peaks")
        t_on,  t_pk, t_off = safe(waves, "ECG_T_Onsets"), safe(waves, "ECG_T_Peaks"), safe(waves, "ECG_T_Offsets")
        p_on,  p_pk, p_off = safe(waves, "ECG_P_Onsets"), safe(waves, "ECG_P_Peaks"), safe(waves, "ECG_P_Offsets")

        n = min(map(len, [r_on, r_pk, r_off, q_pk, s_pk,
                          t_on, t_pk, t_off, p_on, p_pk, p_off])) - 1
        if n <= 0:
            continue   # skip noisy lead

        # beat-wise containers
        aq, ar, as_, at, ap = [], [], [], [], []
        qrs_d, r_d, st_d, stseg_d, t_d, tpseg_d, p_d = [], [], [], [], [], [], []
        qt_d, qtseg_d, qtcseg_d, jt_d, tp_toff_d, rr_d, pr_d = [], [], [], [], [], [], []
        area_qrs, slope_st, skew_rs, idx_t = [], [], [], []
        baseline = np.zeros(n)  # baseline for each beat
        for i in range(n):
            q, s = q_pk[i], s_pk[i]
            ron, roff, r = r_on[i], r_off[i], r_pk[i]
            ton, t, toff = t_on[i], t_pk[i], t_off[i]
            pon, pp, poff = p_on[i], p_pk[i], p_off[i]
            if not np.isnan(poff) and not np.isnan(q) and q - poff > 0:
                baseline[i] = (np.median(clean[poff:q]))
            elif not np.isnan(poff) and np.isnan(q):
                baseline[i] = (np.median(clean[poff: poff + 10]))
            elif not np.isnan(q):
                baseline[i] = (np.median(clean[q - 10:q]))
            else:
                continue  # skip this beat if no baseline
            # --------------- amplitudes (need just one idx each) -----------
            if not np.isnan(q):   aq.append(abs(amp(clean, q) - baseline[i]))
            if not np.isnan(r):   ar.append(abs(amp(clean, r) - baseline[i]))
            if not np.isnan(s):   as_.append(abs(amp(clean, s) - baseline[i]))
            if not np.isnan(t):   at.append(abs(amp(clean, t) - baseline[i]))
            if not np.isnan(pp):  ap.append(abs(amp(clean, pp) - baseline[i]))

            # --------------- durations -------------------------------------
            # QRS (Qpk→Spk)
            if not np.isnan(q) and not np.isnan(s):
                qrs_d.append(ms(s - q))

            # R duration (Ron→Roff)   — requires both on/off
            if not np.isnan(ron) and not np.isnan(roff):
                r_d.append(ms(roff - ron))

            # ST (Spk→Tpk)
            if not np.isnan(s) and not np.isnan(t):
                st_d.append(ms(t - s))

            # ST segment (Roff→Ton)
            if not np.isnan(s) and not np.isnan(ton):
                stseg_d.append(ms(ton - s))

            # T duration
            if not np.isnan(ton) and not np.isnan(toff):
                t_d.append(ms(toff - ton))

            # TP segment
            if i + 1 < len(r_pk) and not np.isnan(toff) and not np.isnan(p_on[i+1]):
                tpseg_d.append(ms(p_on[i+1] - toff))
            # PR segment
            if not np.isnan(ron) and not np.isnan(poff):
                pr_d.append(ms(ron - poff))

            # P duration
            if not np.isnan(pon) and not np.isnan(poff):
                p_d.append(ms(poff - pon))

            # QT & QTseg
            if not np.isnan(q) and not np.isnan(t):
                qt_d.append(ms(t - q))

            if not np.isnan(q) and not np.isnan(toff):
                qtseg_d.append(ms(toff - q))

            if i - 1 >= 0 and not np.isnan(r) and not np.isnan(r_pk[i-1]) and not np.isnan(q) and not np.isnan(toff):
                qtcseg_d.append(1000 * ((toff - q)/SR) / (((r - r_pk[i-1])/SR) ** (1/3)))
            # JT
            if not np.isnan(roff) and not np.isnan(toff):
                jt_d.append(ms(toff - roff))

            # Tp-Toff
            if not np.isnan(t) and not np.isnan(toff):
                tp_toff_d.append(ms(toff - t))

            # RR (needs next R)
            if i + 1 < len(r_pk) and not np.isnan(r) and not np.isnan(r_pk[i+1]):
                rr_d.append(ms(r_pk[i+1] - r))

            # --------------- shape extras ----------------------------------
            if not np.isnan(q) and not np.isnan(s):
                area_qrs.append(np.trapz(clean[q:s+1], dx=1 / SR))

            if not np.isnan(ton) and not np.isnan(roff):
                slope_st.append((amp(clean, ton) - amp(clean, roff)) /
                                (ton - roff + 1e-6))

            if not np.isnan(r) and not np.isnan(s):
                skew_rs.append((amp(clean, s) - amp(clean, r)) /
                            (s - r + 1e-6))

            if not np.isnan(ton) and not np.isnan(t) and not np.isnan(toff):
                idx_t.append((t - ton) / (toff - t + 1e-6))


        m   = lambda x: np.nan if len(x)==0 else float(np.mean(x))
        md  = lambda x: np.nan if len(x)==0 else float(np.median(x))
        sdx = lambda x: np.nan if len(x)==0 else float(np.std(x))

        per_lead_rows.append({
            "lead": lead,
            # amplitudes (6 + 2 ratios + 4 extras)
            "mean_Q": md(aq), "mean_R": md(ar), "mean_S": md(as_),
            "mean_T": md(at), "mean_P": md(ap),
            "R_S_ratio": md(np.abs(ar)) / md(np.abs(as_) + 1e-6),
            "T_R_ratio": md(np.abs(at)) / md(np.abs(ar) + 1e-6),
            "QRS_area": md(area_qrs), "ST_slope": m(slope_st),
            "RS_skew": m(skew_rs),   "T_index": m(idx_t),
            # durations (12)
            "dur_QSpk": m(qrs_d), "dur_R": m(r_d), "dur_STpk": m(st_d),
            "dur_spk-ton": m(stseg_d), "dur_T": m(t_d), "dur_TPseg": m(tpseg_d),
            "dur_P": m(p_d), "dur_QT": m(qt_d), "dur_QTseg": m(qtseg_d), "dur_QTc_seg": m(qtcseg_d),
            "dur_JT": m(jt_d), "dur_TpToff": m(tp_toff_d), "dur_RR": m(rr_d), "dur_PR": m(pr_d),
            # SDs (3)
            "sd_QT": sdx(qt_d), "sd_RR": sdx(rr_d), "sd_TpToff": sdx(tp_toff_d),
            "mean_freq": mean_freq,
            "median_freq": median_freq,
            "skewness_time": skw,
            "kurtosis_time": krt
        })

    

    # ───────── 3) per-lead → one-row wide  ─────────
    df_leads = pd.DataFrame(per_lead_rows).set_index("lead")
    wide = {}
    for lead, row in df_leads.iterrows():
        for feat, val in row.items():
            wide[f"{feat}_{lead}"] = val
    df_wide = pd.DataFrame([wide])

    # ───────── 4) global dispersion / variability ─────────
    df_globals = pd.DataFrame([{
        "QT_dispersion":  df_leads["dur_QT"].max()  - df_leads["dur_QT"].min(),
        "QRS_dispersion": df_leads["dur_QSpk"].max() - df_leads["dur_QSpk"].min(),
    }])
    duration_cols = [col for col in df_leads.columns if col.startswith('dur_') or col.startswith('sd_') or col.startswith('T_index')]

    # Compute mean across leads, skipping NaNs
    global_means = df_leads[duration_cols].mean(axis=0, skipna=True)

    # Rename indices to indicate global mean
    global_means.index = [f"{col}" for col in duration_cols]

    # Create a one-row DataFrame for global features
    df_global = pd.DataFrame([global_means.to_dict()])
    drop_cols = []
    for col in duration_cols:
        for lead in df_leads.index:
            drop_cols.append(f"{col}_{lead}")
    df_wide = df_wide.drop(columns=drop_cols, errors="ignore")

    # ───────── 5) concatenate & return ─────────
    return pd.concat([df_wide, df_global, df_globals], axis=1)
pd.set_option('display.max_columns', None)





feature_list   = []
failed_paths   = []        # keep the offending paths (optional)
failed_count   = 0
total          = len(df)   # total number of ECGs

for idx, path in enumerate(df['path'], 1):
    print(f"[{idx:>4}/{total}] {path}")          # progress line
    try:
        feat = ecg_features_one_record(
            f"data/mimic-iv-ecg/{path}/{path[-8:]}"
        )
    except Exception as e:
        failed_count += 1
        failed_paths.append(path)                # optional
        print(f"   └─➤ ⚠️  exception: {e}")
        continue                                 # skip this record and carry on

    if feat is None:                             # returned but unusable
        continue

    feat['path'] = path                          # tag with source path
    feature_list.append(feat)

# ───────────────────────────────────────────────────────────── summary
print(f"\nFinished: {failed_count} / {total} ECGs "
      f"({failed_count/total:.1%}) raised exceptions and were skipped.")
# Uncomment if you want to see which files failed
# for p in failed_paths:
#     print("  •", p)

# 3) Combine all feature rows into one DataFrame
features_df = pd.concat(feature_list, ignore_index=True) if feature_list else pd.DataFrame()

# 4) Merge on 'path'
df_combined = df.merge(features_df, on='path', how='left')

# 5) Save to CSV
df_combined.to_csv('full_cohort.csv', index=False)
