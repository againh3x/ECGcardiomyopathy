import numpy as np
import neurokit2 as nk
import wfdb
import matplotlib.pyplot as plt
import pandas as pd
# ----- Constants -----
FS      = 500
PRE_MS  = 250
POST_MS = 450
MIN_QRS = 60
MAX_QRS = 200
MIN_T   = 80
df = pd.read_csv('cohort.csv')

#initialize A-ECG columns
for col in ["T_peak_horizontal_azimuth_deg",
            "Spatial_peak_QRS_T_angle_deg",
            "ln_sigma2_T",
            "Q_depth_Z"]:
    if col not in df.columns:
        df[col] = np.nan

#chosen Kors regression matrix 
KORS_3x8 = np.array([
   [ 0.38, -0.07, -0.13,  0.05, -0.01,  0.14,  0.06,  0.54],
   [-0.07,  0.93,  0.06, -0.02, -0.05,  0.06, -0.17,  0.13],
   [ 0.11, -0.23, -0.43, -0.06, -0.14, -0.20, -0.11,  0.31]
])


def load_ecg_wfdb(path, lead_order):
   rec = wfdb.rdrecord(path)
   sigs = rec.p_signal
   idx = {n:i for i,n in enumerate(rec.sig_name)}
   return np.vstack([sigs[:, idx[l]] for l in lead_order])

#nk.clean used to clean ECGs before processing
def clean_all(ecg12):
   return np.vstack([nk.ecg_clean(ecg12[i], sampling_rate=FS) for i in range(ecg12.shape[0])])

#hardcoded fallback method if delineate method discrete wavelet transform (dwt) does not mark R-onset + R-offset as Q-onset + S-offset (it usually does)
def estimate_qrs_bounds(p_off, q_peak, r_on, s_peak, r_off, t_on, fs,
                       pad_left_ms=6, pad_right_ms=35, allow_wide=False):
   to_samp = lambda ms: int(ms*fs/1000)
   pad_left  = to_samp(pad_left_ms)
   pad_right = to_samp(pad_right_ms)
   gap_PT = to_samp(10)
   min_qrs = to_samp(60)
   max_qrs = to_samp(200)
   qrs_on = (q_peak - pad_left) if not np.isnan(q_peak) else (r_on - pad_left)
   if not np.isnan(p_off):
       qrs_on = max(qrs_on, p_off + gap_PT)
   qrs_on = max(0, qrs_on)
   if not np.isnan(s_peak):
       qrs_off = s_peak + pad_right
   elif not np.isnan(r_off):
       qrs_off = r_off + max(3, pad_right//2)
   else:
       qrs_off = (t_on - gap_PT) if not np.isnan(t_on) else (r_on + to_samp(100))
   if not np.isnan(t_on):
       qrs_off = min(qrs_off, t_on - gap_PT)
   if qrs_off <= qrs_on:
       return np.nan, np.nan
   dur = qrs_off - qrs_on
   if dur < min_qrs or dur > max_qrs:
       return np.nan, np.nan
   return int(qrs_on), int(qrs_off)


def collect_relative_offsets_direct(r_peaks, waves, fs):
   """
   For each beat:
     If fused R_Onset / R_Offset bracket Q and S (already enforced when fusing),
     use them directly as QRS onset/offset (NO padding estimation).
     Else fall back to estimate_qrs_bounds().
   Returns robust (median) relative offsets.
   """
   arr = lambda k: np.array(waves.get(k, []), dtype=float)
   Poff = arr("ECG_P_Offsets")
   Qpk  = arr("ECG_Q_Peaks")
   Ron  = arr("ECG_R_Onsets")     # fused
   Spk  = arr("ECG_S_Peaks")
   Roff = arr("ECG_R_Offsets")    # fused
   Ton  = arr("ECG_T_Onsets")
   Toff = arr("ECG_T_Offsets")


   n = min(len(r_peaks), len(Qpk), len(Spk), len(Ron), len(Roff), len(Ton), len(Toff), len(Poff))
   if n == 0:
       return None


   rels = []
   used_direct = 0
   used_fallback = 0


   for i in range(n):
       q_peak = Qpk[i]; s_peak = Spk[i]
       r_on   = Ron[i]; r_off  = Roff[i]
       t_on   = Ton[i]; t_off  = Toff[i]
       p_off  = Poff[i]


       # Need T boundaries
       if any(np.isnan(x) for x in [t_on, t_off]) or t_off <= t_on:
           continue


       direct = False
       # Check direct condition
       if (not np.isnan(r_on) and not np.isnan(r_off) and
           not np.isnan(q_peak) and not np.isnan(s_peak) and
           r_on < q_peak and r_off > s_peak and r_off > r_on):
           q_on  = int(r_on)
           q_off = int(r_off)
           direct = True
           used_direct += 1
       else:
           # Fallback to estimation
           q_on, q_off = estimate_qrs_bounds(p_off, q_peak, r_on, s_peak, r_off, t_on, fs)
           if np.isnan(q_on) or np.isnan(q_off):
               continue
           used_fallback += 1


       # Physiologic sanity & ordering with T
       if t_on <= q_off:
           continue


       rels.append({
           "q_on_rel":  q_on  - r_peaks[i],
           "q_off_rel": q_off - r_peaks[i],
           "t_on_rel":  t_on  - r_peaks[i],
           "t_off_rel": t_off - r_peaks[i]
       })


   if not rels:
       return None


   def robust(key):
       v = np.array([r[key] for r in rels])
       if v.size < 3:
           return int(np.median(v))
       q1,q3 = np.percentile(v,[25,75]); iqr = q3-q1
       mask = (v >= q1-1.5*iqr) & (v <= q3+1.5*iqr)
       return int(np.median(v[mask]))


   return {
       "q_on_rel":  robust("q_on_rel"),
       "q_off_rel": robust("q_off_rel"),
       "t_on_rel":  robust("t_on_rel"),
       "t_off_rel": robust("t_off_rel"),
       "n_beats_used": len(rels),
       "direct_qrs_beats": used_direct,
       "fallback_qrs_beats": used_fallback
   }

#mean beat used for processing A-ECG
def build_mean_beat(ecg12, r_peaks, fs, pre_ms=PRE_MS, post_ms=POST_MS):
   pre  = int(pre_ms/1000*fs)
   post = int(post_ms/1000*fs)
   segs = []
   kept_r = []
   for r in r_peaks:
       s = r - pre; e = r + post
       if s < 0 or e >= ecg12.shape[1]:
           continue
       seg = ecg12[:, s:e]
       baseline = seg[1, :int(0.05*fs)]
       if np.std(baseline) > 5 * np.median(np.abs(baseline - np.median(baseline)) + 1e-9):
           continue
       segs.append(seg)
       kept_r.append(r)
   if not segs:
       return None
   segs = np.stack(segs, axis=0)   # (Nbeats, 12, L)
   mean12 = segs.mean(axis=0)
   return mean12, pre, np.array(kept_r)

#A-ECG feature computation in 3D transformed vector plane
def compute_vcg_features(X, Y, Z, q_on, q_off, t_on, t_off, fs):
   L = X.shape[0]
   clamp = lambda x: max(0, min(int(x), L-1))
   q_on  = clamp(q_on); q_off = clamp(q_off)
   t_on  = clamp(max(t_on, q_off+1)); t_off = clamp(t_off)
   if t_off <= t_on + 1: t_off = min(L-1, t_on+2)
   if (t_off - t_on) < int(0.08*fs):
       return {}
   mag = np.sqrt(X**2 + Y**2 + Z**2)
   q_peak_idx = q_on + np.argmax(mag[q_on:q_off])
   t_peak_idx = t_on + np.argmax(mag[t_on:t_off])
   V_Q = np.array([X[q_peak_idx], Y[q_peak_idx], Z[q_peak_idx]])
   V_T = np.array([X[t_peak_idx], Y[t_peak_idx], Z[t_peak_idx]])
   t_az = np.degrees(np.arctan2(V_T[1], V_T[0]))
   cosang = np.dot(V_Q, V_T)/(np.linalg.norm(V_Q)*np.linalg.norm(V_T)+1e-12)
   cosang = np.clip(cosang, -1, 1)
   qrs_t_angle = np.degrees(np.arccos(cosang))
   M_t = np.vstack([X[t_on:t_off], Y[t_on:t_off], Z[t_on:t_off]]).T
   M_t -= M_t.mean(axis=0)
   _, S, _ = np.linalg.svd(M_t, full_matrices=False)
   ln_sigma2 = np.log(max(S[1],1e-12)) if len(S)>1 else np.nan
   Q_depth_Z = np.min(Z[q_on:q_peak_idx+1])
   return {
       "T_peak_horizontal_azimuth_deg": t_az,
       "Spatial_peak_QRS_T_angle_deg": qrs_t_angle,
       "ln_sigma2_T": ln_sigma2,
       "Q_depth_Z": Q_depth_Z,
       "q_peak_idx": q_peak_idx,
       "t_peak_idx": t_peak_idx
   }


#plotting (for optional visualization)
def plot_mean_beat(mean12, lead_name, lead_index, fs, r_index,
                  q_on_idx, q_off_idx, t_on_idx, t_off_idx):
   L = mean12.shape[1]
   rel_ms = (np.arange(L) - r_index) * 1000.0 / fs
   sig = mean12[lead_index]
   plt.figure(figsize=(10,4))
   plt.plot(rel_ms, sig, linewidth=1.0)
   plt.axvspan((q_on_idx - r_index)*1000/fs,
               (q_off_idx - r_index)*1000/fs,
               color='green', alpha=0.25, label='QRS')
   plt.axvspan((t_on_idx - r_index)*1000/fs,
               (t_off_idx - r_index)*1000/fs,
               color='orange', alpha=0.20, label='T')
   for x,c in [(q_on_idx,'green'),(q_off_idx,'green'),
               (t_on_idx,'orange'),(t_off_idx,'orange')]:
       plt.axvline((x - r_index)*1000/fs, color=c, linestyle='--', linewidth=1)
   plt.axvline(0, color='k', linestyle=':', linewidth=1, label='R peak')
   plt.title(f"Mean Beat – Lead {lead_name}")
   plt.xlabel("Time (ms, relative to R)")
   plt.ylabel("Amplitude")
   plt.legend(loc='upper right')
   plt.tight_layout()
   plt.show()


def plot_vcg(X, Y, Z, q_on_idx, q_off_idx, t_on_idx, t_off_idx,
            q_peak_idx, t_peak_idx):
   mag = np.sqrt(X**2 + Y**2 + Z**2)
   fig, axes = plt.subplots(1, 2, figsize=(11,4))
   ax = axes[0]
   ax.plot(X[q_on_idx:q_off_idx], Y[q_on_idx:q_off_idx],
           color='green', label='QRS loop')
   ax.plot(X[t_on_idx:t_off_idx], Y[t_on_idx:t_off_idx],
           color='orange', label='T loop')
   ax.scatter(X[q_peak_idx], Y[q_peak_idx], c='green', s=25, label='Q peak')
   ax.scatter(X[t_peak_idx], Y[t_peak_idx], c='red', s=25, label='T peak')
   ax.set_aspect('equal','box')
   ax.set_title("VCG Horizontal (X–Y)")
   ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.legend()
   ax2 = axes[1]
   V_Q = np.array([X[q_peak_idx], Y[q_peak_idx]])
   V_T = np.array([X[t_peak_idx], Y[t_peak_idx]])
   angle_Q = np.degrees(np.arctan2(V_Q[1], V_Q[0]))
   angle_T = np.degrees(np.arctan2(V_T[1], V_T[0]))
   def wrap(a):
       if a > 180: a -= 360
       if a <= -180: a += 360
       return a
   angle_Q = wrap(angle_Q); angle_T = wrap(angle_T)
   circle = plt.Circle((0,0), 1.0, color='lightgray', fill=False, linewidth=1)
   ax2.add_patch(circle)
   uQ = V_Q / (np.linalg.norm(V_Q)+1e-12)
   uT = V_T / (np.linalg.norm(V_T)+1e-12)
   ax2.arrow(0,0,uQ[0],uQ[1], color='green', width=0.007, length_includes_head=True)
   ax2.arrow(0,0,uT[0],uT[1], color='red', width=0.007, length_includes_head=True)
   a1 = np.radians(angle_Q)
   diff = (angle_T - angle_Q + 540) % 360 - 180
   arc_angles = np.linspace(a1, a1 + np.radians(diff), 120)
   ax2.plot(np.cos(arc_angles), np.sin(arc_angles), color='blue', linewidth=1)
   ax2.text(0.05, 0.05, f"Δ={abs(diff):.1f}°", color='blue')
   ax2.set_aspect('equal','box')
   ax2.set_xlim(-1.1,1.1); ax2.set_ylim(-1.1,1.1)
   ax2.set_xticks([]); ax2.set_yticks([])
   ax2.set_title("Horizontal Azimuths")
   plt.tight_layout()
   plt.show()


# fusion function to check whether fallback is necessary
def fuse_r_on_off(prom_waves, dwt_waves):
   """
   Replace R_Onsets / R_Offsets with DWT values if:
     DWT_R_Onset < Q_Peak  AND  DWT_R_Offset > S_Peak for that beat.
   """
   arr_p = lambda k: np.array(prom_waves.get(k, []), dtype=float)
   arr_d = lambda k: np.array(dwt_waves.get(k, []), dtype=float)


   Qp = arr_p("ECG_Q_Peaks")
   Sp = arr_p("ECG_S_Peaks")
   Rp_on_p  = arr_p("ECG_R_Onsets")
   Rp_off_p = arr_p("ECG_R_Offsets")
   Rd_on_d  = arr_d("ECG_R_Onsets")
   Rd_off_d = arr_d("ECG_R_Offsets")


   n = min(len(Qp), len(Sp), len(Rp_on_p), len(Rp_off_p), len(Rd_on_d), len(Rd_off_d))
   if n == 0:
       return Rp_on_p, Rp_off_p, 0


   fused_on  = Rp_on_p[:n].copy()
   fused_off = Rp_off_p[:n].copy()
   replaced = 0
   for i in range(n):
       q = Qp[i]; s = Sp[i]
       d_on  = Rd_on_d[i]; d_off = Rd_off_d[i]
       cond_on  = (not np.isnan(d_on))  and (not np.isnan(q)) and (d_on  < q)
       cond_off = (not np.isnan(d_off)) and (not np.isnan(s)) and (d_off > s)
       if cond_on and cond_off:
           fused_on[i]  = d_on
           fused_off[i] = d_off
           replaced += 1
   return fused_on, fused_off, replaced

def extract_vcg_features(ecg_path,
                         lead_order = ["I","II","V1","V2","V3","V4","V5","V6"],
                         delineation_lead = "V5",
                         fs=FS):

        ecg12_raw = load_ecg_wfdb(ecg_path, lead_order)
        ecg12     = clean_all(ecg12_raw)
        
        del_idx = lead_order.index(delineation_lead)
        v5_sig = ecg12[del_idx]

        MIN_V5_R = 6          #if neurokit finds less than 6 R-peaks then use R-peak location from the lead with the most R-peaks + slight realignment (should be more than 6 normally, we dont want to miss R-peaks)
        SEARCH_REALIGN_MS = 20  # window to snap borrowed peaks to nearest V5 max
        min_distance_samples = int(0.22 * FS)  # for pruning (not implemented)

     
        _, v5_info = nk.ecg_peaks(v5_sig, sampling_rate=FS)
        v5_r = v5_info["ECG_R_Peaks"]

        if len(v5_r) >= MIN_V5_R:
            r_peaks = v5_r
        else:
            rpeaks_per_lead = {}
            for li, lead_name in enumerate(lead_order):
                sig = ecg12[li]
                try:
                    _, info = nk.ecg_peaks(sig, sampling_rate=FS)
                    rpeaks_per_lead[lead_name] = info["ECG_R_Peaks"]
                except Exception:
                    rpeaks_per_lead[lead_name] = np.array([], dtype=int)

            #remove V5 so we can pick a different “best” if V5 was weak
            others = {k:v for k,v in rpeaks_per_lead.items() if k != delineation_lead}
            if all(len(arr) == 0 for arr in others.values()):
                # Nothing better—fallback to whatever V5 had (even if < MIN_V5_R)
                r_peaks = v5_r
            else:
                ref_lead = max(others, key=lambda L: len(others[L]))
                ref_r = others[ref_lead]

                # Realign each borrowed R to the nearest local maximum in V5 (peaks shift through leads due to the vector nature of electrode locations)
       
                if len(ref_r):
                    win = int(SEARCH_REALIGN_MS * FS / 1000)
                    aligned = []
                    L = v5_sig.shape[0]
                    for r in ref_r:
                        s = max(0, r - win)
                        e = min(L, r + win + 1)
                        if e - s < 3:
                            continue
                        local = v5_sig[s:e]
                        # index of highest absolute slope/peak in window (simplest: amplitude max)
                        local_peak = np.argmax(local) + s
                        aligned.append(local_peak)
                    if aligned:
                        r_peaks = np.array(sorted(set(aligned)), dtype=int)
                    else:
                        r_peaks = ref_r.copy()
                else:
                    r_peaks = v5_r  # fallback

        #I find prominence more accurate: first try prominence method for delineation baseline then see is dwt R-offset/onset can be used for QRS interval approximation

        prom_signals, prom_waves = nk.ecg_delineate(v5_sig, r_peaks, sampling_rate=fs, method="prominence")
        try:
            dwt_signals,  dwt_waves  = nk.ecg_delineate(v5_sig, r_peaks, sampling_rate=fs, method="dwt")
        except Exception:
            dwt_waves = {} #only ~2 ECGs actually raised exception here when run with this cohort
        fused_on, fused_off, _ = fuse_r_on_off(prom_waves, dwt_waves)

        waves_fused = dict(prom_waves)
        waves_fused["ECG_R_Onsets"]  = fused_on
        waves_fused["ECG_R_Offsets"] = fused_off

        offsets = collect_relative_offsets_direct(
            r_peaks,
            {
              "ECG_P_Offsets": waves_fused.get("ECG_P_Offsets", []),
              "ECG_Q_Peaks":   waves_fused.get("ECG_Q_Peaks", []),
              "ECG_R_Onsets":  waves_fused.get("ECG_R_Onsets", []),
              "ECG_S_Peaks":   waves_fused.get("ECG_S_Peaks", []),
              "ECG_R_Offsets": waves_fused.get("ECG_R_Offsets", []),
              "ECG_T_Onsets":  waves_fused.get("ECG_T_Onsets", []),
              "ECG_T_Offsets": waves_fused.get("ECG_T_Offsets", [])
            },
            fs
        )
        if not offsets:
            return None

        res = build_mean_beat(ecg12, r_peaks, fs, PRE_MS, POST_MS)
        if res is None:
            return None
        mean12, r_index, kept_r = res

        q_on_idx  = r_index + offsets["q_on_rel"]
        q_off_idx = r_index + offsets["q_off_rel"]
        t_on_idx  = r_index + offsets["t_on_rel"]
        t_off_idx = r_index + offsets["t_off_rel"]

        X, Y, Z = (KORS_3x8 @ mean12)
        features = compute_vcg_features(X, Y, Z,
                                        q_on_idx, q_off_idx,
                                        t_on_idx, t_off_idx, fs)

        if not features:
            return None

        # Only return the four requested
        return {k: features[k] for k in [
            "T_peak_horizontal_azimuth_deg",
            "Spatial_peak_QRS_T_angle_deg",
            "ln_sigma2_T",
            "Q_depth_Z"
        ]}



for i, row in df.iterrows():
    p = f'../ecg/{row["path"]}'
    print(p)
    feats = extract_vcg_features(p)
    if feats:
        for k, v in feats.items():
            df.at[i, k] = v
    # (optional) simple progress feedback every 50
    if i % 50 == 0:
        print(f"{i}/{len(df)} processed")
        
df.to_csv('cohort.csv', index=False)
