# Install into the exact Python that’s running this notebook

import numpy as np
import neurokit2 as nk
from tqdm.auto import tqdm

# Ensure wfdb is installed into this Python environment
import wfdb
import matplotlib.pyplot as plt
import pandas as pd
# ----- Constants -----
FS      = 500
PRE_MS  = 400
POST_MS = 450
MIN_QRS = 60
MAX_QRS = 200
MIN_T   = 80


#initialize A-ECG columns
df = pd.read_csv('full_cohort.csv')
df['qrs_duration'] = df['qrs_end'] - df['qrs_onset']
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



def collect_relative_offsets_direct(r_peaks, waves, fs, path, lead_name=None):
   """
   For each beat:
     If fused R_Onset / R_Offset bracket Q and S (already enforced when fusing),
     use them directly as QRS onset/offset (NO padding estimation).
     Else fall back to estimate_qrs_bounds().
   Returns robust (median) relative offsets.
   """
   arr = lambda k: np.array(waves.get(k, []), dtype=float)
   Pon = arr("ECG_P_Onsets")
   Poff = arr("ECG_P_Offsets")
   Qpk  = arr("ECG_Q_Peaks")
   Ron  = arr("ECG_R_Onsets")     # fused
   Spk  = arr("ECG_S_Peaks")
   Roff = arr("ECG_R_Offsets")    # fused
   Ton  = arr("ECG_T_Onsets")
   Toff = arr("ECG_T_Offsets")
   Tpk  = arr("ECG_T_Peaks")


   n = min(len(r_peaks), len(Qpk), len(Spk), len(Ron), len(Roff), len(Ton), len(Toff))
   if n == 0:
       return None


   rels = []
   used_direct = 0
   used_fallback = 0
   real_path = path[-49:-9]
   qrs_duration = df.loc[df['path'] == real_path, ['qrs_duration']].values[0]
   baseline = np.zeros(n)  # baseline for each beat
   baselineT = np.zeros(n)  # baseline for T-wave
   
   # Access the current lead signal using the current function scope
   # This assumes clean is defined in the parent scope
   current_lead_signal = clean  # Using the global clean variable set in extract_vcg_features

   for i in range(n):
       q_peak = Qpk[i]; s_peak = Spk[i]
       r_on   = Ron[i]; r_off  = Roff[i]
       t_on   = Ton[i]; t_off  = Toff[i]
       t_peak = Tpk[i]
       p_on  = Pon[i]  if i < len(Pon)  else np.nan
       p_off = Poff[i] if i < len(Poff) else np.nan
       q_on = np.nan
       q_off = np.nan
       mimic_end = q_peak + qrs_duration * 1000 / fs

       if not np.isnan(p_off) and not np.isnan(q_peak) and q_peak - p_off > 0:
           baseline[i] = np.median(current_lead_signal[int(p_off):int(q_peak)])
       elif not np.isnan(p_off) and np.isnan(q_peak):
           baseline[i] = np.median(current_lead_signal[int(p_off):int(p_off + 10)])  # Fixed syntax error
       elif not np.isnan(q_peak):
           baseline[i] = np.median(current_lead_signal[int(q_peak - 15):int(q_peak)])
       else:
            continue  # skip this beat if no baseline
       
       # Need T boundaries
       if any(np.isnan(x) for x in [t_on, t_off]) or t_off <= t_on:
           continue

       if not np.isnan(t_on):
            baselineT[i] = np.median(current_lead_signal[int(t_on - 10):int(t_on)])

       amplitude_threshold = current_lead_signal[int(s_peak)] + 0.9 * abs(baseline[i] - current_lead_signal[int(s_peak)])
       amplitude_threshold_T = current_lead_signal[int(s_peak)] + 0.95 * abs(baselineT[i] - current_lead_signal[int(s_peak)])
       s_amplitude = abs(baseline[i] - current_lead_signal[int(s_peak)])
       q_amplitude = abs(baseline[i] - current_lead_signal[int(q_peak)])
       direct = False
       # Check direct condition

       if (not np.isnan(q_peak) and q_amplitude < 0.075):
           q_on = int(q_peak - 1)
       if (not np.isnan(q_peak) and not np.isnan(r_on) and not np.isnan(p_off)):
            if (r_on < q_peak and r_on > p_off):
                q_on = int(r_on) + 1
            else:
                q_on = int(q_peak - 4 * fs / 1000)
       elif (not np.isnan(q_peak) and not np.isnan(r_on) and np.isnan(p_off)):
            if (r_on < q_peak and r_on > q_peak - 7):
                q_on = int(r_on) + 1
            else:
                q_on = int(q_peak - 4 * fs / 1000)
       elif (not np.isnan(q_peak)):
            q_on = int(q_peak - 4 * fs / 1000)
       else: 
           continue
       found_q_off = False
       if (not np.isnan(r_off) and not np.isnan(s_peak) and r_off > s_peak + 1):
            if (current_lead_signal[int(r_off)] > amplitude_threshold or 
                current_lead_signal[int(r_off)] > amplitude_threshold_T or 
                s_amplitude < 0.1):
                q_off = int(r_off)
                found_q_off = True
       elif (not np.isnan(r_off) and not np.isnan(s_peak) and not np.isnan(q_peak) and 
            mimic_end > r_off and mimic_end > s_peak and mimic_end < t_on and 
            (current_lead_signal[int(mimic_end)] > amplitude_threshold or 
            current_lead_signal[int(mimic_end)] > amplitude_threshold_T or 
            s_amplitude < 0.1)):
            q_off = int(mimic_end)
            found_q_off = True
       else:
            # Forward search for q_off from s_peak to t_on
            search_start = int(s_peak) + 1
            search_end = int(t_on) if not np.isnan(t_on) else int(s_peak + 100)  # Limit search if t_on is NaN
            
            for j in range(search_start, search_end):
                if j >= len(current_lead_signal):
                    break
                    
                # Check if this point satisfies amplitude criteria
                if (current_lead_signal[j] > amplitude_threshold or 
                    current_lead_signal[j] > amplitude_threshold_T):
                    q_off = j
                    found_q_off = True
                    #print(f"Found q_off by forward search at {j} for beat {i} in {lead_name}")
                    break
            
            if not found_q_off:
                #print(f"Couldn't find q_off by forward search for beat {i} in {lead_name}")
                if t_on > s_peak + 10:
                    q_off = int(s_peak + 10)
                else:
                    q_off = s_peak + 1

        # Physiologic sanity & ordering with T
       if t_on <= q_off:  # Fixed the problematic condition
            continue

       if not np.isnan(t_peak) and t_peak + 15 > t_off and t_peak - 15 > t_on:
            t_off = t_peak + 0.9 * (t_peak - t_on)
        
       if not np.isnan(t_peak) and t_peak - 10 < t_on and t_peak + 15 < t_off:
            if (t_peak - 0.85 * (t_off - t_peak)) > q_off:
                t_on = t_peak - 0.85 * (t_off - t_peak)

        
       if np.isnan(q_off) or np.isnan(q_on) or np.isnan(t_on) or np.isnan(t_off):
            continue

       rel = {                       # ←  keep the beat
        "q_on_rel": q_on  - r_peaks[i],
        "q_off_rel": q_off - r_peaks[i],
        "t_on_rel": t_on  - r_peaks[i],
        "t_off_rel": t_off - r_peaks[i],
        "p_on_rel": p_on - r_peaks[i] if not np.isnan(p_on) else np.nan,
        "p_off_rel": p_off - r_peaks[i] if not np.isnan(p_off) else np.nan
        }

       rels.append(rel)

   if not rels:
        return None

   def robust(key):
        v = np.asarray([r[key] for r in rels], dtype=float)

        # If everything is NaN, just return NaN
        if np.all(np.isnan(v)):
            return np.nan

        # Drop NaNs before IQR filtering
        v = v[~np.isnan(v)]
        if v.size < 3:
            return float(np.median(v))

        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        good = (v >= q1 - 1.5*iqr) & (v <= q3 + 1.5*iqr)
        return float(np.median(v[good]))


   def top75_off():
        v = np.array([r["q_off_rel"] for r in rels])
        if v.size < 3:
            return int(np.percentile(v, 75))
        q1, q3 = np.percentile(v, [25, 75])
        iqr = q3 - q1
        mask = (v >= q1 - 1.5*iqr) & (v <= q3 + 1.5*iqr)
        return int(np.percentile(v[mask], 75))

   return {
        "q_on_rel":  robust("q_on_rel"),
        "q_off_rel": top75_off(),
        "t_on_rel":  robust("t_on_rel"),
        "t_off_rel": robust("t_off_rel"),
        "p_on_rel":  robust("p_on_rel"),
        "p_off_rel": robust("p_off_rel"),
        "n_beats_used": len(rels),
        "baseline": baseline,
        "baselineT": baselineT
    }

#mean beat used for processing A-ECG
def build_mean_beat(ecg12, r_peaks, waves, fs, pre_ms=PRE_MS, post_ms=POST_MS,
                    pvc_flag=False, qrs_thresh_ms=225):
    pre  = int(pre_ms/1000 * fs)
    post = int(post_ms/1000 * fs)
    segs = []
    kept_r = []

    # 1) collect all R amplitudes
    r_amplitudes = [abs(clean[r]) for r in r_peaks]

    Qpk, Spk = waves["ECG_Q_Peaks"], waves["ECG_S_Peaks"]
    s_amplitudes = [abs(clean[s]) for s in Spk]
    for i, r in enumerate(r_peaks):
        # PVC width filter
    
        if pvc_flag and i < len(Qpk) and i < len(Spk):
            dur_ms = (Spk[i] - Qpk[i]) * 1000.0 / fs
            if dur_ms > qrs_thresh_ms:
                continue
            # amplitude‐outlier filter
            if r_amplitudes[i] > 1.4 * np.median(r_amplitudes) or s_amplitudes[i] > 1.4 * np.median(s_amplitudes):
                continue

        # now extract the segment
        s, e = r - pre, r + post
        if s < 0 or e > ecg12.shape[1]:
            continue
        seg = ecg12[:, s:e]

        # baseline‐noise filter, etc.
        baseline = seg[1, :int(0.05*fs)]
        if np.std(baseline) > 5 * np.median(np.abs(baseline - np.median(baseline)) + 1e-9):
            continue

        segs.append(seg)
        kept_r.append(r)

    if not segs:
        return None

    mean12 = np.stack(segs, axis=0).mean(axis=0)
    return mean12, pre, np.array(kept_r)

def compute_p_vcg_features(X, Y, Z,
                           p_on, p_off,
                           mean_QRS_vec, V_Q_peak,
                           fs):
    """
    Return a dict with P‑loop A‑ECG features.
    Every amplitude is already in mV.
    """
    # ─────────────────────────────────── validity
    if np.isnan(p_on) or np.isnan(p_off) or p_off <= p_on + int(0.04 * fs):
        return {}          # no valid P loop

    # ─────────────────────────────────── build P matrices
    P_xyz = np.vstack([X[p_on:p_off],
                       Y[p_on:p_off],
                       Z[p_on:p_off]]).T            # shape (N,3)
    P_mag = np.linalg.norm(P_xyz, axis=1)

    # ─────────────────────────────────── centre the loop *** FIXED ***
    P_centered = P_xyz - P_xyz.mean(axis=0, keepdims=True)

    # mean & peak vectors AFTER centring
    mean_P_vec = P_centered.mean(axis=0)
    mean_P_vec /= (np.linalg.norm(mean_P_vec) + 1e-12)

    p_peak_idx = p_on + np.argmax(P_mag)
    V_P_peak   = np.array([X[p_peak_idx],
                           Y[p_peak_idx],
                           Z[p_peak_idx]])

    # ─────────────────────────────────── SVD on the centred loop
    Up, Sp, Vp = np.linalg.svd(P_centered, full_matrices=False)

    # ─────────────────────────────────── helpers …
    def azimuth_deg(v):   return np.degrees(np.arctan2(v[1], v[0]))
    def elevation_deg(v): return np.degrees(np.arctan2(v[2],
                                    np.linalg.norm(v[:2]) + 1e-12))
    def vec_angle_deg(a, b):
        c = np.clip(np.dot(a, b) /
                    (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12),
                    -1, 1)
        return np.degrees(np.arccos(c))
    def loop_area_2d(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, -1)) -
                            np.dot(y, np.roll(x, -1)))

    # ─────────────────────────────────── loop‑shape metrics
    Yp, Zp = P_centered[:, 1], P_centered[:, 2]      # already centred
    rl_mask = (Yp < 0) & (Zp < 0)
    total_area = loop_area_2d(Yp, Zp)
    pct_rl = (loop_area_2d(Yp[rl_mask], Zp[rl_mask]) /
              (total_area + 1e-12) * 100.0) if rl_mask.any() else 0.0

    # ─────────────────────────────────── feature dict
    featsP = {
        "P_axis_azimuth_deg":          azimuth_deg(mean_P_vec),
        "P_axis_elevation_deg":        elevation_deg(mean_P_vec),

        "spatial_peak_P_QRS_angle_deg":
            vec_angle_deg(V_P_peak, V_Q_peak),
        "spatial_mean_P_QRS_angle_deg":
            vec_angle_deg(mean_P_vec, mean_QRS_vec),

        "time_voltage_P_mVms":
            np.trapz(P_mag, dx=1 / fs) * 1000.0,

        "ratio_P_max_to_mean":
            np.max(P_mag) / (np.mean(P_mag) + 1e-12),

        "ln_amp_dipolar_P_ln_mV": np.log(max(Sp[0], 1e-12)),
        "norm_amp_e2_P":          Sp[1] / Sp[0] if len(Sp) > 1 else np.nan,
        "ln_amp_e3_P_ln_mV":
            np.log(max(Sp[2], 1e-12)) if len(Sp) > 2 else np.nan,

        "pct_P_area_RL_sagittal_percent": pct_rl
    }
    return featsP



def compute_vcg_features(X, Y, Z, q_on, q_off, t_on, t_off, p_on=None,p_off=None, fs=FS):
    """
    Compute a comprehensive set of VCG features from 3D (X,Y,Z) loops.
    Inputs
    ------
    X, Y, Z : 1D numpy arrays
        Spatial leads derived via the Kors matrix (or other transform).
    q_on, q_off, t_on, t_off : int
        Sample indices (absolute) for QRS and T boundaries on the mean beat.
    fs : int
        Sampling frequency (Hz).

    Returns
    -------
    feats : dict
        Dictionary of all requested features (unit noted in key).
    """

    # ------------------------------#
    # 0) Helpers                    #
    # ------------------------------#
    def clamp_idx(i, L):
        # Prevent indices from going out of bounds
        return max(0, min(int(i), L - 1))

    def norm(v):
        # Euclidean norm
        return np.linalg.norm(v) + 1e-12

    def unit(v):
        # Normalize vector safely
        n = norm(v)
        return v / n

    def vec_angle_deg(a, b):
        # 3D angle in degrees between vectors a and b
        c = np.dot(a, b) / (norm(a) * norm(b))
        c = np.clip(c, -1.0, 1.0)
        return np.degrees(np.arccos(c))

    def plane_angle_deg(v1, v2):
        # 2D angle between two planar vectors (e.g., XY plane)
        c = np.dot(v1, v2) / (norm(v1) * norm(v2))
        c = np.clip(c, -1.0, 1.0)
        return np.degrees(np.arccos(c))

    def trapz_time(voltage_array):
        # Time-voltage integral in mV*ms (assuming mV input)
        # trapz integral -> area (mV*s), multiply by 1000 for mV*ms
        return np.trapz(voltage_array, dx=1.0 / fs) * 1000.0

    def azimuth_deg(v):
        # Azimuth angle (XY plane) in degrees: atan2(Y, X)
        return np.degrees(np.arctan2(v[1], v[0]))

    def elevation_deg(v):
        # Elevation angle from XY plane toward +Z
        return np.degrees(np.arctan2(v[2], np.sqrt(v[0]**2 + v[1]**2)))

    def project_to_plane(vec, plane="frontal"):
        # Returns a 2D vector in the chosen plane
        # frontal plane (XY), horizontal plane (XZ), sagittal plane (YZ)
        if plane == "frontal":      # XY
            return np.array([vec[0], vec[1]])
        elif plane == "horizontal": # XZ
            return np.array([vec[0], vec[2]])
        elif plane == "sagittal":   # YZ
            return np.array([vec[1], vec[2]])
        else:
            raise ValueError("Unknown plane.")

    def loop_area_2d(x_arr, y_arr):
        # Polygon area using Shoelace formula
        return 0.5 * np.abs(np.dot(x_arr, np.roll(y_arr, -1)) -
                            np.dot(y_arr, np.roll(x_arr, -1)))

    # ------------------------------#
    # 1) Clamp indices and segments #
    # ------------------------------#
    L = X.shape[0]                                    # total samples
    q_on  = clamp_idx(q_on, L)
    q_off = clamp_idx(q_off, L)
    t_on  = clamp_idx(max(t_on, q_off + 1), L)        # ensure no overlap
    t_off = clamp_idx(t_off, L)
    if t_off <= t_on + 1:                             # tiny T segment fix
        t_off = min(L - 1, t_on + 2)

    # Minimum T duration check
    if (t_off - t_on) < int(0.08 * fs):
        return {}

    # Extract QRS and T segments (Nx3 matrices: rows are time points)
    QRS_xyz = np.vstack([X[q_on:q_off], Y[q_on:q_off], Z[q_on:q_off]]).T
    T_xyz   = np.vstack([X[t_on:t_off], Y[t_on:t_off], Z[t_on:t_off]]).T

    # Magnitudes to find peaks
    mag = np.sqrt(X**2 + Y**2 + Z**2)

    # Peak indices (spatial magnitude peak inside each window)
    q_peak_idx = q_on + np.argmax(mag[q_on:q_off])
    t_peak_idx = t_on + np.argmax(mag[t_on:t_off])

    # Peak vectors (3D)
    V_Q_peak = np.array([X[q_peak_idx], Y[q_peak_idx], Z[q_peak_idx]])
    V_T_peak = np.array([X[t_peak_idx], Y[t_peak_idx], Z[t_peak_idx]])

    # ------------------------------#
    # 2) SVD-based features         #
    # ------------------------------#
    # Center each loop before SVD (remove mean vector)
    QRS_centered = QRS_xyz - QRS_xyz.mean(axis=0, keepdims=True)
    T_centered   = T_xyz   - T_xyz.mean(axis=0, keepdims=True)

    # Perform SVD: QRS_centered = Uq * Sq * Vq^T
    Uq, Sq, Vq = np.linalg.svd(QRS_centered, full_matrices=False)
    Ut, St, Vt = np.linalg.svd(T_centered,   full_matrices=False)

    e1_Q = Vq[0]  # First eigenvector (direction) for QRS
    e1_T = Vt[0]  # First eigenvector (direction) for T
    e2_T = Vt[1] if len(St) > 1 else np.array([np.nan, np.nan, np.nan])
    e2_Q = Vq[1] if len(Sq) > 1 else np.array([np.nan, np.nan, np.nan])
    e3_T = Vt[2] if len(St) > 2 else np.array([np.nan, np.nan, np.nan])

    # Dipolar (first singular value) and non-dipolar sums
    dip_Q = Sq[0]
    dip_T = St[0]
    nondip_Q = np.sum(Sq[1:]) if len(Sq) > 1 else 0.0
    nondip_T = np.sum(St[1:]) if len(St) > 1 else 0.0

    # ------------------------------#
    # 3) Mean directions & planes   #
    # ------------------------------#
    # Spatial mean direction vectors
    mean_QRS_vec = unit(QRS_centered.sum(axis=0))  # direction of average QRS loop
    mean_T_vec   = unit(T_centered.sum(axis=0))    # direction of average T loop

    # Angles between various vectors
    angle_e1_QRS_T = vec_angle_deg(e1_Q, e1_T)                        # (1)
    angle_peak_QRS_T = vec_angle_deg(V_Q_peak, V_T_peak)              # (2)
    angle_mean_QRS_T = vec_angle_deg(mean_QRS_vec, mean_T_vec)        # (3)

    # QRS-T angle in frontal and horizontal planes (project each mean vector)
    mean_QRS_fr = project_to_plane(mean_QRS_vec, "frontal")
    mean_T_fr   = project_to_plane(mean_T_vec, "frontal")
    qrs_t_frontal = plane_angle_deg(mean_QRS_fr, mean_T_fr)            # (19)

    mean_QRS_h  = project_to_plane(mean_QRS_vec, "horizontal")
    mean_T_h    = project_to_plane(mean_T_vec, "horizontal")
    qrs_t_horizontal = plane_angle_deg(mean_QRS_h, mean_T_h)           # (20)

    # Max QRS-T angle among the three planes
    mean_QRS_s  = project_to_plane(mean_QRS_vec, "sagittal")
    mean_T_s    = project_to_plane(mean_T_vec, "sagittal")
    angles_planes = [
        plane_angle_deg(mean_QRS_fr, mean_T_fr),
        plane_angle_deg(mean_QRS_h,  mean_T_h),
        plane_angle_deg(mean_QRS_s,  mean_T_s)
    ]
    max_qrs_t_plane = max(angles_planes)                               # (22)

    # ------------------------------#
    # 4) Time-based features        #
    # ------------------------------#
    # Spatial ventricular activation time (ms):
    # from QRS onset to R-peak-time in the spatial magnitude.
    vact_ms = (q_peak_idx - q_on) * 1000.0 / fs                        # (4)

    # Time-voltage of spatial mean T wave (mV*ms):
    # integrate |T_vec| (magnitude) over T segment.
    T_mag = np.sqrt(T_xyz[:, 0]**2 + T_xyz[:, 1]**2 + T_xyz[:, 2]**2)
    time_voltage_T = trapz_time(T_mag)                                 # (5)

    # ------------------------------#
    # 5) Ventricular gradient (VG)  #
    # ------------------------------#
    # VG = integral(QRS_vec) + integral(T_vec) over their segments
    QRS_int = np.trapz(QRS_xyz, dx=1.0 / fs, axis=0)  # vector in mV*s
    T_int   = np.trapz(T_xyz,   dx=1.0 / fs, axis=0)
    VG_vec  = QRS_int + T_int                                    # vectorial sum
    VG_mag  = norm(VG_vec) * 1000.0                               # mV*ms

    # Elevation angle of VG (degrees)
    VG_elev_deg = elevation_deg(VG_vec)                            # (6)

    # “Optimized for RV pressure overload” component (elevation=27°, azimuth=155°)
    # Build that unit direction vector and project VG onto it
    az_rv = np.radians(155.0)  # azimuth in XY
    el_rv = np.radians(27.0)   # elevation from XY
    dir_rv = np.array([
        np.cos(el_rv) * np.cos(az_rv),
        np.cos(el_rv) * np.sin(az_rv),
        np.sin(el_rv)
    ])
    VG_rvpo = np.dot(VG_vec, dir_rv) * 1000.0                       # (26) mV*ms

    # Spatial mean T wave minus spatial mean QRS (vector subtraction)
    mean_T_minus_Q = (T_int - QRS_int) * 1000.0                     # (27) mV*ms

    # Arithmetic subtraction of mean T from VG (VG - meanT)
    VG_minus_T = (VG_vec - T_int) * 1000.0                          # (28) mV*ms

    # ------------------------------#
    # 6) Eigen-based log/ratios     #
    # ------------------------------#
    ln_e3_T = np.log(max(St[2], 1e-12))   # St is the vector of singular values
    # (7) Ln µV
    # (approx amplitude: using dip_T as scale; feel free to adjust)

    # Normalized second eigenvector amplitudes
    norm_e2_T = (St[1] / St[0]) if len(St) > 1 else np.nan          # (8) µV (normalized)
    norm_e2_Q = (Sq[1] / Sq[0]) if len(Sq) > 1 else np.nan          # (9) µV (normalized)

    # T-wave dipolar voltage (ln µV) ≈ ln of dip_T
    T_dipolar_ln = np.log(max(dip_T, 1e-12))                        # (10)

    # Natural log normalized QRS non-dipolar voltage
    # "normalized" here -> sum(non-dip)/sum(total)
    total_Q = dip_Q + nondip_Q
    qrs_non_dip_norm = (nondip_Q / total_Q) if total_Q > 0 else np.nan
    ln_qrs_non_dip = np.log(max(qrs_non_dip_norm, 1e-12))           # (18)

    # Ratio between ln(QRS non-dip) and ln(T non-dip)
    total_T = dip_T + nondip_T
    t_non_dip_norm = (nondip_T / total_T) if total_T > 0 else np.nan
    ln_t_non_dip  = np.log(max(t_non_dip_norm, 1e-12))
    ln_ratio_qrs_t_nondip = ln_qrs_non_dip / ln_t_non_dip           # (21)

    # Natural log amplitude of 2nd eigenvector of T
    ln_e2_T = np.log(max(St[1], 1e-12)) if len(St) > 1 else np.nan  # (31)

    # ------------------------------#
    # 7) Areas, quadrants, angles   #
    # ------------------------------#
    # % of total QRS area in left sagittal plane (YZ) that is in right-lower quadrant
    # Left sagittal plane (YZ) -> X is "out of plane".
    # Right-lower quadrant in YZ plane: Y < 0 (inferior), Z < 0 (posterior)
    Y_qrs = QRS_xyz[:, 1]
    Z_qrs = QRS_xyz[:, 2]
    total_area_YZ = loop_area_2d(Y_qrs, Z_qrs)
    mask_rl = (Y_qrs < 0) & (Z_qrs < 0)
    if np.any(mask_rl):
        area_rl = loop_area_2d(Y_qrs[mask_rl], Z_qrs[mask_rl])
        pct_rl = 100.0 * area_rl / (total_area_YZ + 1e-12)
    else:
        pct_rl = 0.0                                                   # (11)

    # Amplitude of R wave in lead Y (µV) & Q wave in Z (µV)
    # We'll use the sample at R-peak for R, and at Q onset (q_on_idx) for Q.
    R_amp_Y = Y[q_peak_idx]                                           # (12)
    Q_amp_Z = Z[q_on]                                                 # (13)

    # Max amplitude of 3D QRS loop (µV)
    max_QRS_amp = np.max(mag[q_on:q_off])                             # (14)

    # Directions at max planar voltage
    # Frontal plane (XY) at QRS max
    V_qrs_max = np.array([X[q_peak_idx], Y[q_peak_idx], Z[q_peak_idx]])
    dir_frontal_max = project_to_plane(V_qrs_max, "frontal")
    dir_sagittal_max = project_to_plane(V_qrs_max, "sagittal")
    angle_dir_frontal_max = azimuth_deg([dir_frontal_max[0], dir_frontal_max[1], 0])  # (15)
    angle_dir_sagittal_max = np.degrees(np.arctan2(dir_sagittal_max[1], dir_sagittal_max[0]))  # (16)

    # Azimuth angle of 3D QRS loop when 2/8 (25%) into the loop
    idx_2_8 = q_on + int(0.25 * (q_off - q_on))
    V_qrs_2_8 = np.array([X[idx_2_8], Y[idx_2_8], Z[idx_2_8]])
    az_2_8 = azimuth_deg(V_qrs_2_8)                                   # (17)

    # QRS-T angle (already computed in 3D: angle_peak_QRS_T). (18–22 covered)

    # Angles of VG in sagittal/horizontal planes converted to sine (radians)
    VG_sag = project_to_plane(VG_vec, "sagittal")
    VG_hor = project_to_plane(VG_vec, "horizontal")
    angle_VG_sag_rad = np.arctan2(VG_sag[1], VG_sag[0])
    angle_VG_hor_rad = np.arctan2(VG_hor[1], VG_hor[0])
    sin_VG_sag = np.sin(angle_VG_sag_rad)                              # (23)
    sin_VG_hor = np.sin(angle_VG_hor_rad)                              # (24)

    # Max T-wave angle in horizontal plane (convert to sine radians)
    # We'll find the maximum deviation angle within T loop in horizontal plane.
    X_T_h = project_to_plane(T_xyz.T, "horizontal")  # shape (2, N_T)
    angles_T_h = np.arctan2(X_T_h[1], X_T_h[0])      # radians
    sin_max_T_h = np.sin(angles_T_h[np.argmax(np.abs(angles_T_h))])    # (25)

    # Azimuth angle in sine radians of QRS loop at 3/8 (37.5%) in loop
    idx_3_8 = q_on + int(0.375 * (q_off - q_on))
    V_qrs_3_8 = np.array([X[idx_3_8], Y[idx_3_8], Z[idx_3_8]])
    az_3_8_rad = np.radians(azimuth_deg(V_qrs_3_8))
    sin_az_3_8 = np.sin(az_3_8_rad)                                    # (26b-ish)

    # Vectorcardiographic max T amplitude / mean T amplitude (unitless)
    max_T_amp = np.max(T_mag)
    mean_T_amp = np.mean(T_mag) + 1e-12
    ratio_T_max_mean = max_T_amp / mean_T_amp                          # (29)

    # Max T-wave angle in sagittal plane, sine radians
    X_T_s = project_to_plane(T_xyz.T, "sagittal")
    angles_T_s = np.arctan2(X_T_s[1], X_T_s[0])
    sin_max_T_s = np.sin(angles_T_s[np.argmax(np.abs(angles_T_s))])     # (30)




    # --------------------------------------#
    # 8) Assemble feature dictionary        #
    # --------------------------------------#
    feats = {
        # (1) Angle between e1 of QRS and e1 of T (deg)
        "angle_e1_QRS_T_deg": angle_e1_QRS_T,

        # (2) Spatial peaks QRS-T angle (deg)
        "spatial_peak_QRS_T_angle_deg": angle_peak_QRS_T,

        # (3) Spatial mean QRS-T angle (deg)
        "spatial_mean_QRS_T_angle_deg": angle_mean_QRS_T,

        # (4) Spatial ventricular activation time (ms)
        "spatial_ventricular_activation_time_ms": vact_ms,

        # (5) Time-voltage of spatial mean T wave (mV*ms)
        "time_voltage_T_mean_mVms": time_voltage_T,

        # (6) Elevation angle of VG (deg)
        "VG_elevation_deg": VG_elev_deg,

        # (7) ln amplitude of 3rd eigenvector of T (Ln mV)
        "ln_amp_e3_T_ln_mV": ln_e3_T,

        # (8) Normalized amplitude of 2nd eigenvector (T)
        "norm_amp_e2_T": norm_e2_T,

        # (9) Normalized amplitude of 2nd eigenvector (QRS)
        "norm_amp_e2_QRS": norm_e2_Q,

        # (10) T-wave dipolar voltage (ln µV)
        "T_wave_dipolar_ln_mV": T_dipolar_ln,

        # (11) % QRS loop area in left sagittal plane RL quadrant (%)
        "pct_QRS_area_RL_sagittal_percent": pct_rl,

        # (12) R wave amplitude lead Y (µV)
        "R_wave_amp_lead_Y_mV": R_amp_Y,

        # (13) Q wave amplitude lead Z (µV)
        "Q_wave_amp_lead_Z_mV": Q_amp_Z,

        # (14) Max amplitude of 3D QRS loop (µV)
        "max_amp_QRS_3D_mV": max_QRS_amp,

        # (15) Direction of QRS loop in frontal plane at max voltage (deg azimuth)
        "dir_QRS_max_frontal_deg": angle_dir_frontal_max,

        # (16) Direction of QRS loop in sagittal plane at max voltage (deg)
        "dir_QRS_max_sagittal_deg": angle_dir_sagittal_max,

        # (17) Azimuth of QRS loop at 2/8 into loop (deg)
        "azimuth_QRS_2_8_deg": az_2_8,

        # (18) ln normalized QRS non-dipolar voltage
        "ln_QRS_non_dipolar_norm": ln_qrs_non_dip,

        # (19) QRS-T angle in frontal plane (deg)
        "QRS_T_angle_frontal_deg": qrs_t_frontal,

        # (20) QRS-T angle in horizontal plane (deg)
        "QRS_T_angle_horizontal_deg": qrs_t_horizontal,

        # (21) Ratio of ln(QRS non-dipolar) / ln(T non-dipolar)
        "ratio_ln_QRS_T_non_dipolar": ln_ratio_qrs_t_nondip,

        # (22) Max QRS-T angle among planes (deg)
        "max_QRS_T_angle_across_planes_deg": max_qrs_t_plane,

        # (23) sin(azimuth VG) in sagittal plane
        "sin_azimuth_VG_sagittal": sin_VG_sag,

        # (24) sin(azimuth VG) in horizontal plane
        "sin_azimuth_VG_horizontal": sin_VG_hor,

        # (25) Max T-wave angle in horizontal plane (sin radians)
        "sin_max_T_angle_horizontal": sin_max_T_h,

        # (26) VG optimized for RV pressure overload (mV*ms)
        "VG_RVPO_mVms": VG_rvpo,

        # (27) Spatial mean T wave minus spatial mean QRS (mV*ms) – vector magnitude
        "mean_T_minus_mean_QRS_mVms": norm(mean_T_minus_Q),

        # (28) VG minus spatial mean T (mV*ms) – vector magnitude
        "VG_minus_T_mVms": norm(VG_minus_T),

        # (29) Max T amplitude / mean T amplitude
        "ratio_T_max_to_mean": ratio_T_max_mean,

        # (30) sin(max T-wave angle in sagittal plane)
        "sin_max_T_angle_sagittal": sin_max_T_s,

        # (31) ln amplitude 2nd eigenvector of T (ln µV)
        "ln_amp_e2_T_ln_mV": ln_e2_T,

        # Extra bookkeeping
        "q_peak_idx": q_peak_idx,
        "t_peak_idx": t_peak_idx
    }

    p_feats = {}
    if p_on is not None and p_off is not None:
        p_feats = compute_p_vcg_features(
            X, Y, Z, p_on, p_off,
            mean_QRS_vec, V_Q_peak, fs
        ) 
    feats.update(p_feats)

    return feats





def plot_vcg(X, Y, Z, q_on_idx, q_off_idx, t_on_idx, t_off_idx, p_on_idx=None, p_off_idx=None,
            q_peak_idx=None, t_peak_idx=None):
   mag = np.sqrt(X**2 + Y**2 + Z**2)
   fig, axes = plt.subplots(1, 2, figsize=(11,4))
   ax = axes[0]
   if not np.isnan(p_on_idx):
        ax.plot(X[p_on_idx:p_off_idx], Y[p_on_idx:p_off_idx],
                color='blue', label='P loop') 
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


def plot_all_mean_beats_grid(mean12, lead_order, fs, r_index,
                           q_on_idx, q_off_idx, t_on_idx, t_off_idx, p_on_idx=None, p_off_idx=None, 
                           title="Mean Beats with Median Boundaries"):
    """
    Plots the mean beat for all leads in a grid layout with median boundaries highlighted.
    """
    n_leads = len(lead_order)
    L = mean12.shape[1]
    rel_ms = (np.arange(L) - r_index) * 1000.0 / fs
    
    # Create a grid of subplots (2x4 for 8 leads)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, lead_name in enumerate(lead_order):
        ax = axes[idx]
        sig = mean12[idx]
        
        # Plot the signal
        ax.plot(rel_ms, sig, linewidth=1)
        
        # QRS window
        ax.axvspan((q_on_idx - r_index)*1000/fs,
                  (q_off_idx - r_index)*1000/fs,
                  color='green', alpha=0.25, label='QRS')
        
        # T-wave window
        ax.axvspan((t_on_idx - r_index)*1000/fs,
                  (t_off_idx - r_index)*1000/fs,
                  color='orange', alpha=0.20, label='T')
        
        if not np.isnan(p_on_idx) and not np.isnan(p_off_idx):
            ax.axvspan((p_on_idx - r_index)*1000/fs,
                       (p_off_idx - r_index)*1000/fs,
                       color='blue', alpha=0.15, label='P')

        # Boundary lines (add P boundaries)
        for x, c in [(q_on_idx,'green'), (q_off_idx,'green'),
                     (t_on_idx,'orange'), (t_off_idx,'orange'),
                     (p_on_idx,'blue'),  (p_off_idx,'blue')]:
            if not np.isnan(x):
                ax.axvline((x - r_index)*1000/fs, color=c,
                           linestyle='--', linewidth=1)
        # R-peak line
        ax.axvline(0, color='k', linestyle=':', linewidth=1, label='R peak')
        
        ax.set_title(f"Lead {lead_name}")
        ax.set_xlabel("Time (ms)")
        
        # Only add y-label for leftmost plots
        if idx % 4 == 0:
            ax.set_ylabel("Amplitude (mV)")
        
        # Only add legend to the first plot
        if idx == 0:
            ax.legend(loc='upper right', fontsize='small')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Adjust to make room for suptitle
    plt.show()

def extract_and_visualize_vcg(ecg_path, pvc_flag, qrs_thresh_ms,
                            lead_order=["I","II","V1","V2","V3","V4","V5","V6"],
                            fs=FS, no_p_waves=False, visualize=True):
    """
    Extracts VCG features and visualizes:
    1. Raw V5 ECG
    2. All mean beats with median boundaries
    3. VCG plots
    """
    ecg12_raw = load_ecg_wfdb(ecg_path, lead_order)
    ecg12 = clean_all(ecg12_raw)
    
    # Arrays to store offsets from each lead
    all_q_on_rel = []
    all_q_off_rel = []
    all_t_on_rel = []
    all_t_off_rel = []
    all_p_on_rel = []
    all_p_off_rel = []
    
    # --- Plot raw V5 ECG (full 10 s strip) ---
    v5_idx = lead_order.index("V5")
    v5_sig = ecg12[v5_idx]
    if visualize:
        plt.figure(figsize=(12, 3))
        plt.plot(v5_sig, linewidth=0.8)
        plt.title("Raw V5 ECG (10 s)")
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude (mV)")
        plt.tight_layout()
        plt.show()
    
    # Process each lead to get boundaries
    for lead_idx, lead_name in enumerate(lead_order):
        lead_sig = ecg12[lead_idx]
        
        # Set the global clean variable for this lead
        global clean
        clean = lead_sig
        
        MIN_R_PEAKS = 6  # Minimum required R-peaks
        SEARCH_REALIGN_MS = 40
        
        # First try to get R-peaks from this lead
        _, lead_info = nk.ecg_peaks(lead_sig, sampling_rate=FS)
        lead_r_peaks = lead_info["ECG_R_Peaks"]
        
        # If this lead has enough R-peaks, use them
        if len(lead_r_peaks) >= MIN_R_PEAKS:
            r_peaks = lead_r_peaks
        else:
            # Not enough R-peaks in this lead, check other leads
            rpeaks_per_lead = {}
            for li, ln in enumerate(lead_order):
                if li == lead_idx:  # Skip the current lead, we already checked it
                    continue
                    
                sig = ecg12[li]
                try:
                    _, info = nk.ecg_peaks(sig, sampling_rate=FS)
                    rpeaks_per_lead[ln] = info["ECG_R_Peaks"]
                except Exception:
                    rpeaks_per_lead[ln] = np.array([], dtype=int)
            
            if all(len(arr) == 0 for arr in rpeaks_per_lead.values()):
                # Nothing better—fallback to whatever the current lead had
                r_peaks = lead_r_peaks
            else:
                ref_lead = max(rpeaks_per_lead, key=lambda L: len(rpeaks_per_lead[L]))
                ref_r = rpeaks_per_lead[ref_lead]
                
                # Realign each borrowed R to the nearest local maximum in current lead
                if len(ref_r):
                    win = int(SEARCH_REALIGN_MS * FS / 1000)
                    aligned = []
                    L = lead_sig.shape[0]
                    for r in ref_r:
                        s = max(0, r - win)
                        e = min(L, r + win + 1)
                        if e - s < 3:
                            continue
                        local = lead_sig[s:e]
                        # index of highest absolute slope/peak in window
                        local_peak = np.argmax(local) + s
                        aligned.append(local_peak)
                    if aligned:
                        r_peaks = np.array(sorted(set(aligned)), dtype=int)
                    else:
                        r_peaks = ref_r.copy()
                else:
                    r_peaks = lead_r_peaks  # fallback
        
        # Get delineation for this lead using its R-peaks
        prom_signals, prom_waves = nk.ecg_delineate(lead_sig, r_peaks, sampling_rate=fs, 
                                                 method="prominence", show=False, show_type="all")
        try:
            dwt_signals, dwt_waves = nk.ecg_delineate(lead_sig, r_peaks, sampling_rate=fs, 
                                                   method="dwt", show=False, show_type="all")
        except Exception:
            dwt_waves = {}
        
        # Fuse R onsets and offsets
        fused_on, fused_off, _ = fuse_r_on_off(prom_waves, dwt_waves)
        
        waves_fused = dict(prom_waves)
        waves_fused["ECG_R_Onsets"] = fused_on
        waves_fused["ECG_R_Offsets"] = fused_off
        
        # Get relative offsets for this lead
        offsets = collect_relative_offsets_direct(
            r_peaks,
            {
                "ECG_P_Onsets": waves_fused.get("ECG_P_Onsets", []),
                "ECG_P_Offsets": waves_fused.get("ECG_P_Offsets", []),
                "ECG_Q_Peaks": waves_fused.get("ECG_Q_Peaks", []),
                "ECG_R_Onsets": waves_fused.get("ECG_R_Onsets", []),
                "ECG_S_Peaks": waves_fused.get("ECG_S_Peaks", []),
                "ECG_R_Offsets": waves_fused.get("ECG_R_Offsets", []),
                "ECG_T_Onsets": waves_fused.get("ECG_T_Onsets", []),
                "ECG_T_Offsets": waves_fused.get("ECG_T_Offsets", []),
                "ECG_T_Peaks": waves_fused.get("ECG_T_Peaks", [])
            },
            fs, ecg_path, lead_name
        )
        
        # Store the offsets if valid
        if offsets:
            all_q_on_rel.append(offsets["q_on_rel"])
            all_q_off_rel.append(offsets["q_off_rel"])
            all_t_on_rel.append(offsets["t_on_rel"])
            all_t_off_rel.append(offsets["t_off_rel"])
            if not no_p_waves:
                all_p_on_rel.append(offsets["p_on_rel"])
                all_p_off_rel.append(offsets["p_off_rel"])
            else:
                all_p_on_rel.append(np.nan)
                all_p_off_rel.append(np.nan)

    # If no valid offsets were found across any leads, return None
    if not all_q_on_rel:
        return None
    
    # Calculate median offsets across all leads
    median_q_on_rel = int(np.median(all_q_on_rel))
    median_q_off_rel = int(np.median(all_q_off_rel))
    median_t_on_rel = int(np.median(all_t_on_rel))
    median_t_off_rel = int(np.median(all_t_off_rel))
    if no_p_waves or np.all(np.isnan(all_p_on_rel)):
        median_p_on_rel  = np.nan
        median_p_off_rel = np.nan
    else:
        median_p_on_rel  = int(np.nanmedian(all_p_on_rel))
        median_p_off_rel = int(np.nanmedian(all_p_off_rel))
    
    # Use V5 for final mean beat calculation (as in original code)
    v5_sig = ecg12[lead_order.index("V5")]
    _, v5_info = nk.ecg_peaks(v5_sig, sampling_rate=FS)
    v5_r = v5_info["ECG_R_Peaks"]
    
    # Get V5 R-peaks or best alternative for final processing
    MIN_R_PEAKS = 6
    if len(v5_r) >= MIN_R_PEAKS:
        final_r_peaks = v5_r
    else:
        rpeaks_per_lead = {}
        for li, lead_name in enumerate(lead_order):
            sig = ecg12[li]
            try:
                _, info = nk.ecg_peaks(sig, sampling_rate=FS)
                rpeaks_per_lead[lead_name] = info["ECG_R_Peaks"]
            except Exception:
                rpeaks_per_lead[lead_name] = np.array([], dtype=int)
        
        others = {k:v for k,v in rpeaks_per_lead.items() if k != "V5"}
        if all(len(arr) == 0 for arr in others.values()):
            final_r_peaks = v5_r
        else:
            ref_lead = max(others, key=lambda L: len(others[L]))
            ref_r = others[ref_lead]
            
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
                    local_peak = np.argmax(local) + s
                    aligned.append(local_peak)
                if aligned:
                    final_r_peaks = np.array(sorted(set(aligned)), dtype=int)
                else:
                    final_r_peaks = ref_r.copy()
            else:
                final_r_peaks = v5_r
    
    # Build mean beat using all leads
    res = build_mean_beat(ecg12, final_r_peaks, waves_fused, fs, PRE_MS, POST_MS, pvc_flag, qrs_thresh_ms)
    if res is None:
        return None
    mean12, r_index, kept_r = res
         
    # Calculate absolute indices using median relative offsets
    q_on_idx = r_index + median_q_on_rel
    q_off_idx = r_index + median_q_off_rel
    t_on_idx = r_index + median_t_on_rel
    t_off_idx = r_index + median_t_off_rel
    if not no_p_waves:
        p_on_idx = r_index + median_p_on_rel
        p_off_idx = r_index + median_p_off_rel
    else:
        p_on_idx = np.nan
        p_off_idx = np.nan

    if visualize:
    # Display all leads with median boundaries in a grid
        plot_all_mean_beats_grid(
            mean12, 
            lead_order, 
            fs, 
            r_index, 
            q_on_idx, 
            q_off_idx, 
            t_on_idx, 
            t_off_idx,
            p_on_idx,
            p_off_idx,
            title=f"Mean Beats with Median Boundaries (QRS: {median_q_off_rel - median_q_on_rel}ms)"
        )
    
    # Project to VCG space and compute features
    X, Y, Z = (KORS_3x8 @ mean12)
    features = compute_vcg_features(X, Y, Z,
                                  q_on_idx, q_off_idx,
                                  t_on_idx, t_off_idx, 
                                  p_on_idx, p_off_idx,
                                  fs)
    
    if not features:
        return None
    
    if visualize:
        plot_vcg(
            X, Y, Z,
            q_on_idx, q_off_idx,
            t_on_idx, t_off_idx,
            p_on_idx, p_off_idx,
            features["q_peak_idx"], features["t_peak_idx"]
        )
    
    # Return the four requested features
    return features
    




# Container for per-row features
rows_features = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Extracting VCG features"):
    pvc_flag      = bool(row.get("PVC", False))
    qrs_thresh_ms = 225
    no_p = bool(row.get("Atrial Fibrillation", False) or
                row.get("Atrial Flutter", False))
    path_str      = row["path"]
    ecg_path      = f'data/mimic-iv-ecg/{path_str}/{path_str[-8:]}'

    try: 
        feats = extract_and_visualize_vcg(
                ecg_path, pvc_flag, qrs_thresh_ms, no_p_waves=no_p, visualize=False  # turn off plotting in batch
            )
        if feats is None:
                feats = {}
    except Exception as e:
        # In case of an error, fill NaNs for this row
        feats = {}


    # Ensure 'path' is retained for merging back
    feats["path"] = path_str
    rows_features.append(feats)


feat_df = pd.DataFrame(rows_features)

df = df.merge(feat_df, on="path", how="left")
df.drop(columns=['q_peak_idx', 't_peak_idx', '2116Unnamed: 0'], inplace=True, errors='ignore')
df.to_csv('full_cohort.csv', index=False)
