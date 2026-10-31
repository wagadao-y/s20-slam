# S20 embedded PPK

`ROVER_PPKRAW.bin` (Unicore AA44B5) and `BASE.BIN` (RTCM3) are decoded in memory.
RTKLIB demo5 performs kinematic forward and backward solutions and combines them
with its consistency check. No external executables, RINEX conversion, ROS,
Python or GNSS network access are needed at runtime.

GNSS UTC is mapped into the SLAM clock using `/ssync/sync_time` in `info/*.bag`.
Interpolation requires monotonic clocks, gaps no longer than 3 s, and clock-rate
agreement within 10 ms. Out-of-range or discontinuous synchronisation is omitted.
UTC conversion uses RTKLIB's leap-second table; filenames and bag record timestamps
are not clock anchors. PPK height is ellipsoidal and is not geoid-corrected again.

CRC failures and truncated frames are skipped with byte-wise resynchronisation.
Sampling retains cycle-slip flags from skipped epochs. Unicore's extended GPS L2S
is used in the second band when L2W is absent. Extended GLONASS L2C is likewise
used when the primary L2 signal is absent, retaining its slip and quality fields.
BeiDou satellite IDs C01..C63 are supported, including base C56/C58 ephemerides
previously rejected by the vendored C46 limit. The existing orbit model handles
GEO C59+ as well as C01..C05. S20's legacy BeiDou B1I/B2I/B3I
observations are placed in corresponding bands in both receivers; B1C/B2a data
not available on the rover are omitted. Each partial base MSM is preserved before
the next fragment reuses extended slots, and duplicate satellite records at the
same epoch are merged. RTCM 1005/1006 ECEF antenna reference positions are used
directly, without adding the 1006 antenna height.

Default processing: three frequencies, GPS/GLO/GAL/BDS/QZSS, elevation 15 degrees,
continuous integer ambiguity resolution, ratio 3, ambiguity elevation 25 degrees
and at least 2 s lock, phase/code innovation limits 0.2/10 m, two filter iterations,
broadcast ionosphere and Saastamoinen troposphere. Additional atmospheric states
are optional; a longer baseline does not automatically make them converge faster.
GLONASS is used for FLOAT positioning with ambiguity resolution disabled because
the base and rover receivers differ. Both the AR manager and the integer-pair
selector explicitly enforce OFF; the unpatched demo5 selector admitted GLONASS
integer pairs even with `glomodear=0`.

The combined solution checks directional agreement at RTKLIB's existing 4-sigma
limit, evaluated in local east/north/up. The sigmas are floored at
`fix_check_sd_h` / `fix_check_sd_v` (default 0.01 / 0.02 m): FIX covariance is
often a few millimetres while correct forward and backward FIX solutions differ
by 1-2 cm, below the accuracy of network corrections, and the unfloored check
downgraded most epochs. A wrong integer set still separates the directions by
decimetres. Zero floors use the filter covariance alone. FIX/FIX disagreement
retains the original downgrade to FLOAT. A one-sided FIX is checked against
the other direction's FLOAT with the FLOAT sigmas also floored, at
`float_check_sd_h` / `float_check_sd_v` (default 0.10 / 0.20 m, added in
quadrature to the FIX floors, so about 0.4 m per horizontal axis and 0.8 m
vertical at 4 sigma): FLOAT covariance is millimetres while FLOAT converging
from a cold start or over a long baseline is often 0.1-0.3 m off, and the FIX
floors alone discarded such FIX solutions for minutes, although FIX/FIX pairs
and the LIO trajectory agreed with the FIX side. Zero floors check FIX/FLOAT
like FIX/FIX. FIX/FLOAT
disagreement beyond that adds the outer product of the ECEF directional separation to the
chosen FIX covariance (including cross terms). The normal ENU uncertainty limits
then decide acceptance, and accepted positions carry the increased uncertainty
into SLAM. FLOAT covariance need not cover unresolved ambiguity bias, so using
it as ground truth to downgrade every disagreeing one-sided FIX was excessively
restrictive. An unpaired epoch or a SINGLE counterpart does not receive this
relative-position test. This is a conservative S20 uncertainty policy extension,
not a claim that RTKLIB's original quality-selection policy is an implementation
error or that directional agreement proves surveyed accuracy.

`phase_snr_error` optionally adds RTKLIB's SNR-dependent observation variance at
the 52 dB-Hz reference (metres, 0 disables it). It affects phase and code via the
existing code/phase ratio. `prefer_base_l2c` optionally selects base GPS L2X for
satellites tracked only as L2S by the rover. Legacy L2W or mixed rover tracking
retains the original selection. For a selected satellite, missing L2X epochs
are omitted instead of switching tracking families; bases without L2X retain
L2W. Both options default to disabled: neither improved this dataset reliably.
CLI: `--phase_snr_error <m>` and `--prefer_base_l2c`; GUI: detailed PPK settings.

Only FIX positions passing uncertainty and correction-age limits are used by
default. FLOAT is opt-in; it uses additional 0.20 m horizontal / 0.40 m vertical
uncertainty floors because filter covariance does not describe unresolved integer
ambiguities. RTK and PPK are alternative sources, not independent factors at the
same epoch. Exclusion intervals and the antenna lever arm apply to both. FIX by
itself does not prove surveyed accuracy. Zero accepted positions is reported as
an optimisation error instead of silently switching to receiver RTK.

Cache identity includes engine revision, solver settings and canonical input
paths, sizes and modification times, including sync bags. Acceptance settings
can change without repeating the solver. Processing is cancellable; incomplete
computations are not saved as a completed cache. RTKLIB global state is serialised
within a process.

`solutions` contains only the combined solution used by SLAM. The JSON also
stores `forward_solutions` and `backward_solutions` in chronological order for
epoch-level diagnosis; they never add duplicate GNSS constraints. Counts
distinguish downgraded FIX/FIX pairs from uncertainty-increased FIX/FLOAT pairs. Decoding-error logs
identify message IDs separately from CRC/truncation errors. Completed pipeline
optimisations also record the PPK engine revision, so an engine update invalidates
old optimisation results even when user settings and raw files are unchanged.

Upstream: https://github.com/rtklibexplorer/RTKLIB
Revision: `75a2e56275485b21a67bd35bc94bbeb8936e1a74` (demo5 b34L).
See `vendor/rtklib/license.txt` (BSD 2-clause). Upstream changes: MSVC static-library
EXPORT and TLS in rtklib.h; S20 B3 band in unicore.c; optional `-S20BDS` mapping in
rtcm3.c; BeiDou C63 limit in rtklib.h; GLONASS AR OFF enforcement in rtkpos.c;
S20 FIX/FLOAT consistency check in postpos.c; the static SOFA routines in sofa.c renamed
with an `rk` prefix in place of `iau`, as the SOFA license asks of derived routines. The application adapter is Apache-2.0. The GUI parameter window and
`s20 licenses` include the RTKLIB license for binary distributions.
