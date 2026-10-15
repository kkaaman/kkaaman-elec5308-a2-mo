# Baseline batch validation — 15 October 2026

I ran the Windows Python 3.12.10 regression suite: 394 passed, 1 live-CARLA
marked test deselected, 3 subtests passed. Separate live checks are recorded here.

## BL-B-02

CARLA 0.9.16, Town10HD_Opt, parked Tesla Model 3, rigid matched 640x360,
90-degree cameras. 24 exact-frame RGB/depth/ego joins completed. Missing-frame
injection applied full brake; no callback errors. Previous world settings were
restored and no vehicle/sensor actors leaked. One queue overflow was recorded;
bounded eviction worked and no mismatched frames were joined. The 95 discarded
stream entries include retirement after successful joins; this is not a count
of 95 failed captures. The one timeout is the intentional missing-frame test.

The initial smoke read wheel geometry before physics advanced and failed
validation. The corrected script brakes, advances 20 frames, then constructs
the adapter; order and fail-closed geometry behavior have regression tests.
Raw sample.npz remains in the Downloads validation archive. Raw depth convention,
metric accuracy and moving-ego capture accuracy are not established here.

## BL-A-03

The installed map audit accepted 60 maneuver routes: right 18, straight 37,
left 5. All 11 rejected cases had two legal exits of the requested maneuver.
This is expected fail-closed ambiguity behavior, not proof those roads are
illegal. Explicit route/exit configuration is required before using them.
Saved topology replay independently reproduced all 60 accepted routes and all
11 error reasons. No vehicle was driven by this map audit. Geometry checks
and synthetic T/4-way tests do not establish closed-loop route following.

## BL-C-03 and dependencies

The local YOLO asset matches SHA256
9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef
and size 5544453 bytes. Installed package versions: ultralytics 8.4.177,
torch 2.14.1, numpy 2.5.4. Backend construction took 0.0400 s; first call
1.5023 s; subsequent calls 0.0554, 0.0274 and 0.0268 s (mean 0.0365 s).
Each sample call returned 8 detections. These are wall-time sample checks,
not detector accuracy evaluation or a sustained performance benchmark.
The raw detector report captured the manifest before its version fields were
updated; its top-level versions are the measured installed package values.
Synthetic offline sequence fixtures remain explicitly labelled synthetic.

## BL-D-03 / RV-04

Synthetic tests cover inner foreground vs outer background, mixed surfaces,
invalid/insufficient depth, off-axis projection, legacy constructors and same-
source identity checks. Ground/known-size fallbacks remain flagged invalid.
No inferred distance is automatically interpreted as road clearance.

## G0-01 / review

See ../../gates/G0.md for proposed follow-up owners and dates. G0 is REVIEW
PENDING: scope response, other-PC software verification, team archive access
and reviewer acknowledgement must not be claimed complete without evidence.
BL-E-02 remains a separate earlier commit with its measured steering limits.

ChatGPT assisted with implementation, tests and analysis. I ran the Windows
commands and CARLA experiments and inspected the reported results.
