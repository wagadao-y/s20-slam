# Validation

[日本語](validation_jp.md)

Results on data recorded with the author's one S20. The data is not published. The "Python version" compared is the author's Python prototype made before this app (LIO → pose graph → BA).

## Indoors (a house, 650 s)

The reference trajectory is a BA refined from the trajectory processed by PointClouds Studio.

| | Pose-induced error | Surface thickness | 1 s error, max | 5 s error, max | 60 s error, median |
|---|---:|---:|---:|---:|---:|
| Python version (LIO→PGO→BA×3) | 3.42 mm | 9.84 mm | 37.5 cm | 42.8 cm | 3.2 cm |
| s20-slam (LIO + trajectory optimization) | 3.15 mm | 9.15 mm | 3.5 cm | 5.0 cm | 1.4 cm |

Image consistency (independent of the LiDAR: the median epipolar error of SIFT matches in the fisheye images):

| | Neighbors | Revisits |
|---|---:|---:|
| BA from PointClouds Studio's trajectory (reference) | 2.44 px | 2.62 px |
| Python version | 2.58 px | 2.87 px |
| s20-slam | 2.28 px | 2.62 px |

The effect of a loop: a LIO broken on purpose by keeping the map shorter (start and end 2.1 m apart, up to 1.4 m from the true trajectory) came, with one **Start = end** loop, to 0.9 cm between start and end and 1.0 cm RMS from the true trajectory (4.9 cm at most).

## Outdoors (a park, with RTK)

The numbers in this section were measured with a version before the gravity correction and other changes.

An 1100 s recording:

| | RTK residual, horizontal | Vertical | 3D, 95% | Pose-induced error |
|---|---:|---:|---:|---:|
| Python version (BA + RTK) | 27.4 mm | 18.3 mm | 57.0 mm | 2.71 mm |
| s20-slam (RTK) | 14.8 mm | 19.6 mm | 42.6 mm | 2.54 mm |

A recording with false fixes: RTK residuals horizontal / vertical / 3D 95% of 30.9 / 16.0 / 61.4 mm for the Python version, 12.9 / 19.4 / 41.2 mm for s20-slam (the validity check used 9815 of 14213 epochs).

Two sessions joined (with RTK). The Python version adds loops across the sessions in a pose graph, then runs BA; s20-slam uses no loops, fitting each session to RTK and solving them together with the plane coupling:

| | RTK residual, horizontal | Vertical | 3D, 95% | Pose-induced error | Rigid residual of the overlap | Plane misfit, median / 90% |
|---|---:|---:|---:|---:|---:|---:|
| Python version | 27.4 mm | 17.0 mm | 56.3 mm | 2.72 mm | 1.1 cm | 0.4 / 1.4 mm |
| s20-slam | 14.1 mm | 19.0 mm | 41.6 mm | 2.69 mm | 0.8 cm | 0.4 / 1.4 mm |

PPK (a 30-minute recording with a **Start = end** loop): 99% within 33 mm of PPK horizontally, and the loop closed to 6.5 mm.

## Processing time and size

Indoors, 650 s, `s20 run` all at once, RTX 3060:

| Step | Time |
|---|---:|
| Import (a pass over the whole bag) | 21 s |
| LIO | 15 s |
| Trajectory optimization | 1 min 38 s |
| Point cloud (95 million observations → 30.3 million points) | 207 s |
| Coloring (GPU) | 216 s |

The work folder is 704 MB. With the CPU only, the point cloud and the coloring took 417 s and 465 s.

LAZ compression: 19.62 million colored points, 0.94 GB → 0.39 GB in 4.5 s. The normals (floats) compress poorly, so about 40% of the LAS remains. The output has been checked to read in PDAL and PotreeConverter 2.

## Findings

- The plane coupling converges faster and is more accurate, but with a loose IMU weight, short back-and-forth motions in narrow places (about 0.5 m) are flattened. Tightening the IMU noise to the real device's (gyro 2e-4 rad/s/√Hz, accelerometer 0.005 m/s²/√Hz) keeps them.
- Solving alternately with the planes fixed hardly moves low-frequency deformations such as the whole trajectory tilting up or down. For RTK to act, the planes and the trajectory need to be coupled.
- Indoor's fine LIO values used outdoors improve the local accuracy but increase the overall heading drift and roughly double the processing time. No single set of values fits both indoors and outdoors, so the presets are separate.
