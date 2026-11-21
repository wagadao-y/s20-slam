# Command line (`s20`)

[日本語](cli_jp.md)

The same processing as the GUI, from the command line. `s20 help` shows every command and option.

## All the steps at once

```
s20 run <raw folder> <work folder> [--outdoor | --params file.json] [--no_rtk] [--ppk] [--no_color]
        [--las] [--images [--3dgs_poses] [--sky_sphere | --no_sky_sphere]] [--no_refine] [--device_calibration file.yaml] [--redo] [--until step]
s20 run <raw 1>,<raw 2>,... <work folder> --outdoor   # multiple sessions
```

The steps are extract → lio → opt → cloud → export → refine → colorize.

| Option | |
|---|---|
| `--outdoor` | Outdoor settings (indoor by default) |
| `--params` | Settings saved by the GUI (the detailed parameters of `settings.json`) |
| `--no_rtk` | Do not use the device's RTK (by default it is used when there are fixes) |
| `--ppk` | Compute and use PPK ([PPK](ppk.md), in Japanese) |
| `--no_color` | Do not color |
| `--las` | Write the point clouds as LAS (LAZ by default) |
| `--images` | After coloring, write the photos with their refined poses to `output/images` (in PointClouds Studio's layout) and a COLMAP model beside them. The COLMAP model takes the poses the coloring uses (fitted to the point cloud) |
| `--3dgs_poses` | Experimental. With `--images`, image pose refinement also finds the poses for 3DGS (the photos agreeing best with one another) for the COLMAP model, and `image_points.txt` is written too. Image pose refinement takes longer. Not with `--no_refine` |
| `--sky_sphere`, `--no_sky_sphere` | With `--images`, add / do not add the sky sphere (see below) to the COLMAP model's initial points. By default added with `--outdoor` |
| `--no_refine` | Do not refine the image poses |
| `--device_calibration` | Use the device calibration (made by `calibrate-device`) instead of the maker's factory calibration file (`calibration.yaml`) |
| `--redo` | Redo the import and LIO even when they are there |
| `--until` | Stop after that step |

With multiple sessions, the import and LIO run per session in `<work folder>/sessions/<session name>/` (at first the raw folder's name; only the sessions added are run), and the trajectory optimization and the later steps once for all the sessions.

## One step at a time

```
s20 extract  <raw folder> <data>                                # the bag's index (where the points and images are), IMU and GNSS
s20 lio      <data> runs/lio.txt [--indoor] [--hold a:b]        # odometry. --hold: a section with the heading held to the IMU (s, repeatable)
s20 opt      <data> runs/lio.txt runs/opt.txt [--rtk] [--ppk runs/ppk.json] [--rtk_exclude a:b] [--loop a:b|start-end]
s20 cloud    <data> runs/opt.txt out/cloud.laz [--config params.json]
s20 export   <data> runs/opt.txt out --calib <raw folder>/info/calibration.yaml
s20 refine-images --data <data> --imgpose runs/ImgPose.txt --calib calibration.yaml --cloud out/cloud.laz runs/ImgPose_refined.txt
s20 colorize out/cloud.laz out/cloud_col.laz --images <data>/images --imgpose runs/ImgPose.txt --calib calibration.yaml
```

- Times are seconds from the session's first scan; negative values are seconds back from its end. With multiple sessions, give `<session name>@<s>` or `<number>@<s>`.
- `opt` given multiple LIO trajectories, separated by commas, solves them as multiple sessions (`data` needs `sessions.txt` and the joined `imu.npy` / `gnss.npy`, which `s20 run` makes). Without GNSS, the first session is the reference, and the others are placed by loops across sessions or `--match [session:]x,y,z=[session:]x,y,z` (coordinates of the same feature, two pairs or more per pair of sessions).
- `opt` uses revisits ([How it works](processing.md), in Japanese); `--no_revisits` turns them off. `--revisit_dist m` (the distance to pair places), `--revisit_inliers f` (the least overlap), `--revisit_path_sd k` (the factor of the standard deviation along the path), `--robust sd` and others change the settings. `s20 eval-revisits <data> <opt_knots.txt> [out.csv]` registers the trajectory's pairs of the same place again and writes the width of the doubling per pair to a CSV.
- `cloud`'s `--config` is a JSON of the point cloud settings; missing items take the defaults.
- `colorize` reads the images from `<data>/images` when the files are there, else the JPEGs in the bag.

## Other commands

```
s20 ppk <raw folder> runs/ppk.json [options]                    # PPK positioning (see the PPK page)
s20 calibrate-device <work folder>... --out calibration.yaml    # a device calibration from processed work folders (the GUI's Device calibration)
s20 export-colmap <colored point cloud> <output> --data <data> --imgpose ImgPose.txt --calib calibration.yaml   # a COLMAP model for 3DGS
s20 images   <data> <folder>                                    # write the camera images (JPEG) from the bag
s20 convert  in.las out.laz                                     # convert between LAS and LAZ (by the extension)
s20 potree   <in.las|laz> [--no_rgb] [--gps_time]               # make a Potree 2.0 octree
s20 eval-traj <estimate.txt> <reference.txt>                    # evaluate a trajectory (ATE, relative errors over 1 / 5 / 30 s, jumps)
s20 cut-raw <raw folder> <output> [--from s] --to s             # cut part of a recording out as a short raw folder (for tests)
s20 licenses                                                    # show the licenses
s20 version                                                     # show the version (also --version)
```

`cut-raw`'s times are seconds from the bag's first scan. Only `all_*.bag` is cut; the other files (`info/`, `frame_pose.txt`, `BASE.BIN` and so on) are copied as they are. LIO initializes assuming the first 2 seconds are still, so put `--from` a few seconds before the device starts moving. The device's own trajectory (`frame_pose.txt`) may start earlier than the bag, so its times are no guide; check when the motion starts in the imported IMU.

Point cloud files are LAS or LAZ by their extension (`.laz` is compressed). Both are read.

## COLMAP output (for 3D Gaussian Splatting)

`export-colmap` (or `s20 run --images`) writes the recording as a COLMAP model, for external 3DGS tools.

| Output | Contents |
|---|---|
| `images/left/`, `images/right/` | The recorded JPEGs (not recompressed) |
| `sparse/0/cameras.txt` | The left and right cameras: `OPENCV_FISHEYE` (Kannala-Brandt) fitted to the S20's fisheye polynomial |
| `sparse/0/images.txt` | The images' poses (those the coloring uses; with `s20 run --images --3dgs_poses`, the poses for 3DGS, `ImgPose-3dgs.txt`) |
| `sparse/0/points3D.txt` | Initial points thinned from the colored point cloud (500,000 by default), followed by the sky sphere's when it is added (`export-colmap` adds it with `--sky_sphere`) |
| `masks/` | The image corners beyond `--max_angle` (85°) from the optical axis (outside the range the calibration was fitted to) |
| `gauge.txt` | Z up, meters |
| `export.json` | The fit's error, the number of points |
| `image_points.txt` | With `s20 run --images --3dgs_poses` after the image refinement: the photos' feature tracks triangulated with the poses of `ImgPose-3dgs.txt` (see below) |

Kannala-Brandt does not match the S20's polynomial exactly. The fit's error is about 0.1–0.2 px RMS.

The sky sphere is points placed on a large sphere around the scene. The point cloud has no points in the sky and the far background, so in 3DGS training the Gaussians in front tend to stretch out to show the sky and become floaters; the sky sphere gives them their initial points instead. Its center is the mean of the camera positions and its radius twice the distance within which 99% of the point cloud lies (50 m at least); its points are about 1° apart down to 10° below the horizon, and those no image sees are left out. They are gray: training finds the sky's colors. It is of no use indoors, so by default it is added only outdoors (`--outdoor`).

`image_points.txt` is for filling, with points from the photos, where the LiDAR measured nothing (right next to the device, the low sides of furniture and the like). It is a file of its own beside `points3D.txt` (the LiDAR points), one point per line: the position (x y z), how many photos see it, the time between its first and last photo (s), the widest angle between its photos' rays (degrees), the reprojection error's median and largest (pixel-equivalents), the nearest photo's distance (m), the distance to the nearest point of the point cloud (m), and where the point comes from (0: the matches the image refinement used, searched with the point cloud's help; 1: matches searched along the epipolar lines, without the point cloud). The point cloud misses surfaces such as those right next to the device, so set 0, whose search relies on the point cloud's prediction, has hardly any points there; set 1 searches there too. Points far from the point cloud lie on surfaces it misses. Points with a small angle (short baselines) spread along their rays, and those with a short time span include what was seen only while passing once (people and such). Points seen by only two photos cannot be told from wrong matches, so choose by the number of photos, the angle and the error when using them.
