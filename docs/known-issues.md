# Known issues

[日本語](known-issues_jp.md)

The issues known so far, and what is planned.

## Accuracy

- **Large heading drift of LIO**: on round hills with few features, LIO's heading alone can be dragged one way, over ten meters in 30 minutes. With GNSS the trajectory optimization absorbs it, but without GNSS it distorts the point cloud. For now, mark the section as a **section with heading held to the IMU** by hand.
- **Jumps on returning to a place passed before**: in long recordings, returning to an old part of the map that has drifted can leave large registration residuals, and the trajectory may jump or be dragged.
- **Convergence where constraints are weak**: in areas of vegetation only, the trajectory can keep moving a few cm at each iteration of the optimization's last stage and not reach the convergence test.
- **Systematic vertical offsets**: the difference from PPK is mostly within 1 cm horizontally, but vertically it shifts by a few cm from section to section.
- **Detecting false fixes**: it has been checked on few real false fixes so far.
- **Haze and thickness of surfaces**: compared with a point cloud processed by PointClouds Studio, a little more haze (points a few cm off the surface) remains at the edges of things.

## Features

- **Finding loops automatically**: revisits find and join the same place when it is off by up to about 2 m. Places off by more, and the start and end of a recording that starts and ends at the same place when they are far apart, need to be given by hand (**Start = end**).
- **Several sessions without GNSS**: the first placement of the sessions needs loops across sessions or point matches given by hand. Where the pairs are only along a path, as a trail in a wood, a few tens of cm can remain even with revisits.
- **Time offset between the IMU and the LiDAR** is not estimated.
- **Recordings for 3DGS** (0.5 s intervals, far and near views) are still to be checked.

## Stability

- **GPU errors**: a wgpu validation error or running out of memory could stop processing with an "internal error" (it has not happened with the author's data).
- **End-to-end regression tests**: there are no automatic tests that run the steps through yet.

## What it has been tested on

One S20 (the author's), Windows 11, RTX 3060. On other devices (other firmware or calibration), the initial accelerometer bias (`ba0` in the detailed parameters) and other values may not fit. If it does not work, please open an Issue with the version (**Help → About**), the raw data's layout (**Files** and **Topics** in **Tools → View raw data…**) and the log. The logs are in the work folder's `logs/`. Paths in the log are relative to the work folder, and your user folder is written `~`. The **GNSS** tab of **View raw data…** shows the base station's latitude and longitude: hide them in screenshots if you do not want to make the place public.
