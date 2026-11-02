# Using the GUI

[日本語](gui_jp.md)

What `s20-gui.exe` does, screen by screen. For the basic flow, see the [README](../README.md).

## Work folder

The start screen offers **New work folder…**, opening a work folder, or one of the recent work folders. **File** at the top also creates, opens and closes one (back to the start screen).

A work folder holds the intermediate results (`data/`, `runs/`), the products (`output/`), the settings (`settings.json`) and the trajectory corrections (`s20_edits.json`). Points and images are not copied: the app reads the raw data's bag directly, read only. Keep the raw data in the same place while you work on it (`data/source.txt` records its path and size, and the app tells you to import again if they change).

The display index is in `cache/preview/`. You can delete it when the app is closed; it is made again the next time.

## The window

- **Left panel**: the sessions, the steps and **Processing settings**. Fold it with « in its header, open it again from the strip on the left edge.
- **Toolbar** (top): the preview, **Show all**, and the right panel (**Display**, **Trajectory**, **Images**).
- **Time bar** (bottom): the time chosen. Step it with the ◀▶ buttons or the left and right keys (1 s with Shift), or type a time in the box on the right.
- **Status bar** (very bottom): points drawn and the limit, loading state.
- **Log**: opened and closed with the button at the bottom right. It turns red when processing fails.

Saving, undo, input errors, and processing finishing or failing are shown at the bottom right of the view for a few seconds (they stay in the processing log; click to close).

**Help → About** shows the version, and links to the source, the latest release and problem reports (Issues).

## Processing settings

The GNSS (none / the device's RTK / PPK) is chosen in **Processing settings** in the left panel. Indoor / outdoor, the output format (LAZ / LAS), the gravity correction, the detailed parameters and so on are in **Other settings…** (or **Process → Processing settings…**). The settings are saved in the work folder's `settings.json`; a new work folder starts from the last settings used.

Indoor and outdoor differ only in four LIO values (downsampling, the map's voxel, the minimum range, the LiDAR weight). The trajectory optimization, PPK, point cloud and coloring use the same settings.

## Running the steps

| Step | What it does |
|---|---|
| Import | The bag's index (where the points and images are), the IMU and GNSS tables |
| Odometry (LIO) | Tightly coupled LiDAR–inertial odometry |
| Trajectory optimization | The whole trajectory optimized with the IMU, the planes of the point cloud, GNSS and loops |
| Point cloud | Free-space test, projection onto surfaces, removal of things passing by |
| Trajectory and image pose export | trajectory.txt / las, ImgPose.txt |
| Image pose refinement | Each image's pose adjusted to the point cloud |
| Colorization | The point cloud colored from the camera images |
| Image export | Optional. In PointClouds Studio's layout and as a COLMAP model |

**Run from unfinished or unapplied steps** runs only the steps needed, in order. Each step's ▷ reruns that step alone, or from it to the end (from an earlier unfinished step if there is one; the menu shows what will run).

A result older than an earlier step's (or, for the trajectory optimization, than the corrections) is "unapplied": a yellow ↻ with the reason (settings changed, trajectory corrected, an earlier step redone). The status bar also says so when the preview shown is based on an unapplied result.

Closing the window while processing asks whether to cancel and exit.

## Preview

Chosen with **Preview** in the toolbar. What has not been made yet cannot be chosen.

| Preview | What it shows |
|---|---|
| Scan point cloud (device) | The raw points placed by the trajectory the device estimated while recording (the raw data's `frame_pose.txt`). Available right after the import |
| Scan point cloud (LIO) | Placed by the LIO trajectory |
| Scan point cloud (optimized) | Placed by the optimized trajectory. Follows the trajectory as soon as it is made again |
| Output point cloud (uncolored) | The result of the point cloud step |
| Output point cloud (colored) | The result of the coloring. Compared with the uncolored one, it shows the points dropped for lack of a color |

Mouse: left drag rotates, right drag moves, the wheel zooms, a double click makes that point the center of rotation. **Show all** fits the whole point cloud. Switching the preview keeps the time mark and the zoom, and moves the view to the pose at that time.

The **Display** settings in the right panel:

- **Point colors**: colorized (the camera images' colors), reflectivity, height, session, plain. Remembered separately for the scan and output point clouds. The height range can be changed at the legend's ends; **Fit to view** fits it to the points visible, **Auto** goes back.
- **Range**: limits the scan point cloud to some seconds around the time chosen, or from the start to the time chosen.
- **Adjust the point size automatically**: as Potree's adaptive size, makes the points larger to the spacing of the points drawn. Off by default for the scan point clouds.
- **Drawn points** (5 million by default): the most points drawn on the whole screen. Coarse points of the whole point cloud always stay; zooming and moving loads the details needed in the background.

The output point cloud is read from a Potree octree. When there is none, or it is older than the point cloud, it is made when opened (about 40 seconds for 100 million points).

## Checking the quality

The trajectory optimization writes the quality of each section (about 0.1 s) to `opt_quality.txt`:

- plane misfit (RMS, and the share of points on planes)
- disagreement with the IMU (the preintegration's χ² per degree of freedom)
- RTK residuals (horizontal, vertical)
- correction (how much the odometry's motion over 1 s was corrected, in rotation and translation)

Choosing a metric in **Color by** in the right panel's **Trajectory** paints the trajectory from green to red and lists the sections that look wrong (the highest 1 s averages, at least 5 s apart). Clicking one moves there and marks it. The value shown red can be changed. The time bar is painted in the same colors, with the listed sections as triangles above and the loops' times as dots below.

Choosing a time on the time bar, or with Ctrl+click on the trajectory, shows the camera images (left and right) of that time in the right panel's **Images**.

## Correcting the trajectory

The right panel's **Trajectory** has a list of each kind under **Corrections**. Press **Add**, then click the trajectory twice (a section's start and end; for a loop, two times it passed the same place). Esc cancels.

| Kind | Color | When to use it |
|---|---|---|
| Sections without GNSS | purple | The RTK / PPK positions in the section are not used. For false fixes, and bad positions under buildings or trees |
| Sections with heading held to the IMU | light blue | In the section, LIO does not update the heading and the gyro bias from the LiDAR, and keeps the IMU's prediction. For round hills (concentric paths, round fences) and featureless slopes where the LiDAR slowly drags the heading |
| Loops | yellow | Joins two times at the same place. There is also a **Start = end** button |

- The marks at a correction's ends can be dragged along the trajectory. Click a section or a mark to select it, and Delete removes it. Selecting a row of the list lets you type the times of its ends; ⌖ goes to that time.
- **Apply the corrections** redoes the trajectory optimization, or from LIO when heading sections changed.
- The GNSS positions (white: used, orange: rejected by the validity check, purple: excluded) can be drawn over the trajectory from the right panel's **Display**. Hovering over one shows the solution's state and why it was used or not.
- The corrections are written to `s20_edits.json` at each change. Ctrl+Z / Ctrl+Y (Ctrl+Shift+Z) undo and redo (up to 200 times). Ctrl+S (**Edit → Save corrections**) keeps the corrections as a revision in `edits_revisions/`; running the trajectory optimization also keeps one when they differ from the previous revision. **Correction history** renames them, goes back to one, and deletes them.

## Multiple sessions

Recordings of the same place made in multiple sessions are joined into one trajectory and point cloud by adding their raw folders with **Add session…**. The import and LIO run per session (`sessions/<session name>/`), and the trajectory optimization and the later steps once for all the sessions.

- With GNSS (RTK / PPK), the sessions are placed by it.
- Without GNSS (indoors and so on), the first session is the reference, and the others are connected to it by loops across sessions (two clicks on the trajectory) or point matches (clicks on the same feature in two sessions). Loops are easier when the paths pass the same spot. When they do not come close, use point matches, two pairs or more, far apart.
- After the first placement, places passed again, within and across sessions, are found and registered by matching their point clouds (revisits: **Processing settings → Trajectory optimization → Revisits**, on by default; see [How it works](processing.md), in Japanese). So loops are needed only for the placement. A loop fits best at times when the two sessions see the same place in the same direction.
- Work folders made by earlier versions open with revisits off (so their results do not change). To use them, turn on **Use revisits** and run the optimization again.
- While adding point matches, the point a click would pick is shown at the pointer with its session's name and its distance from the camera (the distance tells you when it would pick a point behind). The points placed are marked P1, P2…; click a mark to select it, drag it onto another point of the same session to move it, Delete removes it. The list is by pairs of sessions, with each pair's current gap.

A session's name is at first its raw folder's name. **⋯ → Rename…** on the session's row changes it (also with one session). The session's folder in the work folder and its name in the corrections are renamed, and the results stay usable. The raw folder keeps its name. The output point clouds have fixed names (`pointcloud_colorized.laz` and so on), whatever the session's or the work folder's name.

## Viewing the raw data

**Tools → View raw data…** (or **⋯ → View raw data…** on a session's row) opens the session's raw folder, read only.

- **Files**: whether the files used in processing are there, and their sizes.
- **Topics**: per topic of the bag, the type, count, start, end, mean rate and largest gap (gaps over 10 times the usual are yellow).
- **Messages**: the list of messages, and every field of the one selected (read with the definitions embedded in the bag). **Go to this time** moves the time bar to that message's time.
- **GNSS**: decodes `ROVER_PPKRAW.bin` and `BASE.BIN` and shows, on a time axis, the observations' time span and rate, gaps, satellites (per system) and signal strength, cycle slips, the base station's position, and the solution state (FIX, FLOAT, DGPS, single).

With a large bag (around 20 GB) on an external HDD, reading the index the first time can take over a minute.

## Device calibration

**Tools → Device calibration…** with several work folders of the same device, already processed, makes a calibration file with the cameras' mounting, fisheye and principal points solved again. Image pose export, image pose refinement and coloring then use it. Include both outdoor recordings and indoor ones up close (outdoors alone, the last few mm of the mounting are not determined).

## Screenshots (for checking)

```
s20-gui --smoke-screenshot out.png --work <work folder> [--raw <raw folder>] [--smoke-start <extract|lio|opt|cloud|export|colorize>]
        [--smoke-points n] [--smoke-view scans|cloud|uncolored] [--smoke-trajectory device|lio|opt] [--smoke-adaptive 0|1]
        [--smoke-zoom f] [--smoke-metric n] [--smoke-goto-worst] [--smoke-lang en|ja]
```

Saves the preview (after running the step given) and exits.
