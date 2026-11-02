# s20-slam

[日本語](README_jp.md)

A post-processing app for the **SHARE SLAM S20**. From the S20's raw recording folder it computes the trajectory and makes a coloured point cloud, without the vendor's software (PointClouds Studio). Windows, GUI and command line.

![s20-slam](docs/images/screenshot.png)

> An unofficial tool. It is not made, endorsed or supported by SHARE.

## What it does

- **Trajectory**: LiDAR–inertial odometry, then a global optimisation with the IMU, the planes of the cloud and, outdoors, the device's RTK or PPK computed from `BASE.BIN` / `ROVER_PPKRAW.bin` (the positioning engine is built in).
- **Point cloud**: passers-by, cars, mirror images and haze removed, surfaces sharpened, then coloured from the two fisheye cameras. Written as LAZ / LAS with a Potree 2.0 octree beside it.
- **Checking and fixing**: preview the scans and the output cloud, colour the trajectory by quality, and fix problem sections yourself (loops, sections without GNSS, sections with the heading held to the IMU).
- **Several sessions** of the same place joined into one trajectory and cloud.
- **Images for 3D Gaussian Splatting**: the photos with refined poses as a COLMAP model.

## Download

Get the latest zip from [Releases](../../releases/latest), unzip it anywhere and run `s20-gui.exe`. No installer is needed.

The executables are not code-signed, so Windows may show "Windows protected your PC" the first time. Choose **More info → Run anyway**. They are built by GitHub Actions from the public source of this repository. To check the download, compare `Get-FileHash <zip>` in PowerShell with the `.sha256` file attached to the release.

| File | |
|---|---|
| `s20-gui.exe` | The app |
| `s20.exe` | Command line (the same processing, for batches) |

### Requirements

- Windows 10 / 11, 64-bit
- A GPU with DirectX 12 (for the preview, and for parts of the processing)
- Memory: 16 GB or more is recommended. A long outdoor recording uses about 8 GB while the cloud is made.
- Disk: besides the copy of the raw data, room for the results. For example, a 16-minute outdoor recording (11 GB of raw data) takes 3.5 GB up to the colouring, 8 GB with the image export. Nearly all of it is the clouds and the photos; the intermediate files are under 100 MB.

Processing time, for example: the same 16-minute recording takes about 24 minutes up to the colouring (with an RTX 3060): odometry 2 min, trajectory optimisation 1.5 min, point cloud 6 min, image pose refinement 7.5 min, colouring 6.5 min.

## Quick start

1. Copy the S20's recording folder (the one holding `all_*.bag`) to your PC as a whole. The app reads the points and images from the bag instead of copying them into the work folder, so leave the copy in place while you work on it.
2. Start `s20-gui.exe`, choose **New work folder…**, then pick the raw folder and an empty folder for the results.
3. In **Processing settings** on the left, choose the GNSS (none / the device's RTK / PPK). **Other settings…** has indoor / outdoor and the output format.
4. Press **Run from unfinished or unapplied steps**. The progress and the log are shown, and you can cancel at any time.
5. When it finishes, the results are in `<work folder>/output/`.

The language is chosen from the menu bar (English / 日本語). Until then it follows Windows' display language.

### Output

| File | |
|---|---|
| `<work folder name>_colorized.laz` | The coloured point cloud (with normals) |
| `<work folder name>_uncolorized.laz` | The cloud before colouring |
| `*_converted/` | Potree 2.0 octrees of the clouds and the trajectory |
| `trajectory.txt`, `trajectory.las` | The trajectory (with GPS times) |
| `coordinate_frame.json` | When GNSS was used: the origin of the local coordinates |
| `images/`, `sparse/0/`, `masks/` | Optional: the photos with their poses, in PointClouds Studio's layout and as a COLMAP model |

With GNSS the coordinates are local, in metres around an origin near the site: X = east, Y = north, Z = up. Z is the height above the origin, not above sea level. The origin's latitude, longitude and ellipsoidal height are in `coordinate_frame.json`; keep it with the cloud. Without GNSS each recording has its own coordinates: Z is up, X and Y have no fixed direction.

### Better colours: the device calibration

Once you have processed a few recordings of your device, **Tools → Device calibration** solves the cameras' mounting, fisheye and principal points again. The colours get fewer doubled edges and less blur than with the maker's factory calibration file (`calibration.yaml`).

- Choose some processed work folders. Include both outdoor recordings and indoor ones that see things up close (the last few mm of the mounting are only determined close up). Each recording takes 2-4 minutes.
- Once saved, it is used for every recording of the same device. Recordings already processed use it when redone from **Trajectory and image pose export**.

## If the result is wrong

- **The trajectory drifts or the cloud is doubled**: open **Trajectory** in the right panel, colour it by a quality metric and look at the listed sections. Add a **loop** where you passed the same place twice (or **Start = end**), then **Apply the corrections**.
- **Bad GNSS in a section**: add it to **Sections without GNSS**.
- **The heading drifts on a round hill or a featureless slope**: add it to **Sections with heading held to the IMU**.
- **To look at the raw data**: **Tools → View raw data…** shows the bag's topics and messages and the GNSS observations.

## Command line

```
s20 run <raw folder> <work folder> [--outdoor] [--ppk] [--no_rtk] [--no_color] [--las] [--images]
s20 run <raw 1>,<raw 2>,... <work folder> --outdoor      # several sessions of one place
s20 help                                                 # every command and option
```

## Documentation

The detailed documentation is in Japanese:

- [GUI guide](docs/gui.md)
- [Command line](docs/cli.md)
- [How it works](docs/processing.md)
- [PPK](docs/ppk.md)
- [Validation](docs/validation.md)
- [Known issues](docs/known-issues.md)

## Building from source

Needs Rust (rustup installs the version in `rust-toolchain.toml`) and the MSVC C/C++ build tools (for RTKLIB).

```
cargo build --release
```

This builds `target/release/s20-gui.exe` and `target/release/s20.exe`.

## License

Apache License 2.0 ([LICENSE](LICENSE)). The third-party software included or ported (RTKLIB, small_gicp, Potree, PotreeConverter) is listed in [NOTICE](NOTICE), and the licenses of the crates and fonts in the executables are in [THIRD-PARTY-LICENSES.txt](THIRD-PARTY-LICENSES.txt). The app shows both under **Help → Licenses**, and so does `s20 licenses`.
