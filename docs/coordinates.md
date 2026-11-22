# Placing the point cloud in a coordinate system

With GNSS (RTK or PPK) the results are in local coordinates around an origin where the GNSS solution first met the quality conditions (usually near where the recording started): X = east, Y = north, Z = up, in meters. The origin's latitude, longitude and ellipsoidal height (WGS84) are in `coordinate_frame.json` in the output folder. Z is the height above the origin's tangent plane, not above sea level.

To put the results in UTM, a national grid or another coordinate reference system, transform them with [PDAL](https://pdal.org/). The output folder has a pipeline for it, `pdal_to_crs.json`, with the origin filled in (only when GNSS was used; written by **Trajectory and image pose export**).

## Transforming

See [PDAL's site](https://pdal.org/) to install it. Run it in the output folder (PDAL looks for the files in the current folder):

```
cd <work folder>/output
pdal pipeline pdal_to_crs.json
```

This transforms, for example, `pointcloud_colorized.laz` into `pointcloud_colorized_crs.laz`, with the coordinate reference system written in the LAS file.

- **Target**: by default the UTM zone of the origin (WGS84, EPSG:326xx / 327xx; UPS near the poles). For another system, give its EPSG code: change `out_srs` of `filters.reprojection` in the JSON, or give it on the command line. Japan's plane rectangular systems are EPSG:6669 (zone I) to 6687 (zone XIX).

  ```
  pdal pipeline pdal_to_crs.json --filters.reprojection.out_srs=EPSG:6678
  ```

  A target changed in the JSON is kept when the app writes the JSON again (when **Trajectory and image pose export**, **Point cloud** or **Colorization** is redone). The coordinates are written in steps of 0.1 mm, so choose a projected system in meters. For latitude and longitude (such as EPSG:4326), also give the steps, for example `--writers.las.scale_x=1e-9 --writers.las.scale_y=1e-9` (otherwise they are 0.0001 degrees, about 10 m).

- **Other files**: give the input and output names.

  ```
  pdal pipeline pdal_to_crs.json --readers.las.filename=trajectory.las --writers.las.filename=trajectory_crs.las
  ```

The input is the newest point cloud in the output folder made in the current coordinates (else the trajectory, `trajectory.las`). It is written again when **Point cloud** or **Colorization** is redone.

Extra dimensions such as the normals (and the trajectory's heading) are left out by default: point clouds placed in a coordinate system mostly go to GIS, and LAS 1.4 files with extra dimensions (LAS 1.4 Extra Bytes) have failed to open in QGIS 3.44 in some setups ([QGIS issue #63906](https://github.com/qgis/QGIS/issues/63906)). To keep them, add `--writers.las.extra_dims=all`.

## Notes

- **Transformations that need grids**: a target with heights above sea level, and in some countries even a horizontal transformation, needs a PROJ grid (such as a geoid model). Without the grid, PROJ **falls back to a coarser transformation without an error** (for heights: ellipsoidal heights labeled as heights above sea level). With the environment variable `PROJ_ONLY_BEST_DEFAULT=YES` it stops with an error instead (PROJ 9.2 or later). Do not use an output file left by such a stop. Install grids into PROJ's data folder with `projsync --file <name> --system-directory` (without `--system-directory` they go to a user folder, which PDAL may not search).

  ```
  $env:PROJ_ONLY_BEST_DEFAULT = "YES"       # PowerShell (bash: export PROJ_ONLY_BEST_DEFAULT=YES)
  ```

- **Height**: Z stays ellipsoidal. For heights above sea level, choose a target that includes them, such as a national grid combined with the country's height system (in Japan, a plane rectangular system with JGD2024 heights, EPSG:11328 for zone X, with the geoid grid `jp_gsi_jpgeo2024.tif`). After transforming, check that Z went down by the geoid height.
- **The position is only as good as the base station's coordinates**: an RTK or PPK position is the base station's coordinates plus the vector from it. An error in those coordinates, or the datum and epoch they are given in, can shift the whole point cloud by meters. To match maps or earlier surveys, use a base station whose coordinates you can trust.
- **Datum transformation**: PROJ chooses how to go from WGS84 to the target's datum. Where the ground moves a lot, a time-dependent transformation may be needed.
- **Orientation**: only the positions are transformed. Kept normals and the trajectory's heading (yaw) are not rotated, so they are off by the angle between true north and the target's grid north (usually under 1°).
