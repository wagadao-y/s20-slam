# コマンドライン（`s20`）

GUI と同じ処理をコマンドラインで実行します。すべてのコマンドとオプションは `s20 help` で表示します。

## 一括処理

```
s20 run <生データフォルダー> <作業フォルダー> [--outdoor | --params file.json] [--no_rtk] [--ppk] [--no_color]
        [--las] [--images] [--no_refine] [--device_calibration file.yaml] [--name 名前] [--redo] [--until 工程]
s20 run <生データ1>,<生データ2>,... <作業フォルダー> --outdoor   # 複数セッション
```

工程は extract → lio → opt → cloud → export → refine → colorize の順です。

| オプション | |
|---|---|
| `--outdoor` | 屋外の設定（既定は屋内） |
| `--params` | GUI で保存した設定（`settings.json` の詳細パラメータ） |
| `--no_rtk` | 機体の RTK を使わない（既定では FIX があれば使う） |
| `--ppk` | PPK を計算して使う（[PPK](ppk.md)） |
| `--no_color` | 着色しない |
| `--las` | 点群を LAS で出力する（既定は LAZ） |
| `--images` | 着色のあと、姿勢合わせ後の写真を `output/images`（PointClouds Studio と同じ形式）に、COLMAP のモデルを横に出力する |
| `--no_refine` | 画像の姿勢合わせをしない |
| `--device_calibration` | 機体の校正（`calibrate-device` で作ったもの）を、メーカーの校正ファイル（出荷時の `calibration.yaml`）の代わりに使う |
| `--redo` | 抽出と LIO があってもやり直す |
| `--until` | その工程で止める |

複数セッションでは、抽出と LIO をセッションごとに `<作業フォルダー>/sessions/<生データのフォルダー名>/` で行い（足したセッションだけ実行します）、軌跡の最適化から後を全セッションで 1 回行います。

## 工程ごとの実行

```
s20 extract  <生データ> <data>                                  # bag の索引（点・画像の位置）と IMU・GNSS
s20 lio      <data> runs/lio.txt [--indoor] [--hold a:b]        # オドメトリ。--hold：方位を IMU に任せる区間（秒、繰り返し可）
s20 opt      <data> runs/lio.txt runs/opt.txt [--rtk] [--ppk runs/ppk.json] [--rtk_exclude a:b] [--loop a:b|start-end]
s20 cloud    <data> runs/opt.txt out/cloud.laz [--config params.json]
s20 export   <data> runs/opt.txt out --calib <生データ>/info/calibration.yaml
s20 refine-images --data <data> --imgpose runs/ImgPose.txt --calib calibration.yaml --cloud out/cloud.laz runs/ImgPose_refined.txt
s20 colorize out/cloud.laz out/cloud_col.laz --images <data>/images --imgpose runs/ImgPose.txt --calib calibration.yaml
```

- 時刻は、セッションの最初のスキャンからの秒です。負の値は終わりからの秒です。複数セッションでは `セッション名@秒` か `番号@秒` で指定します。
- `opt` に複数の LIO の軌跡をカンマ区切りで渡すと、複数セッションとして解きます（`data` に `sessions.txt` と連結した `imu.npy`／`gnss.npy` が必要です。`s20 run` が作ります）。GNSS がないときは、最初のセッションを基準に、セッションをまたぐループか `--match [session:]x,y,z=[session:]x,y,z`（同じ特徴の座標、セッションの組ごとに 2 組以上）で残りを置きます。
- `cloud` の `--config` は点群の設定の JSON です。ない項目は既定値になります。
- `colorize` の画像は、`<data>/images` にファイルがあればそれを、なければ bag の中の JPEG を読みます。

## そのほか

```
s20 ppk <生データ> runs/ppk.json [オプション]                  # PPK 測位（PPK のページを参照）
s20 calibrate-device <作業フォルダー>... --out calibration.yaml  # 処理済みの作業フォルダーから機体の校正を作る（GUI の「機体の校正」）
s20 export-colmap <着色済み点群> <出力> --data <data> --imgpose ImgPose.txt --calib calibration.yaml   # 3DGS 用の COLMAP モデル
s20 images   <data> <フォルダー>                                # カメラ画像（JPEG）を bag から書き出す
s20 convert  in.las out.laz                                     # LAS と LAZ の変換（拡張子で判断）
s20 potree   <in.las|laz> [--no_rgb] [--gps_time]               # Potree 2.0 の 8 分木を作る
s20 eval-traj <推定.txt> <基準.txt>                             # 軌跡の評価（ATE、1／5／30 秒の相対誤差、跳び）
s20 cut-raw <生データ> <出力> [--from 秒] --to 秒               # 撮影の一部を、短い生データフォルダーとして切り出す（テスト用）
s20 licenses                                                    # ライセンスの表示
```

`cut-raw` の時刻は bag の最初のスキャンからの秒です。切り出すのは `all_*.bag` だけで、ほかのファイル（`info/`、`frame_pose.txt`、`BASE.BIN` など）はそのままコピーします。LIO は最初の 2 秒が静止していることを前提に初期化するので、`--from` は機体が動き出す数秒前にしてください。本体の軌跡（`frame_pose.txt`）は bag より早く始まることがあり、その時刻は目安になりません。抽出した IMU で動き出しを確かめてください。

点群のファイルは拡張子で LAS と LAZ を切り替えます（`.laz` なら圧縮）。読み込みはどちらも受け付けます。

## COLMAP 形式の出力（3D Gaussian Splatting 用）

`export-colmap`（または `s20 run --images`）は、撮影を COLMAP のモデルとして書き出します。外部の 3DGS ツールに渡すためのものです。

| 出力 | 中身 |
|---|---|
| `images/left/`、`images/right/` | 記録された JPEG（再圧縮しません） |
| `sparse/0/cameras.txt` | 左右のカメラ。S20 の魚眼多項式に当てはめた `OPENCV_FISHEYE`（Kannala-Brandt） |
| `sparse/0/images.txt` | 画像の姿勢 |
| `sparse/0/points3D.txt` | 着色済み点群を間引いた初期点（既定 50 万点） |
| `masks/` | 光軸から `--max_angle`（85°）より外の画面の隅（校正を当てはめていない範囲） |
| `gauge.txt` | Z 軸が上、単位はメートル |
| `export.json` | 当てはめの誤差、点の数 |

Kannala-Brandt は S20 の多項式と完全には一致しません。当てはめの誤差は RMS で 0.1〜0.2 px 程度です。
