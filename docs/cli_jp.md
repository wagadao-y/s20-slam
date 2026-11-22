# コマンドライン（`s20`）

[English](cli.md)

GUI と同じ処理をコマンドラインで実行します。すべてのコマンドとオプションは `s20 help` で表示します。

## 一括処理

```
s20 run <生データフォルダー> <作業フォルダー> [--outdoor | --params file.json] [--no_rtk] [--ppk] [--no_color]
        [--las] [--images [--3dgs_poses] [--sky_sphere | --no_sky_sphere]] [--no_refine] [--device_calibration file.yaml] [--redo] [--until 工程]
s20 run <生データ1>,<生データ2>,... <作業フォルダー> --outdoor   # 複数セッション
```

工程は extract → lio → opt → cloud → export → refine → colorize の順です。

| オプション | |
|---|---|
| `--outdoor` | 屋外の設定（既定は屋内） |
| `--params` | GUI で保存した設定（`settings.json` の詳細パラメータ） |
| `--no_rtk` | 機体の RTK を使わない（既定では FIX が 100 件以上あれば使う） |
| `--ppk` | PPK を計算して使う（[PPK](ppk.md)） |
| `--no_color` | 着色しない（`--images` とは併用できない：画像の出力は着色のあとに行うため） |
| `--las` | 点群を LAS で出力する（既定は LAZ） |
| `--images` | 着色のあと、姿勢合わせ後の写真を `output/images`（PointClouds Studio と同じ形式）に、COLMAP のモデルを横に出力する。COLMAP のモデルの姿勢は着色と同じ姿勢（点群に合わせた姿勢） |
| `--3dgs_poses` | 実験的。`--images` で、画像の姿勢合わせで 3DGS 用の姿勢（写真どうしが一番合う姿勢）も求め、COLMAP のモデルに使い、`image_points.txt` も出力する。画像の姿勢合わせが長くなる。`--no_refine` とは併用できない |
| `--sky_sphere`、`--no_sky_sphere` | `--images` で、COLMAP のモデルの初期点に sky sphere（下を参照）を足す／足さない。既定は `--outdoor` のときに足す |
| `--no_refine` | 画像の姿勢合わせをしない |
| `--device_calibration` | 機体の校正（`calibrate-device` で作ったもの）を、メーカーの校正ファイル（出荷時の `calibration.yaml`）の代わりに使う |
| `--redo` | 抽出と LIO があってもやり直す（LIO の設定を変えたとき、`--outdoor` や `--params` を変えたときは、なくても LIO からやり直す） |
| `--until` | その工程で止める |

複数セッションでは、抽出と LIO をセッションごとに `<作業フォルダー>/sessions/<セッション名>/`（セッション名ははじめ生データのフォルダー名）で行い（足したセッションだけ実行します）、軌跡の最適化から後を全セッションで 1 回行います。

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
- `opt` は再訪（[処理の仕組み](processing.md)）を使います。`--no_revisits` で使いません。`--revisit_dist m`（組にする距離）、`--revisit_inliers f`（重なりの下限）、`--revisit_path_sd k`（道沿いの標準偏差の倍率）、`--robust sd` などで設定を変えられます。`s20 eval-revisits <data> <opt_knots.txt> [out.csv]` は、軌跡の同じ場所の組を位置合わせし直して、二重の幅を組ごとに CSV に書きます。
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
s20 version                                                     # バージョンの表示（--version でも）
```

`cut-raw` の時刻は bag の最初のスキャンからの秒です。切り出すのは `all_*.bag` だけで、ほかのファイル（`info/`、`frame_pose.txt`、`BASE.BIN` など）はそのままコピーします。LIO は最初の 2 秒が静止していることを前提に初期化するので、`--from` は機体が動き出す数秒前にしてください。本体の軌跡（`frame_pose.txt`）は bag より早く始まることがあり、その時刻は目安になりません。抽出した IMU で動き出しを確かめてください。

点群のファイルは拡張子で LAS と LAZ を切り替えます（`.laz` なら圧縮）。読み込みはどちらも受け付けます。

## COLMAP 形式の出力（3D Gaussian Splatting 用）

`export-colmap`（または `s20 run --images`）は、撮影を COLMAP のモデルとして書き出します。外部の 3DGS ツールに渡すためのものです。

| 出力 | 中身 |
|---|---|
| `images/left/`、`images/right/` | 記録された JPEG（再圧縮しません） |
| `sparse/0/cameras.txt` | 左右のカメラ。S20 の魚眼多項式に当てはめた `OPENCV_FISHEYE`（Kannala-Brandt） |
| `sparse/0/images.txt` | 画像の姿勢（着色と同じ姿勢。`s20 run --images --3dgs_poses` では 3DGS 用の姿勢 `ImgPose-3dgs.txt`） |
| `sparse/0/points3D.txt` | 着色済み点群を間引いた初期点（既定 50 万点）。sky sphere を足すときは、その後ろに続く（`export-colmap` は `--sky_sphere` で足す） |
| `masks/` | 光軸から `--max_angle`（85°）より外の画面の隅（校正を当てはめていない範囲） |
| `gauge.txt` | Z 軸が上、単位はメートル |
| `export.json` | 当てはめの誤差、点の数 |
| `image_points.txt` | `s20 run --images --3dgs_poses` で画像の姿勢合わせを行ったとき：写真の特徴点のトラックを、`ImgPose-3dgs.txt` の姿勢で三角測量した点（下を参照） |

Kannala-Brandt は S20 の多項式と完全には一致しません。当てはめの誤差は RMS で 0.1〜0.2 px 程度です。

sky sphere は、シーンを囲む大きな球の上に置いた点です。空や遠くの背景には点群の点がなく、3DGS の学習で手前の Gaussian が空を表そうとして、浮遊物（floater）になりやすいので、その初期値にします。球の中心はカメラ位置の平均、半径は点群の 99% が収まる距離の 2 倍（50 m 以上）で、水平より 10° 下までに約 1° おきに並べ、どの画像にも写らない点は除きます。色は灰色で、空の色は学習で決まります。屋内では役に立たないので、既定では屋外（`--outdoor`）のときだけ足します。

`image_points.txt` は、LiDAR が点を取れない所（機体のすぐ近くや、低い所の側面など）を写真の点で補うためのものです。`points3D.txt`（LiDAR の点）とは別のファイルで、1 行が 1 点です：位置（x y z）、写っている写真の枚数、最初と最後の写真の時間差（秒）、写真の光線どうしの最大の角度（度）、再投影誤差の中央値と最大（画素相当）、最も近い写真までの距離（m）、最寄りの点群の点までの距離（m）、点の出どころ（0：画像の姿勢合わせに使った対応、点群を手がかりに探したもの。1：点群を使わずにエピポーラ線に沿って探した対応）。点群は機体のすぐ近くの面などを取れないので、点群に予測を頼る 0 には、そうした面の点がほとんどありません。1 はそこも探します。点群からの距離が大きい点が、点群にない面の点です。角度の小さい点（基線の短い点）は光線の向きにばらつき、時間差の短い点は一度通っただけで写ったもの（人など）を含みます。写真 2 枚だけの点は誤対応を見分けられないので、使う側で写真の枚数・角度・誤差を見て選んでください。
