//! Decoders for the S20 message types (fixed layouts, little endian, ROS1 serialization).

use anyhow::{Context, Result, bail};

pub struct Cursor<'a> {
    b: &'a [u8],
    pub p: usize,
}

impl<'a> Cursor<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Cursor { b, p: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self
            .b
            .get(self.p..self.p + n)
            .context("message truncated")?;
        self.p += n;
        Ok(s)
    }
    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn i8(&mut self) -> Result<i8> {
        Ok(self.take(1)?[0] as i8)
    }
    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into()?))
    }
    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into()?))
    }
    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into()?))
    }
    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into()?))
    }
    pub fn f64(&mut self) -> Result<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into()?))
    }
    pub fn bytes(&mut self) -> Result<&'a [u8]> {
        let n = self.u32()? as usize;
        self.take(n)
    }
    pub fn rest(&self) -> &'a [u8] {
        &self.b[self.p..]
    }
}

/// std_msgs/Header -> stamp in seconds.
pub fn header(c: &mut Cursor) -> Result<f64> {
    let _seq = c.u32()?;
    let s = c.u32()? as f64;
    let n = c.u32()? as f64;
    let _frame = c.bytes()?;
    Ok(s + n * 1e-9)
}

/// sensor_msgs/Imu -> [stamp, wx, wy, wz, ax, ay, az] (S20: acceleration in g).
pub fn imu(b: &[u8]) -> Result<[f64; 7]> {
    let mut c = Cursor::new(b);
    let t = header(&mut c)?;
    for _ in 0..4 + 9 {
        c.f64()?; // orientation + covariance
    }
    let w = [c.f64()?, c.f64()?, c.f64()?];
    for _ in 0..9 {
        c.f64()?;
    }
    let a = [c.f64()?, c.f64()?, c.f64()?];
    Ok([t, w[0], w[1], w[2], a[0], a[1], a[2]])
}

/// app_osd/LidarBaseInfoMsg (info bag `/livox/base_info`) -> LiDAR core temperature in degrees C
/// (`core_temp` is in 0.01 degrees).
pub fn lidar_core_temp(b: &[u8]) -> Result<f64> {
    let mut c = Cursor::new(b);
    header(&mut c)?;
    let _state = c.u16()?;
    Ok(c.u16()? as i16 as f64 * 0.01)
}

/// Size of one livox_ros_driver2/CustomPoint (u32 offset_time, 3*f32, reflectivity, tag, line).
pub const POINT_SIZE: usize = 19;

pub struct LidarScan<'a> {
    pub stamp: f64,
    pub timebase_ns: u64,
    /// packed CustomPoint records (POINT_SIZE bytes each)
    pub points: &'a [u8],
}

/// livox_ros_driver2/CustomMsg
pub fn lidar(b: &[u8]) -> Result<LidarScan<'_>> {
    let mut c = Cursor::new(b);
    let stamp = header(&mut c)?;
    let timebase_ns = c.u64()?;
    let _point_num = c.u32()?;
    let _lidar_id = c.u8()?;
    for _ in 0..3 {
        c.u8()?;
    }
    let n = c.u32()? as usize;
    let rest = c.rest();
    if rest.len() < n * POINT_SIZE {
        bail!("lidar message truncated");
    }
    Ok(LidarScan {
        stamp,
        timebase_ns,
        points: &rest[..n * POINT_SIZE],
    })
}

/// One decoded Livox point.
#[derive(Debug, Clone, Copy)]
pub struct LivoxPoint {
    pub offset_ns: u32,
    pub xyz: [f32; 3],
    pub reflectivity: u8,
    pub tag: u8,
    pub line: u8,
}

pub fn livox_point(rec: &[u8]) -> LivoxPoint {
    let f = |i: usize| f32::from_le_bytes(rec[i..i + 4].try_into().unwrap());
    LivoxPoint {
        offset_ns: u32::from_le_bytes(rec[0..4].try_into().unwrap()),
        xyz: [f(4), f(8), f(12)],
        reflectivity: rec[16],
        tag: rec[17],
        line: rec[18],
    }
}

/// rtk_agent/PVTSLNMsg -> row in the gnss.npy layout:
/// [t, type, lat, lon, hgt, latstd, lonstd, hgtstd, undulation, psr_lat, psr_lon, psr_hgt, psr_solnsvs, diffage]
pub fn pvtsln(b: &[u8]) -> Result<[f64; 14]> {
    let mut c = Cursor::new(b);
    let t = header(&mut c)?;
    let best_type = c.u8()? as f64;
    let hgt = c.f32()? as f64;
    let lat = c.f64()?;
    let lon = c.f64()?;
    let hgtstd = c.f32()? as f64;
    let latstd = c.f32()? as f64;
    let lonstd = c.f32()? as f64;
    let diffage = c.f32()? as f64;
    let _psr_type = c.u8()?;
    let psr_hgt = c.f32()? as f64;
    let psr_lat = c.f64()?;
    let psr_lon = c.f64()?;
    let undulation = c.f32()? as f64;
    let _best_svs = c.u8()?;
    let _best_solnsvs = c.u8()?;
    let _psr_svs = c.u8()?;
    let psr_solnsvs = c.u8()? as f64;
    Ok([
        t,
        best_type,
        lat,
        lon,
        hgt,
        latstd,
        lonstd,
        hgtstd,
        undulation,
        psr_lat,
        psr_lon,
        psr_hgt,
        psr_solnsvs,
        diffage,
    ])
}

/// sensor_msgs/CompressedImage -> (stamp, jpeg bytes)
pub fn compressed_image(b: &[u8]) -> Result<(f64, &[u8])> {
    let mut c = Cursor::new(b);
    let t = header(&mut c)?;
    let _format = c.bytes()?;
    let data = c.bytes()?;
    Ok((t, data))
}
