fn main() {
    let root = "vendor/rtklib/src";
    let mut b = cc::Build::new();
    b.include(root)
        .warnings(false)
        .define("ENAGLO", None)
        .define("ENAGAL", None)
        .define("ENAQZS", None)
        .define("ENACMP", None)
        .define("NFREQ", "3")
        .define("NEXOBS", "3")
        .define("S20_STATIC", None)
        .define("_CRT_SECURE_NO_WARNINGS", None);
    for n in [
        "rtkcmn",
        "trace",
        "rinex",
        "sbas",
        "preceph",
        "rcvraw",
        "rtcm",
        "rtcm2",
        "rtcm3",
        "rtcm3e",
        "pntpos",
        "ephemeris",
        "ionex",
        "rtkpos",
        "lambda",
        "ppp",
        "ppp_ar",
        "solution",
        "geoid",
        "tides",
    ] {
        b.file(format!("{root}/{n}.c"));
    }
    for n in [
        "novatel",
        "ublox",
        "swiftnav",
        "crescent",
        "skytraq",
        "javad",
        "nvs",
        "binex",
        "rt17",
        "septentrio",
        "unicore",
    ] {
        b.file(format!("{root}/rcv/{n}.c"));
    }
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        b.define("WIN32", None);
    }
    b.file("native/ppk.c").compile("s20_ppk");
    if std::env::var("CARGO_CFG_TARGET_OS").unwrap() == "windows" {
        println!("cargo:rustc-link-lib=winmm");
    }
    println!("cargo:rerun-if-changed=native");
    println!("cargo:rerun-if-changed=vendor/rtklib");
}
