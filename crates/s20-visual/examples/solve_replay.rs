//! The tracks' adjustment alone, on the inputs `refine-images` wrote with `S20_DUMP_TRACKS` set:
//! for trying changes to `bundle::solve` without the features and the matching before it.
//!
//! cargo run --release -p s20-visual --example solve_replay -- <inputs> [<out poses.txt>]
//!
//! The output (optional): per image, the adjusted pose (x y z, then the rotation's nine entries by
//! row), to compare runs.

use anyhow::{Context, Result};
use s20_visual::bundle::{self, SolveInput};
use std::io::Write;
use std::time::Instant;

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let input = args
        .get(1)
        .context("usage: solve_replay <inputs> [<out poses.txt>]")?;
    let started = Instant::now();
    let p = SolveInput::load(std::path::Path::new(input))?;
    eprintln!(
        "read {} images, {} frames, {} tracks ({:.1} s{})",
        p.poses.len(),
        p.frames,
        p.tracks.len(),
        started.elapsed().as_secs_f64(),
        s20_visual::memory::note()
    );
    let started = Instant::now();
    let log = |m: &str| eprintln!("[{:7.1}s] {m}", started.elapsed().as_secs_f64());
    let sol = bundle::solve(
        &p.poses,
        &p.frame,
        p.frames,
        &p.chain,
        &p.tracks,
        &bundle::Settings::default(),
        &log,
    );
    eprintln!(
        "{} iterations, cost {:.6e} -> {:.6e} ({:.1} s{})",
        sol.iterations,
        sol.cost[0],
        sol.cost[1],
        started.elapsed().as_secs_f64(),
        s20_visual::memory::note()
    );
    if let Some(out) = args.get(2) {
        let mut w = std::io::BufWriter::new(std::fs::File::create(out)?);
        for (i, start) in p.poses.iter().enumerate() {
            let q = sol.pose(p.frame[i], start);
            let r = q.r;
            writeln!(
                w,
                "{} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12} {:.12}",
                i,
                q.p.x,
                q.p.y,
                q.p.z,
                r[(0, 0)],
                r[(0, 1)],
                r[(0, 2)],
                r[(1, 0)],
                r[(1, 1)],
                r[(1, 2)],
                r[(2, 0)],
                r[(2, 1)],
                r[(2, 2)]
            )?;
        }
    }
    Ok(())
}
