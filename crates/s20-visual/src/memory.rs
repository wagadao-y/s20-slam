//! The memory this process holds, for the log.

/// The memory the process holds now and at its highest (bytes), or None where it is not known:
/// on Windows what it allocated (its private commit), without the pages of the files it maps (a
/// recording's images read through a map counted in its working set).
#[cfg(windows)]
pub fn process() -> Option<(u64, u64)> {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
    c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
    // SAFETY: c is a PROCESS_MEMORY_COUNTERS_EX with its size set, which the call takes in place
    // of a PROCESS_MEMORY_COUNTERS; the pseudo handle of the current process needs no closing
    (unsafe {
        GetProcessMemoryInfo(
            GetCurrentProcess(),
            (&raw mut c).cast::<PROCESS_MEMORY_COUNTERS>(),
            c.cb,
        )
    } != 0)
        .then_some((c.PrivateUsage as u64, c.PeakPagefileUsage as u64))
}

/// The process's resident memory now and at its highest (bytes), or None where it is not known.
#[cfg(not(windows))]
pub fn process() -> Option<(u64, u64)> {
    let s = std::fs::read_to_string("/proc/self/status").ok()?;
    let kb = |key: &str| -> Option<u64> {
        s.lines()
            .find_map(|l| l.strip_prefix(key))?
            .trim()
            .trim_end_matches("kB")
            .trim()
            .parse()
            .ok()
    };
    Some((kb("VmRSS:")? * 1024, kb("VmHWM:")? * 1024))
}

/// ", memory 12.3 GB (highest 20.1 GB)" for a log line, or nothing where it is not known.
pub fn note() -> String {
    match process() {
        Some((now, peak)) => format!(
            ", memory {:.1} GB (highest {:.1} GB)",
            now as f64 / 1e9,
            peak as f64 / 1e9
        ),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    /// (Known on Windows and Linux; elsewhere the log goes without it.)
    #[cfg(any(windows, target_os = "linux"))]
    #[test]
    fn the_process_holds_some_memory() {
        let (now, peak) = super::process().unwrap();
        assert!(now > 0 && peak >= now);
        assert!(super::note().contains("GB"));
    }
}
