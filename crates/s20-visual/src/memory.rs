//! The memory this process holds, for the log.

/// The process's resident memory now and at its highest (bytes), or None where it is not known.
#[cfg(windows)]
pub fn process() -> Option<(u64, u64)> {
    use windows_sys::Win32::System::ProcessStatus::{
        GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c: PROCESS_MEMORY_COUNTERS = unsafe { std::mem::zeroed() };
    c.cb = std::mem::size_of::<PROCESS_MEMORY_COUNTERS>() as u32;
    // SAFETY: c is a PROCESS_MEMORY_COUNTERS with its size set, as the call requires; the
    // pseudo handle of the current process needs no closing
    (unsafe { GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) } != 0)
        .then_some((c.WorkingSetSize as u64, c.PeakWorkingSetSize as u64))
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
