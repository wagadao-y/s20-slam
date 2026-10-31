//! The memory available to a process.

/// The physical memory available now (bytes), or None where it is not known.
#[cfg(windows)]
pub fn available() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut s: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    s.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: s is a MEMORYSTATUSEX with its length set, as the call requires
    (unsafe { GlobalMemoryStatusEx(&mut s) } != 0).then_some(s.ullAvailPhys)
}

/// The physical memory available now (bytes), or None where it is not known.
#[cfg(not(windows))]
pub fn available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kb: u64 = s
        .lines()
        .find_map(|l| l.strip_prefix("MemAvailable:"))?
        .trim()
        .trim_end_matches("kB")
        .trim()
        .parse()
        .ok()?;
    Some(kb * 1024)
}

#[cfg(test)]
mod tests {
    #[test]
    fn some_memory_is_available() {
        assert!(super::available().is_some_and(|b| b > 0));
    }
}
