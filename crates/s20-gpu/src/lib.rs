//! The process's GPU (wgpu compute, no window) for the heavy loops, shared by the crates that use
//! it. Without a usable GPU the callers keep their CPU paths. The GPU computes in single
//! precision.
use anyhow::{Context, Result};
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

/// The process-wide GPU, created on first use (None when there is no usable one).
pub fn shared() -> Option<&'static Gpu> {
    static GPU: OnceLock<Option<Gpu>> = OnceLock::new();
    GPU.get_or_init(|| Gpu::new().ok()).as_ref()
}

/// A compute device and its queue.
pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub name: String,
    pub limits: wgpu::Limits,
}

impl Gpu {
    /// The high-performance adapter with its own limits (large buffers), or an error. The backend
    /// is fixed (DX12 on Windows), so that the same adapter and shader compiler are used.
    pub fn new() -> Result<Self> {
        let mut desc = wgpu::InstanceDescriptor::new_without_display_handle();
        desc.backends = if cfg!(windows) {
            wgpu::Backends::DX12
        } else {
            wgpu::Backends::PRIMARY
        };
        let instance = wgpu::Instance::new(desc.with_env());
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))
        .context("no GPU adapter")?;
        let limits = adapter.limits();
        let info = adapter.get_info();
        let name = format!("{}, {:?}", info.name, info.backend);
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("s20 compute"),
            required_limits: limits.clone(),
            ..Default::default()
        }))
        .context("GPU device")?;
        // Errors outside a scope (see `checked`) and a lost device are noted, not a panic (wgpu's
        // default): the GPU is not used again.
        device.on_uncaptured_error(std::sync::Arc::new(|e: wgpu::Error| {
            eprintln!("GPU error: {e}");
            BROKEN.store(true, Ordering::Relaxed);
        }));
        device.set_device_lost_callback(|reason, msg| {
            eprintln!("GPU lost: {reason:?} {msg}");
            BROKEN.store(true, Ordering::Relaxed);
        });
        Ok(Gpu {
            device,
            queue,
            name,
            limits,
        })
    }

    /// `f`'s GPU work with its validation, out-of-memory and internal errors as an error (the
    /// caller goes on on the CPU), not a panic.
    pub fn checked<T>(&self, f: impl FnOnce() -> Result<T>) -> Result<T> {
        anyhow::ensure!(!BROKEN.load(Ordering::Relaxed), "the GPU failed before");
        let scopes = [
            wgpu::ErrorFilter::OutOfMemory,
            wgpu::ErrorFilter::Validation,
            wgpu::ErrorFilter::Internal,
        ]
        .map(|e| self.device.push_error_scope(e));
        let r = f();
        let mut first = None;
        for s in scopes.into_iter().rev() {
            if let Some(e) = pollster::block_on(s.pop()) {
                first.get_or_insert(e);
            }
        }
        if let Some(e) = first {
            anyhow::bail!("GPU: {e}");
        }
        anyhow::ensure!(!BROKEN.load(Ordering::Relaxed), "the GPU failed");
        r
    }
}

/// Set once the GPU failed outside a checked scope or was lost.
static BROKEN: AtomicBool = AtomicBool::new(false);
