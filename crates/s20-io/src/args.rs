//! Tiny command line parser: positional arguments, `--flag`, `--key value`.
use anyhow::{Context, Result, bail};
use std::str::FromStr;

pub struct Args {
    items: Vec<Option<String>>,
}

impl Args {
    pub fn from_env() -> Self {
        Args {
            items: std::env::args().skip(1).map(Some).collect(),
        }
    }
    pub fn from_vec(v: Vec<String>) -> Self {
        Args {
            items: v.into_iter().map(Some).collect(),
        }
    }
    /// Next unused positional argument (not starting with "--").
    pub fn positional(&mut self) -> Option<String> {
        let mut i = 0;
        while i < self.items.len() {
            match &self.items[i] {
                Some(s) if s.starts_with("--") => {
                    // skip "--key value" pairs whose value is still unconsumed: values are consumed by get()
                    i += 1;
                }
                Some(_) => return self.items[i].take(),
                None => i += 1,
            }
        }
        None
    }
    pub fn positional_req(&mut self, what: &str) -> Result<String> {
        self.positional()
            .with_context(|| format!("missing argument: {what}"))
    }
    pub fn flag(&mut self, name: &str) -> bool {
        for it in self.items.iter_mut() {
            if it.as_deref() == Some(name) {
                *it = None;
                return true;
            }
        }
        false
    }
    pub fn get<T: FromStr>(&mut self, name: &str, default: T) -> Result<T>
    where
        T::Err: std::fmt::Display,
    {
        for i in 0..self.items.len() {
            if self.items[i].as_deref() == Some(name) {
                let v = self
                    .items
                    .get_mut(i + 1)
                    .and_then(Option::take)
                    .with_context(|| format!("{name} needs a value"))?;
                self.items[i] = None;
                return v
                    .parse::<T>()
                    .map_err(|e| anyhow::anyhow!("{name} {v}: {e}"));
            }
        }
        Ok(default)
    }
    pub fn get_str(&mut self, name: &str) -> Result<Option<String>> {
        for i in 0..self.items.len() {
            if self.items[i].as_deref() == Some(name) {
                let v = self
                    .items
                    .get_mut(i + 1)
                    .and_then(Option::take)
                    .with_context(|| format!("{name} needs a value"))?;
                self.items[i] = None;
                return Ok(Some(v));
            }
        }
        Ok(None)
    }
    pub fn finish(&self) -> Result<()> {
        let left: Vec<&str> = self.items.iter().flatten().map(String::as_str).collect();
        if !left.is_empty() {
            bail!("unknown arguments: {}", left.join(" "));
        }
        Ok(())
    }
}
