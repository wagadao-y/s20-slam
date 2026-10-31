//! Times the user gives (loops, excluded GNSS intervals): seconds on a session, which do not depend
//! on how the sessions' clocks are shifted onto one time axis (see `s20_io::dataset::build_set`).

use anyhow::{Context, Result, bail};

/// A time on a session: seconds from its first scan, or back from its end when negative. The
/// session is named as in the work folder (`sessions/<name>`), or by its number; empty is the first.
/// Written as a bare number for the first session (as in one-session work folders).
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(from = "AtRepr", into = "AtRepr")]
pub struct At {
    pub session: String,
    pub t: f64,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
enum AtRepr {
    T(f64),
    At { session: String, t: f64 },
}

impl From<AtRepr> for At {
    fn from(r: AtRepr) -> Self {
        match r {
            AtRepr::T(t) => At::new(t),
            AtRepr::At { session, t } => At { session, t },
        }
    }
}

impl From<At> for AtRepr {
    fn from(a: At) -> Self {
        if a.session.is_empty() {
            AtRepr::T(a.t)
        } else {
            AtRepr::At {
                session: a.session,
                t: a.t,
            }
        }
    }
}

impl At {
    /// A time on the first session.
    pub fn new(t: f64) -> Self {
        At {
            session: String::new(),
            t,
        }
    }

    pub fn on(session: &str, t: f64) -> Self {
        At {
            session: session.to_string(),
            t,
        }
    }

    /// `t` or `<session>@t` (as on the command line).
    pub fn parse(s: &str) -> Result<Self> {
        let (session, t) = s.rsplit_once('@').unwrap_or(("", s));
        Ok(At {
            session: session.to_string(),
            t: t.trim()
                .parse()
                .with_context(|| format!("not a time: {s:?}"))?,
        })
    }

    /// `a:b`, each as for [`At::parse`].
    pub fn parse_pair(s: &str) -> Result<(At, At)> {
        let (a, b) = s.split_once(':').context("a:b")?;
        Ok((At::parse(a)?, At::parse(b)?))
    }
}

/// A point picked on a session: in the body frame of the session's pose at time `t` (seconds as
/// for [`At`]), or, without a time, in the session's world as its odometry gives it.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct PointAt {
    #[serde(default)]
    pub session: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t: Option<f64>,
    pub p: [f64; 3],
}

impl PointAt {
    /// `x,y,z` or `<session>:x,y,z` (the session's world), as on the command line.
    pub fn parse(s: &str) -> Result<Self> {
        let (session, xyz) = s.rsplit_once(':').unwrap_or(("", s));
        let v: Vec<f64> = xyz
            .split(',')
            .map(|x| x.trim().parse::<f64>())
            .collect::<Result<_, _>>()
            .with_context(|| format!("not a point: {s:?}"))?;
        let p: [f64; 3] = v
            .try_into()
            .map_err(|_| anyhow::anyhow!("a point has three coordinates: {s:?}"))?;
        Ok(PointAt {
            session: session.to_string(),
            t: None,
            p,
        })
    }

    /// `a=b`, each as for [`PointAt::parse`].
    pub fn parse_pair(s: &str) -> Result<(PointAt, PointAt)> {
        let (a, b) = s.split_once('=').context("a=b")?;
        Ok((PointAt::parse(a)?, PointAt::parse(b)?))
    }
}

/// The sessions of an optimisation, to turn [`At`] into times on the common axis.
pub struct Clock {
    pub names: Vec<String>,
    /// per session: start of its first scan, end of its trajectory (s, shifted)
    pub range: Vec<(f64, f64)>,
}

impl Clock {
    pub fn session(&self, name: &str) -> Result<usize> {
        if name.is_empty() {
            return Ok(0);
        }
        if let Some(k) = self.names.iter().position(|n| n == name) {
            return Ok(k);
        }
        match name.parse::<usize>() {
            Ok(k) if k < self.names.len() => Ok(k),
            _ => bail!("no session {name:?} (sessions: {})", self.names.join(", ")),
        }
    }

    pub fn abs(&self, a: &At) -> Result<f64> {
        let (s, e) = self.range[self.session(&a.session)?];
        Ok(if a.t < 0.0 { e + a.t } else { s + a.t })
    }

    /// `12.3 s`, with the session's name when there are several.
    pub fn label(&self, a: &At) -> String {
        match self.session(&a.session) {
            Ok(k) if self.names.len() > 1 => format!("{} {:.1} s", self.names[k], a.t),
            _ => format!("{:.1} s", a.t),
        }
    }

    /// The label of time `t` on session `k`.
    pub fn label_at(&self, k: usize, t: f64) -> String {
        self.label(&At::on(&k.to_string(), t - self.range[k].0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_session_times_stay_bare_numbers() {
        let e = vec![
            (At::new(0.5), At::new(-0.5)),
            (At::on("b", 3.0), At::new(7.0)),
        ];
        let s = serde_json::to_string(&e).unwrap();
        assert_eq!(s, r#"[[0.5,-0.5],[{"session":"b","t":3.0},7.0]]"#);
        let back: Vec<(At, At)> = serde_json::from_str(&s).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn times_resolve_on_their_session() {
        let c = Clock {
            names: vec!["a".into(), "b".into()],
            range: vec![(100.0, 200.0), (300.0, 400.0)],
        };
        assert_eq!(c.abs(&At::new(5.0)).unwrap(), 105.0);
        assert_eq!(c.abs(&At::on("b", 5.0)).unwrap(), 305.0);
        assert_eq!(c.abs(&At::on("1", -1.0)).unwrap(), 399.0);
        assert!(c.abs(&At::on("c", 1.0)).is_err());
        assert_eq!(At::parse("b@-2.5").unwrap(), At::on("b", -2.5));
        assert_eq!(
            At::parse_pair("1:b@2").unwrap(),
            (At::new(1.0), At::on("b", 2.0))
        );
        assert_eq!(c.label_at(1, 312.0), "b 12.0 s");
        let (a, b) = PointAt::parse_pair("1.5,-2,3=room 2:4,5,6").unwrap();
        assert_eq!((a.session.as_str(), a.p), ("", [1.5, -2.0, 3.0]));
        assert_eq!(
            (b.session.as_str(), b.p, b.t),
            ("room 2", [4.0, 5.0, 6.0], None)
        );
        assert!(PointAt::parse("1,2").is_err());
    }
}
