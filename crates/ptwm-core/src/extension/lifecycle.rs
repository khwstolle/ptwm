//! Lifecycle declares the instance-allocation policy the host should use.

use serde::{Deserialize, Serialize};

use super::ExtensionError;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lifecycle {
    /// No instance, no state; called directly as a function.
    None,
    /// One instance per rayon worker thread; never accessed cross-thread.
    Thread,
    /// One shared instance per process; contribution handles synchronization.
    Process,
}

impl Lifecycle {
    pub fn as_u8(self) -> u8 {
        match self {
            Lifecycle::None => 0,
            Lifecycle::Thread => 1,
            Lifecycle::Process => 2,
        }
    }

    pub fn from_u8(v: u8) -> Result<Self, ExtensionError> {
        match v {
            0 => Ok(Lifecycle::None),
            1 => Ok(Lifecycle::Thread),
            2 => Ok(Lifecycle::Process),
            other => Err(ExtensionError::UnknownLifecycle(other)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u8_roundtrip() {
        for l in [Lifecycle::None, Lifecycle::Thread, Lifecycle::Process] {
            assert_eq!(Lifecycle::from_u8(l.as_u8()).unwrap(), l);
        }
    }

    #[test]
    fn unknown_lifecycle_rejected() {
        assert!(Lifecycle::from_u8(99).is_err());
    }

    #[test]
    fn toml_deserializes_lowercase_names() {
        // Deserialize enum from a TOML string value
        use toml::Value;
        let val = Value::String("thread".to_string());
        let l: Lifecycle = val.try_into().unwrap();
        assert_eq!(l, Lifecycle::Thread);
    }
}
