//! Closed-but-versioned taxonomy of contribution kinds.

use serde::{Deserialize, Serialize};

use super::ExtensionError;

/// The 13 contribution kinds. New kinds may be added in minor PTWM
/// releases; readers refuse to decode files referencing unknown kinds.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Kind {
    Transform,
    PlaneCodec,
    ChainBuilder,
    ChainExplorer,
    Classifier,
    Scorer,
    DeltaScheme,
    IntegrationAdapter,
    ContainerLayout,
    HardwareBackend,
    BenchmarkMetric,
    TrainingHook,
    RawBinary,
}

impl Kind {
    pub fn as_u16(self) -> u16 {
        match self {
            Kind::Transform => 0x0001,
            Kind::PlaneCodec => 0x0002,
            Kind::ChainBuilder => 0x0003,
            Kind::ChainExplorer => 0x0004,
            Kind::Classifier => 0x0005,
            Kind::Scorer => 0x0006,
            Kind::DeltaScheme => 0x0007,
            Kind::IntegrationAdapter => 0x0008,
            Kind::ContainerLayout => 0x0009,
            Kind::HardwareBackend => 0x000A,
            Kind::BenchmarkMetric => 0x000B,
            Kind::TrainingHook => 0x000C,
            Kind::RawBinary => 0x000D,
        }
    }

    pub fn from_u16(v: u16) -> Result<Self, ExtensionError> {
        match v {
            0x0001 => Ok(Kind::Transform),
            0x0002 => Ok(Kind::PlaneCodec),
            0x0003 => Ok(Kind::ChainBuilder),
            0x0004 => Ok(Kind::ChainExplorer),
            0x0005 => Ok(Kind::Classifier),
            0x0006 => Ok(Kind::Scorer),
            0x0007 => Ok(Kind::DeltaScheme),
            0x0008 => Ok(Kind::IntegrationAdapter),
            0x0009 => Ok(Kind::ContainerLayout),
            0x000A => Ok(Kind::HardwareBackend),
            0x000B => Ok(Kind::BenchmarkMetric),
            0x000C => Ok(Kind::TrainingHook),
            0x000D => Ok(Kind::RawBinary),
            other => Err(ExtensionError::UnknownKind(other)),
        }
    }

    /// Whether this kind participates only at encode time.
    pub fn is_encode_only(self) -> bool {
        matches!(
            self,
            Kind::ChainBuilder
                | Kind::ChainExplorer
                | Kind::Classifier
                | Kind::Scorer
                | Kind::TrainingHook
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn u16_roundtrip_for_every_kind() {
        for k in [
            Kind::Transform,
            Kind::PlaneCodec,
            Kind::ChainBuilder,
            Kind::ChainExplorer,
            Kind::Classifier,
            Kind::Scorer,
            Kind::DeltaScheme,
            Kind::IntegrationAdapter,
            Kind::ContainerLayout,
            Kind::HardwareBackend,
            Kind::BenchmarkMetric,
            Kind::TrainingHook,
            Kind::RawBinary,
        ] {
            assert_eq!(Kind::from_u16(k.as_u16()).unwrap(), k);
        }
    }

    #[test]
    fn unknown_kind_rejected() {
        assert!(Kind::from_u16(0x00FF).is_err());
        assert!(Kind::from_u16(0x0000).is_err());
    }
}
