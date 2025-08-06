//! Sealed-but-extensible capability vocabulary.
//!
//! v1 baseline keys: determinism, native_deps, host_imports,
//! hardware_class, sandbox_class, mem_factor, fuel_factor.
//! Unknown keys are parse-time allow, verify-time deny (the verifier
//! refuses to invoke a contribution declaring a capability key the
//! host doesn't recognize unless the user explicitly opts in via
//! policy).

use std::collections::BTreeMap;

use ciborium::Value as CborValue;
use serde::{Deserialize, Serialize};

use super::ExtensionError;

/// Set of canonical baseline capability keys recognized by v1.
pub const BASELINE_KEYS: &[&str] = &[
    "determinism",
    "native_deps",
    "host_imports",
    "hardware_class",
    "sandbox_class",
    "mem_factor",
    "fuel_factor",
];

/// Capability map; preserves declared order via `BTreeMap`'s canonical sort.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct CapabilityMap(pub BTreeMap<String, CapabilityValue>);

/// Capability values are restricted to a small typed set so verification
/// can be exhaustive on baseline keys.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CapabilityValue {
    Bool(bool),
    Int(i64),
    Float(F64Wrap),
    Text(String),
    List(Vec<CapabilityValue>),
    Map(BTreeMap<String, CapabilityValue>),
}

/// Hashable, Eq-comparable wrapper around `f64` (NaN treated as unequal).
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct F64Wrap(pub f64);

impl PartialEq for F64Wrap {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for F64Wrap {}

/// Single entry of a `Capability` for ergonomic field-style construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Capability {
    pub key: String,
    pub value: CapabilityValue,
}

impl CapabilityMap {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, key: impl Into<String>, value: CapabilityValue) {
        self.0.insert(key.into(), value);
    }

    pub fn get(&self, key: &str) -> Option<&CapabilityValue> {
        self.0.get(key)
    }

    /// Encode to canonical CBOR (sorted keys, deterministic).
    pub fn to_cbor(&self) -> Result<Vec<u8>, ExtensionError> {
        let map: Vec<(CborValue, CborValue)> = self
            .0
            .iter()
            .map(|(k, v)| (CborValue::Text(k.clone()), to_cbor_value(v)))
            .collect();
        let mut buf = Vec::new();
        ciborium::ser::into_writer(&CborValue::Map(map), &mut buf)
            .map_err(|e| ExtensionError::CapabilitySer(e.to_string()))?;
        Ok(buf)
    }

    pub fn from_cbor(bytes: &[u8]) -> Result<Self, ExtensionError> {
        let value: CborValue = ciborium::de::from_reader(bytes)
            .map_err(|e| ExtensionError::CapabilityParse(e.to_string()))?;
        let CborValue::Map(entries) = value else {
            return Err(ExtensionError::CapabilityParse("expected map".into()));
        };
        let mut map = BTreeMap::new();
        for (k, v) in entries {
            let CborValue::Text(key) = k else {
                return Err(ExtensionError::CapabilityParse(
                    "non-text capability key".into(),
                ));
            };
            map.insert(key, from_cbor_value(v)?);
        }
        Ok(Self(map))
    }

    /// Return any keys declared but not recognized by the v1 baseline.
    pub fn unknown_keys(&self) -> Vec<&str> {
        self.0
            .keys()
            .filter(|k| !BASELINE_KEYS.contains(&k.as_str()))
            .map(String::as_str)
            .collect()
    }
}

fn to_cbor_value(v: &CapabilityValue) -> CborValue {
    match v {
        CapabilityValue::Bool(b) => CborValue::Bool(*b),
        CapabilityValue::Int(i) => CborValue::Integer((*i).into()),
        CapabilityValue::Float(f) => CborValue::Float(f.0),
        CapabilityValue::Text(s) => CborValue::Text(s.clone()),
        CapabilityValue::List(xs) => CborValue::Array(xs.iter().map(to_cbor_value).collect()),
        CapabilityValue::Map(m) => CborValue::Map(
            m.iter()
                .map(|(k, v)| (CborValue::Text(k.clone()), to_cbor_value(v)))
                .collect(),
        ),
    }
}

const MAX_CAPABILITY_DEPTH: usize = 32;

fn from_cbor_value(v: CborValue) -> Result<CapabilityValue, ExtensionError> {
    from_cbor_value_depth(v, 0)
}

fn from_cbor_value_depth(v: CborValue, depth: usize) -> Result<CapabilityValue, ExtensionError> {
    if depth > MAX_CAPABILITY_DEPTH {
        return Err(ExtensionError::CapabilityParse(format!(
            "capability nesting exceeds depth {MAX_CAPABILITY_DEPTH}"
        )));
    }
    Ok(match v {
        CborValue::Bool(b) => CapabilityValue::Bool(b),
        CborValue::Integer(i) => CapabilityValue::Int(
            i.try_into()
                .map_err(|_| ExtensionError::CapabilityParse("int overflow".into()))?,
        ),
        CborValue::Float(f) => CapabilityValue::Float(F64Wrap(f)),
        CborValue::Text(s) => CapabilityValue::Text(s),
        CborValue::Array(xs) => CapabilityValue::List(
            xs.into_iter()
                .map(|x| from_cbor_value_depth(x, depth + 1))
                .collect::<Result<_, _>>()?,
        ),
        CborValue::Map(entries) => {
            let mut m = BTreeMap::new();
            for (k, v) in entries {
                let CborValue::Text(key) = k else {
                    return Err(ExtensionError::CapabilityParse(
                        "non-text capability key in nested map".into(),
                    ));
                };
                m.insert(key, from_cbor_value_depth(v, depth + 1)?);
            }
            CapabilityValue::Map(m)
        }
        other => {
            return Err(ExtensionError::CapabilityParse(format!(
                "unsupported CBOR value: {other:?}"
            )));
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cbor_roundtrip_baseline_keys() {
        let mut m = CapabilityMap::new();
        m.set("determinism", CapabilityValue::Bool(true));
        m.set("hardware_class", CapabilityValue::Text("cpu".into()));
        m.set(
            "native_deps",
            CapabilityValue::List(vec![CapabilityValue::Text("libcuda>=12.0".into())]),
        );

        let bytes = m.to_cbor().unwrap();
        let back = CapabilityMap::from_cbor(&bytes).unwrap();
        assert_eq!(m, back);
    }

    #[test]
    fn unknown_keys_reported() {
        let mut m = CapabilityMap::new();
        m.set("determinism", CapabilityValue::Bool(true));
        m.set("future_thing", CapabilityValue::Int(42));
        let unknown = m.unknown_keys();
        assert_eq!(unknown, vec!["future_thing"]);
    }

    #[test]
    fn cbor_serialization_is_deterministic() {
        let mut m1 = CapabilityMap::new();
        m1.set("b", CapabilityValue::Int(2));
        m1.set("a", CapabilityValue::Int(1));
        let mut m2 = CapabilityMap::new();
        m2.set("a", CapabilityValue::Int(1));
        m2.set("b", CapabilityValue::Int(2));
        assert_eq!(m1.to_cbor().unwrap(), m2.to_cbor().unwrap());
    }

    #[test]
    fn deeply_nested_capability_value_is_rejected() {
        fn nested(depth: usize) -> CapabilityValue {
            if depth == 0 {
                CapabilityValue::Int(0)
            } else {
                CapabilityValue::List(vec![nested(depth - 1)])
            }
        }
        let mut m = CapabilityMap::new();
        m.set("x", nested(64));
        let bytes = m.to_cbor().unwrap();
        let res = CapabilityMap::from_cbor(&bytes);
        assert!(res.is_err(), "deeply nested capability should be rejected");
    }
}
