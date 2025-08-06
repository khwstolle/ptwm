//! JSON Canonicalization Scheme (RFC 8785) serialization for the restricted
//! manifest schema.
//!
//! The PTWM-JSON manifest restricts scalars to integers, strings, booleans,
//! and `null` (real-valued data is carried as opaque base64), so the only JCS
//! rules that matter here are:
//!
//! * object members sorted by key, ordered by UTF-16 code unit;
//! * no insignificant whitespace;
//! * strings escaped exactly as JSON (handled by `serde_json`);
//! * integers emitted in their shortest form (trivial — no fractions/exponents).
//!
//! Floats are explicitly rejected so the schema-restriction invariant ("no
//! real-valued scalars in the manifest") is enforced at serialization time.

use serde_json::Value;

use crate::error::PtwmCoreError;

/// Serialize `value` to a canonical (RFC 8785) JSON string.
///
/// Returns an error if the value contains a non-integer number, which would
/// violate the integers-only-scalar schema restriction.
pub fn to_string(value: &Value) -> Result<String, PtwmCoreError> {
    let mut out = String::new();
    write_value(value, &mut out)?;
    Ok(out)
}

fn write_value(value: &Value, out: &mut String) -> Result<(), PtwmCoreError> {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        Value::Number(n) => {
            if n.is_f64() {
                return Err(PtwmCoreError::InvalidContainer(
                    "manifest canonicalization: floating-point scalars are not permitted".into(),
                ));
            }
            out.push_str(&n.to_string());
        }
        Value::String(s) => write_string(s, out),
        Value::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_value(item, out)?;
            }
            out.push(']');
        }
        Value::Object(map) => {
            // Sort keys by their UTF-16 code-unit sequence, per RFC 8785.
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort_by(|a, b| utf16_cmp(a, b));
            out.push('{');
            for (i, key) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(',');
                }
                write_string(key, out);
                out.push(':');
                write_value(&map[*key], out)?;
            }
            out.push('}');
        }
    }
    Ok(())
}

/// Emit `s` as a JSON string with canonical escaping. `serde_json` already
/// produces RFC 8259 / RFC 8785-compatible minimal escaping for string values.
fn write_string(s: &str, out: &mut String) {
    // `serde_json::to_string` on a string never fails.
    let encoded = serde_json::to_string(s).expect("string serialization is infallible");
    out.push_str(&encoded);
}

/// Compare two strings by their UTF-16 code-unit sequences.
fn utf16_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn sorts_object_keys() {
        let v = json!({"b": 1, "a": 2, "c": 3});
        assert_eq!(to_string(&v).unwrap(), r#"{"a":2,"b":1,"c":3}"#);
    }

    #[test]
    fn no_whitespace_nested() {
        let v = json!({"z": [1, 2, {"y": "x"}], "a": null, "ok": true});
        assert_eq!(
            to_string(&v).unwrap(),
            r#"{"a":null,"ok":true,"z":[1,2,{"y":"x"}]}"#
        );
    }

    #[test]
    fn rejects_floats() {
        let v = json!({"n": 1.5});
        assert!(to_string(&v).is_err());
    }

    #[test]
    fn escapes_strings() {
        let v = json!({"k": "a\"b\n"});
        assert_eq!(to_string(&v).unwrap(), r#"{"k":"a\"b\n"}"#);
    }
}
