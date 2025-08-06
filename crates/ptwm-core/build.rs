fn main() {
    println!("cargo:rerun-if-changed=data/bundled-keys.toml");
    println!("cargo:rerun-if-changed=data/vendor_table.toml");
}
