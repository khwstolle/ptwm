//! Chain DAG types — Chain, ChainNode, ChainEdge, TerminalRef.

use crate::transforms::op::OpId;
use crate::types::role::Role;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainNode {
    pub op: OpId,
    pub params: Vec<u8>, // op-specific TLV payload (encoded by Op::write_params)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChainEdge {
    pub src_node: u8,
    pub src_output_idx: u8,
    pub dst_node: u8,
    pub dst_input_idx: u8,
    pub role_override: Option<Role>,
    pub vendor_bytes: Vec<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalRef {
    pub node_idx: u8,
    pub output_idx: u8,
    pub role: Role,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chain {
    pub nodes: Vec<ChainNode>,
    pub edges: Vec<ChainEdge>,
    pub terminals: Vec<TerminalRef>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::role::ScaleFormat;

    #[test]
    fn chain_construction() {
        let chain = Chain {
            nodes: vec![
                ChainNode {
                    op: OpId::Source,
                    params: vec![],
                },
                ChainNode {
                    op: OpId::BytePassthrough,
                    params: vec![],
                },
            ],
            edges: vec![ChainEdge {
                src_node: 0,
                src_output_idx: 0,
                dst_node: 1,
                dst_input_idx: 0,
                role_override: None,
                vendor_bytes: vec![],
            }],
            terminals: vec![TerminalRef {
                node_idx: 1,
                output_idx: 0,
                role: Role::Scale {
                    format: ScaleFormat::E4M3,
                },
            }],
        };
        assert_eq!(chain.nodes.len(), 2);
        assert_eq!(chain.edges.len(), 1);
        assert_eq!(chain.terminals.len(), 1);
    }
}
