use super::*;
use crate::extension::{WILSON_ACTION_JVP_FAMILY, WILSON_FORCE_FAMILY};

#[test]
fn rules_register_both_first_order_roles_on_the_action_family_only() {
    let rules = ad_rules().unwrap();
    assert!(rules.lookup_linearize(WILSON_ACTION_FAMILY).is_some());
    assert!(rules
        .lookup_linear_transpose(WILSON_ACTION_FAMILY)
        .is_some());
    assert!(rules.lookup_primal_vjp(WILSON_ACTION_FAMILY).is_none());
    assert!(rules.lookup_linearize(WILSON_ACTION_JVP_FAMILY).is_none());
    assert!(rules
        .lookup_linear_transpose(WILSON_ACTION_JVP_FAMILY)
        .is_none());
    assert!(rules.lookup_linearize(WILSON_FORCE_FAMILY).is_none());
    assert!(rules.lookup_linear_transpose(WILSON_FORCE_FAMILY).is_none());
}

#[test]
fn independent_contexts_own_independent_semantic_rule_sets() {
    let first = ad_rules().unwrap();
    let second = ad_rules().unwrap();
    let first_rule = first.lookup_linearize(WILSON_ACTION_FAMILY).unwrap();
    let second_rule = second.lookup_linearize(WILSON_ACTION_FAMILY).unwrap();
    assert!(!std::sync::Arc::ptr_eq(&first_rule, &second_rule));
}
