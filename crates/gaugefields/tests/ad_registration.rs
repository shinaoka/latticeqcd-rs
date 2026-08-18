#![cfg(feature = "autodiff")]

use gaugefields::ad_rules;
use tenferro_ad::AdContext;

const ACTION: &str = "gaugefields.wilson_action.v1";
const FORCE: &str = "gaugefields.wilson_force.v1";

#[test]
fn rule_set_registers_only_the_action_family_for_independent_contexts() {
    for _ in 0..2 {
        let rules = ad_rules().unwrap();
        assert!(rules.lookup_linearize(ACTION).is_some());
        assert!(rules.lookup_linear_transpose(ACTION).is_some());
        assert!(rules.lookup_primal_vjp(ACTION).is_none());
        assert!(rules.lookup_linearize(FORCE).is_none());
        assert!(rules.lookup_linear_transpose(FORCE).is_none());

        let context = AdContext::builder()
            .with_semantic_extension_rules(rules)
            .unwrap()
            .build()
            .unwrap();
        assert!(context
            .semantic_extension_rules()
            .lookup_linearize(ACTION)
            .is_some());
        assert!(context
            .semantic_extension_rules()
            .lookup_linear_transpose(ACTION)
            .is_some());
    }
}
