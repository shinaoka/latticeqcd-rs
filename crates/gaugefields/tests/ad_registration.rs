#![cfg(feature = "autodiff")]

use gaugefields::ad_rules;
use tenferro_ad::AdContext;

const ACTION: &str = "gaugefields.wilson_action.v1";
const JVP: &str = "gaugefields.wilson_action_jvp.v1";
const FORCE: &str = "gaugefields.wilson_force.v1";

#[test]
fn rule_set_registers_only_the_role_split_path_for_independent_contexts() {
    for _ in 0..2 {
        let rules = ad_rules().unwrap();
        assert!(rules.is_linearize_registered(ACTION));
        assert!(rules.is_linear_transpose_registered(JVP));
        assert!(!rules.is_primal_vjp_registered(ACTION));
        assert!(!rules.is_linearize_registered(FORCE));
        assert!(!rules.is_linear_transpose_registered(FORCE));

        let context = AdContext::builder()
            .with_extension_rules(rules)
            .build()
            .unwrap();
        assert!(context.extension_rules().is_linearize_registered(ACTION));
        assert!(context
            .extension_rules()
            .is_linear_transpose_registered(JVP));
    }
}
