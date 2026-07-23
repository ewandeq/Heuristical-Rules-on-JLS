DecisionTree post-grid-search report (dt_v2)
Source: analyse/decisiontree/<task>/{metrics.csv, confusion.csv, feature_importances.csv}

## Agent 4 classes (agent_multiclass_4)
Metrics (metrics.csv):

Confusion matrix (confusion.csv):
```csv
,pred_TOPIC,pred_LAST_AGENT,pred_LAST_PATIENT,pred_UNKNOWN
true_TOPIC,30,9,14,38
true_LAST_AGENT,10,11,2,11
true_LAST_PATIENT,5,6,33,14
true_UNKNOWN,81,21,53,107
```

Feature importances (feature_importances.csv):
```csv
feature,importance
d_topic_last_agent,0.20107154809144653
agent_margin,0.15687250368729957
verb_index,0.12058326181998597
d_last_agent_last_patient,0.1124518160894164
noun_count,0.07294462607056997
postverb_is_pt,0.05247250319224601
one_noun,0.05033900823241175
last_agent_conf,0.045988246967066986
d_topic_last_patient,0.04571462014482603
pt_is_bare,0.028711495085909335
agent_top2,0.021743622985946797
last_patient_conf,0.021686536625723712
verb_post_pt,0.013760633344547822
postverb_pt_resolved,0.011637804398988282
agent_max_non_unknown,0.011176702964817709
topic_conf,0.0104635866634084
verb_missing,0.006152051698737854
discourse_max,0.0057724468177875075
agent_top1,0.0037523519206884145
no_noun,0.0029977874721763623
discourse_margin,0.0026590602860854014
low_discourse,0.0010477854399129785
has_verb,0.0
unknown_conf,0.0
action_conf,0.0
```

## Agent UNKNOWN vs non-UNKNOWN (agent_unknown_binary)
Metrics (metrics.csv):

Confusion matrix (confusion.csv):
```csv
,pred_neg,pred_pos
true_neg,90,93
true_pos,129,133
```

Feature importances (feature_importances.csv):
```csv
feature,importance
agent_top2,0.4217294924466949
d_topic_last_patient,0.2279786429158594
last_patient_conf,0.2262279897637496
agent_top1,0.12406387487369608
action_conf,0.0
no_noun,0.0
agent_margin,0.0
verb_post_pt,0.0
pt_is_bare,0.0
postverb_pt_resolved,0.0
postverb_is_pt,0.0
verb_missing,0.0
one_noun,0.0
verb_index,0.0
noun_count,0.0
topic_conf,0.0
has_verb,0.0
low_discourse,0.0
discourse_max,0.0
discourse_margin,0.0
d_last_agent_last_patient,0.0
d_topic_last_agent,0.0
unknown_conf,0.0
last_agent_conf,0.0
agent_max_non_unknown,0.0
```

## Agent non-UNKNOWN (3 classes) (agent_nonunknown_3)
Metrics (metrics.csv):

Confusion matrix (confusion.csv):
```csv
,pred_TOPIC,pred_LAST_AGENT,pred_LAST_PATIENT
true_TOPIC,54,24,13
true_LAST_AGENT,15,15,4
true_LAST_PATIENT,8,10,40
```

Feature importances (feature_importances.csv):
```csv
feature,importance
d_last_agent_last_patient,0.3667878625930291
d_topic_last_agent,0.23653653473409234
discourse_margin,0.12823334462452235
last_agent_conf,0.09672604066920383
postverb_is_pt,0.09433789327656643
agent_max_non_unknown,0.07737832410258608
one_noun,0.0
agent_margin,0.0
agent_top2,0.0
agent_top1,0.0
verb_post_pt,0.0
pt_is_bare,0.0
postverb_pt_resolved,0.0
verb_missing,0.0
action_conf,0.0
no_noun,0.0
noun_count,0.0
topic_conf,0.0
has_verb,0.0
low_discourse,0.0
discourse_max,0.0
d_topic_last_patient,0.0
unknown_conf,0.0
last_patient_conf,0.0
verb_index,0.0
```

## Patient binary (- vs rest) (patient_binary)
Metrics (metrics.csv):

Confusion matrix (confusion.csv):
```csv
,pred_no_patient,pred_has_patient
true_no_patient,123,4
true_has_patient,34,284
```

Feature importances (feature_importances.csv):
```csv
feature,importance
discourse_margin,0.8885889338806838
one_noun,0.03947691022231474
discourse_max,0.02854436722411968
postverb_pt_resolved,0.02402958448409379
d_topic_last_patient,0.019360204188788147
action_conf,0.0
no_noun,0.0
patient_margin,0.0
patient_top2,0.0
patient_top1,0.0
verb_post_pt,0.0
pt_is_bare,0.0
postverb_is_pt,0.0
verb_missing,0.0
verb_index,0.0
noun_count,0.0
topic_conf,0.0
has_verb,0.0
low_discourse,0.0
d_last_agent_last_patient,0.0
d_topic_last_agent,0.0
unknown_conf,0.0
last_patient_conf,0.0
last_agent_conf,0.0
patient_max_non_unknown,0.0
```


