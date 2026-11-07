# TEST-18 offline fault report

Overall: **BLOCKED**

Synthetic command-level integration only; not physical stopping, cloud transport, release or admission evidence

{'executed': 216, 'passed': 216, 'failed': 0, 'blocked_combinations': 12}

## Executed fixture cases

| Profile | Behaviour | Track | Fault | Result | Failed checks |
|---|---|---|---|---|---|
| local_fixture | cruise | False | missing_depth | PASS |  |
| local_fixture | cruise | False | invalid_depth | PASS |  |
| local_fixture | cruise | False | missing_capture_clock | PASS |  |
| local_fixture | cruise | False | stale_capture | PASS |  |
| local_fixture | cruise | False | old_sim_frame | PASS |  |
| local_fixture | cruise | False | duplicate_frame | PASS |  |
| local_fixture | cruise | False | foreign_observation | PASS |  |
| local_fixture | cruise | False | old_session_result | PASS |  |
| local_fixture | cruise | False | wrong_result_frame | PASS |  |
| local_fixture | cruise | False | wrong_result_source | PASS |  |
| local_fixture | cruise | False | detector_exception | PASS |  |
| local_fixture | cruise | False | deadline_miss | PASS |  |
| local_fixture | cruise | False | generation_restart | PASS |  |
| local_fixture | cruise | False | missing_track_id | PASS |  |
| local_fixture | cruise | False | invalid_state | PASS |  |
| local_fixture | cruise | False | controller_exception | PASS |  |
| local_fixture | cruise | False | controller_delay | PASS |  |
| local_fixture | cruise | False | invalid_command | PASS |  |
| local_fixture | cruise | True | missing_depth | PASS |  |
| local_fixture | cruise | True | invalid_depth | PASS |  |
| local_fixture | cruise | True | missing_capture_clock | PASS |  |
| local_fixture | cruise | True | stale_capture | PASS |  |
| local_fixture | cruise | True | old_sim_frame | PASS |  |
| local_fixture | cruise | True | duplicate_frame | PASS |  |
| local_fixture | cruise | True | foreign_observation | PASS |  |
| local_fixture | cruise | True | old_session_result | PASS |  |
| local_fixture | cruise | True | wrong_result_frame | PASS |  |
| local_fixture | cruise | True | wrong_result_source | PASS |  |
| local_fixture | cruise | True | detector_exception | PASS |  |
| local_fixture | cruise | True | deadline_miss | PASS |  |
| local_fixture | cruise | True | generation_restart | PASS |  |
| local_fixture | cruise | True | missing_track_id | PASS |  |
| local_fixture | cruise | True | invalid_state | PASS |  |
| local_fixture | cruise | True | controller_exception | PASS |  |
| local_fixture | cruise | True | controller_delay | PASS |  |
| local_fixture | cruise | True | invalid_command | PASS |  |
| local_fixture | approach | False | missing_depth | PASS |  |
| local_fixture | approach | False | invalid_depth | PASS |  |
| local_fixture | approach | False | missing_capture_clock | PASS |  |
| local_fixture | approach | False | stale_capture | PASS |  |
| local_fixture | approach | False | old_sim_frame | PASS |  |
| local_fixture | approach | False | duplicate_frame | PASS |  |
| local_fixture | approach | False | foreign_observation | PASS |  |
| local_fixture | approach | False | old_session_result | PASS |  |
| local_fixture | approach | False | wrong_result_frame | PASS |  |
| local_fixture | approach | False | wrong_result_source | PASS |  |
| local_fixture | approach | False | detector_exception | PASS |  |
| local_fixture | approach | False | deadline_miss | PASS |  |
| local_fixture | approach | False | generation_restart | PASS |  |
| local_fixture | approach | False | missing_track_id | PASS |  |
| local_fixture | approach | False | invalid_state | PASS |  |
| local_fixture | approach | False | controller_exception | PASS |  |
| local_fixture | approach | False | controller_delay | PASS |  |
| local_fixture | approach | False | invalid_command | PASS |  |
| local_fixture | approach | True | missing_depth | PASS |  |
| local_fixture | approach | True | invalid_depth | PASS |  |
| local_fixture | approach | True | missing_capture_clock | PASS |  |
| local_fixture | approach | True | stale_capture | PASS |  |
| local_fixture | approach | True | old_sim_frame | PASS |  |
| local_fixture | approach | True | duplicate_frame | PASS |  |
| local_fixture | approach | True | foreign_observation | PASS |  |
| local_fixture | approach | True | old_session_result | PASS |  |
| local_fixture | approach | True | wrong_result_frame | PASS |  |
| local_fixture | approach | True | wrong_result_source | PASS |  |
| local_fixture | approach | True | detector_exception | PASS |  |
| local_fixture | approach | True | deadline_miss | PASS |  |
| local_fixture | approach | True | generation_restart | PASS |  |
| local_fixture | approach | True | missing_track_id | PASS |  |
| local_fixture | approach | True | invalid_state | PASS |  |
| local_fixture | approach | True | controller_exception | PASS |  |
| local_fixture | approach | True | controller_delay | PASS |  |
| local_fixture | approach | True | invalid_command | PASS |  |
| local_fixture | hold | False | missing_depth | PASS |  |
| local_fixture | hold | False | invalid_depth | PASS |  |
| local_fixture | hold | False | missing_capture_clock | PASS |  |
| local_fixture | hold | False | stale_capture | PASS |  |
| local_fixture | hold | False | old_sim_frame | PASS |  |
| local_fixture | hold | False | duplicate_frame | PASS |  |
| local_fixture | hold | False | foreign_observation | PASS |  |
| local_fixture | hold | False | old_session_result | PASS |  |
| local_fixture | hold | False | wrong_result_frame | PASS |  |
| local_fixture | hold | False | wrong_result_source | PASS |  |
| local_fixture | hold | False | detector_exception | PASS |  |
| local_fixture | hold | False | deadline_miss | PASS |  |
| local_fixture | hold | False | generation_restart | PASS |  |
| local_fixture | hold | False | missing_track_id | PASS |  |
| local_fixture | hold | False | invalid_state | PASS |  |
| local_fixture | hold | False | controller_exception | PASS |  |
| local_fixture | hold | False | controller_delay | PASS |  |
| local_fixture | hold | False | invalid_command | PASS |  |
| local_fixture | hold | True | missing_depth | PASS |  |
| local_fixture | hold | True | invalid_depth | PASS |  |
| local_fixture | hold | True | missing_capture_clock | PASS |  |
| local_fixture | hold | True | stale_capture | PASS |  |
| local_fixture | hold | True | old_sim_frame | PASS |  |
| local_fixture | hold | True | duplicate_frame | PASS |  |
| local_fixture | hold | True | foreign_observation | PASS |  |
| local_fixture | hold | True | old_session_result | PASS |  |
| local_fixture | hold | True | wrong_result_frame | PASS |  |
| local_fixture | hold | True | wrong_result_source | PASS |  |
| local_fixture | hold | True | detector_exception | PASS |  |
| local_fixture | hold | True | deadline_miss | PASS |  |
| local_fixture | hold | True | generation_restart | PASS |  |
| local_fixture | hold | True | missing_track_id | PASS |  |
| local_fixture | hold | True | invalid_state | PASS |  |
| local_fixture | hold | True | controller_exception | PASS |  |
| local_fixture | hold | True | controller_delay | PASS |  |
| local_fixture | hold | True | invalid_command | PASS |  |
| delayed_fixture | cruise | False | missing_depth | PASS |  |
| delayed_fixture | cruise | False | invalid_depth | PASS |  |
| delayed_fixture | cruise | False | missing_capture_clock | PASS |  |
| delayed_fixture | cruise | False | stale_capture | PASS |  |
| delayed_fixture | cruise | False | old_sim_frame | PASS |  |
| delayed_fixture | cruise | False | duplicate_frame | PASS |  |
| delayed_fixture | cruise | False | foreign_observation | PASS |  |
| delayed_fixture | cruise | False | old_session_result | PASS |  |
| delayed_fixture | cruise | False | wrong_result_frame | PASS |  |
| delayed_fixture | cruise | False | wrong_result_source | PASS |  |
| delayed_fixture | cruise | False | detector_exception | PASS |  |
| delayed_fixture | cruise | False | deadline_miss | PASS |  |
| delayed_fixture | cruise | False | generation_restart | PASS |  |
| delayed_fixture | cruise | False | missing_track_id | PASS |  |
| delayed_fixture | cruise | False | invalid_state | PASS |  |
| delayed_fixture | cruise | False | controller_exception | PASS |  |
| delayed_fixture | cruise | False | controller_delay | PASS |  |
| delayed_fixture | cruise | False | invalid_command | PASS |  |
| delayed_fixture | cruise | True | missing_depth | PASS |  |
| delayed_fixture | cruise | True | invalid_depth | PASS |  |
| delayed_fixture | cruise | True | missing_capture_clock | PASS |  |
| delayed_fixture | cruise | True | stale_capture | PASS |  |
| delayed_fixture | cruise | True | old_sim_frame | PASS |  |
| delayed_fixture | cruise | True | duplicate_frame | PASS |  |
| delayed_fixture | cruise | True | foreign_observation | PASS |  |
| delayed_fixture | cruise | True | old_session_result | PASS |  |
| delayed_fixture | cruise | True | wrong_result_frame | PASS |  |
| delayed_fixture | cruise | True | wrong_result_source | PASS |  |
| delayed_fixture | cruise | True | detector_exception | PASS |  |
| delayed_fixture | cruise | True | deadline_miss | PASS |  |
| delayed_fixture | cruise | True | generation_restart | PASS |  |
| delayed_fixture | cruise | True | missing_track_id | PASS |  |
| delayed_fixture | cruise | True | invalid_state | PASS |  |
| delayed_fixture | cruise | True | controller_exception | PASS |  |
| delayed_fixture | cruise | True | controller_delay | PASS |  |
| delayed_fixture | cruise | True | invalid_command | PASS |  |
| delayed_fixture | approach | False | missing_depth | PASS |  |
| delayed_fixture | approach | False | invalid_depth | PASS |  |
| delayed_fixture | approach | False | missing_capture_clock | PASS |  |
| delayed_fixture | approach | False | stale_capture | PASS |  |
| delayed_fixture | approach | False | old_sim_frame | PASS |  |
| delayed_fixture | approach | False | duplicate_frame | PASS |  |
| delayed_fixture | approach | False | foreign_observation | PASS |  |
| delayed_fixture | approach | False | old_session_result | PASS |  |
| delayed_fixture | approach | False | wrong_result_frame | PASS |  |
| delayed_fixture | approach | False | wrong_result_source | PASS |  |
| delayed_fixture | approach | False | detector_exception | PASS |  |
| delayed_fixture | approach | False | deadline_miss | PASS |  |
| delayed_fixture | approach | False | generation_restart | PASS |  |
| delayed_fixture | approach | False | missing_track_id | PASS |  |
| delayed_fixture | approach | False | invalid_state | PASS |  |
| delayed_fixture | approach | False | controller_exception | PASS |  |
| delayed_fixture | approach | False | controller_delay | PASS |  |
| delayed_fixture | approach | False | invalid_command | PASS |  |
| delayed_fixture | approach | True | missing_depth | PASS |  |
| delayed_fixture | approach | True | invalid_depth | PASS |  |
| delayed_fixture | approach | True | missing_capture_clock | PASS |  |
| delayed_fixture | approach | True | stale_capture | PASS |  |
| delayed_fixture | approach | True | old_sim_frame | PASS |  |
| delayed_fixture | approach | True | duplicate_frame | PASS |  |
| delayed_fixture | approach | True | foreign_observation | PASS |  |
| delayed_fixture | approach | True | old_session_result | PASS |  |
| delayed_fixture | approach | True | wrong_result_frame | PASS |  |
| delayed_fixture | approach | True | wrong_result_source | PASS |  |
| delayed_fixture | approach | True | detector_exception | PASS |  |
| delayed_fixture | approach | True | deadline_miss | PASS |  |
| delayed_fixture | approach | True | generation_restart | PASS |  |
| delayed_fixture | approach | True | missing_track_id | PASS |  |
| delayed_fixture | approach | True | invalid_state | PASS |  |
| delayed_fixture | approach | True | controller_exception | PASS |  |
| delayed_fixture | approach | True | controller_delay | PASS |  |
| delayed_fixture | approach | True | invalid_command | PASS |  |
| delayed_fixture | hold | False | missing_depth | PASS |  |
| delayed_fixture | hold | False | invalid_depth | PASS |  |
| delayed_fixture | hold | False | missing_capture_clock | PASS |  |
| delayed_fixture | hold | False | stale_capture | PASS |  |
| delayed_fixture | hold | False | old_sim_frame | PASS |  |
| delayed_fixture | hold | False | duplicate_frame | PASS |  |
| delayed_fixture | hold | False | foreign_observation | PASS |  |
| delayed_fixture | hold | False | old_session_result | PASS |  |
| delayed_fixture | hold | False | wrong_result_frame | PASS |  |
| delayed_fixture | hold | False | wrong_result_source | PASS |  |
| delayed_fixture | hold | False | detector_exception | PASS |  |
| delayed_fixture | hold | False | deadline_miss | PASS |  |
| delayed_fixture | hold | False | generation_restart | PASS |  |
| delayed_fixture | hold | False | missing_track_id | PASS |  |
| delayed_fixture | hold | False | invalid_state | PASS |  |
| delayed_fixture | hold | False | controller_exception | PASS |  |
| delayed_fixture | hold | False | controller_delay | PASS |  |
| delayed_fixture | hold | False | invalid_command | PASS |  |
| delayed_fixture | hold | True | missing_depth | PASS |  |
| delayed_fixture | hold | True | invalid_depth | PASS |  |
| delayed_fixture | hold | True | missing_capture_clock | PASS |  |
| delayed_fixture | hold | True | stale_capture | PASS |  |
| delayed_fixture | hold | True | old_sim_frame | PASS |  |
| delayed_fixture | hold | True | duplicate_frame | PASS |  |
| delayed_fixture | hold | True | foreign_observation | PASS |  |
| delayed_fixture | hold | True | old_session_result | PASS |  |
| delayed_fixture | hold | True | wrong_result_frame | PASS |  |
| delayed_fixture | hold | True | wrong_result_source | PASS |  |
| delayed_fixture | hold | True | detector_exception | PASS |  |
| delayed_fixture | hold | True | deadline_miss | PASS |  |
| delayed_fixture | hold | True | generation_restart | PASS |  |
| delayed_fixture | hold | True | missing_track_id | PASS |  |
| delayed_fixture | hold | True | invalid_state | PASS |  |
| delayed_fixture | hold | True | controller_exception | PASS |  |
| delayed_fixture | hold | True | controller_delay | PASS |  |
| delayed_fixture | hold | True | invalid_command | PASS |  |

## Blocked combinations

- remote / cruise / tracking=False: Canonical client/hybrid/server implementations now exist; this fixture profile is not a selected/admitted production flag combination. E4-10 evidence and G3 flag admission pending
- remote / cruise / tracking=True: Canonical client/hybrid/server implementations now exist; this fixture profile is not a selected/admitted production flag combination. E4-10 evidence and G3 flag admission pending
- remote / approach / tracking=False: Canonical client/hybrid/server implementations now exist; this fixture profile is not a selected/admitted production flag combination. E4-10 evidence and G3 flag admission pending
- remote / approach / tracking=True: Canonical client/hybrid/server implementations now exist; this fixture profile is not a selected/admitted production flag combination. E4-10 evidence and G3 flag admission pending
- remote / hold / tracking=False: Canonical client/hybrid/server implementations now exist; this fixture profile is not a selected/admitted production flag combination. E4-10 evidence and G3 flag admission pending
- remote / hold / tracking=True: Canonical client/hybrid/server implementations now exist; this fixture profile is not a selected/admitted production flag combination. E4-10 evidence and G3 flag admission pending
- hybrid / cruise / tracking=False: Canonical client/hybrid/server implementations now exist; this fixture profile is not a selected/admitted production flag combination. E4-10 evidence and G3 flag admission pending
- hybrid / cruise / tracking=True: Canonical client/hybrid/server implementations now exist; this fixture profile is not a selected/admitted production flag combination. E4-10 evidence and G3 flag admission pending
- hybrid / approach / tracking=False: Canonical client/hybrid/server implementations now exist; this fixture profile is not a selected/admitted production flag combination. E4-10 evidence and G3 flag admission pending
- hybrid / approach / tracking=True: Canonical client/hybrid/server implementations now exist; this fixture profile is not a selected/admitted production flag combination. E4-10 evidence and G3 flag admission pending
- hybrid / hold / tracking=False: Canonical client/hybrid/server implementations now exist; this fixture profile is not a selected/admitted production flag combination. E4-10 evidence and G3 flag admission pending
- hybrid / hold / tracking=True: Canonical client/hybrid/server implementations now exist; this fixture profile is not a selected/admitted production flag combination. E4-10 evidence and G3 flag admission pending
