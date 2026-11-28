# ROUND-028 独立科研验收合同

本合同由固定 ROUND-027 科研协议与 ROUND-028 阶段 A 指令在任何预检前制定。正式训练和评价均未授权；合同不得根据测试结果回填或放宽。逐字段机器可读内容以同名 JSON 为准。

固定 D_grad SHA256 `93c4c834c3fd3c8b92a64caa88da8a671e27be23dc2df11999c8e5f3dfbe47d8`；种子 12451 的训练首批 SHA256 `d6a2a69cb6b00d60f5f449c0c8a9ffdd3dbdaff0073d0e6e0deca9ddc6d61213`；真实 2+1 首三张 SHA256 `14cee780dd97cb7ad27477c904c63a8d3e83eb2393787408c9c56497ea4575f7`。

| 条目 | 阶段 | 对象 | 样本 | 证据内容与断言 |
|---|---|---|---:|---|
| A01 | GPU | B0,P0 | 0 | teacher_sha256 eq cd6e151efe68ee77ef57ce474d0e83737321f271d4bace1ef8809013a66ae60d, P0_sha256 eq 95073e1c9b3731a792fc72d596b3fa1c96b86989b328241b00bb663020c80958 |
| A02 | GPU | B0,P0 | 0 | teacher_hidden eq 2048, P0_hidden eq 512, parameters eq 11399524, registrations_match eq True, cupy_lif_config_match eq True |
| A03 | CPU | ALL | 512 | split_sha256 eq 69dccfe992236766aea575a0a66495802ca889ca329d61f51e8de3c1ab7f3165, train_count eq 45000, validation_count eq 4500, disjoint eq True |
| A04 | GPU | K0,K1,K2,K3,M6 | 0 | all_initial_state_equal eq True, independent_loads eq 5 |
| A05 | GPU | K0,K1,K2,K3,M6 | 0 | K0_scope eq ['fc2.bias', 'fc2.weight'], K1_scope eq ['fc2.bias', 'fc2.weight'], K2_scope eq ['fc2.bias', 'fc2.weight'], K3_scope eq ['fc2.bias', 'fc2.weight'], M6_scope eq ['fc2.bias', 'fc2.weight'] |
| A06 | CPU | K0,K1,K2,K3,M6 | 0 | seed eq 12450, batch eq 256, epochs eq 5, momentum eq 0.9, weight_decay eq 0, rho eq 0.5, beta eq 0.5, tau eq 4, lr_sequence eq [0.01, 0.009045084971874737, 0.006545084971874737, 0.003454915028125263, 0.0009549150281252633] |
| B01 | CPU | B0,P0 | 0 | max_error finite_le 1e-10 |
| B02 | CPU | B0,P0 | 0 | max_error finite_le 1e-10 |
| B03 | CPU | B0,P0 | 0 | max_error finite_le 1e-10 |
| B04 | GPU | B0,P0 | 512 | max_normalized_error finite_le 1e-05 |
| B05 | GPU | B0,P0 | 512 | max_normalized_error finite_le 1e-05 |
| B06 | GPU | B0,P0 | 512 | max_normalized_error finite_le 1e-05 |
| B07 | GPU | B0,P0 | 512 | mismatch_count eq 0 |
| B08 | CPU | ALL | 0 | zero_fallback eq True |
| B09 | CPU | ALL | 0 | single_damage eq True |
| B10 | CPU | ALL | 0 | equal_damage eq True |
| B11 | CPU | ALL | 0 | wrong_teacher_zero eq True |
| B13 | CPU | ALL | 0 | rho0_uniform eq True |
| B14 | CPU | ALL | 0 | cyclic_shift_multiset eq True |
| B12 | GPU | B0,P0 | 512 | min_weight finite_ge 0.499999, max_weight finite_le 2.500001, mean_error finite_le 1e-06 |
| B15 | CPU | K1,K2,K3,M6 | 0 | max_abs_kd finite_le 1e-06 |
| B16 | CPU | K1,K2,K3,M6 | 0 | K1_abs_error finite_le 1e-05, K2_abs_error finite_le 1e-05, K3_abs_error finite_le 1e-05, M6_abs_error finite_le 1e-05 |
| B17 | GPU | B0 | 3 | actual_batch_sizes eq [2, 1], normalized_error finite_le 1e-05 |
| B18 | GPU | B0,P0 | 512 | nonuniform_fraction finite_ge 0.1, shift_different_elements finite_ge 1 |
| C_K0 | GPU | K0 | 256 | input_shared eq True, amp_enabled eq True, finite_loss eq True, finite_gradient eq True, nonzero_gradient eq True, optimizer_step_success eq True, readout_updated eq True, frozen_changed_count eq 0, threshold_changed_count eq 0, teacher_grad_count eq 0, P0_grad_count eq 0, teacher_state_equal eq True, P0_state_equal eq True, manual_penalty_calls eq 0, spike_mismatch_count eq 0, parameters eq 11399524, amp_CE_repeat_error finite_le 1e-06 |
| D01_K0 | GPU | K0 | 256 | checkpoint_written eq True |
| D02_K0 | GPU | K0 | 256 | all_model_state_exact eq True |
| D03_K0 | GPU | K0 | 256 | param_groups_exact eq True |
| D04_K0 | GPU | K0 | 256 | optimizer_state_exact eq True |
| D05_K0 | GPU | K0 | 256 | momentum_tensor_count finite_ge 1, all_momentum_dtype_shape_match eq True, max_momentum_abs_diff finite_le 0 |
| D06_K0 | GPU | K0 | 256 | scheduler_state_exact eq True |
| D07_K0 | GPU | K0 | 256 | scaler_state_exact eq True |
| D08_K0 | GPU | K0 | 256 | continuous_vs_restored_exact eq True, max_parameter_abs_diff finite_le 0 |
| C_K1 | GPU | K1 | 256 | input_shared eq True, amp_enabled eq True, finite_loss eq True, finite_gradient eq True, nonzero_gradient eq True, optimizer_step_success eq True, readout_updated eq True, frozen_changed_count eq 0, threshold_changed_count eq 0, teacher_grad_count eq 0, P0_grad_count eq 0, teacher_state_equal eq True, P0_state_equal eq True, manual_penalty_calls eq 0, spike_mismatch_count eq 0, parameters eq 11399524, amp_CE_repeat_error finite_le 1e-06 |
| D01_K1 | GPU | K1 | 256 | checkpoint_written eq True |
| D02_K1 | GPU | K1 | 256 | all_model_state_exact eq True |
| D03_K1 | GPU | K1 | 256 | param_groups_exact eq True |
| D04_K1 | GPU | K1 | 256 | optimizer_state_exact eq True |
| D05_K1 | GPU | K1 | 256 | momentum_tensor_count finite_ge 1, all_momentum_dtype_shape_match eq True, max_momentum_abs_diff finite_le 0 |
| D06_K1 | GPU | K1 | 256 | scheduler_state_exact eq True |
| D07_K1 | GPU | K1 | 256 | scaler_state_exact eq True |
| D08_K1 | GPU | K1 | 256 | continuous_vs_restored_exact eq True, max_parameter_abs_diff finite_le 0 |
| C_K2 | GPU | K2 | 256 | input_shared eq True, amp_enabled eq True, finite_loss eq True, finite_gradient eq True, nonzero_gradient eq True, optimizer_step_success eq True, readout_updated eq True, frozen_changed_count eq 0, threshold_changed_count eq 0, teacher_grad_count eq 0, P0_grad_count eq 0, teacher_state_equal eq True, P0_state_equal eq True, manual_penalty_calls eq 0, spike_mismatch_count eq 0, parameters eq 11399524, amp_CE_repeat_error finite_le 1e-06 |
| D01_K2 | GPU | K2 | 256 | checkpoint_written eq True |
| D02_K2 | GPU | K2 | 256 | all_model_state_exact eq True |
| D03_K2 | GPU | K2 | 256 | param_groups_exact eq True |
| D04_K2 | GPU | K2 | 256 | optimizer_state_exact eq True |
| D05_K2 | GPU | K2 | 256 | momentum_tensor_count finite_ge 1, all_momentum_dtype_shape_match eq True, max_momentum_abs_diff finite_le 0 |
| D06_K2 | GPU | K2 | 256 | scheduler_state_exact eq True |
| D07_K2 | GPU | K2 | 256 | scaler_state_exact eq True |
| D08_K2 | GPU | K2 | 256 | continuous_vs_restored_exact eq True, max_parameter_abs_diff finite_le 0 |
| C_K3 | GPU | K3 | 256 | input_shared eq True, amp_enabled eq True, finite_loss eq True, finite_gradient eq True, nonzero_gradient eq True, optimizer_step_success eq True, readout_updated eq True, frozen_changed_count eq 0, threshold_changed_count eq 0, teacher_grad_count eq 0, P0_grad_count eq 0, teacher_state_equal eq True, P0_state_equal eq True, manual_penalty_calls eq 0, spike_mismatch_count eq 0, parameters eq 11399524, amp_CE_repeat_error finite_le 1e-06 |
| D01_K3 | GPU | K3 | 256 | checkpoint_written eq True |
| D02_K3 | GPU | K3 | 256 | all_model_state_exact eq True |
| D03_K3 | GPU | K3 | 256 | param_groups_exact eq True |
| D04_K3 | GPU | K3 | 256 | optimizer_state_exact eq True |
| D05_K3 | GPU | K3 | 256 | momentum_tensor_count finite_ge 1, all_momentum_dtype_shape_match eq True, max_momentum_abs_diff finite_le 0 |
| D06_K3 | GPU | K3 | 256 | scheduler_state_exact eq True |
| D07_K3 | GPU | K3 | 256 | scaler_state_exact eq True |
| D08_K3 | GPU | K3 | 256 | continuous_vs_restored_exact eq True, max_parameter_abs_diff finite_le 0 |
| C_M6 | GPU | M6 | 256 | input_shared eq True, amp_enabled eq True, finite_loss eq True, finite_gradient eq True, nonzero_gradient eq True, optimizer_step_success eq True, readout_updated eq True, frozen_changed_count eq 0, threshold_changed_count eq 0, teacher_grad_count eq 0, P0_grad_count eq 0, teacher_state_equal eq True, P0_state_equal eq True, manual_penalty_calls eq 0, spike_mismatch_count eq 0, parameters eq 11399524, amp_CE_repeat_error finite_le 1e-06 |
| D01_M6 | GPU | M6 | 256 | checkpoint_written eq True |
| D02_M6 | GPU | M6 | 256 | all_model_state_exact eq True |
| D03_M6 | GPU | M6 | 256 | param_groups_exact eq True |
| D04_M6 | GPU | M6 | 256 | optimizer_state_exact eq True |
| D05_M6 | GPU | M6 | 256 | momentum_tensor_count finite_ge 1, all_momentum_dtype_shape_match eq True, max_momentum_abs_diff finite_le 0 |
| D06_M6 | GPU | M6 | 256 | scheduler_state_exact eq True |
| D07_M6 | GPU | M6 | 256 | scaler_state_exact eq True |
| D08_M6 | GPU | M6 | 256 | continuous_vs_restored_exact eq True, max_parameter_abs_diff finite_le 0 |
| E01 | GPU | K0,K1,K2,K3,M6 | 256 | steady_batches_per_group finite_ge 3, all_group_timings_finite eq True, estimated_formal_wall_seconds finite_le 1800 |
| F01 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq missing_group |
| F02 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq teacher_changed |
| F03 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq optimizer_scope |
| F04 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq config_changed |
| F05 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq training_remote |
| F06 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq missing_endpoint |
| F07 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq missing_reload |
| F08 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq endpoint_changed |
| F09 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq training_incomplete |
| F10 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq evaluation_remote |
| F11 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq missing_B01 |
| F12 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq missing_B02 |
| F13 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq missing_B03 |
| F14 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq missing_momentum |
| F15 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq altered_momentum |
| F16 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq joint_count_256 |
| F17 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq same_count_wrong_ids |
| F18 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq passed_but_error |
| F19 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq missing_evidence |
| F20 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq changed_evidence_hash |
| F21 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq changed_source |
| F22 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq wrong_candidate |
| F23 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq old_lock |
| F24 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq missing_training_authorization |
| F25 | CPU | ALL | 0 | rejected eq True, training_calls eq 0, optimizer_formal_updates eq 0, validation_loader_calls eq 0, evaluation_calls eq 0, scenario eq missing_review_authorization |
