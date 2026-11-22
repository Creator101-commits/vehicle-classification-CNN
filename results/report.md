# Vehicle CNN benchmark (full)

- git_sha: `85fcf2a20b76ff777df1f68bca82f130d0eb503a`
- ordering_ok: True
- images: 26378, classes: Bicycle, Bus, Car, Motorcycle, NonVehicles, Taxi, Truck, Van
- pHash threshold: 6
- weighting: unweighted
- jpeg quality: 75, subsampling: 2
- split counts: {'train': 18465, 'val': 3958, 'test': 3955}

## Test metrics (grouped, original formats)

| run | acc | bal-acc | macro-F1 |
| --- | ---: | ---: | ---: |
| custom_A_s0 | 0.8809 | 0.7772 | 0.7961 |
| custom_A_s1 | 0.8731 | 0.7731 | 0.7870 |
| custom_A_s2 | 0.8756 | 0.7851 | 0.7961 |
| resnet_scratch_s0 | 0.8900 | 0.7965 | 0.8069 |
| resnet_scratch_s1 | 0.8857 | 0.7925 | 0.8063 |
| resnet_scratch_s2 | 0.8968 | 0.8057 | 0.8197 |
| resnet_pretrained_s0 | 0.9128 | 0.8481 | 0.8511 |
| resnet_pretrained_s1 | 0.9193 | 0.8465 | 0.8617 |
| resnet_pretrained_s2 | 0.9178 | 0.8483 | 0.8601 |
| custom_B_s0 | 0.8771 | 0.7704 | 0.7884 |
| custom_B_s1 | 0.8733 | 0.7769 | 0.7914 |
| custom_B_s2 | 0.8771 | 0.7726 | 0.7911 |

## Format 2x2 (custom arms)

- arm_A_test_orig seed 0: acc=0.8809 macro-F1=0.7961 NonVehicles recall=0.9985
- arm_A_test_orig seed 1: acc=0.8731 macro-F1=0.7870 NonVehicles recall=0.9926
- arm_A_test_orig seed 2: acc=0.8756 macro-F1=0.7961 NonVehicles recall=0.9955
- arm_A_test_harmonized seed 0: acc=0.8809 macro-F1=0.7961 NonVehicles recall=0.9985
- arm_A_test_harmonized seed 1: acc=0.8728 macro-F1=0.7868 NonVehicles recall=0.9918
- arm_A_test_harmonized seed 2: acc=0.8759 macro-F1=0.7963 NonVehicles recall=0.9963
- arm_B_test_orig seed 0: acc=0.8771 macro-F1=0.7884 NonVehicles recall=0.9963
- arm_B_test_orig seed 1: acc=0.8733 macro-F1=0.7914 NonVehicles recall=0.9955
- arm_B_test_orig seed 2: acc=0.8771 macro-F1=0.7911 NonVehicles recall=0.9963
- arm_B_test_harmonized seed 0: acc=0.8766 macro-F1=0.7882 NonVehicles recall=0.9948
- arm_B_test_harmonized seed 1: acc=0.8736 macro-F1=0.7916 NonVehicles recall=0.9963
- arm_B_test_harmonized seed 2: acc=0.8771 macro-F1=0.7911 NonVehicles recall=0.9963

## Inflation demo

- random_model_random_test: {'macro_f1': 0.770891296128763, 'accuracy': 0.8642568250758341, 'balanced_accuracy': 0.7497071413472995, 'n': 3956}
- random_model_clean_subset: {'macro_f1': 0.7541638184371442, 'accuracy': 0.9485387547649301, 'balanced_accuracy': 0.7404099303963922, 'n': 1574}
- grouped_model_grouped_test_macro_f1: 0.7961101459552786
- headline_inflation_macro_f1: -0.025218849826515677
