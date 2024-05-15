
The official pytorch code repository for paper [MiGCL: Multi-granular Contrastive Learning for Self-supervised Pre-training]()

![MIGCL_fig_github-1](https://github.com/vangorade/MIGCL_code/assets/71941335/f6b6fbc8-8ba5-4645-ad34-aa98eac85ed5)

Abstract. Self-supervised learning (SSL) has become increasingly
promising across various domains, particularly in vision tasks,
owing to its capability to learn from data without explicit
annotations. However, prevailing SSL methods, primarily based
on the contrastive learning paradigm, often utilize a single type
of encoder such as Convolutional Neural Networks (CNNs).
This reliance on a singular encoder type restricts the true
potential of SSL in capturing more discriminative and task-
specific representations. The limitation arises from the fact that
diverse encoders inherently capture different inductive biases
and can learn complementary representations. To address this
limitation, we introduced Multi-Granular Contrastive Learning
for Self-supervised Pre-training (MiGCL) framework designed to
effectively model both global and local contexts in a hierarchical
multi-task fashion. By leveraging multiple encoders, MiGCL
captures a broader spectrum of information, leading to the
learning of semantically rich representations applicable across
various tasks. We validate the effectiveness of our proposed
method through extensive experimental studies conducted on
seven publicly available datasets spanning natural vision and
chest X-ray modalities. Our empirical findings demonstrate the
superiority of MiGCL over state-of-the-art methods across all
evaluated tasks, demonstrating its ability in learning generalize
representations.

# Results

![MIGCL_CAMs_guthub-1](https://github.com/vangorade/MIGCL_code/assets/71941335/90a8b0eb-4d0b-4bfc-b5aa-e090d27e0b94)

