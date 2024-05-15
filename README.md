The official pytorch code repository for paper [MiGCL: Multi-granular Contrastive Learning for Self-supervised Pre-training]()

[embed]https://github.com/vangorade/MIGCL_code/files/15319722/MIGCL_fig_github.pdf[/embed]

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

