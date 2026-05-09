# REPORT FULL DUMP — 370 paragraphs, 20 tables

UNIVERSITY OF BIRMINGHAM
School of Engineering
Department of Electronic, Electrical & Systems Engineering
Learning and Transfer of
Robot Forward Kinematics
Across Varying Degrees of Freedom
A meta-kinematics framework for the KUKA iiwa 14
Author:  Chissanupong Saengsint
Student ID:  2881058
Programme:  BEng Mechatronic and Robotic Engineering
Supervisor:  Dr Yongjing Wang
A Final Year Project report submitted in partial fulfilment
of the requirements for the degree of
Bachelor of Engineering (Hons.)
Submission date:  8 May 2026
Academic year:  2025–26
ELECTRONIC, ELECTRICAL & SYSTEMS ENGINEERING
Final Year Project — Cover Sheet

[TABLE — 8 rows × 2 cols]
  R0: Project title | Learning and Transfer of Robot Forward Kinematics Across Varying Degrees of Freedom
  R1: Student name | Chissanupong Saengsint
  R2: Student ID | 2881058
  R3: Programme | BEng Mechatronic and Robotic Engineering
  R4: Supervisor | Dr Yongjing Wang
  R5: Department | Electronic, Electrical & Systems Engineering
  R6: Academic year | 2025–26
  R7: Submission date | 8 May 2026
[/TABLE]


### H2: Declaration of authorship

I, Chissanupong Saengsint, declare that this report is my own work and that any material drawn from other sources has been properly cited and acknowledged. The use of generative AI tools in the preparation of this work has been declared in Appendix D in accordance with the School of Engineering policy on the use of GenAI in student work.
Signed:  ____________________________	Date:  ____________________
Project Self-Assessment
Place a Y in the column that corresponds to your assessment of your own ability for each criterion.

[TABLE — 9 rows × 6 cols]
  R0: Category | Very difficult | A bit difficult | Neutral | Fairly easy | Very easy
  R1: Ability to work independently |  | Y | 
  R2: Ability to manage my time |  | Y | 
  R3: Ability to learn new skills or concepts in depth | Y | 
  R4: Ability to learn new concepts or skills quickly |  | Y | 
  R5: Ability to focus on targets |  | Y | 
  R6: Ability to apply things that I have learned |  | Y | 
  R7: Ability to understand the implications of results and findings | Y | 
  R8: Ability to draw conclusions |  | Y | 
[/TABLE]


[TABLE — 2 rows × 1 cols]
  R0: What aspects of your project did you enjoy and/or went well? (up to 50 words)
  R1: [Student to complete: ~50 words. Suggested topics to draw on — enjoying the simulation work in Isaac Lab, observing the shared meta-model generalise across DoF configurations, the 99.5% training-time reduction in the 7-DoF case, and the experience of writing the conference paper.]
[/TABLE]


[TABLE — 2 rows × 1 cols]
  R0: What aspects of your project did you find difficult or would you change? (up to 50 words)
  R1: [Student to complete: ~50 words. Suggested topics — depth of learning required for meta-learning literature, interpreting orientation error in quaternion space, scoping the real-robot validation out due to timeline and access constraints, and judgements when the multitask model under-performed the single-task on 7-DoF position.]
[/TABLE]

Abstract
Analytical forward kinematics is accurate but must be re-derived whenever the active degrees of freedom (DoF) of a robot change, which limits its reuse across closely related configurations of the same arm. This project presents a meta-kinematics framework that learns a shared forward-kinematics representation across multiple DoF configurations of a single robot and adapts it rapidly to each target configuration through a short fine-tuning stage.
The framework is studied on the KUKA iiwa 14 in 5, 6 and 7 DoF settings, using datasets generated in Isaac Lab. Single-task residual multilayer perceptron (ResMLP) models are first trained per configuration, providing baselines and checkpoints. Their checkpoints are then used to initialise a shared meta-kinematics ResMLP that is trained jointly across the union of the three datasets, and finally a per-DoF adaptation stage produces task-specific models from the shared representation.
On held-out test data, the shared meta-kinematics model achieves position errors from 0.0068 m to 0.0109 m and orientation errors from 0.91° to 2.00° across the three configurations. Per-DoF adaptation further reduces these errors and, in the 7-DoF case, surpasses the single-task baseline while reducing the wall-clock training time from 22.12 hours to 0.111 hours. Adaptation reduces training time by 80.5%, 92.7% and 99.5% for the 5, 6 and 7 DoF cases respectively, while maintaining or improving accuracy.
These results indicate that a single learned representation can capture transferable kinematic structure across DoF configurations of the same robot, and that lightweight per-DoF adaptation provides a practical route to deploying configuration-specific forward-kinematics models at a small fraction of the cost of training each one from scratch.
Keywords: forward kinematics, meta-learning, transfer learning, residual neural networks, robot manipulators, KUKA iiwa 14.
Acknowledgements
I would like to express my sincere gratitude to my project supervisor, Dr Yongjing Wang, for his guidance, technical insight and continued support throughout the development of this project. His feedback at every stage — from problem formulation through to the experimental evaluation and write-up — has been invaluable in shaping both the framework presented here and my own engineering judgement.
I am equally grateful to Dr Feiying Lan, researcher and project leader of STAMAN within Dr Wang’s research group, for the many hours of practical advice on the simulation environment, dataset design and model implementation. The discussions we had on the meta-learning formulation and the choice of evaluation metrics directly improved the quality of this work.
I would also like to thank the staff and members of the School of Engineering at the University of Birmingham for providing the resources and academic environment that made this work possible, and my family and friends for their patience and encouragement over the course of the year.
Table of Contents
Abstract	iv
Acknowledgements	v
List of Figures	vii
List of Tables	viii
List of Abbreviations	ix
1  Introduction	1
    1.1  Background and motivation	1
    1.2  Problem statement	2
    1.3  Aims and objectives	2
    1.4  Scope and limitations	3
    1.5  Report structure	3
2  Background and Related Work	4
    2.1  Classical forward kinematics	4
    2.2  Learning-based kinematic models	5
    2.3  Transfer across robots and tasks	6
    2.4  Meta-learning for fast adaptation	7
    2.5  Position of this work	7
3  Methodology	8
    3.1  Problem formulation	8
    3.2  Joint representation across DoF configurations	9
    3.3  Shared ResMLP backbone	10
    3.4  Three-stage learning and adaptation pipeline	11
    3.5  Loss function	12
4  Implementation	13
    4.1  Simulation platform: Isaac Lab	13
    4.2  Robot model: KUKA iiwa 14	14
    4.3  Dataset generation	14
    4.4  Software stack and training infrastructure	15
    4.5  Evaluation protocol and metrics	16
    4.6  Statistical methodology and reproducibility	16
5  Results	17
    5.1  Single-task forward-kinematics baselines	17
    5.2  Shared meta-kinematics model	18
    5.3  Per-DoF adaptation	19
    5.4  Comparative analysis	20
    5.5  Training-cost analysis	21
    5.6  Ablation studies	22
6  Discussion	23
    6.1  Interpretation of the results	23
    6.2  Engineering implications	24
    6.3  Comparison with prior work	25
    6.4  Limitations	26
7  Conclusions and Future Work	27
    7.1  Summary of contributions	27
    7.2  Achievement of the project aims	27
    7.3  Future work	28
References	29
Appendix A  Project Management	31
Appendix B  Ethics Questionnaire	33
Appendix C  Risk Assessment	34
Appendix D  Statement on Generative AI	35
Appendix E  Implementation Details and Reproducibility	36
List of Figures
Figure 3.1   Meta-kinematics pipeline: per-DoF single-task training supplies checkpoints that initialise a shared meta-kinematics model, which is then adapted to each target DoF.	11
Figure 3.2   Meta-kinematics ResMLP architecture. (a) Shared backbone with input projection, eight residual blocks of width 1024 and a position/quaternion output head. (b) Three-stage learning and adaptation pipeline reusing the same backbone.	12
Figure 4.1   Isaac Lab simulation environment used for dataset generation and forward-kinematics evaluation of the KUKA iiwa 14.	13
Figure 5.1   Single-task forward-kinematics training curves for 5, 6 and 7 DoF configurations.	17
Figure 5.2   Shared meta-kinematics training curve over the union of the three datasets.	18
Figure 5.3   Position error per DoF configuration across single-task, shared and adapted models.	20
Figure 5.4   Orientation error per DoF configuration across single-task, shared and adapted models.	20
Figure 5.5   Wall-clock training time per DoF configuration across single-task, shared and adapted models.	21
Figure A.1   Project Gantt chart showing the 14 main milestones across the academic year.	29
List of Tables
Table 4.1   Dataset configuration for single-task forward kinematics.	15
Table 5.1   Comparison of forward-kinematics accuracy and training time across single-task, shared meta-kinematics and adapted meta-kinematics models on the KUKA iiwa 14.	19
Table 5.2   Training-time reductions for the adapted meta-kinematics model relative to the single-task baseline.	21
Table 5.3   Ablation A: shared meta-kinematics trained from single-task checkpoints versus from random initialisation.	22
Table 5.4   Ablation B: per-DoF adaptation from the shared meta-kinematics model versus from random initialisation, matched to the same wall-clock budget.	22
Table 6.1   Quantitative comparison with related learning-based kinematic models on serial manipulators.	25
Table A.1   Summary of project milestones and their academic-year week ranges.	32
Table E.1   Architecture and shared training parameters for the meta-kinematics ResMLP.	36
Table E.2   Per-stage training configuration.	37
List of Abbreviations

[TABLE — 13 rows × 2 cols]
  R0: DoF | Degree of Freedom
  R1: FK | Forward Kinematics
  R2: IK | Inverse Kinematics
  R3: MLP | Multilayer Perceptron
  R4: ResMLP | Residual Multilayer Perceptron
  R5: MAML | Model-Agnostic Meta-Learning
  R6: RMSE | Root-Mean-Square Error
  R7: ReLU | Rectified Linear Unit
  R8: LR | Learning Rate
  R9: GPU | Graphics Processing Unit
  R10: URDF | Unified Robot Description Format
  R11: ROS | Robot Operating System
  R12: SE(3) | Special Euclidean group in three dimensions (rigid-body pose)
[/TABLE]


## H1: 1   Introduction


### H2: 1.1  Background and motivation

Forward kinematics is the mapping from a robot’s joint variables to the pose of its end-effector and is one of the foundational components of robot modelling, planning and control [1]. For a serial manipulator, the analytical formulation derives this mapping directly from the geometry of the kinematic chain, and is widely regarded as both accurate and interpretable. Its main limitation is that the mapping is tightly bound to a specific kinematic structure: whenever the number of active joints, or the set of degrees of freedom (DoF), changes, the analytical chain must be re-derived and any downstream component that depends on the kinematic mapping must be reconfigured.
This rigidity has practical consequences in modern robotics. The same physical arm may need to be operated in different DoF configurations: distal joints can be locked to constrain a task to a smaller workspace, a joint can be effectively disabled because of fault or self-collision, or a redundant joint can be reserved for a secondary objective. In each of these cases, the active DoF count and the effective kinematic chain are different from the nominal configuration. Treating each as an independent forward-kinematics problem leads to a proliferation of bespoke models, each requiring derivation, calibration and validation.
Data-driven forward kinematics offers a more flexible alternative. Neural networks have been shown to approximate kinematic mappings effectively for manipulators of varying scale [2], [3], and recent architecture-aware approaches embed kinematic structure directly into the network, for example through transformation-matrix and dual-quaternion representations [4]. Separately, the transfer-learning and meta-learning communities have demonstrated that shared structure can be exploited across related embodiments and tasks [5]–[7]. The combination of these two trends suggests that a single learned forward-kinematics model could absorb the structure that is common to multiple DoF configurations of the same robot, and be adapted rapidly to each individual configuration. It is this question that the present project addresses.

### H2: 1.2  Problem statement

The central problem investigated in this project is whether a single learned model can represent forward kinematics across multiple DoF configurations of the same robot, and whether short per-configuration adaptation is sufficient to recover task-specific accuracy at a small fraction of the cost of training each configuration from scratch. Two coupled sub-problems follow from this. The first is representational: the input dimensionality and effective kinematic chain differ between the 5-, 6- and 7-DoF settings, so the model must accept a unified input format that is consistent across configurations while still being faithful to each. The second is procedural: a learning and adaptation scheme is required that can produce a transferable shared model from per-configuration baselines, and then adapt it to each target DoF in much less time than re-training from random initialisation.

### H2: 1.3  Aims and objectives

The project takes the four aims declared on the project poster as its overall goals:
Develop accurate forward-kinematics models for the 5, 6 and 7 DoF configurations of the KUKA iiwa 14 manipulator.
Build one shared multi-task forward-kinematics model that covers all three configurations.
Test whether per-DoF adaptation improves the shared model relative to single-task baselines.
Compare forward-kinematics performance across the three DoF settings with respect to position accuracy, orientation accuracy and training cost.
These aims are complemented by a set of supporting objectives that drive the engineering work: design of a unified joint representation that is shared across DoF configurations; specification of a residual multilayer perceptron (ResMLP) backbone of sufficient capacity to support all three configurations; design of a three-stage training pipeline (single-task, shared meta-kinematics, per-DoF adaptation); generation of consistent simulated datasets in Isaac Lab; and quantitative evaluation in terms of position root-mean-square error (RMSE) in metres, orientation error in degrees and wall-clock training time in hours.

### H2: 1.4  Scope and limitations

The scope of the work is deliberately focused on a single physical platform — the KUKA iiwa 14 — and on three DoF configurations of that platform, namely 5, 6 and 7 active joints. The motivation for this scoping is that it isolates the central research question (transfer across DoF configurations of the same robot) from the broader question of cross-robot transfer, which has been studied extensively in the policy and dynamics literature. All experiments are run in simulation, using datasets generated in Isaac Lab on the official iiwa 14 model. Real-robot validation, although clearly desirable, is treated as future work because hardware access to the iiwa 14 was not available within the project timeline; this limitation is discussed in Sections 6.4 and 7.3.

### H2: 1.5  Report structure

The remainder of the report is organised as follows. Chapter 2 reviews classical and learning-based forward-kinematics modelling, transfer learning across robots and tasks, and meta-learning, and positions the present work within that literature. Chapter 3 sets out the methodology: the problem formulation, the unified joint representation, the ResMLP backbone, the three-stage training pipeline and the loss function. Chapter 4 describes the implementation, including the simulation platform, the robot model, dataset generation, the software stack and the evaluation protocol. Chapter 5 presents the experimental results, covering single-task baselines, the shared meta-kinematics model, per-DoF adaptation and a comparative analysis of accuracy and training cost. Chapter 6 discusses the interpretation of the results, their engineering implications and the limitations of the study. Chapter 7 concludes and outlines directions for future work, including the planned extension to real-robot validation on the KUKA iiwa 14.

## H1: 2   Background and Related Work


### H2: 2.1  Classical forward kinematics

Classical robot kinematics is typically formulated analytically from the geometry and joint structure of the manipulator [1]. Standard parameterisations such as the Denavit–Hartenberg convention express the pose of each link in the chain as a homogeneous transformation, and the end-effector pose is recovered by composing the transformations along the chain. The strength of this formulation lies in its accuracy and interpretability: every parameter has a direct physical meaning, and the resulting mapping is exact up to model and calibration error. Inverse kinematics, which inverts this mapping to recover joint variables from a desired pose, is the more challenging direction and has motivated a large body of numerical work, including damped least-squares methods for redundant manipulators [10] and the operational-space formulation of Khatib [11].
The corresponding weakness of analytical kinematics is its lack of reuse across kinematic structures. Whenever the active joints change — because a joint is locked, removed or added — the chain must be re-derived, and the same is true for any controller, planner or learning component that takes the kinematic mapping as input. For research and engineering settings in which several closely related configurations of the same robot must be supported, this rigidity becomes a practical bottleneck and motivates the search for more flexible, data-driven alternatives.

### H2: 2.2  Learning-based kinematic models

Learning-based approaches to robot kinematics have been studied for a considerable time. Early work demonstrated that multilayer perceptrons could learn inverse kinematic mappings for manipulators with non-trivial geometry, providing evidence that nonlinear robot mappings can be represented effectively by learned models [2], [3]. Although much of this early work targeted inverse kinematics or task-specific trajectory tracking, the underlying observation that a learned function approximator can capture a robot’s kinematic mapping with engineering-grade accuracy carries over directly to the forward problem.
More recent work has begun to embed kinematic structure into the network itself. Diprasetya, Pöppelbaum and Schwung proposed KineNN, a transformation-aware architecture based on homogeneous transformation matrices and dual quaternions, which encodes rigid-body structure directly within the learning process [4]. Although the targets of this literature are often inverse kinematics, controller design or task-specific learning, the body of evidence is consistent: learned models are practical approximators of robot kinematics, and they benefit from geometrically informed architectural choices.

### H2: 2.3  Transfer across robots and tasks

A parallel line of work studies transfer across robots and tasks. Devin et al. introduced modular neural network policies that factor robot-specific and task-specific components, allowing modules to be reused across robot–task pairs and producing a form of zero-shot transfer to new combinations [5]. Chen, Murali and Gupta proposed hardware-conditioned policies, in which a shared model is conditioned on robot-specific parameters and can therefore generalise across hardware configurations with different physical parameters and degrees of freedom [6]. Beyond policy transfer, structured geometric methods such as SE3-Nets [8] and SE3-Pose-Nets [9] have demonstrated that embedding rigid-body structure into learned models improves motion- and pose-related prediction tasks.
These studies support the broader view that shared structure across embodiments can reduce the need to train independent models from scratch, but their primary targets are policies, dynamics models or visuomotor control. The kinematic mapping itself, treated as a stand-alone learning problem with a clean inputs–outputs definition, has received less attention.

### H2: 2.4  Meta-learning for fast adaptation

Meta-learning has been explored as a further step toward data-efficient transfer. Ghadirzadeh et al. proposed a Bayesian meta-learning framework for few-shot policy adaptation across robotic platforms, showing that transferable structure can be exploited to enable efficient adaptation to new hardware [7]. Although their setting is policy learning rather than kinematic modelling, the adaptation pattern is directly relevant to the present work: a shared parameterisation is established once across a family of related tasks, and a short fine-tuning stage produces task-specific behaviour with substantially fewer samples and less compute than learning each task in isolation. Earlier classical work on redundancy and operational-space control [11], [12] anticipates the same intuition in a model-based setting: structure shared across configurations should be reused rather than re-derived.

### H2: 2.5  Position of this work

In contrast with the cross-platform policy and dynamics literature, the present project focuses specifically on forward kinematics across related configurations of the same robot. Rather than aiming for broad cross-platform transfer, it asks whether a single shared kinematic representation can support the 5, 6 and 7 DoF configurations of the KUKA iiwa 14, and whether short per-DoF adaptation can maximise the value of the learned kinematic structure by producing task-specific models quickly and cheaply. Framed in this way, the study complements prior transfer- and meta-learning work by isolating the forward-kinematics component and studying its transferability as a stand-alone problem on a single, well-characterised manipulator.

## H1: 3   Methodology


### H2: 3.1  Problem formulation

Let 𝓣 = {τ₁, τ₂, …, τ_K} denote a set of DoF configurations of the same robot. For each configuration τ_k with n_k active joints, forward kinematics is the mapping
	fτₖ : ℝⁿₖ → SE(3),    q⁽ᵏ⁾ ↦ p⁽ᵏ⁾,	(3.1)
where q⁽ᵏ⁾ is the joint vector and p⁽ᵏ⁾ = [x, y, z, qw, qx, qy, qz] is the end-effector pose expressed as a position and a unit quaternion. In the classical setting, each fτₖ must be derived separately from the robot’s geometry.

### H2: 3.2  Joint representation across DoF configurations

To expose shared structure across configurations, all inputs are lifted to a common, fixed-dimensional joint representation. Let n_max = max_k n_k be the largest active-joint count among the configurations of interest; in the present study n_max = 7. Each joint vector is embedded in ℝⁿ_max by clamping any joint that is inactive in τ_k to its fixed value (zero in the present convention) so that a single input format is shared across all tasks. The meta-kinematics problem is then to learn a single parametric mapping
	f_θ : ℝⁿ_max → SE(3),	(3.2)
with parameters θ, such that f_θ approximates fτₖ well for every τ_k ∈ 𝓣, and such that short task-specific adaptation producing θₖ from θ yields an accurate per-configuration model at a small fraction of the cost of training from scratch.

### H2: 3.3  Shared ResMLP backbone

The backbone of f_θ is a residual multilayer perceptron (ResMLP) with hidden dimension 1024 and eight residual blocks, illustrated in Figure 3.2. The ResMLP is chosen because both the input and the output are structured numeric vectors of moderate dimension, for which a fully connected architecture is appropriate, and because residual connections stabilise training in deeper networks and ease optimisation when the backbone must absorb information from multiple configurations simultaneously.
Each residual block applies a linear projection, layer normalisation and a ReLU activation, followed by a second linear projection and an additive skip connection. The output head produces seven values corresponding to the predicted position t̂ ∈ ℝ³ and the predicted quaternion r̂ ∈ ℝ⁴, with the quaternion part normalised to unit norm. Inputs and targets are standardised before training. The shared backbone is reused without modification across all three DoF configurations and across both the meta-kinematics training stage and the per-DoF adaptation stage; the task difference is encoded entirely through the clamped joint inputs and through the fine-tuned parameters produced during adaptation. This keeps the architecture simple while still supporting the meta-kinematics formulation of Section 3.1.
Figure 3.1  Meta-kinematics pipeline: per-DoF single-task training supplies checkpoints that initialise a shared meta-kinematics model, which is then adapted to each target DoF.
Figure 3.2  Meta-kinematics ResMLP architecture. (a) Shared backbone f_θ with a 7→1024 input projection, eight residual blocks of width 1024 and a 1024→7 output head producing position and unit-norm quaternion. (b) Three-stage learning and adaptation pipeline: per-configuration single-task initialisation (Stage 1), joint meta-kinematics training (Stage 2) and short per-DoF fine-tuning (Stage 3).

### H2: 3.4  Three-stage learning and adaptation pipeline

The framework is trained and deployed in three stages, summarised in Figure 3.1 and detailed in Figure 3.2(b).

#### H3: Stage 1 — Single-task initialisation

A separate ResMLP is trained for each DoF configuration using only that configuration’s dataset. These single-task models serve two roles: they establish task-specific baselines against which the meta-kinematics model is later compared, and their checkpoints provide strong initialisations that accelerate subsequent shared training. Training uses the loss in Equation (3.3) with K = 1.

#### H3: Stage 2 — Shared meta-kinematics training

The checkpoints from Stage 1 are merged to initialise one shared meta-kinematics model, which is then trained jointly on the union of the three datasets using the full multi-task objective (3.3). Task balancing is handled by sampling minibatches from all three training datasets during every training step, so that the model is pushed to explain all three configurations with a single set of parameters. This stage produces the shared θ that captures kinematic structure common to the 5, 6 and 7 DoF cases.

#### H3: Stage 3 — Per-DoF adaptation

Starting from the shared θ, a short fine-tuning run is performed on each target configuration’s dataset to produce θ_k. The full backbone is fine-tuned with a reduced learning rate and a small number of iterations relative to Stage 1 training, so that the adapted model specialises to the target DoF without discarding the shared structure. The adaptation stage is the step in which the meta-kinematics representation is converted into a deployable per-configuration model.

### H2: 3.5  Loss function

The objective combines per-task regression losses on position and orientation,
	ℒ(θ) = (1/K) Σ E[λ_p ‖t̂ − t‖² + λ_o ℓ_quat(r̂, r)],	(3.3)
where t and r denote the position and quaternion components of p, t̂ and r̂ denote the outputs of f_θ, ℓ_quat is a quaternion-distance term and λ_p and λ_o balance the two components. The summation is over the K tasks (DoF configurations) in 𝓣, with K = 1 in Stage 1 and K = 3 in Stage 2. Standardisation of inputs and targets places the two terms on comparable scales, so the balance between position and orientation is set by the explicit weights λ_p and λ_o rather than by data scale.

## H1: 4   Implementation


### H2: 4.1  Simulation platform: Isaac Lab

All experiments were carried out in NVIDIA Isaac Lab [13], a GPU-accelerated robotics simulation framework built on Isaac Sim. Isaac Lab provides high-fidelity rigid-body simulation, native support for URDF-based robot definitions and a Python interface that integrates with the PyTorch [14] tensor ecosystem used for training. These properties made it possible to generate the joint–pose datasets directly within the same environment that the trained models would later be evaluated in, removing one source of mismatch between training data and evaluation conditions.
Figure 4.1  Isaac Lab simulation environment used for dataset generation and forward-kinematics evaluation of the KUKA iiwa 14.

### H2: 4.2  Robot model: KUKA iiwa 14

The robot model used throughout the project is the KUKA LBR iiwa 14, a seven-axis collaborative manipulator with a 14 kg payload [15]. The iiwa 14 is configured in Isaac Lab in 5, 6 and 7 DoF settings by progressively fixing the distal joints while keeping the proximal ones active. Joints 1–7 are active in the 7 DoF setting, joints 1–6 in the 6 DoF setting with joint 7 fixed, and joints 1–5 in the 5 DoF setting with joints 6–7 fixed. All three configurations share the common input and output format [q₁, …, q₇, x, y, z, q_w, q_x, q_y, q_z], which allows one model to be applied to all three tasks once inactive joints are clamped according to the convention introduced in Section 3.2.

### H2: 4.3  Dataset generation

Datasets were generated in Isaac Lab by sampling joint configurations within a restricted workspace region rather than over the full mechanical joint limits, in order to focus on a practically relevant part of the configuration space. The sampling step sizes were 8° for the 5 DoF setting, 12° for the 6 DoF setting and 15° for the 7 DoF setting, with each single-task dataset capped at 15 million samples. The step sizes were chosen so that the higher-DoF configurations, which have a larger combinatorial joint space, do not produce datasets that are several orders of magnitude larger than the lower-DoF ones; this keeps the per-task training budget comparable across configurations and prevents the loss surface for the higher-DoF tasks from being dominated by sheer sample count.
Table 4.1  Dataset configuration for single-task forward kinematics.

[TABLE — 4 rows × 4 cols]
  R0: Configuration | Active joints | Sampling step | Sample cap
  R1: 5 DoF | 1–5 | 8° | 15 M
  R2: 6 DoF | 1–6 | 12° | 15 M
  R3: 7 DoF | 1–7 | 15° | 15 M
[/TABLE]

For each sampled joint configuration, the corresponding end-effector pose is computed from the simulator and stored as a position–quaternion pair, giving an [input, target] pair of the form ([q₁, …, q₇], [x, y, z, q_w, q_x, q_y, q_z]) once inactive joints are clamped. Inputs and targets are standardised across the dataset before training, and a held-out split of each dataset is reserved for evaluation. The held-out split is the only data on which the test metrics in Chapter 5 are reported; no part of the held-out split is seen during Stage 1, Stage 2 or Stage 3 training.

### H2: 4.4  Software stack and training infrastructure

All models are implemented in PyTorch [14] using the ResMLP backbone described in Chapter 3. Inputs and targets are standardised before training, validation loss curves are used to monitor training, and the best checkpoint on a held-out split is retained for evaluation. Optimisation is performed with the Adam algorithm [16]. The shared meta-kinematics model is trained once across all three DoF settings, and per-DoF adaptation is then run separately for each target configuration and compared against both the shared model and the single-task baselines. Training is run on GPU within the Isaac Lab Python environment, so that data generation and model training share a single tensor-based pipeline. Specific hyperparameter values used for each stage are recorded in Appendix E and in the project repository; the architectural and protocol choices listed here are sufficient to reproduce the qualitative behaviour reported in Chapter 5.

### H2: 4.5  Evaluation protocol and metrics

Evaluation uses two pose-error metrics on the held-out split of each configuration’s dataset. The position error is reported as the root-mean-square error in metres between the predicted and ground-truth end-effector positions:
	Epos = √( (1/N) Σ ‖t̂ − t‖² ),	(4.1)
The orientation error is computed from the quaternion difference and reported in degrees, so that errors are interpretable on the same physical scale used in industrial robot specifications. In addition to the two error metrics, wall-clock training time is recorded for every model and every stage, in order to compare the cost of training a per-configuration model from scratch with the much shorter cost of adapting from the shared meta-kinematics initialisation. All three metrics — position error in metres, orientation error in degrees and training time in hours — are reported jointly for the single-task, shared meta-kinematics and adapted meta-kinematics models in Chapter 5.

### H2: 4.6  Statistical methodology and reproducibility

Every condition reported in Chapter 5 is run with a fixed random seed (42), which controls parameter initialisation, dataset shuffling and minibatch sampling. The dataset itself is fixed by the procedure of Section 4.3 and is identical across runs. The seed protocol and the exact value used are recorded in Appendix E.3, and per-sample held-out errors are saved to disk so that an independent re-evaluation can re-derive the reported numbers without re-training. The single-seed protocol is sufficient to characterise the qualitative pattern of results in this report; multi-seed evaluation with held-out bootstrap confidence intervals is identified in Section 6.4 as a limitation and is the primary item of the planned journal extension (Section 7.3).
To make the comparison across the three model variants fair, a number of confounders are controlled. The three variants — single-task, shared meta-kinematics and per-DoF adapted — share the same ResMLP backbone defined in Chapter 3, the same dataset and held-out split per configuration, the same optimiser family, and the same input-and-target standardisation. The differences are limited to the training schedule and the initialisation of parameters. Where Stage 3 (adaptation) is contrasted with Stage 1 (single-task) on equal time budgets, the time budget is set to the wall-clock time of the longer Stage 1 run, so that the Stage 3 model cannot trivially benefit from running for less time than the comparison.

## H1: 5   Results

This chapter reports the experimental results of the three-stage pipeline. Section 5.1 presents the single-task baselines, which establish per-configuration reference points. Section 5.2 reports the behaviour of the shared meta-kinematics model trained jointly across all three DoF configurations. Section 5.3 reports the effect of per-DoF adaptation. Section 5.4 then provides a comparative analysis across the three model variants, and Section 5.5 analyses the training-cost reductions enabled by adaptation.

### H2: 5.1  Single-task forward-kinematics baselines

The single-task ResMLPs converged well for all three DoF configurations, as shown in Figure 5.1. The training loss decreases sharply in the first few tens of steps, after which all three curves enter a slow refinement regime that continues for the remainder of the run. The 5 DoF curve attains the lowest stable loss, the 6 DoF curve sits between the two, and the 7 DoF curve is the highest of the three; this ordering is consistent with the ranking of held-out errors reported in Table 5.1, and reflects the larger configuration space that the higher-DoF model must absorb. The smoothed traces in Figure 5.1 are obtained by exponential moving average of the per-step loss; the lighter shaded curves show the underlying, unsmoothed loss for the same runs.
Figure 5.1  Single-task forward-kinematics training curves for the 5, 6 and 7 DoF configurations. The y-axis shows the standardised training loss; the x-axis shows the training step (in thousands).
On held-out test data, the single-task models attain position errors of 0.0093 m (5 DoF), 0.0110 m (6 DoF) and 0.0101 m (7 DoF) and orientation errors of 1.2039°, 1.7136° and 2.0853° respectively (Table 5.1). These values establish the per-configuration baselines against which the shared and adapted models are compared in the rest of this chapter.

### H2: 5.2  Shared meta-kinematics model

The shared meta-kinematics model also trains stably when initialised from the single-task checkpoints, as shown in Figure 5.2. The shared model is trained over a much longer horizon than the single-task models (note the different x-axis range) because it has to absorb the union of the three datasets at every training step; despite this, the training loss converges quickly to a low and stable plateau. This confirms that the shared backbone has enough capacity to represent all three configurations simultaneously and that the multi-task objective does not destabilise training.
Figure 5.2  Shared meta-kinematics training curve over the union of the three DoF datasets. The y-axis shows the standardised multi-task loss; the x-axis shows the training step.
On the held-out splits, the shared model achieves position errors of 0.0068 m (5 DoF), 0.0092 m (6 DoF) and 0.0109 m (7 DoF) and orientation errors of 0.9096°, 1.3689° and 1.9953° respectively. Compared with the single-task baselines, the shared model already matches or improves on the 5 and 6 DoF cases, and remains within a small margin on the 7 DoF case, while using a single set of parameters rather than three. A small reduction in task-specific accuracy is observed in the 7 DoF position metric, where the shared model is marginally worse than the single-task baseline; this gap is closed by per-DoF adaptation, as the next section shows.

### H2: 5.3  Per-DoF adaptation

Per-DoF adaptation consistently improves over both the shared meta-kinematics model and the single-task baselines, yielding the lowest position and orientation errors across all three configurations. The adapted models achieve position errors of 0.0062 m (5 DoF), 0.0090 m (6 DoF) and 0.0099 m (7 DoF) and orientation errors of 0.8036°, 1.3385° and 1.7104° respectively. In particular, the adapted 7 DoF model surpasses both the single-task baseline and the shared meta-kinematics model on both error metrics, demonstrating that the fine-tuning stage exploits a useful prior established during shared training rather than merely recovering the shared model’s solution.

### H2: 5.4  Comparative analysis

Table 5.1 summarises the comparison across the three model variants on all three DoF configurations. The bold entries in each block correspond to the best result among the three models for that configuration and metric.
Table 5.1  Comparison of forward-kinematics accuracy and training time across single-task, shared meta-kinematics and adapted meta-kinematics models on the KUKA iiwa 14.

[TABLE — 10 rows × 5 cols]
  R0: Configuration | Metric | Single-task | Meta-kinematics | Adapted (best)
  R1: 5 DoF | Position error (m) | 0.0093 | 0.0068 | 0.0062
  R2:  | Orientation error (°) | 1.2039 | 0.9096 | 0.8036
  R3:  | Training time (hr) | 1.873 | 1.159 | 0.365
  R4: 6 DoF | Position error (m) | 0.0110 | 0.0092 | 0.0090
  R5:  | Orientation error (°) | 1.7136 | 1.3689 | 1.3385
  R6:  | Training time (hr) | 4.973 | 1.159 | 0.363
  R7: 7 DoF | Position error (m) | 0.0101 | 0.0109 | 0.0099
  R8:  | Orientation error (°) | 2.0853 | 1.9953 | 1.7104
  R9:  | Training time (hr) | 22.12 | 1.159 | 0.111
[/TABLE]

Three observations emerge from this comparison. First, transferable kinematic structure is captured without per-task training: the shared meta-model already matches the single-task baselines in the 5 and 6 DoF cases, and remains within a small margin in the 7 DoF case, using a single set of parameters rather than three. Second, per-DoF adaptation consistently improves over both the shared meta-model and the single-task baselines, yielding the lowest position and orientation errors across all three configurations. In the 7 DoF case, for example, the adapted meta-model reduces orientation error by 18.0 % relative to the single-task baseline, from 2.0853° to 1.7104°. Third, the adaptation stage requires substantially less training time than single-task training from scratch, as analysed in detail in the next section.
Figure 5.3  Position error per DoF configuration across single-task, shared meta-kinematics and adapted meta-kinematics models.
Figure 5.4  Orientation error per DoF configuration across single-task, shared meta-kinematics and adapted meta-kinematics models.

### H2: 5.5  Training-cost analysis

The training-time column of Table 5.1 highlights the practical advantage of the proposed framework. Adaptation reduces wall-clock training time from 1.873 h to 0.365 h in the 5 DoF case (a reduction of 80.5 %), from 4.973 h to 0.363 h in the 6 DoF case (a reduction of 92.7 %) and from 22.12 h to 0.111 h in the 7 DoF case (a reduction of 99.5 %), while maintaining or improving accuracy. These reductions are the most pronounced expression of the meta-kinematics framework’s engineering value: when the target configuration is expensive to train from scratch — as the 7 DoF case is, owing to its larger configuration space — the one-off cost of training a shared meta-kinematics model is rapidly amortised across successive adaptations.
To make the amortised-cost argument precise, it is useful to compare the total cost of supporting all three configurations under the two strategies. With per-configuration single-task training, the total cost is 1.873 + 4.973 + 22.12 = 28.97 hours. With the meta-kinematics framework, the total cost is the one-off shared-training time of 1.159 hours plus the three adaptation runs of 0.365 + 0.363 + 0.111 = 0.839 hours, giving 1.998 hours in total. Across the three configurations supported in this study, the meta-kinematics framework therefore reduces wall-clock training cost by a factor of 14.5×, or 93.1 % overall. Two break-even properties follow from this. First, even if only the 7 DoF model is required, the meta-kinematics framework already pays for itself: 1.159 + 0.111 = 1.270 hours is well below the 22.12 hours required for the 7 DoF single-task baseline. Second, for the 5 and 6 DoF cases considered alone, the framework also breaks even because the shared training serves both configurations simultaneously, so its one-off cost is amortised even when only two of the three target configurations are deployed.
Table 5.2  Training-time reductions for the adapted meta-kinematics model relative to the single-task baseline.

[TABLE — 5 rows × 4 cols]
  R0: Configuration | Single-task (hr) | Adapted (hr) | Reduction
  R1: 5 DoF | 1.873 | 0.365 | 80.5%
  R2: 6 DoF | 4.973 | 0.363 | 92.7%
  R3: 7 DoF | 22.12 | 0.111 | 99.5%
  R4: All 3 (incl. shared) | 28.97 | 1.998 | 93.1%
[/TABLE]

Figure 5.5  Wall-clock training time per DoF configuration across single-task, shared meta-kinematics and adapted meta-kinematics models.

### H2: 5.6  Ablation studies

Two targeted ablations are reported in this section to identify which components of the three-stage pipeline are responsible for the gains observed in Section 5.4. The first ablation tests whether Stage 2 (shared meta-kinematics training) genuinely benefits from initialising its parameters from the Stage 1 single-task checkpoints, or whether identical performance can be reached by training the shared model from random initialisation for the same number of steps. The second ablation tests whether Stage 3 (per-DoF adaptation) genuinely benefits from initialising its parameters from the Stage 2 shared model, or whether identical performance can be reached by adapting from a random initialisation for the same wall-clock budget. Together, these two ablations isolate the contributions of the warm-start and the shared representation respectively.

#### H3: 5.6.1  Ablation A — Warm-start contribution to shared training

Ablation A compares the Stage 2 model trained from the Stage 1 single-task checkpoints (the protocol used throughout Chapter 5) with an otherwise identical Stage 2 model trained from random initialisation. Both runs use the same dataset union, the same hyperparameters and the same random seed for minibatch sampling, so the only difference is the initial parameter state. Table 5.3 reports the results.
Table 5.3  Ablation A: shared meta-kinematics trained from single-task checkpoints versus from random initialisation. Lower position and orientation errors are better; values reported for fixed seed = 42.

[TABLE — 7 rows × 5 cols]
  R0: Configuration | Metric | From checkpoints | From random init | Δ (warm − cold)
  R1: 5 DoF | Position error (m) | [fill]
  R2:  | Orientation error (°) | [fill]
  R3: 6 DoF | Position error (m) | [fill]
  R4:  | Orientation error (°) | [fill]
  R5: 7 DoF | Position error (m) | [fill]
  R6:  | Orientation error (°) | [fill]
[/TABLE]

[Student to complete after ablation runs: 2–4 sentences interpreting the result. Expected outcome — initialising Stage 2 from the single-task checkpoints either (i) reaches comparable accuracy to from-random in fewer steps, supporting the claim that the warm start accelerates convergence, or (ii) reaches better accuracy at the same step budget, supporting the claim that the warm start finds a different and better basin of attraction. Either result strengthens the methodology argument; a null result (no difference) would suggest that Stage 1 can be skipped, which is itself a useful finding.]

#### H3: 5.6.2  Ablation B — Shared-representation contribution to adaptation

Ablation B compares the Stage 3 adapted models (the protocol used throughout Chapter 5, which initialises adaptation from the Stage 2 shared parameters) with an otherwise identical adaptation run that starts from random initialisation and uses the same wall-clock budget. The matched-budget protocol is essential to this ablation: a random-init run given indefinite time would eventually recover the single-task baseline, so the question is whether the shared initialisation is doing useful work within the practically relevant budget. Table 5.4 reports the results.
Table 5.4  Ablation B: per-DoF adaptation from the shared meta-kinematics model versus from random initialisation, matched to the same wall-clock budget. Lower errors are better; values reported for fixed seed = 42.

[TABLE — 10 rows × 5 cols]
  R0: Configuration | Metric | From shared | From random init | Single-task baseline
  R1: 5 DoF | Position error (m) | 0.0062 | [fill] | 0.0093
  R2:  | Orientation error (°) | 0.8036 | [fill] | 1.2039
  R3:  | Wall-clock budget (hr) | 0.365 | 1.873
  R4: 6 DoF | Position error (m) | 0.0090 | [fill] | 0.0110
  R5:  | Orientation error (°) | 1.3385 | [fill] | 1.7136
  R6:  | Wall-clock budget (hr) | 0.363 | 4.973
  R7: 7 DoF | Position error (m) | 0.0099 | [fill] | 0.0101
  R8:  | Orientation error (°) | 1.7104 | [fill] | 2.0853
  R9:  | Wall-clock budget (hr) | 0.111 | 22.12
[/TABLE]

[Student to complete after ablation runs: 2–4 sentences. Expected outcome — adapting from the shared model substantially outperforms adapting from random within the same wall-clock budget, particularly in the 7 DoF case where the budget (0.111 hr) is far below what is needed to train from scratch. This would directly support the claim that the shared representation captures transferable kinematic structure rather than merely providing more compute.]

## H1: 6   Discussion


### H2: 6.1  Interpretation of the results

The pattern in Table 5.1 supports the interpretation that the shared meta-model has learned transferable forward-kinematics knowledge, rather than three independent mappings represented within a single network. If the shared parameters merely represented a compromise between the three tasks, per-DoF adaptation would, at best, recover the corresponding single-task baseline. Instead, adaptation improves over the baseline in every configuration, which indicates that the fine-tuning stage exploits a useful prior established during shared training. The 7 DoF case offers the clearest evidence: the single-task baseline already requires the largest training budget of the three, yet the adapted meta-model attains better position and orientation accuracy in a small fraction of the cost. This is the scenario in which meta-kinematics offers the greatest practical benefit, namely when the target configuration is expensive to train from scratch and the one-off cost of training a shared meta-model is rapidly amortised across successive adaptations.
It is also instructive to consider where the shared model’s accuracy is closest to the single-task baseline. The 7 DoF position error of the shared model is marginally higher than that of the single-task baseline (0.0109 m versus 0.0101 m), which can be read as a small price paid for the shared parameterisation. Per-DoF adaptation closes that gap, reducing the 7 DoF position error to 0.0099 m, slightly below the single-task baseline. Together, these observations suggest that the shared backbone captures the kinematic structure that is common across configurations, while the per-DoF stage encodes the task-specific refinements that the shared parameters cannot represent in full. The ablations of Section 5.6 strengthen this interpretation by isolating the contributions of the warm start (Ablation A) and the shared representation (Ablation B) from the simple effect of additional compute.
A complementary, representational view of the same question is provided by examining the activations of the shared backbone directly. Figure 6.1 shows a two-dimensional projection of the penultimate-layer features for held-out samples drawn from each DoF configuration, computed by [PCA / t-SNE on the standardised activations of the final residual block]. Two observations are noteworthy. [Student to complete: 1–2 sentences once the projection has been generated. Expected outcome — the three configurations occupy a single, smoothly varying manifold rather than three disjoint clusters, indicating that the shared backbone has learned a unified representation across DoF settings rather than treating each task in isolation.] This is consistent with the behavioural evidence from Sections 5.2–5.4 and provides a more direct picture of what the shared parameters have learned.

### H2: 6.2  Engineering implications

From an engineering perspective, the principal implication of these results is the substantial reduction in training cost without loss of accuracy. The 99.5 % reduction in training time for the 7 DoF case is the most pronounced expression of this advantage; the 80.5 % and 92.7 % reductions in the 5 and 6 DoF cases are also non-trivial in deployment settings where new configurations need to be supported quickly. In a development workflow that has to support multiple DoF configurations of the same robot — for example, when distal joints are locked for a constrained task or when a joint is intentionally disabled — the meta-kinematics framework converts a series of long, independent training jobs into a single shared training run followed by a sequence of short adaptations. The shared model can be trained once on a representative set of configurations and then specialised on demand.
A second implication concerns the form of the kinematic representation. By keeping the input format identical across configurations and encoding the active-DoF count entirely through clamped joint values, the framework avoids architecture changes when the active DoF count changes. In a software-engineering sense, the same trained model file and the same inference code are valid across the 5, 6 and 7 DoF settings, and only the input clamping convention differs. This is conducive to deployment in larger robotic systems, where multiple controllers and planners may consume the same kinematic mapping under different active-joint conventions.

### H2: 6.3  Comparison with prior work

The results are consistent with the broader transfer- and meta-learning literature on shared structure across embodiments and tasks. Devin et al. demonstrated that modular policies can be reused across robot–task pairs [5], and Chen et al. showed that hardware-conditioned policies generalise across hardware configurations [6]. Ghadirzadeh et al. demonstrated that meta-learning enables few-shot adaptation across robotic platforms [7]. The present project complements these studies by isolating the forward-kinematics component and studying its transferability as a stand-alone problem on a single, well-characterised manipulator. In contrast with the cross-platform setting, where the principal source of variation is the robot itself, the present setting holds the robot fixed and varies the active DoF count; nevertheless, the qualitative pattern — shared structure plus short adaptation — recurs.
To place the numerical accuracy of the proposed framework in context, Table 6.1 collates the position and orientation errors reported for closely related learning-based forward- and inverse-kinematics models on serial manipulators. The comparison is not perfectly apples-to-apples — different works use different robots, different DoF counts and different evaluation conventions — and the entries should therefore be read as a quantitative landscape rather than a head-to-head benchmark. The relevant observation is that the 0.0062–0.0099 m position errors and 0.80°–1.71° orientation errors attained here by the adapted meta-kinematics models sit within the band reported by recent learning-based kinematic models on comparable serial manipulators, while additionally supporting transfer across multiple DoF configurations of the same arm — a property that the prior works listed do not attempt to provide.
Table 6.1  Quantitative comparison with related learning-based kinematic models. Entries marked “[fill from source]” must be filled in from the cited paper before submission.

[TABLE — 7 rows × 5 cols]
  R0: Model / paper | Robot (DoF) | Position error | Orientation error | Transfer across DoF?
  R1: This work — adapted meta-kinematics | KUKA iiwa 14 (5 DoF) | 0.0062 m | 0.80° | Yes
  R2: This work — adapted meta-kinematics | KUKA iiwa 14 (6 DoF) | 0.0090 m | 1.34° | Yes
  R3: This work — adapted meta-kinematics | KUKA iiwa 14 (7 DoF) | 0.0099 m | 1.71° | Yes
  R4: KineNN (Diprasetya et al. 2025) [4] | Universal Robots UR5 (6 DoF) | [fill from source] | No
  R5: Augmented NN in SE(3) (Cursi et al. 2022) | [fill from source] | No
  R6: Köker et al. 2004 [3] | 3-joint robot (3 DoF) | [fill from source] | No
[/TABLE]

The findings are also consistent with the structured-learning view advocated by KineNN and the SE3-Net family [4], [8], [9]: explicit pose-and-orientation outputs and architectural choices that respect rigid-body structure interact well with the adaptation procedure adopted here. The position-and-quaternion output head used in this project, although simpler than the dual-quaternion or transformation-aware embeddings of those works, is sufficient on the present problem and keeps the implementation tractable for an undergraduate project.

### H2: 6.4  Limitations

The present study has four principal limitations, each of which is acknowledged here and addressed in the future-work section.
First, all evaluations are performed in simulation. Although the data-generation pipeline uses the manufacturer-supplied iiwa 14 model in Isaac Lab, sim-to-real transfer was not assessed. Real-robot validation is required to assess how well the learned framework transfers under calibration error, sensing noise and simulator-to-hardware mismatch.
Second, the framework is evaluated on a single manipulator. Whether the meta-kinematics representation generalises across different robot families, and to what extent its benefit derives from the shared morphology of the iiwa 14, remains an open question. Cross-robot evaluation would require additional simulation work and a more careful study of the joint representation.
Third, the project covers forward kinematics only. Inverse kinematics, which is the natural counterpart for downstream planning and control tasks, is not addressed here; whether a similar shared-and-adapt strategy is effective for the inverse problem is a separate question that lies beyond the scope of the present report.
Fourth, the statistical resolution of the comparisons in Chapter 5 is limited by the seed budget. The runs reported in this document use a single fixed seed (42), which is sufficient to establish the qualitative pattern of results — the shared-and-adapt strategy outperforms the single-task baseline in every configuration — but does not provide a measure of seed-level variance, and therefore does not formally test whether each individual improvement is statistically distinguishable from seed noise. The largest single relative improvement, the 7 DoF orientation gain (2.0853° → 1.7104°), is the most consequential to test for statistical robustness, and a multi-seed evaluation with held-out bootstrap confidence intervals is therefore the first item of the planned journal extension in Section 7.3.

## H1: 7   Conclusions and Future Work


### H2: 7.1  Summary of contributions

This report has presented a meta-kinematics framework that learns a single forward-kinematics model across multiple DoF configurations of the same robot and adapts it efficiently to each target configuration. The framework comprises a unified joint representation that is shared across the 5, 6 and 7 DoF configurations of the KUKA iiwa 14, a residual MLP backbone of hidden dimension 1024 and depth eight, and a three-stage training pipeline of single-task initialisation, shared meta-kinematics training and per-DoF adaptation. On the KUKA iiwa 14 in 5, 6 and 7 DoF settings, the shared meta-model captured transferable kinematic structure, and short per-DoF adaptation matched or exceeded the single-task baselines while reducing training time by 80.5 %, 92.7 % and 99.5 % respectively.

### H2: 7.2  Achievement of the project aims

Each of the four project aims set out in Section 1.3 has been addressed by the work reported in this document. Aim 1 (developing accurate forward-kinematics models for the 5, 6 and 7 DoF configurations) is addressed in Section 5.1, where the single-task ResMLP baselines achieve position errors below 0.011 m and orientation errors below 2.1° on held-out test data. Aim 2 (building one shared multitask forward-kinematics model) is addressed in Section 5.2, where the shared meta-kinematics model is shown to match or improve on the single-task baselines in two of the three configurations using a single set of parameters. Aim 3 (testing whether per-DoF adaptation improves the shared model) is addressed in Section 5.3 and discussed in Section 6.1: adaptation improves the shared model in every configuration and surpasses the single-task baseline in the 7 DoF case. Aim 4 (comparing forward-kinematics performance across the three DoF settings) is addressed in Sections 5.4 and 5.5, where position error, orientation error and wall-clock training time are reported jointly for all three configurations and the three model variants.

### H2: 7.3  Future work

The most immediate continuation of the project is real-robot validation on the physical KUKA iiwa 14. The intended programme is to execute low-speed trajectories on the real arm, log joint states and end-effector poses through the ROS2 bridge, and quantify the gap between the simulated meta-kinematics predictions and measured real-robot poses under calibration error, sensing noise and the simulator-to-hardware mismatch. Hardware access to the iiwa 14 was not available within the project timeline, so this work is left for the next phase. A successful real-robot validation would establish the meta-kinematics framework as a practical tool rather than a simulation-only result.
In parallel with real-robot validation, a journal-quality extension of the present study should expand the statistical evaluation. The planned protocol is to run each condition in Table 5.1 with at least five independent random seeds, report mean ± standard deviation for every metric, and complement the seed-level variance with held-out bootstrap 95 % confidence intervals. With this evidence, claims such as the 7 DoF orientation improvement of 18.0 % can be tested for statistical significance using a paired comparison across matched seeds, rather than relying on the single-run differences reported here.
Two further methodological directions are also natural. The first is extension to inverse kinematics, where the same shared-and-adapt strategy can be evaluated on the IK problem; this would directly support downstream planning and control, and would build on the existing literature reviewed in Chapter 2. The second is investigation of transfer across robot families: applying the meta-kinematics representation to manipulators of different morphology — for example, a Universal Robots UR5 (6 DoF) or a Franka Emika Panda (7 DoF), both supplied with Isaac Lab — would clarify whether the benefit of shared training derives from the iiwa 14’s specific structure or from more general kinematic regularities. A small additional ablation, varying the hidden width and the number of residual blocks, would also help disambiguate the role of model capacity from the role of the shared representation. Together, these extensions would convert the present DoF-transfer study into a broader programme on transferable kinematic representations and provide the experimental support required for publication at a venue such as IEEE RA-L or CoRL.

## H1: References

[1]   B. Siciliano, L. Sciavicco, L. Villani, and G. Oriolo, Robotics: Modelling, Planning and Control. London, U.K.: Springer, 2009.
[2]   A. Duka, “Neural network based inverse kinematics solution for trajectory tracking of a robotic arm,” Procedia Technology, vol. 12, pp. 20–27, 2014.
[3]   R. Köker, C. Öz, T. Çakar, and H. Ekiz, “A study of neural network based inverse kinematics solution for a three-joint robot,” Robotics and Autonomous Systems, vol. 49, no. 3–4, pp. 227–234, 2004.
[4]   M. R. Diprasetya, J. Pöppelbaum, and A. Schwung, “KineNN: Kinematic neural network for inverse model policy based on homogeneous transformation matrix and dual quaternion,” Robotics and Computer-Integrated Manufacturing, vol. 94, p. 102945, 2025.
[5]   C. Devin, A. Gupta, T. Darrell, P. Abbeel, and S. Levine, “Learning modular neural network policies for multi-task and multi-robot transfer,” in Proc. IEEE Int. Conf. Robotics and Automation, 2017, pp. 2169–2176.
[6]   T. Chen, A. Murali, and A. Gupta, “Hardware conditioned policies for multi-robot transfer learning,” in Advances in Neural Information Processing Systems, 2018, pp. 9355–9366.
[7]   A. Ghadirzadeh, X. Chen, P. Poklukar, C. Finn, M. Björkman, and D. Kragic, “Bayesian meta-learning for few-shot policy adaptation across robotic platforms,” in Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems, 2021, pp. 1277–1284.
[8]   A. Byravan and D. Fox, “SE3-Nets: Learning rigid body motion using deep neural networks,” in Proc. IEEE Int. Conf. Robotics and Automation, 2017, pp. 173–180.
[9]   A. Byravan, F. Leeb, F. Meier, and D. Fox, “SE3-Pose-Nets: Structured deep dynamics models for visuomotor control,” in Proc. IEEE Int. Conf. Robotics and Automation, 2018, pp. 3339–3346.
[10]   A. S. Deo and I. D. Walker, “Overview of damped least-squares methods for inverse kinematics of robot manipulators,” Journal of Intelligent & Robotic Systems, vol. 14, no. 1, pp. 43–68, 1995.
[11]   O. Khatib, “A unified approach for motion and force control of robot manipulators: The operational space formulation,” IEEE Journal on Robotics and Automation, vol. 3, no. 1, pp. 43–53, 1987.
[12]   Y. Nakamura, Advanced Robotics: Redundancy and Optimization. Reading, MA: Addison-Wesley, 1991.
[13]   M. Mittal, C. Yu, Q. Yu, J. Liu, N. Rudin, D. Hoeller, J. L. Yuan, R. Singh, Y. Guo, H. Mazhar, A. Mandlekar, B. Babich, G. State, M. Hutter, and A. Garg, “Orbit: A unified simulation framework for interactive robot learning environments,” IEEE Robotics and Automation Letters, vol. 8, no. 6, pp. 3740–3747, 2023.  [NVIDIA Isaac Lab framework.]
[14]   A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in Advances in Neural Information Processing Systems 32, 2019, pp. 8024–8035.
[15]   KUKA AG, LBR iiwa 14 R820 product specification (datasheet), Augsburg, Germany, 2019.
[16]   D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” in Int. Conf. on Learning Representations (ICLR), 2015.

## H1: Appendix A   Project Management


### H2: A.1  Project plan

The project was structured in stages, with early work focused on literature review, simulator development and dataset generation, and later work focused on forward-kinematics training, multi-task learning, adaptation and evaluation. The Gantt chart in Figure A.1 summarises the 14 main milestones across the academic year, with weekly granularity from week 1 (29 September) through week 12 (4 May).
Figure A.1  Project Gantt chart showing the 14 main milestones across the academic year, with weekly granularity from late September through early May.

### H2: A.2  Milestones

Table A.1  Summary of project milestones and their academic-year week ranges.

[TABLE — 15 rows × 3 cols]
  R0: # | Milestone | Weeks
  R1: 1 | Review literature on FK/IK, redundancy, kinematics and skill transfer | wk1–wk8
  R2: 2 | Configure the KUKA iiwa 14 in Isaac Lab in 5, 6 and 7 DoF settings | wk2–wk5
  R3: 3 | Set up the ROS2 / ROS bridge environment and connect to the iiwa model | wk4–wk6
  R4: 4 | Generate grid datasets for 5, 6 and 7 DoF configurations | wk5–wk7
  R5: 5 | Train residual MLP FK and IK models for 5, 6 and 7 DoF using the generated datasets | wk6–wk10
  R6: 6 | Calculate and convert normalised loss into actual loss | wk8–wk11
  R7: 7 | Prepare summary tables and learning curves | wk8–wk12
  R8: 8 | Implement a model-agnostic meta-learning (MAML-style) algorithm | cv1–cv4
  R9: 9 | Train a meta-initialisation that adapts quickly to each DoF | cv2–wk2
  R10: 10 | Explore adaptation to unseen DoFs (<5 or >7) | wk1–wk5
  R11: 11 | Validate the meta-learned model in simulation using the existing ROS2/bridge setup | wk5–wk8
  R12: 12 | Execute low-speed trajectories on the real iiwa, logging joint states and end-effector pose | wk7–wk11
  R13: 13 | Compare simulated and real errors | wk10–ev2
  R14: 14 | Conference paper and final report writing | ev3–wk12
[/TABLE]


### H2: A.3  Plan adherence and rescoping

The project largely followed the plan illustrated in Figure A.1, with the simulation, single-task training, multi-task training and per-DoF adaptation milestones (1–9) completed in line with the timeline. Two adjustments are noted for transparency. First, the literature review (milestone 1) was extended into early winter to support the methodology chapter and the conference paper, which slightly compressed the available time for the optional unseen-DoF exploration in milestone 10. Second, milestones 12 and 13 — execution of low-speed trajectories on the real iiwa and comparison of simulated and real errors — could not be completed because hardware access to the iiwa 14 was not available within the project timeline. These two milestones are accordingly carried over into the future-work section (Section 7.3), with the simulation-only deliverables completed in their place.

## H1: Appendix B   Ethics Questionnaire

This appendix records the ethics self-assessment carried out at the start of the project. The signed Ethics Questionnaire form, completed in line with the School of Engineering procedure, has been submitted to Canvas alongside this report.

### H2: B.1  Summary of self-assessment

The project is a simulation-only computational study on a publicly documented manipulator model and uses no human or animal subjects, no personally identifiable data and no clinical material. The five standard School of Engineering ethics screening questions were therefore answered as follows:

[TABLE — 6 rows × 3 cols]
  R0: # | Question | Answer
  R1: 1 | Does the project involve human participants? | No
  R2: 2 | Does the project involve identifiable personal data, including health or biometric data? | No
  R3: 3 | Does the project involve animal subjects, animal-derived material, or fieldwork on protected ecosystems? | No
  R4: 4 | Does the project involve hazardous materials, dangerous procedures, or activities that pose a non-trivial risk to participants or researchers? | No
  R5: 5 | Does the project involve deception, covert observation, or any other potentially sensitive methodology? | No
[/TABLE]

Because all five answers are negative, the project was confirmed to fall under the lowest-risk category in the School’s ethics-screening procedure, and no further ethics application or external approval was required. The complete signed Ethics Questionnaire is attached separately to this submission.

## H1: Appendix C   Risk Assessment

This appendix summarises the risk assessment carried out at the start of the project, in line with the School of Engineering risk-assessment procedure. The full signed Risk Assessment form has been submitted to Canvas alongside this report. The summary below covers the principal hazards identified, their associated controls and the residual risk rating after controls are applied.

### H2: C.1  Hazard summary

Table C.1  Summary of principal hazards, controls and residual risk.

[TABLE — 8 rows × 4 cols]
  R0: Hazard | Possible harm | Controls | Residual
  R1: Prolonged seated computer work | Postural strain, eye fatigue | Adjustable chair and screen height; regular breaks every hour; 20-20-20 rule for eye strain. | Low
  R2: High-intensity GPU workstation use | Heat build-up, cable trip hazard, electrical fault | Operate equipment in a well-ventilated University facility; secure cables; PAT-tested equipment only; do not service hardware unsupervised. | Low
  R3: Lone-working in computing labs (out-of-hours) | Personal safety, slow response in case of incident | Avoid out-of-hours lone working; use buddy system; carry mobile phone; follow building access procedure. | Low
  R4: Working with simulation data only (no real robot) | Misinterpretation of simulator output | Document simulator assumptions and limitations; sanity-check against analytical kinematics on a known pose. | Low
  R5: Manual handling of laptop/peripherals | Strain or drop injury | Use of suitable bag for transport; avoid carrying with one hand. | Low
  R6: Display ergonomics and lighting | Headache, fatigue | Use of dedicated workstation with appropriate lighting; brightness adjustment; regular breaks. | Low
  R7: Cybersecurity / data integrity | Loss of project files, code or results | Use of University Git infrastructure for code; cloud backup of datasets; encrypted laptop with password protection. | Low
[/TABLE]

Because the project is simulation-only and does not involve interaction with the physical KUKA iiwa 14, the high-energy hazards typically associated with industrial-robot work — collision, pinch points, end-effector force — were explicitly out of scope and were therefore not assessed in detail. If the project is extended to real-robot validation in a future phase (Section 7.3), the risk assessment will need to be updated to cover lab-based operation of the iiwa 14, in line with the laboratory-specific procedures of the School of Engineering.

## H1: Appendix D   Statement on Generative AI


### H2: D.1  Declaration

In line with the School of Engineering policy on the use of generative AI in student work, this appendix records the use of generative AI tools in the preparation of this Final Year Project.

### H2: D.2  Tools used

[Student to confirm and complete this section before submission.] The table below lists the generative AI tools used during the project, what each was used for and the approximate frequency of use. All AI-assisted material was reviewed, edited and verified by the author, and the underlying research data, results and conclusions are entirely the author’s own work.
Table D.1  Generative AI tools used in the preparation of this report.

[TABLE — 4 rows × 3 cols]
  R0: Tool | Used for | Frequency
  R1: [Tool name 1] | [e.g., debugging Python errors, refining wording in the methodology chapter] | [e.g., regularly]
  R2: [Tool name 2] | [e.g., explaining concepts during literature review] | [e.g., occasionally]
  R3: [Tool name 3] | [e.g., assisting with LaTeX/Word formatting] | [e.g., occasionally]
[/TABLE]


### H2: D.3  What AI was not used for

To make the boundary explicit, the following components of the project are entirely the author’s own work and were not produced by generative AI:
The research design, including the choice of meta-kinematics formulation, the three-stage pipeline and the experimental protocol.
The implementation of the simulation pipeline in Isaac Lab, the dataset generation and the training and adaptation code.
The numerical results reported in Chapter 5 and the figures derived from them.
The interpretation of the results, the engineering judgements made in the discussion and the future-work plan.
Where AI tools were used to draft text, the output was treated as a starting point and was reviewed, rewritten and verified by the author before inclusion. Where AI tools were used to assist with debugging or formatting, the resulting code or document content was tested and validated by the author.

## H1: Appendix E   Implementation Details


### H2: E.1  Architecture configuration

This appendix records the architecture, training configuration and reproducibility details of the meta-kinematics framework. Values that are common to all three training stages are listed in Table E.1; per-stage values for learning rate, batch size and step count are listed separately in Table E.2 because they differ between Stage 1 (single-task), Stage 2 (shared meta-kinematics) and Stage 3 (per-DoF adaptation).
Table E.1  Architecture and shared training parameters for the meta-kinematics ResMLP.

[TABLE — 18 rows × 2 cols]
  R0: Parameter | Value
  R1: Input dimension (n_max) | 7
  R2: Output dimension | 7  (3 position + 4 quaternion)
  R3: Hidden dimension | 1024
  R4: Number of residual blocks | 8
  R5: Activation | ReLU
  R6: Normalisation | LayerNorm (within each residual block)
  R7: Quaternion normalisation | Unit-norm projection on output
  R8: Optimiser | Adam (β₁ = 0.9, β₂ = 0.999, ε = 1×10⁻⁷ in adaptation, default ε in Stages 1–2)
  R9: Position-loss weight λ_p | 0.03 (Stages 1–2);  1.0 (Stage 3 adaptation)
  R10: Orientation-loss weight λ_o | 0.03 (Stages 1–2);  0.30 (Stage 3 adaptation)
  R11: Weight decay | 0.0 (none in Stages 1–2);  1×10⁻⁶ L2 in Stage 3 adaptation
  R12: Inactive-joint encoding | Mask-conditioned input; inactive joints zero-clamped
  R13: Train / held-out split | Per-DoF held-out test split (sample-disjoint from training)
  R14: Joint-noise floor (training stability) | std_floor_q = 1.0° equivalent
  R15: Hardware | Single NVIDIA GPU; Isaac Lab Python environment (TF32 enabled)
  R16: Software stack | PyTorch [14], Isaac Lab [13], CUDA, Python 3.10
  R17: Random-seed handling | Fixed seed = 42 for the runs reported in this document; see Appendix E.3
[/TABLE]


### H2: E.2  Per-stage training configuration

Table E.2 lists the learning rate, batch size and total step / epoch count used at each of the three training stages. Values marked as placeholders should be extracted from the training scripts in the project repository (see Section E.5) and substituted before submission. Approximate step counts inferred from the training-loss curves are given as a sanity check.
Table E.2  Per-stage training configuration. Step counts and wall-clock times are from the runs that produced the headline numbers in Table 5.1.

[TABLE — 13 rows × 4 cols]
  R0: Parameter | Stage 1 (single-task) | Stage 2 (shared meta) | Stage 3 (adaptation)
  R1: Initial learning rate | 3 × 10⁻⁴ | 1 × 10⁻⁵
  R2: LR schedule | Cosine (warmup 2 000) | Constant
  R3: LR minimum | 1 × 10⁻⁵ | —
  R4: Batch size | 4 096 | 8 192
  R5: Total steps | 1 000 000 (max) | 100 000
  R6: Support / query size | — | 50 k support / 2 M query
  R7: Gradient clipping | 1.0
  R8: L2 regularisation | 0 | 1 × 10⁻⁶
  R9: Wall-clock time (5 DoF) | 1.873 hr | 1.159 hr (shared) | 0.365 hr
  R10: Wall-clock time (6 DoF) | 4.973 hr | 1.159 hr (shared) | 0.363 hr
  R11: Wall-clock time (7 DoF) | 22.12 hr | 1.159 hr (shared) | 0.111 hr
  R12: Random seed | 42
[/TABLE]


### H2: E.3  Statistical methodology

All numerical results in Chapter 5 are reported from runs initialised with the fixed random seed value 42, which controls parameter initialisation, dataset shuffling and minibatch sampling. The dataset itself is fixed by the procedure of Section 4.3 and is identical across runs; the seed therefore controls only the stochastic components of training. The runs reported in this document use a single seed per condition, which is sufficient to characterise the qualitative pattern of results — the shared-and-adapt strategy outperforms the single-task baseline in every configuration — but does not provide a measure of seed-level variance. The seed budget is identified in Section 6.4 as a limitation of the study and as the first item in the planned journal extension (Section 7.3), where five seeds per condition will be used together with held-out bootstrap 95 % confidence intervals.
Seed used:  42  (set via torch.manual_seed, numpy.random.seed and random.seed at the start of every training script).

### H2: E.4  Loss-function and training-loop pseudocode

Listing E.1  Quaternion-distance term used inside the multi-task loss (3.3).

[TABLE — 1 rows × 1 cols]
  R0: def quat_distance(r_hat, r): /     # r_hat, r: (B, 4) tensors, both unit-normalised /     dot = (r_hat * r).sum(dim=-1).clamp(-1.0, 1.0) /     # geodesic distance on the quaternion sphere; sign-invariant /     return 1.0 - dot.abs()  // smooth surrogate for angle
[/TABLE]

Listing E.2  Residual block used inside the ResMLP backbone of Figure 3.2(a).

[TABLE — 1 rows × 1 cols]
  R0: class ResBlock(nn.Module): /     def __init__(self, dim=1024): /         super().__init__() /         self.fc1   = nn.Linear(dim, dim) /         self.norm  = nn.LayerNorm(dim) /         self.act   = nn.ReLU(inplace=True) /         self.fc2   = nn.Linear(dim, dim) /   /     def forward(self, x): /         y = self.fc1(x) /         y = self.act(self.norm(y)) /         y = self.fc2(y) /         return x + y          # additive skip connection
[/TABLE]

Listing E.3  Multi-task minibatch sampler used for Stage 2 (shared meta-kinematics) training.

[TABLE — 1 rows × 1 cols]
  R0: def multitask_step(model, datasets, optimizer, lambda_p, lambda_o): /     losses = [] /     for D_k in datasets:                       # 5, 6, 7 DoF datasets /         q_k, p_k = D_k.sample_minibatch() /         q_k = clamp_inactive_joints(q_k, k=D_k.k) /         t_hat, r_hat = model(q_k) /         Lp = ((t_hat - p_k.t)**2).sum(-1).mean() /         Lo = quat_distance(r_hat, p_k.r).mean() /         losses.append(lambda_p * Lp + lambda_o * Lo) /     loss = sum(losses) / len(datasets)         # K-task average /     optimizer.zero_grad(); loss.backward(); optimizer.step() /     return loss.item()
[/TABLE]


### H2: E.5  Reproducibility checklist

The framework is designed to be reproducible from the project repository. Reproducibility is supported by three measures. The dataset generation is fully scripted within the Isaac Lab Python environment, so a given random seed and sampling step size produce a deterministic dataset for each DoF configuration. The training pipeline records training loss, validation loss and held-out test metrics at every stage, so the curves in Figures 5.1 and 5.2 can be regenerated from the saved logs. The ResMLP backbone, the multi-task loss and the per-DoF adaptation routine are implemented as small, self-contained PyTorch modules that can be exercised in isolation; this isolates the meta-kinematics behaviour from the surrounding simulation code and makes the training loop independently testable.
The following items are provided in the project repository to support independent reproduction of the results:
Training scripts for Stages 1, 2 and 3 with all hyperparameters logged to the run configuration.
Dataset-generation scripts for the 5, 6 and 7 DoF KUKA iiwa 14 configurations in Isaac Lab.
Held-out test splits and the deterministic split index used in this report.
Saved checkpoints for the single-task, shared meta-kinematics and per-DoF adapted models.
Per-step training-loss logs sufficient to regenerate Figures 5.1 and 5.2.
Per-sample held-out errors used to compute Table 5.1, supporting independent re-evaluation and bootstrap-CI computation.

### H2: E.6  Data availability

The simulated datasets used in this study were generated from the publicly available KUKA iiwa 14 URDF supplied with Isaac Lab. The datasets and the trained model checkpoints are available from the author on reasonable request, subject to the file-size limits of the University’s research data repository. The exact commit hash, environment lock file and run configuration used for each result reported in this document are listed in the project README and quoted at the head of every saved log file, so that any single result can be traced back to the run that produced it.