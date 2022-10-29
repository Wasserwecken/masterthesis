
# Analytical
## A Deep Learning Framework for Character Motion Synthesis and Editing
> We present a framework to synthesize character movements based
on high level parameters, such that the produced movements respect the manifold of human motion, trained on a large motion capture dataset. The learned motion manifold, which is represented by
the hidden units of a convolutional autoencoder, represents motion
data in sparse components which can be combined to produce a
wide range of complex movements. To map from high level parameters to the motion manifold, we stack a deep feedforward neural
network on top of the trained autoencoder. This network is trained
to produce realistic motion sequences from parameters such as a
curve over the terrain that the character should follow, or a target
location for punching and kicking. The feedforward control network and the motion manifold are trained independently, allowing
the user to easily switch between feedforward networks according
to the desired interface, without re-training the motion manifold.
Once motion is generated it can be edited by performing optimization in the space of the motion manifold. This allows for imposing
kinematic constraints, or transforming the style of the motion, while
ensuring the edited motion remains natural. As a result, the system
can produce smooth, high quality motion sequences without any
manual pre-processing of the training data.

## Phase-Functioned Neural Networks for Character Control
> We present a real-time character control mechanism using a novel neural
network architecture called a Phase-Functioned Neural Network. In this
network structure, the weights are computed via a cyclic function which
uses the phase as an input. Along with the phase, our system takes as input
user controls, the previous state of the character, the geometry of the scene,
and automatically produces high quality motions that achieve the desired
user control. The entire network is trained in an end-to-end fashion on a
large dataset composed of locomotion such as walking, running, jumping,
and climbing movements fitted into virtual environments. Our system can
therefore automatically produce motions where the character adapts to
different geometric environments such as walking and running over rough
terrain, climbing over large rocks, jumping over obstacles, and crouching
under low ceilings. Our network architecture produces higher quality results
than time-series autoregressive models such as LSTMs as it deals explicitly
with the latent variable of motion relating to the phase. Once trained, our
system is also extremely fast and compact, requiring only milliseconds of
execution time and a few megabytes of memory, even when trained on
gigabytes of motion data. Our work is most appropriate for controlling
characters in interactive scenes such as computer games and virtual reality
systems.

## Mode-Adaptive Neural Networks for Quadruped Motion Control
> Quadruped motion includes a wide variation of gaits such as walk, pace,
trot and canter, and actions such as jumping, sitting, turning and idling.
Applying existing data-driven character control frameworks to such data
requires a significant amount of data preprocessing such as motion labeling
and alignment. In this paper, we propose a novel neural network architecture
called Mode-Adaptive Neural Networks for controlling quadruped characters. The system is composed of the motion prediction network and the
gating network. At each frame, the motion prediction network computes the
character state in the current frame given the state in the previous frame and
the user-provided control signals. The gating network dynamically updates
the weights of the motion prediction network by selecting and blending
what we call the expert weights, each of which specializes in a particular
movement. Due to the increased flexibility, the system can learn consistent
expert weights across a wide range of non-periodic/periodic actions, from
unstructured motion capture data, in an end-to-end fashion. In addition, the
users are released from performing complex labeling of phases in different
gaits. We show that this architecture is suitable for encoding the multimodality of quadruped locomotion and synthesizing responsive motion in
real-time.

## Neural State Machine for Character-Scene Interactions
> We propose Neural State Machine, a novel data-driven framework to guide
characters to achieve goal-driven actions with precise scene interactions.
Even a seemingly simple task such as sitting on a chair is notoriously hard
to model with supervised learning. This difficulty is because such a task
involves complex planning with periodic and non-periodic motions reacting
to the scene geometry to precisely position and orient the character. Our
proposed deep auto-regressive framework enables modeling of multi-modal
scene interaction behaviors purely from data. Given high-level instructions
such as the goal location and the action to be launched there, our system
computes a series of movements and transitions to reach the goal in the
desired state. To allow characters to adapt to a wide range of geometry such
as different shapes of furniture and obstacles, we incorporate an efficient
data augmentation scheme to randomly switch the 3D geometry while maintaining the context of the original motion. To increase the precision to reach
the goal during runtime, we introduce a control scheme that combines egocentric inference and goal-centric inference. We demonstrate the versatility
of our model with various scene interaction tasks such as sitting on a chair,
avoiding obstacles, opening and entering through a door, and picking and
carrying objects generated in real-time just from a single model.

## Local Motion Phases for Learning Multi-Contact Character Movements
> Training a bipedal character to play basketball and interact with objects, or
a quadruped character to move in various locomotion modes, are difficult
tasks due to the fast and complex contacts happening during the motion.
In this paper, we propose a novel framework to learn fast and dynamic
character interactions that involve multiple contacts between the body and
an object, another character and the environment, from a rich, unstructured
motion capture database. We use one-on-one basketball play and character
interactions with the environment as examples. To achieve this task, we
propose a novel feature called local motion phase, that can help neural
networks to learn asynchronous movements of each bone and its interaction
with external objects such as a ball or an environment. We also propose a
novel generative scheme to reproduce a wide variation of movements from
abstract control signals given by a gamepad, which can be useful for changing
the style of the motion under the same context. Our scheme is useful for
animating contact-rich, complex interactions for real-time applications such
as computer games.

## Neural Animation Layering for Synthesizing Martial Arts Movements
> Interactively synthesizing novel combinations and variations of character
movements from different motion skills is a key problem in computer animation. In this paper, we propose a deep learning framework to produce a large
variety of martial arts movements in a controllable manner from raw motion
capture data. Our method imitates animation layering using neural networks
with the aim to overcome typical challenges when mixing, blending and
editing movements from unaligned motion sources. The framework can
synthesize novel movements from given reference motions and simple user
controls, and generate unseen sequences of locomotion, punching, kicking,
avoiding and combinations thereof, but also reconstruct signature motions
of different fighters, as well as close-character interactions such as clinching
and carrying by learning the spatial joint relationships. To achieve this goal,
we adopt a modular framework which is composed of the motion generator
and a set of different control modules. The motion generator functions as
a motion manifold that projects novel mixed/edited trajectories to natural
full-body motions, and synthesizes realistic transitions between different
motions. The control modules are task dependent and can be developed and
trained separately by engineers to include novel motion tasks, which greatly
reduces network iteration time when working with large-scale datasets. Our
modular framework provides a transparent control interface for animators
that allows modifying or combining movements after network training, and
enables iterative adding of control modules for different motion tasks and
behaviors. Our system can be used for offline and online motion generation
alike, and is relevant for real-time applications such as computer games.

## DeepPhase: Periodic Autoencoders for Learning Motion Phase Manifolds
> Learning the spatial-temporal structure of body movements is a fundamental
problem for character motion synthesis. In this work, we propose a novel
neural network architecture called the Periodic Autoencoder that can learn
periodic features from large unstructured motion datasets in an unsupervised manner. The character movements are decomposed into multiple latent
channels that capture the non-linear periodicity of different body segments
while progressing forward in time. Our method extracts a multi-dimensional
phase space from full-body motion data, which effectively clusters animations and produces a manifold in which computed feature distances provide a
better similarity measure than in the original motion space to achieve better
temporal and spatial alignment. We demonstrate that the learned periodic
embedding can significantly help to improve neural motion synthesis in
a number of tasks, including diverse locomotion skills, style-based movements, dance motion synthesis from music, synthesis of dribbling motions
in football, and motion query for matching poses within large animation
databases.

# Pyhsical
## Evolved Controllers for Simulated Locomotion
> We present a system for automatically evolving neural networks as physics-based locomotion controllers for humanoid characters.
Our approach provides two key features: (a) the topology of the neural
network controller gradually grows in size to allow increasingly complex
behavior, and (b) the evolutionary process requires only the physical
properties of the character model and a simple fitness function. No a priori knowledge of the appropriate cycles or patterns of motion is needed.

## Data-Driven Biped Control
> We present a dynamic controller to physically simulate underactuated three-dimensional full-body biped locomotion. Our datadriven controller takes motion capture reference data to reproduce
realistic human locomotion through realtime physically based simulation. The key idea is modulating the reference trajectory continuously and seamlessly such that even a simple dynamic tracking
controller can follow the reference trajectory while maintaining its
balance. In our framework, biped control can be facilitated by a
large array of existing data-driven animation techniques because
our controller can take a stream of reference data generated on-thefly at runtime. We demonstrate the effectiveness of our approach
through examples that allow bipeds to turn, spin, and walk while
steering its direction interactively.

## DeepLoco: Dynamic Locomotion Skills Using Hierarchical Deep Reinforcement Learning
> Learning physics-based locomotion skills is a dicult problem, leading
to solutions that typically exploit prior knowledge of various forms. In
this paper we aim to learn a variety of environment-aware locomotion
skills with a limited amount of prior knowledge. We adopt a two-level
hierarchical control framework. First, low-level controllers are learned that
operate at a ne timescale and which achieve robust walking gaits that
satisfy stepping-target and style objectives. Second, high-level controllers
are then learned which plan at the timescale of steps by invoking desired
step targets for the low-level controller. The high-level controller makes
decisions directly based on high-dimensional inputs, including terrain maps
or other suitable representations of the surroundings. Both levels of the
control policy are trained using deep reinforcement learning. Results are
demonstrated on a simulated 3D biped. Low-level controllers are learned for
a variety of motion styles and demonstrate robustness with respect to forcebased disturbances, terrain variations, and style interpolation. High-level
controllers are demonstrated that are capable of following trails through
terrains, dribbling a soccer ball towards a target location, and navigating
through static or dynamic obstacles

## ASE: Large-Scale Reusable Adversarial Skill Embeddings for Physically
> The incredible feats of athleticism demonstrated by humans are made possible in part by a vast repertoire of general-purpose motor skills, acquired
through years of practice and experience. These skills not only enable humans to perform complex tasks, but also provide powerful priors for guiding
their behaviors when learning new tasks. This is in stark contrast to what
is common practice in physics-based character animation, where control
policies are most typically trained from scratch for each task. In this work,
we present a large-scale data-driven framework for learning versatile and
reusable skill embeddings for physically simulated characters. Our approach
combines techniques from adversarial imitation learning and unsupervised
reinforcement learning to develop skill embeddings that produce life-like
behaviors, while also providing an easy to control representation for use
on new downstream tasks. Our models can be trained using large datasets
of unstructured motion clips, without requiring any task-specific annotation or segmentation of the motion data. By leveraging a massively parallel
GPU-based simulator, we are able to train skill embeddings using over a
decade of simulated experiences, enabling our model to learn a rich and
versatile repertoire of skills. We show that a single pre-trained model can be
effectively applied to perform a diverse set of new tasks.
