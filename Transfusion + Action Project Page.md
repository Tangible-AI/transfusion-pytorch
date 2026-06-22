
# References and what I used them for 
**References used so far:**
- Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model
- Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments

# Transfusion Architecture

**The Uni-modal Approach**: 
In the original paper, the authors attempt to create a multi-modal model that uses **shared** parameters to generate text and images. Think captioning a given image or generating/editing and image based on a text descriptions. 

The text and images are tokenized separately, undergo hybrid attention, and utilize a different loss.

Below: I describe the original transfusion implementation.
### Encoding
#### Text
Text uses a BPE borrowed from llama-2.  https://huggingface.co/meta-llama/Llama-2-7b-hf
#### Images
Images undergo two stages of representation compression.
##### **Stage 1:** Pixels --> latent: $256 \times 256 \rightarrow 32 \times 32 \times 8$
- 256 x 256 pixel image compressed by VAE (Esser et al. 2021) to 32 x 32 x 8 latent image. Each 8-dimensional latent pixel here represents an 8x8 pixel patch in the original image.

##### **Noise:** Add noise to the latent pixels according to the schedule:
Sample $t$, sample $\epsilon$, form $x_t = \sqrt{\bar \alpha} \, x_0 + \sqrt{1-\bar \alpha_t} \, \epsilon$. This happens on the whole latent grid, all patches at one shared $t$. 

###### Noise Scheduler
Derived from Nichol and Dhariwal 2021: $\sqrt{\bar \alpha_t} = \cos( \frac{t}{T} \cdot \frac{\pi}{2})$ 

##### **Timestep:** Add the timestep embedding $\tau(t)$ to the noised latent pixels. 
$$\tilde{p}_j \leftarrow \tilde{p}_j + \tau(t), \quad \tilde{p}_j, \tau(t) \in \mathbb{R}^{k \cdot k \cdot c}$$
\** The authors provide a footnote mentioning the add the timestep $t$ to every patch vector before the linear layer, but there's no description or architectural details for the embedding function $\tau(t)$ . 
###### Timestep Embedding Details
For now: I assume they follow something similar to Ho et al. 2020 (DDPM).
1. Sinusoidal embedding of scalar $t$.
$$\text{emb}(t)_{2i} = \sin  \left( \frac{t}{10000^{2i/d_e}} \right) \, \, \, \text{emb}(t)_{2i+1} = \cos  \left( \frac{t}{10000^{2i/d_e}} \right)$$

2. A small MLP (i.e. 2 linear layers with nonlinearity like SiLU between them). This MLP is learnable

##### **Stage 2:** latent patches --> transformer vectors (the patch encoder)
$32 \times 32 \times 8 \rightarrow \mathbb{R}^d$
The authors explore two approaches. 
- simple linear layer (adds negligible parameters)
- U-Net down blocks (from Nichol & Dhariwal 2021 / Saharia et al. 2022) adds 0.27B parameters
### Loss
- Language modeling loss for text:$$\mathcal{L}_{LM} = \mathbb{E}[- \log P_{\theta} (y_i | y _{<i})]$$
- Noise prediction for images:
$$\mathcal{L}_{DDPM} = \mathbb{E}_{x_0, t, \epsilon}[|| \epsilon - \epsilon_{\theta} (x_t, t, c)||^2]$$


### Hybrid Attention
- Causal attention is applied to every element in the sequence regardless of modality, and bi-directional attention to every other patch within the same image. This allows image patches to attend to every other patch within the same sequence, but only attend to text or patches of other images that previously appeared in the sequence.


![[Pasted image 20260613094516.png|189]]

### Inference
In *LM mode*, the authors follow the standard practice of sampling token by token from the predicted distribution. When a BOI token is sampled, the decoding algorithm switches to *diffusion mode*, where they follow the standard procedure of decoding form diffusion models. They append a pure noise $x_T$ in the form of $n$ image patches to the input sequence, and denoise over $T$ steps. At each step, $t$, they take the noise prediction and use it to produce $x_{T-1}$, which then overwrites $x_T$ in the sequence. 

This means the model is always conditioning its noise prediction on the last timestep of the noised image and can't attend to previous timesteps. Once, the diffusion process ends, they append an EO token to the predicted image, and switch back to $LM$ mode.
# Modifications for Robotics
An everything model for robotics should be able to take in multi-modal inputs and provide multi-modal outputs that can be helpful for robot learning.

Data sources:
- it should  leverage all available robot data in a way that can be useful
	- Simulation
	- Real World Teleoperation Data
	- UMI- collected data
- As well as robot-agnostic data that can be useful for learning dynamics, reasoning, 
	- Text instructions (i.e. fine-grained verbal instructions)
		- WikiHow, Instruction Manuals, ehow, Instructables)
	- Human-Egocentric data (useful for learning skills, real world dynamics, priors)

Outputs:
- Action Chunks
- Next-state predictions (either in a pixel or latent space)
- A value or score for a given trajectory:
	- This can also be achieved by directly attempting to use the model to predict/generate scores or by using the model's text and image capability to query it's observations
- High level task plan
	- Similar to how LLMs can adapt their response such that they reason over a long trajectory without getting into the details of 'how' they will accomplish the goal
-
## Conditioning inputs (adapted from QWEN-VLA)

| Symbol        | Name                                   | Type / structure                                                                                                                                                                                                     | Role                                                                             |
| ------------- | -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| $e$           | Embodiment description                 | Natural-language prompt prepended to $x$, filled from a fixed template:<br>(robot, arm config, control freq, horizon)                                                                                                | Sole platform-specific interface                                                 |
| $x$           | High level task language instruction   | Natural-language string, episode-specific                                                                                                                                                                            | The actual goal to ground and decompress into a trajectory                       |
| $z_{t':t'+s}$ | Fine-grained task language instruction | Natural-language string describing atomic actions for time steps $[t':t'+s]$                                                                                                                                         | Leverages text for steerability                                                  |
| $o_t$         | Visual context                         | One or more image frames, video, or a history window; each wrapped with view-boundary tokens `<tag_start\|>` `<tokens for the tag>` `<\|tag_end>` <br><br><br>(tag $\in$ `ego`, `cam_left_wrist`, `cam_right_wrist`) | What the scene looks like now (and recently); grounds the action prior in vision |
| $a_{t: t+h}$  | Trajectory                             | Action trajectory from time step $t$ to time step $[t:t+h]$                                                                                                                                                          | Serves as a conditioning action or a target action.                              |
* For now: assume that we have chosen the task annotation horizon $s$ to be the same as the action chunk horizon $h$. Thus, the dataset contains sequences of paired (finegrained text annotation, action chunk, observation chunk)
### The embodiment prompt $e$ (template)

> The robot is `{robot_tag}` with `{single arm / dual arms}` \[`, waist`]\[`, and mobile base`]. The control frequency is `{FPS}` Hz. Please predict the next `{chunk_size}` control actions to execute the following task: `{ori_instruction}`.

- `{robot_tag}` and optional modifiers (`waist`, `mobile base`) — set per embodiment
- `{FPS}` — dataset's native control frequency
- `{chunk_size}` — prediction horizon
- `{ori_instruction}` — where $x$ is embedded

### How does the new modality (actions) get generated?
The action chunk is a continuous modality that will get generated through a flow-matching objective similar to images. 

![[Pasted image 20260617154120.png]]

## Stage 1: Using the model to generate cross-embodiment, cross-modality predictions
Input sequence example (Pre-tokenization, tags implicit):
```
<e> , <x>, <z_{0:h-1}>, <o_0>, <a_{0:h-1}>, 

<z_{h:2h-1}>, <o_h>, <a_{h:2h-1}>, 

<z_{2h:3h-1}>, <o_2h> <a_{2h:3h-1}>,

<z_{3h:4h-1}>, <o_3h> <a_{3h:4h-1}>,

```

modality wise this is: 
```
<text>, <text>, <text>, <image(s)>, <action>, 

<text>, <image(s)>, <action>, 

<text>, <image(s)>, <action>,

<text>, <image(s)>, <action>,
```

**How can we use it? Example output sequences to test in stage one**
```
Input: <embodiment>, <high-level task>, <fine-grained text instruction>, <image observation>

Output: <action chunk>
```

```
Input: <embodiment>, <high-level task>, <fine-grained text instruction>, <image observation>

Output: <action chunk>
```

**What insights would stage 1 offer?**
- This architecture lends itself to working across heterogeneous robot embodiments since the embodiment is provided as context:
	- We can test whether the action chunk predictions improves as we increase the diversity of robots.
- Can the textual information can serve as the 'reasoning' modality for the robot?
	- Do the fine-grained text instructions help steer the model towards the right task. I.e. can I steer the robot to perform dexterous tasks well by explicitly describing how to hold an object, how to manipulate, and how to use it?
	- Does the language provide a medium for long-horizon task reasoning?
- Can human videos be treated as 'simply another robot embodiment' or should it be treated as pre-training step?

# Timeline:
- June 17: Wrap up Vanilla Transfusion investigation - OxFordFlowers & CoCo
- June 18: Find out how to use a VLM to annotate ego and robot videos: store this data on S3
- June 19: Convert EpicKitchens, Aria, EgoDex, Ego4D to LeRobotV3
- June 20: Buffer for data to finish converting - Meanwhile: add Action modality to Transfusion
- June 21: Perform a smoke-test for Transfusion: Train for 5-6 hours and observe loss / generated images. 
- June 22: Scale up training to multiple GPUS & Spawn hyper-parameter sweeps
- June 23: Train
- June 24: Train
- June 25: Train
- June 27: Train

# Data Requirements
### Robot Data
- Images
- Task Classification - i.e. the high level goals
- Actions (Joint, EEF)
- Robot Description (embodiment_name, control frequency)
- Atomic Action Text Description - i.e. the fine grained goals
### Diverse non-Robot Pretraining Data
#### What did Transfusion use? 
- Image + Captions. 
	- 380 M licensed Shutterstock images
	- 220M publicly available images w/ captions
	- 80 M unsampled shutterstock images containing people
	- **Conceptual 12M (CC12M)** (Changpinyo et al. 2021)
- What can I use?
#### What can I use?
- **CC12M** (Conceptual 12M) — actually used in Transfusion's large-scale run, fully public, ~12M image-caption pairs.
- **LAION-2B / LAION-400M** — the standard large open image-text datasets (though availability has fluctuated due to content-safety takedowns and re-releases; check current status before relying on them).
- **DataComp / DataComp-1B** — a curated, benchmarked open image-text dataset designed for exactly this kind of training.
- **COYO-700M** — another large open image-text corpus.
- **Conceptual Captions (CC3M)**, **SBU**, **Visual Genome** — smaller, cleaner options.
# Encoder Options - Map from Modality Representation to Transformer inputs
### Text: meta-llama/Llama-2-7b-hf (BPE Token IDs) 
- https://huggingface.co/meta-llama/Llama-2-7b-hf
### Images: 
The overall workflow is: Pixels → latent image → patchify → linear or U-Net-down projection → transformer
We can represent the images using a standard VAE encoder/decoder
- https://huggingface.co/stabilityai/sd-vae-ft-mse
For now: try linear projections, later follow up with U-Net down projects
### Action Encoder (Robots): 

One possible representation for heterogenous robot actions follows from the QWEN-VLA line of thought. 
###### Robots
To enable training on heterogenous robot data, the authors normalize the actions per their respective dataset 1st/99th percentile quartile mapping to $[-1, 1]$. This removes cross-embodiment scale differences and outlier actions. 

For their training, the authors represent every action vector at a given timestep as a length $d_{max}$ vector. For an embodiment requiring $c < k$ action dimensions, they will fill the leading $c$ channels and zero pad the rest. A training-time mask excludes the padding from the gradient. 
###### Human
The human ego data is reparameterized. Each wrist motion is represented as the SE(3) \[6 dof translation and rotation\] transformation of the wrist coordinate frame at a future time step relative to the initial frame.

Hand articulation is compressed from 45 dims to 10 by performing PCA across all human datasets and retaining the weights of the first 10 principal components. 

Thus all bimanual hand actions are $2 \cdot (6 + 10) = 32$ dimensions vectors.


