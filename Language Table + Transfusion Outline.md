
# Goal

I am creating a multi modal model based on the transfusion approach and applying it to robotics. I want to build up to an architecture that can generate actions, state predictions, and text reasoning as outlined in [[Transfusion + Action Project Page]]. For now, the goal is generate meaningful textual reasoning, state predictions, and action chunk predictions on one embodiment, using a simple dataset and environment. Essentially, we are stress testing whether the transfusion approach enables shared parameters across these three modalities.


## Q: What does the language table dataset contain? What are my inputs?
The language table dataset consists of a few variants, listed below along with their path. 
### Summary Table

Dataset | Real/sim | Controlled by | Language-labeled by | # episodes
--------| --------- | ------------- | ----------------- | --------: 
language_table | real | human | human | 442,226
language_table_sim | sim | human | human | 181,020
language_table_blocktoblock_sim | sim | human | scripted | 8,000
language_table_blocktoblock_4block_sim |  sim | human | scripted | 8,298
language_table_blocktoblock_oracle_sim | sim | oracle | scripted | 200,000
language_table_blocktoblockrelative_oracle_sim | sim | oracle | scripted | 200,000
language_table_blocktoabsolute_oracle_sim | sim | oracle | scripted | 200,000
language_table_blocktorelative_oracle_sim | sim | oracle | scripted | 200,000
language_table_separate_oracle_sim | sim | oracle | scripted | 200,000

### Paths

Dataset | Data Location
--------| --------------
language_table | [gs://gresearch/robotics/language_table](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table/0.0.1/)
language_table_sim | [gs://gresearch/robotics/language_table_sim](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_sim/0.0.1/)
language_table_blocktoblock_sim | [gs://gresearch/robotics/language_table_blocktoblock_sim](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_blocktoblock_sim/0.0.1/)
language_table_blocktoblock_4block_sim | [gs://gresearch/robotics/language_table_blocktoblock_4block_sim](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_blocktoblock_4block_sim/0.0.1/)
language_table_blocktoblock_oracle_sim | [gs://gresearch/robotics/language_table_blocktoblock_oracle_sim](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_blocktoblock_oracle_sim/0.0.1/)
language_table_blocktoblockrelative_oracle_sim | [gs://gresearch/robotics/language_table_blocktoblockrelative_oracle_sim](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_blocktoblockrelative_oracle_sim/0.0.1/)
language_table_blocktoabsolute_oracle_sim | [gs://gresearch/robotics/language_table_blocktoabsolute_oracle_sim](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_blocktoabsolute_oracle_sim/0.0.1/)
language_table_blocktorelative_oracle_sim | [gs://gresearch/robotics/language_table_blocktorelative_oracle_sim](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_blocktorelative_oracle_sim/0.0.1/)
language_table_separate_oracle_sim | [gs://gresearch/robotics/language_table_separate_oracle_sim](https://console.cloud.google.com/storage/browser/gresearch/robotics/language_table_separate_oracle_sim/0.0.1/)
All nine are versioned identically at `gs://gresearch/robotics/<name>/0.0.1/` and load via `tfds.builder_from_directory`. The authors also release one checkpoint (BC+ResNet, sim) at `gs://gresearch/robotics/language_table_checkpoints/`.

The Dataset Structure is listed below:
```text
Dataset Structure (element_spec):

episode_ds:
  episode_id: Shape=(), Dtype=<dtype: 'string'>
  steps: (sequence of elements)
    elements:
      action: Shape=(2,), Dtype=<dtype: 'float32'>
      is_first: Shape=(), Dtype=<dtype: 'bool'>
      is_last: Shape=(), Dtype=<dtype: 'bool'>
      is_terminal: Shape=(), Dtype=<dtype: 'bool'>
      observation:
        effector_target_translation: Shape=(2,), Dtype=<dtype: 'float32'>
        effector_translation: Shape=(2,), Dtype=<dtype: 'float32'>
        instruction: Shape=(512,), Dtype=<dtype: 'int32'>
        rgb: Shape=(360, 640, 3), Dtype=<dtype: 'uint8'>
      reward: Shape=(), Dtype=<dtype: 'float32'>
```

---
All nine releases sit in the _same_ environment — the 2D tabletop where a plane-constrained xArm6 with a cylindrical end-effector pushes 8 blocks; observations are 360 x 640 RGB.

- **Domain:** real robot vs. PyBullet simulation.
- **Who controlled the robot:** human teleoperators (diverse, "play"-style long-horizon collect, no resets/segmentation) vs. a scripted **oracle** that solves one task family using privileged simulator state.
- **How the language was produced:** crowdsourced **human** hindsight relabeling (open-vocabulary, tens of thousands of unique strings) vs. **scripted/templated** language synthesized from predefined synonym sets per task condition.

Those axes produce the three groups in the README.
### Real (1 dataset). 
`language_table` — 442,226 human-teleoperated, human-relabeled real-robot episodes. This is the flagship, open-vocabulary set: the real-world $\mathcal{D}_\text{training}$ behind the 93.5%-success LAVA policy. (The paper trained on a 298,782-episode / 87,140-unique-instruction subset of it.)

### Simulation, human-controlled (3 datasets).

- `language_table_sim` — 181,020 episodes, human teleop + human relabeling (78,623 unique instructions). The open-vocabulary sim analogue of the flagship; this is what the sim architecture/data ablations run on.
- `language_table_blocktoblock_sim` — 8,000 episodes, human teleop but **scripted** language, restricted to the single "push block to block" task on the standard 8-block board.
- `language_table_blocktoblock_4block_sim` — 8,298 episodes, same as above but on a reduced 4-block board configuration.

### Simulation, oracle-controlled (5 datasets, 200,000 episodes each). 
Each is a single task family generated by a scripted oracle with synthetic templated language, and the five map one-to-one onto the benchmark's five evaluation task families (Appendix D, 696 task conditions total):

- `language_table_blocktoblock_oracle_sim` → **block2block** (push a block to another block)
- `language_table_blocktoblockrelative_oracle_sim` → **block2blockrel** (push to a relative offset _of another block_)
- `language_table_blocktoabsolute_oracle_sim` → **block2abs** (push to an absolute board location)
- `language_table_blocktorelative_oracle_sim` → **block2rel** (push to a relative offset direction)
- `language_table_separate_oracle_sim` → **separate** (separate two blocks)

The practical distinction for our purposes: the two human-relabeled sets (`language_table`, `language_table_sim`) are where the _language breadth_ lives — open vocabulary, ~80k–120k unique instructions each, but expensive to collect. The five oracle sets are cheap, clean, single-skill demonstrations at volume (1M episodes combined) with low language diversity, aligned to the benchmark's automated success metrics — useful for skill-specific study or benchmark-aligned pretraining/eval, not for vocabulary coverage. The two small `blocktoblock` human sets sit in between.

Two things worth flagging for precision:

1. **A count discrepancy on the real set.** The README lists 442,226 real episodes, but the paper reports 414,798 total relabeled real trajectories (Appendix B.4) and 413k in Table III. The released snapshot is therefore _larger_ than what the paper quotes — consistent with the authors' note that collection and relabeling continued after the trained policy was frozen but before public release. The sim count, by contrast, matches exactly (181,020 in both).
2. **What "~600k" refers to.** The abstract's "nearly 600,000 language-labeled trajectories" is just the two human-relabeled sets (real + sim ≈ 623k, or the paper's 594k using its 413k figure). The five oracle sets add ~1M synthetic episodes on top, so the full release is ≈1.64M episodes.

## Q: Will I pretrain on a dataset?
This can be an ablation: I can setup two experiments, one which takes the CC12M image + text checkpoints, and another which is trained from scratch.

## Q: What data should I use? What becomes my train/test split and what do I hold out?
- I think using all the data makes sense. Need to double check.
## Q: How do I structure the data for Transfusion to leverage it?
A sample sequence will consist of a text instruction, image observation, and action chunk.

The shared transformer needs these modalities to be converted to a shared $x$ dimensional space. 

The **text** (discrete) modality will be tokenized using the Chat-GPT 2 tokenizer. Each token will be mapped to an $x$ dim vector using a learned embedding matrix.

The **image** will be encoded using a pretrained VAE: For now we'll use `stabilityai/sd-vae-ft-mse`.  The image is encoded into a latent pixel representation. This one specifically maps images to a latent 4-channel tensor.

In scenarios where we provide multiple image observations: Let's keep each image as its own block of patches in the sequence, each wrapped in BOI/EOI, with intra-image bidirectional attention and causal attention across images.

The **actions** will be treated as another continuous modality.Here's what I'm thinking

Represent an action chunk as a $h \times d$ array: where $h$ is the action prediction horizon. Let's scale the action dimensions to 1st/99th-percentile→[-1,1]  as is done in RT-2/OpenVLA/π0-FAST.

Use independent diffusion timesteps for image vs action. Don't share t across modalities — that's UWM's key mechanism and it's what lets the same weights serve as policy / dynamics / generator.

## Q: What is the new loss?
- Keep the flow-matching objective for continuous modalities: λ tuning will get harder. We'll now balance three terms (LM + λ_img·diffusion_img + λ_act·diffusion_act). Transfusion used λ=5 for one continuous term; budget real tuning for the action weight, since under-weighting it is an easy way to get a great image model that can't act.
### Note: The VAE may not provide the best latent representation for control, let's ignore this for now but test against it later:
The VAE latent may seem like the right choice when we are generating/predicting images. Transfusion uses the VAE because it's job is image generation and we need a decoadable latent. For a policy with the objective of deriving actions from an understanding of images, the field has universally moved away from recontrsuction-optimzed latents towards semantically pretrained encoders, because the VAE latents preserve pixel fidelity, not the spatial/semantic structure useful for control. 
- OpenVLA fuses **DINOv2** (spatial fidelity) and **SigLIP** (semantic/language alignment), concatenating ~256 patch tokens from each along the hidden dimension, from a single third-person image. The fused vision encoder extracts 256 patch embeddings from each vision transformer, concatenates them along the hidden dimension, and projects them into the language embedding space, and their ablation shows the fused vision encoder improves policy performance (though dataset diversity mattered more). [arxiv](https://arxiv.org/pdf/2502.19645)[arxiv](https://arxiv.org/pdf/2406.09246)
- Diffusion Policy uses a from-scratch **ResNet-18 with two changes: global average pooling → spatial-softmax, and BatchNorm → GroupNorm**, trained end-to-end, one encoder per view. Replace the global average pooling with a spatial softmax pooling to maintain spatial information... Replace BatchNorm with GroupNorm for stable training. This is important when the normalization layer is used in conjunction with Exponential Moving Average. [arxiv](https://arxiv.org/pdf/2303.04137)
- The trend is toward frozen/finetuned self-supervised backbones: finetuned DINOv3 matches or exceeds ResNet-18 on several tasks, frozen DINOv3 remains competitive... self-supervised features improve sample efficiency and robustness. (Caveat: it's task-dependent — JUICER found an ImageNet-pre-trained ResNet18 with global pooling to outperform a ResNet18 trained from scratch with SpatialSoftmax pooling.) [arxiv](https://arxiv.org/pdf/2509.17684)[arxiv](https://arxiv.org/pdf/2404.03729)


**What data am I leaving for Evals?**
# What I've done so far:
I've forked a version of a Transfusion implementation from the community. I've had to make a few changes to the repository. 

# Ablations
Provide two past frames as input (both observed) vs one past frame as input.
- Past frame(s) as input + a future frame as a diffusion target

Provide one past frame + one future frame as input, followed by the action chunks.
- Like GR-1: predict actions and future frames jointly
- in this scenario, inference can act as an IDM

Using the standard one frame as input:
- ablate over the lambda's that govern the influence of each modality towards the loss

## Environment setup

## Changes to Original Transfusion Codebase
- 

## Run Artifact Storage
### wandb
### checkpoints

### conditional results

### sampling resutls

# Open Questions
- Since I'm changing the dataset action inputs to 1st/99th percentile $->$ [-1, 1], how do I decode back into continuous actions at inference?
- Should I finetune the VAE? Is it good enough as is?
-  Is the VAE the proper encoder? Especially if I'm trying to capture semantic information rather than pixel perfect representation

# Improvments / Things to Try