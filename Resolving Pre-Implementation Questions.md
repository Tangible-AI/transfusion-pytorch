
I'm preparing to test an existing community repository called Transfusion to perform multi-modal predictions (text, images, actions). Here's an outline of my plan. What parts are unclear? Give an overview of questions I should answer before setting off on the implementation.

Use the files in the project page to familiarize yourself with the two papers referenced here

---
# Claude
**1. Shared parameters vs. action expert — the unstated central decision.** Your stated thesis is _shared parameters across text/image/action in one transformer_ (Transfusion, extended). But QwenVLA — your action reference — deliberately does the opposite. It is a Qwen3.5 VLM backbone _plus a separate ~1.15B DiT action expert_ that concatenates the VLM's hidden states with a noisy action chunk and runs joint self-attention with AdaLN timestep conditioning. The backbone and expert are distinct parameter sets (VLM frozen in their Stage I, then unfrozen); this is the π0 design (Black et al. 2024, cited inside QwenVLA). So you are borrowing QwenVLA's action _representation and conditioning_ (1st/99th-percentile normalization, channel padding + validity mask, the embodiment-prompt template, the e/x/z/o/a layout) while rejecting its _architecture_. That's a legitimate and genuinely novel hypothesis — I'm not aware of a single shared-weight stream demonstrated to do text+image+action jointly. But name it as the hypothesis, because two things follow: you can't cite QwenVLA's success numbers as evidence your shared-param approach will work, and the timestep/loss mechanics diverge from theirs (point 2). Your docs imply the shared-weight in-stream choice — just make it deliberate rather than incidental.

**2. Does the repo actually support what you need? (highest implementation risk)** The lucidrains multi-modality API was built for _one_ extra continuous modality — images — which are 2D, channel-first, with a conv patchifier. Actions are a 1D h×d sequence, not a spatial grid. Before writing training code, verify in the source:

- A _non-spatial_ continuous modality (`modality_num_dim=1`, a linear/MLP encoder over the h×d chunk, bypassing the conv down/up path that assumes 2D).
- _Two_ distinct continuous modalities coexisting in one sequence with distinct encoders/decoders and distinct `dim_latent` (the tuple API), and that the attention mask, the `[som]/[eom]/[meta]` markers, and the sampler all generalize when two non-text modality types are interleaved.
- _Independent per-modality flow timesteps._ You cite UWM for not sharing t across image and action — confirm the repo samples t per modality _instance_, not one t per sequence. If it shares t, that's a code change, and it's the exact mechanism you're relying on, so verify it directly rather than assume.
- _Per-modality loss weights._ You want LM + λ_img·flow_img + λ_act·flow_act, but the repo exposes a single `flow_loss_weight` plus token-fraction weighting. Splitting it is probably a small patch — confirm.
- _Adding a modality to a checkpoint_ (for the pretraining ablation): loading text+image weights while randomly initializing only the action encoder/decoder and action-modality params, and that the state-dict load tolerates the new tuple entry.

**3. The encoder fork is sharper than the outline states.** You've already flagged VAE-vs-semantic correctly. The Transfusion-specific wrinkle you haven't written down: Transfusion _needs a decodable latent_ because it generates images. DINOv2/SigLIP are _not invertible_ — you can't decode them back to pixels — so a single semantic encoder is incompatible with **state prediction** (future-frame generation), one of your three stated outputs. The fork is therefore: (a) VAE everywhere (keeps state prediction, worse features for control); (b) semantic encoder for input conditioning + VAE for the generation target (two image representations in-stream, more tokens, more plumbing); (c) VAE in, predict future state in VAE-latent space only (what you already support). This is coupled to point 4.

**4. The image pipeline isn't working yet — which changes the readiness bar per output.** Your CC12M 0.16B run produces textures with weak conditioning and the flow loss has plateaued (~0.735). Your three outputs depend on that differently: action prediction only needs the model to _condition on_ images, so weak image _generation_ doesn't necessarily block it; state prediction _is_ image generation and is blocked by exactly the failure you're seeing; reasoning text has its own problem (point 5). So the question is: **is state prediction in scope for run #1, or deferred until generation is fixed?** If deferred, the CC12M result isn't blocking and you can move on actions+text. If in scope, resolve the image-quality issue first (your own listed suspects: capacity, VAE quality, linear-vs-U-Net patchifier) or you'll conflate failure modes across modalities.

**5. Data and supervision reality on Language Table.** Several things degenerate or are underspecified when QwenVLA's rich conditioning meets this dataset:

- _Reasoning text has no supervision._ Language Table has one short instruction per episode (the (512,) int32 field), no chain-of-thought. "Meaningful textual reasoning" is in your goal but has no signal in the data. Either the text target is just the instruction (reproduction, not reasoning), or you synthesize reasoning traces with a VLM annotator (your timeline mentions this), or you defer it. Define what text the model is trained to emit and what counts as success — otherwise "reasoning" is unfalsifiable on this dataset.
- _The QwenVLA conditioning collapses here:_ e is one fixed embodiment, x and z are the same string (no separate per-chunk fine-grained annotation), o is a single third-person RGB. Decide how the template maps onto the real fields and accept that several channels are trivial.
- _Instruction encoding:_ confirm what produced the (512,) int32 buffer and whether you decode→string→re-tokenize with your chosen tokenizer or consume it directly.
- _Chunking/semantics:_ actions are 2D planar pushes. Choose the chunk horizon h and the image-to-chunk cadence (one frame per chunk vs. per step), confirm the native control frequency (for the embodiment prompt) and the action semantics (delta of the effector target?) before normalizing. The QwenVLA pad-to-K + mask scheme is unnecessary for a single 2-dim embodiment — only adopt it if cross-embodiment is in scope now.
- _Decode at inference (your open question):_ store the 1st/99th-percentile bounds (computed on the train split only), invert the affine map, clip. Straightforward.

**6. Evaluation and scope — the biggest "what does success mean."**

- _Open-loop vs closed-loop._ This is the most consequential unanswered question. Flow-matching MSE on held-out chunks is a weak proxy; the established standard in policy learning is closed-loop success in the environment (Diffusion Policy, OpenVLA, and essentially everything you cite evaluate this way) because open-loop error correlates poorly with task success — especially under the multimodal action distributions flow matching exists to model. Closed-loop means wiring the model into the PyBullet Language Table sim as a receding-horizon policy (predict chunk → execute k steps → re-observe). That's real engineering; plan it now, not after training.
- _Which variant?_ `language_table_sim` (open-vocab, human teleop) fits the language-breadth/reasoning story but has noisy, multimodal actions and no clean automated metric. The oracle sets (e.g. `block2block_oracle_sim`) are single-skill, scripted, and map onto the benchmark's 696 task conditions with defined success metrics — far cleaner for closed-loop eval. For run #1 I'd argue the oracle block2block set is the right stress test (one skill, clean success signal, and flow matching still has to model the distribution); reserve the open-vocab set for the reasoning/breadth claims later.
- _Split._ Still "use all the data, TBD." You need held-out episodes — or, stronger, held-out task conditions — and, for closed-loop, held-out scene/instruction configurations.
- _Ablation scope._ The GR-1-style joint future-frame+action, the IDM-at-inference idea, and the past-vs-future-frame variants are good but all presuppose image generation works and they multiply the surface. Commit to one configuration for run #1 and mark the rest deferred.

**7. Loose ends.** Tokenizer drifts across your docs (Llama-2 BPE → byte-level for Flowers → GPT-2 for CC12M); pin one — GPT-2 matches your latest pipeline and is fine. And the pretraining ablation's "from CC12M checkpoint" transfers only text+image priors (action params start fresh), and those priors are weak given the CC12M result — so frame it as "do weak image/text priors still help?" rather than assuming they do.

If I compress this to the three you must answer before writing any code: **(1)** shared-weight action modality or separate expert; **(2)** is state prediction — and therefore working image generation plus a decodable latent — in scope for run #1 or deferred; **(3)** open-loop or closed-loop eval, and on which variant. Most of the rest falls out of those three.

Which would you like to resolve first? I'd suggest (1), since it determines whether the repo audit in point 2 is even the right audit.

# Me
Answers: 
1. I understand that I'm testing a shared parameter model. My hypothesis is that given enough parameters, it's not necessary to train an action expert. My hope is that we can form a space where text, videos, and actions (all forms of reasoning) exist together.
2. The repo was built for more than one continuous modality, I added the relevant python files in the shared files so you can see for yourself (but I've only tested generating one continuous modality). You're right in that I need to confirm that I can incorporate a second continuous modality. As per your suggestions: I should test
    1. That I can replace the image encoder with a linear/mlp encoder over the hxd chunk
    2. That I can have two distinct continous modalities coexisting in one sequence with distinct encoders/decoders (the actions should use an MLP), `dim_latent`, and that the attention mask can recognize the `[som]/[eom]/[meta]` markers, and that the sampler can generalize when two non-text modalities are interleaved.
    3. That I can have independent per-modality flow timesteps. If it shares t, that's a code change.
    4. That we can have per-modality loss weights. You want LM + λ_img·flow_img + λ_act·flow_act, but the repo exposes a single flow_loss_weight plus token-fraction weighting.
    5. Adding a modality to a checkpoint (for the pretraining ablation): loading text+image weights while randomly initializing only the action encoder/decoder and action-modality params, and that the state-dict load tolerates the new tuple entry.
    6. For one ablation: I'd like to test whether adding a goal image as part of the training enables the model to both plan it's own goal state and generate the trajectory to reach it. In this scenario, the input sequence would be text instruction, current image state, goal image, action chunk. At inference, I'd like to query the model with the instruction and current image, and ask it to generate a goal state along with the action required to reach it. I haven't tested generating two images. I need to answer whether the model will need to treat the two images as two separate modalities, or whether they can be treated as the same modality, just with different purposes
3. I won't argue with this point: I want to test with the simplest approach first, which is path (a). Depending on the outcomes, and while this approach trains, I can work on a version with (b) or (c). Let's make a note of them
4. I know the image pipeline isn't working yet, but I'm confident I can get it working. I'm already seeing better results after incorporating the U-Net, without increasing the parameter count beyond 0.16B. We'll start with U-Net when training on the language-table dataset
5. You bring up great points:
    1. reasoning text has no supervision. The most I can do right now is that I can use text as a conditioning mechanism, text is just instruction. What I would be able to test, however, is whether I can stitch together several commands to accomplish a long horizon task.
    2. The QwenVLA conditioning does collapse, since this is just one emobodiment where x and z are the same string per sequence. In my mind, the setup will simply help test whether we can share parameters across these three different modalities. Later, I'll work on increasing the complexity of the textual instructions, incorporating heterogeneous embodiments, and several new points of view
    3. I do need to confirm the structure of the produced the (512,) int32 buffer and whether I decode→string→re-tokenize with your chosen tokenizer or consume it directly.
    4. For this test, I won't utilize the QwenVLA pad-to-K + mask scheme (since we're only using a single 2-dim embodiment). I'm not even sure I need an encoder. I do need to consider how I can map the action chunk to the transformer input dimensions. How should I choose the chunk horizon h?
        1. I also need to identify how to format the data for training and evaluating consumption.
            1. I'll be running several model varients: single current image + action chunk, two consecutive images for history and 1 action chunk, and current image + goal state and one action chunk. If the episodes are all different lengths, the data format will need to be thought out. For example, should the goal image be the final image (success) or the image at the end of the action chunk.
    5. Make a note of your suggestion for the Decode at inference, I'll need to implement this.
6. Valid points: 
	1. I will test the closed loop policy on the langauge_table sim environment.
	2. Regarding which data and split: Explain a few approaches I can take here
	3. For configuration #1: I think the simplest approach is to provide the current frame: and try to estimate the next action chunk. The ablations can come later: though I should track them.
7. For the first pass, GPT-2 text tokenizer works fine.

Help me tighten up the ablations. There seems to be a few directions to explore if the first pass works: which is
1. GR-1 stype joint future-frame + action and 
2. the history as context frames. Are there other ablations or hyperparameters I should experiment with in the first pass?
3. Can I start the training from the latest CC12M checkpoint: Do image/text priors help?

What remaining unresolved questions I should straighten out? Did I cover all of your concerns?

# Claude
Let me look at the training scripts you added so my recommendations are grounded in the actual API you're calling, not a generic picture of the repo.I read `train_cc12m.py`, `unet_patchifier.py`, and `train_toy.py`. Two things worth grounding the rest on: `train_toy.py` confirms the continuous-modality path you need for actions (a raw `(length, dim_latent)` tensor, _no_ encoder — the built-in linear lifts `dim_latent→D`), and `unet_patchifier.py` confirms the image path is inherently 2D-conv, so actions will need their own 1D in/out projection. Both reinforce the recommendations below.

## The concrete decisions you asked about

**Action representation + horizon (your 5.4).** You're right that you don't need a learned encoder. Flow-match directly in normalized action space: `dim_latent_act = action_dim = 2`, one token _per timestep_ so the chunk is a `(h, 2)` tensor, `modality_num_dim=1`, `modality_default_shape=(h,)`. Identity in/out; the repo's linear does 2↔768. This mirrors Diffusion Policy and π0, which diffuse/flow in raw action space rather than a learned action latent — it's the faithful choice and keeps the target interpretable. Per-timestep tokens (not one flattened token) so the intra-modality bidirectional attention models the joint chunk across time, exactly as it does across image patches; RoPE over sequence position preserves order, so `add_pos_emb` is optional. If raw-space flow underfits, a small 2→32 MLP encoder is the first thing to add — but start identity.

For h: seed it from the closest comparable work, then sweep. Diffusion Policy's convention on planar pushing (Push-T, very close to this) is predict ~16, execute ~8, observe ~2; QwenVLA uses 16-step manipulation chunks. Start **h=16, Ta=8** (receding horizon: predict 16, execute 8, re-observe). Avoid two failure modes: Ta=h (no re-planning until the chunk fully executes → compounding error) and Ta=1 (discards the chunking benefit, 16× the inference cost). Confirm Language Table's native rate for the embodiment-prompt FPS field — it's low (~10 Hz), so 16 steps ≈ 1.6 s, reasonable for a push.

**Two images: same modality, two instances (your 2.6).** Both are RGB→VAE→same `dim_latent`/patchifier/geometry; "current" vs "goal" is a _role_, not a representation. Distinguish by fixed sequence position (goal second) plus a cheap text role tag ("current:"/"goal:"), which also feeds the language-as-glue story. The repo already handles multiple image instances in one sequence (the smoke test passes two) — no modality index needed.

The real dependency is the same item as your 2.3, but **stronger than I framed it**: the goal-image ablation needs _per-instance_ timesteps. At training, current is clean conditioning (t→0, no flow loss) while goal and action are targets (their own sampled t); at inference you clamp current=clean and sample goal+action. That's the UWM mechanism (t per modality _instance_; t=0 ⇒ pure conditioning, t=1 ⇒ pure target). Your scripts only ever have one image instance, so per-instance t is **untested** — verify the repo assigns t per instance, not one t shared across all images of a type. If it shares t across instances, current and goal can't carry different t and the ablation needs a code change.

This forces an immediate decision: in the first pass (current→action), keep the current frame clean (pure policy, simplest, correct first pass) — but build the per-instance-t plumbing now, because noising every instance with its own t is what later yields state prediction and the IDM capability "for free," and matches your one-shared-space goal.

**Goal image — final vs end-of-chunk are two different ablations (your 5.4.1.1).** Not two options for one experiment; two experiments with different targets and claims:

- _End-of-chunk frame_ (obs h steps ahead) = short-horizon **forward-dynamics** target. (current, action chunk, frame_{t+h}) teaches "given these actions, predict the near-future state." This is your **GR-1 ablation**; inverting it at inference (two frames → action) is the IDM use. Well-defined for any window with h steps left; truncate near episode end.
- _Final/success frame_ = long-horizon **goal-imagination** target. (current, …, terminal frame) teaches "imagine the success state, then plan toward it" — your **goal-planning ablation**. For human-teleop play data "success" is ill-defined; for the _oracle_ sim sets the terminal frame _is_ the solved state, so it's a clean goal label — another reason to run the first robot experiments on an oracle set, not the open-vocab human set.

So: "goal planning" → terminal frame; "GR-1" → frame_{t+h}. Don't conflate them. For variable-length episodes, sample fixed windows ([t, t+h]) and split at the **episode** level, never the window level, or windows leak across train/test.

**Data and split — approaches (your 6.2).** Four axes, increasing generalization strength:

1. _Held-out episodes (random)._ ~2% (≈4k of 200k); open-loop chunk error + closed-loop in sim. Unseen trajectories, same task family. Easiest, weakest claim.
2. _Held-out task conditions / instructions._ The benchmark defines 696 conditions (App. D); hold out a subset (or, on the human set, unseen instruction strings). Compositional/instruction generalization — the natural one, since the benchmark is built on conditions.
3. _Held-out visual configs._ Block colors/layout/lighting (sim varies these). Scene OOD — what QwenVLA's OOD eval measures.
4. _Cross-domain._ Train sim → eval real `language_table`. Sim-to-real; defer.

First pass: split at episode level (1) for fast open-loop sanity, and report the real number as **closed-loop success on held-out task conditions** (2) from the benchmark's eval families — that's the metric that actually means "it learned a policy." Keep a small fixed (instruction, init-state) set you re-roll every checkpoint as a qualitative tracker. Reserve (3)/(4).

**Decode at inference — the note (your 5.5).** Compute per-dimension 1st/99th percentiles **on the train split only** (q01, q99). Normalize `a_norm = clip(2·(a−q01)/(q99−q01) − 1, −1, 1)`. At inference invert: `a = (a_norm+1)/2·(q99−q01) + q01`, then clip to the env's bounds. Store q01/q99 _in the checkpoint_ and load the identical constants at eval — a silent mismatch here is a classic "trained fine, fails closed-loop." Clipping is lossy by design (can't command beyond [q01,q99]); fine for 2D pushes.

## Tightened ablation plan

Lock the first pass as one fixed config; sweep only cheap high-value knobs around it; sequence the architectural ablations after it works.

**First-pass config (config #1):** current frame (clean) → action chunk; oracle sim set; U-Net patchifier 0.16B; GPT-2 tokenizer; action flow in normalized space; h=16/Ta=8; per-instance t wired in (current still clean); your existing paper LR recipe. _Bar:_ action flow loss decreases and closed-loop success beats a trivial baseline (zero / constant-velocity) on held-out conditions.

**Sweep within the first pass (do these — cheap, high leverage):**

- **λ_act — the one that matters most.** You flagged it: under-weighting action gives a great image model that can't act. The repo exposes one `flow_loss_weight` + token-fraction weighting, so per-modality λ is a code change (your 2.4) — make it, then sweep λ_act at λ_img=1, LM=1. Budget your real tuning here, not on architecture.
- **Ta (execution horizon)** for closed-loop: DP shows this materially moves success; sweep {4,8,12} at fixed h=16.
- **Inference flow steps** for the action ODE: π0 uses ~10 Euler steps; cheap post-hoc sweep {4,10,16}.
- **Instruction dropout / CFG:** reuse the `prob_uncond` knob from CC12M on the action modality so you can apply classifier-free guidance toward the instruction and quantify instruction-following — that's how you'll later test "language steers the policy."

**Architectural ablations, sequenced after #1 works** (what each tests / dependency):

1. **History frames (1 vs 2 inputs)** — does temporal context help (velocity inference)? DP uses obs horizon 2. Cheapest change; both frames clean.
2. **GR-1 joint next-frame + action** (target frame_{t+h}) — does co-predicting dynamics improve the policy and give IDM-at-inference? Needs per-instance t.
3. **Goal-image planning** (target terminal frame) — can it imagine success and plan to it? Same machinery as (2), strongest test of the shared-space hypothesis.
4. **Encoder fork (b)/(c)** — semantic encoder for conditioning + VAE for generation, or VAE-latent state prediction only. The control-vs-reconstruction question we tabled.
5. **Init: CC12M vs scratch** (below).
6. _(Later)_ **long-horizon instruction stitching** — chaining sub-instructions in one episode; your 5.1 reasoning test.

Do 1→3 in order; they share the per-instance-t plumbing, so each one after the first is cheap.

**CC12M init — do priors help? (your Q3).** Run it as a clean two-arm ablation, but calibrate expectations _down_: the checkpoint is a 0.16B model that conditions weakly and generates poorly (thin prior); the action params are fresh regardless (partial-load into a now-two-modality model, your 2.5); and CC12M natural images are far from a synthetic 2D tabletop (limited visual transfer). The field's consensus that visual/VLM pretraining helps VLAs (RT-2, OpenVLA, π0, QwenVLA's reliance on a strong Qwen3.5 backbone) is real but is evidence for _strong_ priors — a pretrained semantic vision encoder would be a more promising transfer source than this in-house checkpoint. Two gotchas: the checkpoint must share the _exact_ architecture you train robots with (U-Net patchifier, 0.16B, same image-token grid) or the state-dict won't align; and decide preprocessing for the non-square 360×640 frames (center-crop vs letterbox vs anisotropic resize) before committing either arm.

## Coverage check + remaining open questions

Closed from my original list: shared-param thesis, encoder fork (a first, b/c noted), image-pipeline plan (U-Net), reasoning scope (text=instruction; stitching as the testable claim), conditioning collapse, tokenizer, eval mode, config #1.

Still genuinely open — straighten before/while building:

- **Per-instance vs per-type timesteps** in the repo (the stronger version of your 2.3). Both headline ablations need current-clean + goal/action-as-targets at once — different t on two instances of the _same_ modality. Untested in your scripts; verify in source.
- **The policy inference path.** Closed loop needs "given a text+image prefix, sample _only_ the action modality" — not the LM-then-switch-at-`[som]` default, and not the unconditional `generate_modality_only` the CC12M script calls. Confirm `sample()` accepts a fixed multimodal prefix and emits the next continuous modality on demand; if not, that's a patch and it gates _all_ closed-loop eval.
- **(512,) instruction buffer (your 5.3).** Almost certainly UTF-8 bytes zero-padded to 512 — Language Table's `decode_inst` does roughly `bytes(x[x>0]).decode('utf-8')`. So decode→string→re-tokenize with GPT-2; verify against the loader's decode helper before trusting it.
- **λ implementation (your 2.4)** — the per-modality-weight change must land before the λ_act sweep is even possible.
- **Action semantics** — confirm the 2D action is a _delta_ of the effector target (vs absolute setpoint) before computing percentiles; it changes what [-1,1] means.

The biggest single risk is the pair **per-instance timesteps + the policy inference path** — together they're the difference between "expressible in this repo" and "needs surgery," and both the goal-image and GR-1 ablations ride on them. I'd verify those two in the source before writing the data pipeline. Want me to go through the installed `transfusion.py` for those two specifically?

# Me
**Action Representation**
It's confirmed: Let's flow-match directly in normalized action space: 

For h: I like your suggestion of seeding with the closest comparable work: then sweeping. We'll start with predict ~16, then at inference execute ~8, and observe ~2 images (when we get to this point in the ablation, since the first pass will only use one image).

**Two images same modality**
When I test the two images ablation: I'll incorporate the architecture to include the *per instance noised timesteps*

**Goal-image: final vs end-of-chunk are two different ablations:**
Agreed: they are two different experiments entirely. 

**Data and split:**
Is the suggestion for the first training and eval pass to use the  `language_table_blocktoblock_oracle_sim` 196k episodes with 4k held out for open loop performance, but ultimately report the closed loop performance that I see on the sim?

**Decode at inference:**
Implementation noted: Compute per-dimension 1st/99th percentiles **on the train split only** (q01, q99). Normalize `a_norm = clip(2·(a−q01)/(q99−q01) − 1, −1, 1)`. At inference invert: `a = (a_norm+1)/2·(q99−q01) + q01`, then clip to the env's bounds. Store q01/q99 _in the checkpoint_ and load the identical constants at eval — a silent mismatch here is a classic "trained fine, fails closed-loop." Clipping is lossy by design (can't command beyond [q01,q99]); fine for 2D pushes.

Let's focus on achieving clarity on the checks I should make and details I need to decide for me to implement the stage 1 along with the hyperparameter sweeps. 

Are there any remaining questions that would block stage 1? 

# Claude
## Data/split — yes, with one nuance

`language_table_blocktoblock_oracle_sim` (200k oracle episodes, single block2block skill, scripted language) is the right Stage-1 set: clean, benchmark-aligned, and its terminal frame is the solved state — free goal labels for the later goal ablation. The 196k/4k episode-level split for open-loop sanity is fine. Two caveats: open-loop chunk error is only a sanity signal (treat closed-loop success as _the_ metric, as you said); and decide whether your closed-loop eval conditions are in-distribution (oracle episodes already cover the block2block conditions you roll out → acceptable for Stage 1, you're first proving it learns the skill at all) or held-out block/target configs (a real generalization claim). Run in-distribution closed-loop first, add held-out configs later.

## Decisions to lock before coding

Recommendation in each so nothing stays open:

1. **Sequence layout:** `[instruction text] [BOI current-frame EOI] [BOM action-chunk EOM]`, GPT-2 tokens. Keep the QwenVLA embodiment prefix even though it's degenerate — forward-compat, costs nothing.
2. **Image role — the one real Stage-1 fork.** Either (A) the current frame is also a flow target (carries λ_img; the model jointly learns to denoise the observation), or (B) pure clean conditioning, no flow loss (never generates images in Stage 1). Given your concern that image generation isn't solved yet, **B is the cleaner isolation of the action question — use it _if_ check 2 shows the repo can exclude a continuous modality from the flow loss; otherwise A with a small λ_img**, which is guaranteed expressible and achieves nearly the same isolation. Either way you feed the clean frame and sample only the action at inference. This is also _why Stage 1 needs no per-instance timesteps_: one image + one action are distinct modality _types_, so independent t per type suffices; per-instance t is only for the two-image ablations.
3. **Action modality:** dim_latent=2, h=16 tokens (`modality_num_dim=1`, `default_shape=(16,)`), identity encoder/decoder, flow target, percentile-normalized, q01/q99 stored in the checkpoint.
4. **Image preprocessing for 360×640:** do **not** anisotropically squish to square — that distorts the geometry a spatial pushing policy depends on (apparent distances and directions change). Use a centered workspace crop or letterbox-pad, then resize; verify the blocks stay in frame.
5. **Resolution/tokens:** start at your CC12M res (256 → 256 tokens) so VAE constants, the U-Net patchifier, and CC12M-init compatibility all carry over. Drop to 128 (64 tokens) only if throughput is tight, and re-check VAE reconstruction at 128 first.
6. **Horizons:** h=16 predict; Stage-1 history = 1 frame; closed-loop Ta=8. (History=2 is a later ablation.)
7. **Loss weights:** implement per-modality weights (LM=1, λ_img, λ_act) — your 2.4 change — and sweep λ_act.

## Checks to run, in order

(✕ = gates the closed-loop metric; the rest are training-setup confirmations)

1. **Two continuous modalities construct + forward.** Build with `dim_latent=(4,2)`, `modality_default_shape=((16,16),(16,))`, `modality_num_dim=(2,1)`. Confirm the **per-modality tuple forms** of what CC12M passes as scalars: `modality_encoder=(VAE, None)`, `modality_decoder=(…)`, and crucially whether `pre_post_transformer_enc_dec` accepts a **tuple/None per modality** (image → U-Net patchifier; action → built-in linear). This is the most likely spot the API forces a small change; the action half is low-risk (the toy confirms a 1D, no-encoder modality works), the open question is mixing it with the image's patchifier.
2. **Per-modality flow loss + optional no-loss conditioning.** Confirm `bd.flow` is a per-modality list aligned to modality index (CC12M's `sum(bd.flow)` implies it already is). If so, your weighting is `LM + Σ λ_i·bd.flow[i]`, computed yourself, bypassing the global `flow_loss_weight`. Also check here whether a continuous modality can be marked conditioning-only / excluded from the flow loss — that decides framing B vs A in decision 2.
3. **Independent t across modalities.** Confirm the forward samples the flow timestep **per modality instance**, not one t for the whole sequence. Per-type is enough for Stage 1; just verify image-t ≠ action-t.
4. **✕ Policy inference path — the one true gate.** Closed loop needs: given a fixed prefix `[text, clean image]`, **force-generate modality index 1 (action), shape (16,)** — not the default `sample()` that runs LM until it _chooses_ to emit `[som]`, and not the unconditional `generate_modality_only`. Confirm there's an API for "condition on this prefix, emit this specific modality now," or plan a small sampler patch. Without it you can't read an action out at inference at all. Check it first — everything else can be working and you're still stuck here.
5. **Meta/shape token on a 1D modality.** Confirm the `[meta]` token + decoder reshape handle `modality_num_dim=1` with dim_latent=2 (the toy's `(2,)` modality says 1D works; just confirm at `(16,)`/dim_latent=2).
6. **Known gotchas:** `use_flex_attn=False`; pass shapes/num_dim as tuples (gotcha #4); pin versions.

## What blocks Stage 1?

**Training: nothing conceptual.** Multi-continuous-modality training is the repo's advertised capability, the toy + CC12M scripts show both halves, and per-modality λ is a small change you control. Assemble it and smoke-test checks 1–3 and 5 — you can start Stage-1 _training_ today.

**The closed-loop metric is gated on two items, both engineering, not conceptual:** check 4 (the forced-modality inference path — verify-or-patch) and the **sim rollout harness** (install the `language_table` PyBullet env, wire receding-horizon control at Ta=8, define the block2block success criterion). Treat these as the long poles and build them in parallel with training, because training can converge and you still can't measure the thing that matters until both exist.




