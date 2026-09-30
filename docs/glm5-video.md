# GLM 5.3 Flash video sampling

GLM's native video processor combines two sampled moments in each temporal
patch. Sparse sampling across a scene cut can therefore combine different
cards or scenes into one visual group. In a bounded two-card clip, the model
reported a crossfade that was not present in the original video.

For tasks that need each sampled frame kept separate, start the engine with:

```sh
VMLX_GLM5_VIDEO_TEMPORAL_MODE=preserve_frames vmlx-engine serve /path/to/model
```

Set the environment before starting the process. Omit it, or set
`native_pairs`, to use the checkpoint's native temporal grouping. This is a
GLM-specific serving option, not a chat-template parameter. It does not change
model weights, reasoning effort, or output-token limits.

Frame preservation repeats each sampled frame within its temporal patch and
retains that frame's timestamp. It does not insert crossfade pixels or discard
sampled frames. It normally doubles the number of visual groups relative to
native pairs. An eight-frame diagnostic used 2,392 visual tokens instead of
1,196 at the same resolution. Token budgets account for this; under a fixed
budget, spatial resolution may need to decrease. An impossible minimum budget
is rejected.

The mode has a separate runtime cache namespace, so native-pair checkpoints
cannot be restored into a frame-preserving process. Normal causal-prefix
reuse, chunked prefill and tool-continuation storage still apply within the
selected mode.

This option corrected the tested scene-cut interpretation on the revised
GLM-5.3-Flash-JANGH2 bundle in the app and streaming API. It is not a claim that
all videos, motion tasks, or model-generated answers are error-free. Native
pairs remain the default; frame preservation trades additional visual tokens
for separation of sampled moments.
