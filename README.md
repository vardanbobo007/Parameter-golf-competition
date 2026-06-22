# Hybrid nGPT / GPT / Mamba Submission

3-seed mean val_bpb = 1.173

!!! Please use Pytorch 2.11 to run the model, it is important.

This submission is a hybrid model mixing nGPT transformer layers, standard GPT layers, and Mamba2 layers.

I have the following layers:

```python
[nT, M, M, T, M, M, T, M, T, nT]
```

where:

- `nT`  stands for modified nGPT layer
- `T`  stands for GPT transformer layer
- `M`  stands for Mamba2 layer

## Main architecture details

### Modified nGPT layers

The nGPT layers are not exactly the same as in the original nGPT paper.

The main differences are:

- I normalize only the Q and K matrices after every optimizer step. For this short training setup, this worked better than normalizing more all matrices.
- The attention block and MLP block both receive the input directly, i.e. MLP gets the input not the result of the attention
- The final nGPT layer output is:

```python
h_att + h_mlp - h
```

where `h` is the layer input, and `h_att`, `h_mlp` are the normalized outputs of the attention and MLP parts.

Putting nGPT layers at the beginning and end of the model helpes a lot. The model became much more stable numerically, and I could move some tensors to bf16 without hurting the loss.

### GPT-style transformer layers

The `T` layers are standard transformer-style layers, with one difference:

- The MLP gets the input of the layer directly, not after attention.
- The T layer output is:

```python
h_att(h) + h_mlp(h)
```

where `h_att` and `h_mlp` are the outputs of the attention and MLP blocks.

The MLP width is:

```python
(3.5) * emb_dim
```

The activation is:

```python
LeakyReLU(x) ** 2
```

I tried to fuse the MLP manually and also looked at existing kernels. But for this embedding size and these tensor shapes, `torch.compile` in PyTorch 2.11 already did a very good job, at least in my setup.

### Mamba2 layers

The Mamba2 layers are standard Mamba2 layers.

Parameters:

```text
d_state  = 128
d_conv   = 4
expand   = 2
head_dim = 64
```

I originally wanted to use Mamba to make 8k or 16k context and train the model using fp8. However, I could not get significant speed-up with fp8 with this model.

I could not get `torch.compile` to work on the Mamba layers without graph breaks. There is a non-contiguous tensor inside the Mamba path. Making it contiguous  makes the model slower. Because of this, I compiled layers separately or only compiled contiguous transformer blocks.
I could not find simple solution to this problem. I did not have enought time to study this problem deeper.

### Numerical stability

The model is very stable. No gradient spikes. We can look at the conditions of matrices (ratioo of the biggest and smallest eigenvalues) and see that it's not big. This means that the model is not sensative to small numerical changes. 
Probably, this is the reason that I can force some elemnts to be in bf16 (when torch.autocast does not want) and without any loss increase.

## Context length and fp8

The model uses:

```text
max_context = 8192
```

The original plan was to mix transformer and Mamba layers, use 8k or 16k context, and then speed things up with fp8. In practice, the fp8 speedup was not enough for this setup, so I dropped the idea.
I think if we have embedding dimesnion = 1024, then fp8 will speed the model up a lot, but due to 16MB we can not afford emb_dim =1024 

## Optimizer

I use  AdamW and Muon.

Weight decay:

```text
weight_decay = 0.75
```
I have 4 different groups of parameters.
1. Emb/out matrix. Embedding and output matrices are tied.  I use AdamW for this group
2. All non 2D parameters. I use AdamW for this group
3. 2D parameters of nGPT and GPT. I use Muon for this group
4. 2D parameters of Mamba. I also, use Muon for this

I thought that GPT and Mamba layers are very different, so for this short run their best optimizer parameters are different and in this short run it is wrong unify them. However, it is required a lot of compute and time to test and find best optimizer parameters. I could not fully use this idea.

## Evaluation

I use sliding-window evaluation. It is already standard technique among all participants.

## Rule compliance

This submission follows the competition rules:

- No TTT.
- No tokenizer changes.
- No data processing changes.
- No validation data used during training.
- Training is under the 600 second limit.
- Evaluation is under the 600 second limit.
- Final artifact is under 16,000,000 bytes.

## Notes

This is probably not the best possible version of this idea.

I think the model could get lower loss by making it narrower and adding another layer. Recursive layers, like in some other submissions, would  help. I did not have enough time to test all these combinations.
For example, adding reccursive layers makes the size of the model much bigger that 16MB (with the same loss). Higher weight decay solves this problem, but you need to retune all existing parameters. This requires a lot of time and compute.

I think this hybrid construction is promising. The nGPT layers at the start and end are especially useful for stability.

One thing: the second run may be faster than the first one because `torch.compile` cache is already saved. Delete cache for fair timing. 


## Results

Final results over 3 independent runs:

| Seed | val_bpb | Artifact size, bytes |
|---:|---:|---:|
| 42  | 1.1729 | 15,472,940 |
| 1337 | 1.1732 | 15,492,405 |
| 2025 | 1.1734 | 15,472,940 |

The 3-seed mean is:

```text
val_bpb = 1.173