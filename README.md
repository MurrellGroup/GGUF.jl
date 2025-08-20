# GGUF

[![Build Status](https://github.com/MurrellGroup/GGUF.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/MurrellGroup/GGUF.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/MurrellGroup/GGUF.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/MurrellGroup/GGUF.jl)

GGUF.jl is a Julia implementation of the [GGUF specification](https://github.com/ggml-org/ggml/blob/323951f1bdcdfbd5b5ff3a9a7c3770e63b1a560e/docs/gguf.md), a file format for quick loading and saving of deep learning models. Unlike tensor-only file formats like [safetensors](https://huggingface.co/docs/safetensors) (see also [SafeTensors.jl](https://github.com/FluxML/SafeTensors.jl)), GGUF encodes both the tensors and a standardized set of metadata.

> [!WARNING]
> GGUF.jl does not yet support quantized tensors. Attempting to load a quantized tensor will result in an error.

## Usage

```julia
using GGUF

path = "Qwen3-0.6B-bf16.gguf"

# Load the file with tensors memory-mapped from disk
gguf = GGUF.deserialize(path)

# Pass mmap=false to load tensors into memory
# gguf = GGUF.deserialize(path, mmap=false)

# Get an ordered dict of the metadata
metadata = gguf.metadata

# Get an ordered dict of the tensors
tensors = gguf.tensors

# do a surgery
delete!(tensors, "output.weight")

# GGUFObject need to be ordered collections of string-keyed pairs
new_gguf = GGUFObject(metadata, tensors)

# Serialize to a new file
GGUF.serialize("my_model.gguf", new_gguf)
```

## Installation

```julia
using Pkg
Pkg.Registry.add(url="https://github.com/MurrellGroup/MurrellGroupRegistry")
Pkg.add("GGUF")
```
