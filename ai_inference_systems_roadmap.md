# AI Inference Systems Roadmap

## Purpose

This roadmap is designed to take a computer engineering student from low-level systems fundamentals to serious AI inference systems work.

The target role family is:

- ML Systems Engineer
- Inference Systems Engineer
- AI Runtime Engineer
- Compiler Engineer for ML Systems
- Performance Engineer
- GPU Runtime Engineer
- AI Infrastructure Engineer

The long-term goal is to become competitive for companies working on high-performance inference, custom AI hardware, runtimes, compilers, and serving infrastructure, such as Cerebras, Groq, NVIDIA, AMD, Tenstorrent, SambaNova, Modular, Google TPU/XLA, AWS Trainium/Inferentia, Apple Neural Engine teams, and similar organizations.

This roadmap is project-driven.

Do not treat it as a list of topics to memorize.

The core loop is:

```text
Build something small.
Measure it.
Find the bottleneck.
Learn the concept that explains the bottleneck.
Improve it.
Write down what changed.
Repeat.
```

The north star question is:

```text
Can I explain and improve the path from model weights to generated tokens?
```

---

# High-Level Roadmap

The correct learning order is:

```text
1. C++ benchmarking and matrix multiplication
2. CPU cache behavior and memory layout
3. Tensor operations
4. Attention from scratch
5. Tiny transformer inference
6. KV cache, prefill, and decode
7. Profiling and performance analysis
8. Quantization and memory optimization
9. Runtime graph design and operator fusion
10. CUDA and GPU acceleration basics
11. Inference serving and batching
12. Production metrics and system design
13. Existing systems deep dives
14. Advanced distributed inference
```

This order matters.

Do not start with TensorRT, MLIR, CUDA wizardry, or giant LLM serving frameworks before you understand the basic execution path.

Start with the fundamentals:

```text
matmul -> tensors -> attention -> tiny transformer -> KV cache -> quantization -> graph runtime -> GPU basics -> serving system
```

---

# Phase 1: C++ Performance Foundations

## Goal

Become comfortable writing, benchmarking, and profiling low-level code.

Inference systems are performance-sensitive. Before touching LLM runtimes, learn how normal code becomes slow.

## Topics

Learn:

- C++ performance basics
- memory layout
- stack vs heap
- cache hierarchy
- cache lines
- spatial locality
- temporal locality
- alignment
- row-major layout
- basic benchmarking
- Linux `perf`
- flamegraphs
- compiler optimization flags
- basic SIMD awareness

Defer:

- deep branch prediction
- interrupt handling
- memory controller internals
- lock-free data structures
- wait-free algorithms
- NUMA deep dives

Those are useful later, but not first.

## Project: Matmul Lab

Build a small C++ project:

```text
matmul-lab/
  include/
    matrix.hpp
    timer.hpp
  src/
    naive_matmul.cpp
    tiled_matmul.cpp
    benchmark.cpp
  results/
    results.csv
  README.md
```

Implement:

```text
1. Naive matrix multiplication
2. Cache-tiled matrix multiplication
3. Loop-reordered matrix multiplication
4. Optional: basic SIMD experiment
5. Benchmark harness
6. CSV output
```

Test across multiple matrix sizes:

```text
64x64
128x128
256x256
512x512
1024x1024
```

Measure:

```text
runtime
GFLOP/s
cache misses if possible
speedup vs naive
```

## Required Questions

By the end of this phase, be able to answer:

```text
Why is naive matmul slow?
Why does loop ordering matter?
What is a cache line?
What is spatial locality?
What is temporal locality?
Why does cache tiling help?
Why do benchmark results vary between runs?
What does a profiler tell you that intuition does not?
```

## Deliverables

- Working C++ matmul benchmark repo
- README explaining naive vs tiled matmul
- Benchmark table
- Short writeup: "Why naive matrix multiplication is slow"

---

# Phase 2: Tensor Operations

## Goal

Understand the primitive operations that neural networks execute.

Inference is mostly tensor operations plus careful memory movement.

## Topics

Learn:

- tensor dimensions
- shapes
- strides
- layouts
- row-major vs column-major
- contiguous vs non-contiguous tensors
- broadcasting
- transposition
- reductions
- elementwise operations
- matrix multiplication
- softmax
- normalization
- embedding lookup

Prioritize for LLM inference:

```text
matmul
embedding lookup
softmax
RMSNorm / LayerNorm
elementwise activation
attention
```

De-prioritize early:

```text
convolutions
pooling
image-specific layers
```

They are useful, but LLM inference should be the main path.

## Project: Tensor Lab

Build:

```text
tensor-lab/
  include/
    tensor.hpp
    ops.hpp
  src/
    tensor.cpp
    matmul.cpp
    softmax.cpp
    rmsnorm.cpp
    elementwise.cpp
    benchmark.cpp
  tests/
    test_tensor.cpp
    test_ops.cpp
  README.md
```

Implement:

```text
1. Basic tensor struct
2. Shape metadata
3. Strides
4. Contiguous storage
5. Matrix multiplication
6. Elementwise addition/multiplication
7. Softmax
8. RMSNorm or LayerNorm
9. Simple tests
```

## Required Questions

Be able to answer:

```text
What is a tensor stride?
Why can transposes be expensive?
What does contiguous memory mean?
Why does layout affect performance?
What is broadcasting?
What is a reduction?
Why is softmax numerically sensitive?
What does normalization do during inference?
```

## Deliverables

- Minimal tensor library in C++
- Unit tests for tensor operations
- README explaining shapes, strides, and layouts
- Benchmarks for matmul and softmax

---

# Phase 3: Attention From Scratch

## Goal

Understand attention as an actual computation, not just a diagram.

Attention is one of the core operations in transformer inference.

## Topics

Learn:

- query, key, value matrices
- scaled dot-product attention
- softmax over attention scores
- causal masking
- multi-head attention
- sequence length scaling
- memory cost of attention
- why attention can become expensive

Formula:

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

## Project: Attention Lab

Build:

```text
attention-lab/
  src/
    attention_numpy.py
    attention.cpp
    benchmark.cpp
  tests/
    test_attention.py
    test_attention.cpp
  README.md
```

Implement attention first in Python/NumPy, then in C++.

Start with:

```text
single-head attention
no KV cache
small tensors
clear tests
```

Then add:

```text
causal mask
multi-head attention
timing vs sequence length
```

## Required Questions

Be able to answer:

```text
What are Q, K, and V?
Why do we divide by sqrt(d_k)?
What does the softmax operate over?
What does causal masking do?
Why does attention scale badly with sequence length?
What changes during autoregressive decoding?
```

## Deliverables

- Python attention implementation
- C++ attention implementation
- Tests comparing outputs
- Benchmark showing cost as sequence length increases
- README explaining attention step-by-step

---

# Phase 4: Tiny Transformer Inference

## Goal

Build a minimal transformer inference runtime.

This is the first major project that proves you understand the path from input tokens to output logits.

## Topics

Learn:

- token IDs
- embeddings
- transformer blocks
- attention
- MLP/feed-forward layers
- residual connections
- RMSNorm/LayerNorm
- logits
- sampling
- greedy decoding
- temperature
- top-k sampling
- top-p sampling

## Project: TinyInfer

Build:

```text
tiny-infer/
  include/
    tensor.hpp
    model.hpp
    linear.hpp
    attention.hpp
    mlp.hpp
    rmsnorm.hpp
    sampler.hpp
  src/
    tensor.cpp
    linear.cpp
    attention.cpp
    mlp.cpp
    rmsnorm.cpp
    sampler.cpp
    main.cpp
  tests/
    test_linear.cpp
    test_attention.cpp
    test_sampler.cpp
  README.md
```

The runtime should execute this path:

```text
tokens
  -> embedding lookup
  -> transformer block
  -> final norm
  -> logits
  -> sampler
  -> next token
```

Start with a toy model. Do not worry about loading Llama weights immediately.

## Implementation Steps

Step 1:

```text
Hardcode tiny random weights.
Run one forward pass.
Print logits.
```

Step 2:

```text
Add greedy sampling.
Generate one next token.
```

Step 3:

```text
Add generation loop.
Generate N tokens.
```

Step 4:

```text
Add timing per operation.
Track tokens/sec.
```

## Required Questions

Be able to answer:

```text
What happens in a transformer block?
What is an embedding lookup?
What is an MLP layer doing?
What are logits?
What does sampling do?
Why does generation happen one token at a time?
Why is autoregressive decoding different from normal batch inference?
```

## Deliverables

- Tiny transformer inference runtime in C++
- Greedy decoding
- Basic sampling
- Timing instrumentation
- README explaining the forward pass

---

# Phase 5: KV Cache, Prefill, and Decode

## Goal

Understand the core memory problem in LLM inference.

The KV cache is one of the most important concepts in LLM serving.

## Topics

Learn:

- autoregressive generation
- prefill phase
- decode phase
- KV cache
- memory growth with sequence length
- batch size effects
- context length effects
- cache fragmentation
- paged KV cache concept
- continuous batching concept

## Key Concepts

Prefill:

```text
The model processes the input prompt tokens.
This phase has more parallelism.
```

Decode:

```text
The model generates one token at a time.
This phase is latency-sensitive and often memory-bound.
```

KV cache:

```text
Stores previous keys and values so the model does not recompute them every token.
Saves compute but consumes memory.
```

## Project: KV Cache Lab

Extend `tiny-infer`:

```text
tiny-infer/
  include/
    kv_cache.hpp
  src/
    kv_cache.cpp
    attention_cached.cpp
  benchmarks/
    kv_cache_benchmark.cpp
```

Implement:

```text
1. Naive attention without cache
2. Cached attention with KV cache
3. Measure time per generated token
4. Measure memory usage as sequence length grows
5. Compare prefill vs decode timing
```

## Experiments

Run experiments varying:

```text
prompt length
generated tokens
number of layers
hidden size
number of heads
batch size
```

Track:

```text
prefill latency
decode latency per token
tokens/sec
KV cache memory
```

## Required Questions

Be able to answer:

```text
What is prefill?
What is decode?
Why does KV cache exist?
Why does KV cache memory grow with sequence length?
Why is decode often memory-bound?
Why does long context make serving harder?
Why can batching improve throughput but hurt latency?
```

## Deliverables

- KV cache implementation
- Prefill vs decode benchmark
- Memory growth chart
- README explaining KV cache tradeoffs

---

# Phase 6: Profiling and Performance Analysis

## Goal

Learn to diagnose performance problems using tools instead of guessing.

## Topics

Learn:

- latency
- throughput
- p50/p95/p99 latency
- memory bandwidth
- cache misses
- flamegraphs
- CPU profiling
- allocation profiling
- instruction-level hotspots
- benchmark methodology

## Tools

Use:

```text
perf
gprof or alternatives
valgrind / massif
sanitizers
flamegraph tools
time
hyperfine
```

Optional later:

```text
Intel VTune
Nsight Systems
Nsight Compute
```

## Project: Inference Profiler

Extend `tiny-infer` with profiling output:

```text
tiny-infer/
  profiler/
    profiler.hpp
    profiler.cpp
  results/
    profile_runs.csv
```

Track:

```text
time in matmul
time in attention
time in MLP
time in sampling
time in memory allocation
peak memory
tokens/sec
```

## Benchmark Discipline

Always include:

```text
warmup runs
multiple trials
average latency
p50 latency
p95 latency
fixed input sizes
fixed generation lengths
compiler flags
machine specs
```

## Required Questions

Be able to answer:

```text
What is the difference between latency and throughput?
What is p95 latency?
Why are warmup runs important?
Why can allocation show up as a bottleneck?
Why is profiling better than guessing?
What does a flamegraph show?
What does a cache miss indicate?
```

## Deliverables

- Profiler integrated into tiny inference runtime
- Benchmark reports
- Flamegraph or perf report
- README explaining bottlenecks found

---

# Phase 7: Quantization and Memory Optimization

## Goal

Understand how inference systems reduce memory usage and improve speed.

Quantization is central to practical inference.

## Topics

Learn:

- fp32
- fp16
- bf16
- int8
- int4
- per-tensor quantization
- per-channel quantization
- scale factors
- zero points
- symmetric vs asymmetric quantization
- weight-only quantization
- activation quantization
- weight packing
- dequantization overhead
- accuracy vs performance tradeoffs

## Key Questions

Quantization is not just:

```text
smaller number = faster model
```

The real question is:

```text
Does reducing memory bandwidth outweigh dequantization overhead?
```

Sometimes yes. Sometimes no.

## Project: QuantBench

Build:

```text
quant-bench/
  include/
    quant.hpp
    packed_weights.hpp
  src/
    fp32_linear.cpp
    int8_linear.cpp
    int4_pack.cpp
    benchmark.cpp
    error_analysis.cpp
  results/
    speed_results.csv
    error_results.csv
  README.md
```

Implement:

```text
1. FP32 linear layer
2. INT8 quantized linear layer
3. Optional INT4 weight packing
4. Quantization/dequantization functions
5. Error measurement
6. Speed benchmark
```

Measure:

```text
runtime
memory usage
model size
average absolute error
max error
speedup vs fp32
```

## Required Questions

Be able to answer:

```text
What does quantization save?
What is dequantization?
What is a scale factor?
What is per-channel quantization?
Why can int4 be harder than int8?
Why might quantization fail to speed things up?
How does quantization affect memory bandwidth?
```

## Deliverables

- Quantized linear layer implementation
- FP32 vs INT8 benchmark
- Error analysis
- README explaining tradeoffs

---

# Phase 8: C++ Runtime and Memory Management

## Goal

Learn the systems techniques used inside runtimes.

## Topics

Learn:

- custom allocators
- arena allocation
- memory pools
- buffer reuse
- object lifetime
- smart pointers
- stack vs heap allocation
- heap fragmentation
- manual alignment
- circular buffers
- zero-copy design
- memory pinning concept
- NUMA awareness later

## Project: Runtime Memory Lab

Build:

```text
runtime-memory-lab/
  include/
    arena_allocator.hpp
    buffer_pool.hpp
  src/
    arena_allocator.cpp
    buffer_pool.cpp
    benchmark.cpp
  README.md
```

Then integrate into `tiny-infer`:

```text
tiny-infer/
  runtime/
    arena_allocator.hpp
    buffer_pool.hpp
```

Implement:

```text
1. Arena allocator
2. Buffer pool for temporary tensors
3. Allocation counter
4. Peak memory tracker
5. Benchmark with and without buffer reuse
```

## Required Questions

Be able to answer:

```text
Why are frequent heap allocations bad?
What is an arena allocator?
What is a memory pool?
What is buffer reuse?
Why does runtime memory planning matter?
What is fragmentation?
Why do inference runtimes avoid unnecessary copies?
```

## Deliverables

- Arena allocator
- Buffer pool
- Runtime benchmark before/after memory reuse
- README explaining allocation strategy

---

# Phase 9: Runtime Graph Design and Operator Fusion

## Goal

Understand how inference runtimes and ML compilers optimize computation graphs.

## Topics

Learn:

- computational graphs
- graph nodes
- graph edges
- tensors as values
- operators
- graph serialization
- constant folding
- dead code elimination
- operator fusion
- memory planning
- subgraph partitioning
- backend code generation
- lowering

## Key Idea

A neural network can be represented as a graph:

```text
input -> matmul -> add bias -> activation -> output
```

Runtime systems optimize this graph before execution.

Operator fusion reduces memory traffic.

Example:

```text
matmul -> add -> activation
```

Can become:

```text
matmul_add_activation
```

This avoids writing intermediate tensors to memory.

## Project: MiniGraph

Build:

```text
mini-graph/
  graph/
    node.py
    tensor.py
    graph.py
  passes/
    constant_folding.py
    dead_code_elimination.py
    op_fusion.py
    memory_planner.py
  codegen/
    cpp_codegen.py
  examples/
    mlp_graph.py
  README.md
```

Implement:

```text
1. Graph representation
2. Tensor metadata
3. Constant folding pass
4. Dead code elimination pass
5. Simple op fusion pass
6. Memory reuse planner
7. Optional C++ code generation
```

## Experiments

Compare:

```text
unoptimized graph
constant-folded graph
fused graph
memory-planned graph
```

Measure:

```text
number of ops
number of temporary tensors
estimated memory traffic
runtime if codegen exists
```

## Required Questions

Be able to answer:

```text
What is an IR?
What is a computational graph?
What is constant folding?
What is dead code elimination?
What is operator fusion?
Why does fusion reduce memory traffic?
What is memory planning?
What does lowering mean?
```

## Deliverables

- Tiny graph runtime or compiler
- Optimization passes
- Before/after graph examples
- README explaining graph optimization

---

# Phase 10: CUDA and GPU Basics

## Goal

Understand the GPU execution model and why GPU inference optimization is hard.

Do not start here. Come here after CPU matmul, tensors, attention, and profiling.

## Topics

Learn:

- CUDA programming model
- threads
- blocks
- grids
- warps
- streaming multiprocessors
- global memory
- shared memory
- registers
- memory coalescing
- occupancy
- warp scheduling
- kernel launch overhead
- synchronization
- CUDA streams
- async execution

## Project: CUDA Lab

Build:

```text
cuda-lab/
  src/
    vector_add.cu
    matmul_naive.cu
    matmul_tiled.cu
    softmax.cu
    benchmark.cu
  README.md
```

Implement:

```text
1. Vector addition
2. Naive CUDA matmul
3. Tiled CUDA matmul using shared memory
4. Simple softmax kernel
5. CPU vs GPU benchmark
```

## Required Questions

Be able to answer:

```text
What is a CUDA thread?
What is a block?
What is a warp?
What is an SM?
What is global memory?
What is shared memory?
What is memory coalescing?
Why does tiled GPU matmul use shared memory?
What is kernel launch overhead?
What is occupancy?
```

## Deliverables

- CUDA matmul implementation
- CPU vs GPU benchmark
- README explaining GPU execution model
- Notes on memory coalescing and shared memory

---

# Phase 11: Inference Serving Systems

## Goal

Move from local runtime to serving many requests.

Inference serving is a systems problem involving queues, batching, latency, throughput, cancellation, and resource management.

## Topics

Learn:

- HTTP APIs
- streaming responses
- request queues
- batching
- continuous batching concept
- request cancellation
- backpressure
- timeout handling
- admission control
- p50/p95/p99 latency
- throughput
- tokens/sec
- load testing
- observability
- model versioning
- resource isolation
- fault tolerance basics

## Project: MiniServe

Build:

```text
mini-serve/
  server/
    main.py or main.cpp
    request_queue.py
    scheduler.py
    batcher.py
    streamer.py
    metrics.py
  load_test/
    load_test.py
  README.md
```

Use one of these backends:

```text
Ollama
llama.cpp
your tiny-infer runtime
```

Start simple:

```text
User request -> queue -> model call -> streaming response
```

Then add:

```text
batching
cancellation
metrics
load testing
```

## Metrics

Track:

```text
request count
queue wait time
time to first token
tokens/sec
p50 latency
p95 latency
p99 latency
active requests
batch size
error rate
```

## Required Questions

Be able to answer:

```text
Why is serving one user different from serving many users?
What is request batching?
Why does batching improve throughput?
Why can batching hurt latency?
What is time to first token?
What is tail latency?
What is backpressure?
What does cancellation require?
Why are metrics necessary in production inference?
```

## Deliverables

- Inference server
- Streaming generation
- Queueing
- Basic batching
- Load test script
- Metrics output
- README with latency/throughput results

---

# Phase 12: Parallelism Models

## Goal

Understand the main forms of parallelism used in inference systems.

## Topics

Learn:

- data parallelism
- model parallelism
- tensor parallelism
- pipeline parallelism
- task parallelism
- asynchronous execution
- synchronization
- load balancing
- distributed inference
- multi-device serving
- communication overhead

## Learn Conceptually First

Do not implement massive distributed inference immediately.

First understand the differences:

```text
Data parallelism:
Same model replicated across devices. Different requests go to different replicas.

Tensor parallelism:
One layer's tensor computation is split across devices.

Pipeline parallelism:
Different layers run on different devices.

Task parallelism:
Different runtime tasks run concurrently.
```

## Project: Parallelism Notes and Simulation

Build a simple simulator:

```text
parallelism-sim/
  simulator.py
  scenarios/
    data_parallel.py
    pipeline_parallel.py
    queueing.py
  README.md
```

Simulate:

```text
request arrivals
processing time
batch size
queue wait
multiple workers
pipeline stages
```

## Required Questions

Be able to answer:

```text
What is data parallelism?
What is tensor parallelism?
What is pipeline parallelism?
Why does communication overhead matter?
Why can pipeline parallelism create bubbles?
Why is load balancing hard?
When is distributed inference worth the complexity?
```

## Deliverables

- Simple inference scheduling simulator
- README comparing parallelism strategies
- Notes on tradeoffs

---

# Phase 13: Production Deployment and System Architecture

## Goal

Understand how inference systems run in production.

## Topics

Learn:

- API design
- service boundaries
- latency budget allocation
- scalability patterns
- model versioning
- rollout strategies
- observability
- logging
- tracing
- metrics
- reliability engineering
- cost optimization
- resource isolation
- fault tolerance
- containerization
- build systems
- deployment basics

De-prioritize early:

```text
serverless architectures
large Kubernetes setups
complex cloud infra
```

Learn normal services first.

## Project: Productionized MiniServe

Extend `mini-serve`:

```text
mini-serve/
  Dockerfile
  docker-compose.yml
  config/
    config.yaml
  observability/
    metrics.md
  README.md
```

Add:

```text
Dockerfile
config file
health check endpoint
metrics endpoint
structured logs
basic model versioning
load test results
```

## Required Questions

Be able to answer:

```text
What is a latency budget?
What is a health check?
What is model versioning?
Why does observability matter?
What is resource isolation?
What is a rollout?
Why does cost matter in inference?
What causes tail latency in production?
```

## Deliverables

- Dockerized inference server
- Metrics endpoint
- Health check endpoint
- Load testing report
- README explaining production considerations

---

# Phase 14: Existing Systems Deep Dives

## Goal

Learn from real inference systems after building your own simplified versions.

Do not read these projects randomly. Read them with specific questions.

## Systems to Study

### llama.cpp

Use it to learn:

```text
local inference
GGUF format
CPU inference
quantization
backend abstraction
tensor representation
memory management
```

Questions to ask:

```text
How are tensors represented?
How are quantized weights stored?
How are backends abstracted?
How does the runtime manage memory?
```

### ggml

Use it to learn:

```text
tensor library design
computation graphs
CPU kernels
backend execution
```

Questions to ask:

```text
How does the graph get built?
How are operations represented?
How does memory planning work?
```

### vLLM

Use it to learn:

```text
high-throughput serving
continuous batching
PagedAttention
KV cache management
request scheduling
```

Questions to ask:

```text
How are requests scheduled?
How is KV cache managed?
Why does paging help?
How does continuous batching work?
```

### TensorRT-LLM

Use it to learn:

```text
NVIDIA inference optimization
engine building
kernel fusion
precision calibration
plugin development
production GPU serving
```

Questions to ask:

```text
What does engine building optimize?
How does TensorRT handle dynamic shapes?
What does precision calibration do?
What are plugins for?
```

### MLC LLM / TVM

Use it to learn:

```text
compilation for inference
graph lowering
operator scheduling
hardware-specific code generation
runtime abstraction
```

Questions to ask:

```text
What is lowered?
What is optimized at graph level?
What is optimized at kernel level?
How does codegen target different backends?
```

### SGLang

Use it to learn:

```text
structured generation
serving abstractions
LLM program execution
runtime scheduling
```

Questions to ask:

```text
How does structured generation affect serving?
How are requests represented?
How does runtime scheduling work?
```

## Required Deliverables

For each system, write a short study note:

```text
What problem does this system solve?
What is the key idea?
What part of my own project does this relate to?
What did I learn from reading it?
What would I copy or avoid?
```

---

# Phase 15: Papers and Technical Writeups

## Goal

Read papers after building enough context to understand why they matter.

Do not start with papers. Build first, then read.

## Recommended Topics

Read about:

```text
FlashAttention
PagedAttention / vLLM
Speculative decoding
Continuous batching
Quantization
KV cache compression
MLIR
TVM
Tensor parallelism
Pipeline parallelism
Mixture-of-Experts inference
```

## Reading Method

For each paper/writeup, answer:

```text
What problem is being solved?
Why does the naive approach fail?
What is the main idea?
What assumptions does the method make?
What are the tradeoffs?
How would I implement a toy version?
How does this relate to my projects?
```

## Deliverable

Create:

```text
paper-notes/
  flashattention.md
  pagedattention.md
  speculative_decoding.md
  quantization.md
  mlir_intro.md
```

Each note should include:

```text
summary
key diagrams if helpful
implementation notes
questions
possible project extensions
```

---

# Six-Month Execution Plan

## Month 1: Matmul and Profiling

Build:

```text
naive matmul
tiled matmul
benchmark harness
perf notes
```

Learn:

```text
cache lines
locality
alignment
basic SIMD
perf
flamegraphs
```

Deliverable:

```text
matmul-lab repo with README explaining why tiled matmul beats naive matmul
```

## Month 2: Tensor Ops and Attention

Build:

```text
tensor class
matmul
softmax
RMSNorm/LayerNorm
attention
```

Learn:

```text
strides
layouts
reductions
broadcasting
attention math
```

Deliverable:

```text
attention forward pass in C++ with tests
```

## Month 3: Tiny Transformer Inference

Build:

```text
embedding lookup
transformer block
MLP
sampling
KV cache basics
```

Learn:

```text
autoregressive generation
prefill
decode
logits
sampling
```

Deliverable:

```text
toy model forward/decode loop with tokens/sec benchmark
```

## Month 4: Quantization

Build:

```text
int8 linear layer
weight packing experiment
quantization error measurement
benchmark comparison
```

Learn:

```text
int8/int4
scale factors
dequantization
memory bandwidth
accuracy/speed tradeoff
```

Deliverable:

```text
fp32 vs int8 benchmark report
```

## Month 5: Runtime/Compiler Mini-Project

Build:

```text
tiny graph IR
constant folding
op fusion
memory reuse
optional C++ codegen
```

Learn:

```text
IR
passes
fusion
memory planning
lowering
```

Deliverable:

```text
mini graph optimizer with before/after benchmarks
```

## Month 6: Serving System

Build:

```text
HTTP inference server
request queue
streaming
batching
metrics
load test
```

Learn:

```text
latency
throughput
p95/p99
backpressure
batch scheduling
```

Deliverable:

```text
mini inference server with load-test results
```

---

# Twelve-Month Execution Plan

## Months 1-2: Systems Foundation

Focus:

```text
C++
benchmarking
matmul
cache behavior
profiling
basic memory management
```

Projects:

```text
matmul-lab
tensor-lab
```

## Months 3-4: Transformer Inference

Focus:

```text
attention
transformer blocks
sampling
KV cache
prefill/decode
```

Projects:

```text
attention-lab
tiny-infer
kv-cache-lab
```

## Months 5-6: Quantization and Runtime Memory

Focus:

```text
int8/int4
weight packing
dequantization
arena allocation
buffer reuse
memory profiling
```

Projects:

```text
quant-bench
runtime-memory-lab
```

## Months 7-8: Compiler/Runtime Design

Focus:

```text
graph IR
operator fusion
constant folding
memory planning
codegen
```

Project:

```text
mini-graph
```

## Months 9-10: GPU Basics

Focus:

```text
CUDA
warps
blocks
shared memory
memory coalescing
CUDA matmul
softmax kernel
```

Project:

```text
cuda-lab
```

## Months 11-12: Serving and Production

Focus:

```text
request scheduling
batching
streaming
metrics
load testing
deployment
existing systems deep dives
```

Projects:

```text
mini-serve
productionized-mini-serve
system study notes
```

---

# Portfolio Targets

By the end of this roadmap, aim to have 3 polished projects.

Do not create 12 half-finished repos.

## Project 1: TinyInfer++

Description:

```text
A C++ transformer inference runtime implementing tensor operations, attention, KV caching, sampling, profiling, and memory tracking.
```

Must include:

```text
C++ implementation
KV cache
sampling
tokens/sec benchmark
memory tracking
clear README
tests
```

Resume bullet:

```text
Built a C++ transformer inference runtime with attention, KV caching, token sampling, memory tracking, and benchmark instrumentation for tokens/sec and per-layer latency.
```

## Project 2: QuantBench

Description:

```text
A quantized inference kernel benchmarking suite comparing fp32 and int8/int4 linear layers.
```

Must include:

```text
fp32 linear layer
int8 linear layer
optional int4 packing
error analysis
speed benchmark
memory comparison
clear README
```

Resume bullet:

```text
Implemented quantized linear layers with int8/int4 weight experiments, benchmarking speed, memory footprint, and numerical error against fp32 baselines.
```

## Project 3: MiniServe

Description:

```text
A toy LLM inference server with request queueing, streaming responses, batching, cancellation, and latency/throughput metrics.
```

Must include:

```text
HTTP API
streaming
request queue
basic batching
cancellation
p50/p95 latency
tokens/sec metrics
load testing
clear README
```

Resume bullet:

```text
Built an inference serving runtime with streaming generation, request queueing, dynamic batching, cancellation, and p95 latency/token-throughput instrumentation.
```

## Optional Project 4: MiniGraph

Description:

```text
A toy ML graph compiler/runtime with graph IR, constant folding, operator fusion, memory planning, and optional C++ codegen.
```

Resume bullet:

```text
Built a neural network graph optimizer with constant folding, operator fusion, memory planning, and C++ backend generation for simplified inference workloads.
```

---

# Skills Checklist

## Low-Level Systems

Know:

```text
C++
CMake
Linux
gdb
perf
valgrind
sanitizers
flamegraphs
cache hierarchy
memory layout
threading basics
atomics basics
```

Do not over-focus early on:

```text
lock-free structures
wait-free algorithms
deep kernel internals
interrupt handling
```

## Math and ML

Know:

```text
linear algebra basics
matrix multiplication
softmax
normalization
attention
MLP layers
embeddings
tokenization
sampling
KV cache
prefill/decode
quantization
```

Do not over-focus early on:

```text
training giant models
backpropagation details
deep ML theory
CV-specific architectures
```

## Runtime and Compiler

Know:

```text
computational graphs
IRs
passes
constant folding
dead code elimination
operator fusion
memory planning
lowering
backend codegen basics
```

## GPU

Know:

```text
CUDA basics
threads
blocks
warps
SMs
global memory
shared memory
memory coalescing
occupancy
streams
kernel launch overhead
```

## Serving Systems

Know:

```text
request queues
batching
streaming
cancellation
backpressure
latency percentiles
throughput
tokens/sec
observability
load testing
model versioning
fault tolerance basics
```

---

# Final Self-Test Questions

You are making real progress if you can answer these clearly:

```text
Why is naive matmul slow?
What does cache tiling improve?
What is a tensor stride?
Why can transposes be expensive?
What does attention compute?
What happens during prefill?
What happens during decode?
Why does KV cache dominate memory?
Why is decode often memory-bound?
Why does batching improve throughput?
Why can batching hurt latency?
What does quantization save?
When can quantization fail to speed things up?
What does op fusion reduce?
What is memory planning?
What does a profiler tell you that intuition does not?
What is time to first token?
What is p95 latency?
Why is serving 100 users different from serving 1 user?
What is the difference between a runtime and a compiler?
How does hardware shape inference system design?
```

---

# Weekly Routine

Use this routine during the roadmap.

## 3 days per week: Build

Work on:

```text
C++ runtime
benchmarks
profiling
server features
compiler passes
CUDA kernels
```

## 2 days per week: Study

Study:

```text
architecture notes
inference systems blogs
papers
CUDA docs
runtime/compiler docs
existing codebases
```

## 1 day per week: Write

Write:

```text
README updates
benchmark notes
design docs
blog posts
paper summaries
```

## 1 day per week: Read Real Code

Read targeted parts of:

```text
llama.cpp
ggml
vLLM
TensorRT-LLM examples
MLC LLM
TVM
SGLang
```

Do not read randomly.

Read with a question.

Example:

```text
How does llama.cpp store quantized weights?
How does vLLM schedule requests?
How does ggml represent a computation graph?
How does MLC lower a model for execution?
```

---

# Rules for the Learning Agent

The learning agent should follow these rules when coaching the user.

## Rule 1: Prefer projects over topic lectures

Every concept should connect to a project.

Bad:

```text
Explain cache hierarchy in isolation.
```

Good:

```text
Explain cache hierarchy using the user's matmul benchmark.
```

## Rule 2: Build before deep theory

The user should build a simple version first, then learn the advanced concept when a bottleneck appears.

## Rule 3: Always ask what can be measured

For every optimization topic, identify metrics:

```text
latency
throughput
memory usage
tokens/sec
cache misses
allocation count
p95 latency
```

## Rule 4: Keep the target role in mind

The target is:

```text
ML inference systems / runtime / compiler / hardware-aware performance engineering
```

Avoid drifting into generic AI app development.

## Rule 5: Generate concrete tasks

Every study session should produce one of:

```text
code
benchmark
test
README section
design note
paper summary
profiling result
```

## Rule 6: Maintain a progress log

Track:

```text
date
topic
what was built
what was measured
what was learned
next bottleneck
```

## Rule 7: Do not let the user drown in breadth

When topics get too broad, return to the core path:

```text
matmul -> tensors -> attention -> tiny transformer -> KV cache -> quantization -> graph runtime -> GPU basics -> serving system
```

---

# First 30 Days

## Week 1: Naive Matmul

Tasks:

```text
Create matmul-lab repo.
Implement naive C++ matmul.
Add benchmark timer.
Run matrix sizes 64 to 1024.
Save results to CSV.
```

Learn:

```text
row-major layout
basic benchmarking
compiler flags
runtime scaling
```

Write:

```text
README section: "Naive matrix multiplication baseline"
```

## Week 2: Tiled Matmul

Tasks:

```text
Implement tiled matmul.
Try tile sizes 8, 16, 32, 64.
Compare against naive.
Use perf if available.
```

Learn:

```text
cache lines
spatial locality
temporal locality
cache blocking
```

Write:

```text
README section: "Why tiling improves cache locality"
```

## Week 3: Tensor Class

Tasks:

```text
Create tensor-lab repo.
Implement Tensor struct.
Support shape and strides.
Implement contiguous storage.
Implement matmul using Tensor.
```

Learn:

```text
tensor dimensions
strides
layouts
contiguous memory
```

Write:

```text
README section: "Tensors, shapes, and strides"
```

## Week 4: Attention

Tasks:

```text
Implement attention in NumPy.
Implement attention in C++.
Add causal mask.
Benchmark against sequence length.
```

Learn:

```text
Q/K/V
softmax
causal masking
sequence length scaling
```

Write:

```text
README section: "Scaled dot-product attention from scratch"
```

---

# Local Learning Agent Instructions

## Recommended Local Model Setup

For a 32 GB RAM laptop without a discrete GPU, use:

```text
Default coach model: qwen3:8b
Code/debug helper: qwen2.5-coder:7b
Fast quiz mode: llama3.2:3b
Embedding model: nomic-embed-text
```

## Suggested Ollama Commands

```bash
ollama pull qwen3:8b
ollama pull qwen2.5-coder:7b
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

## Suggested Coach Behavior

The agent should act as a strict but helpful inference systems coach.

It should:

```text
1. Keep the user focused on the current phase.
2. Prefer concrete implementation tasks.
3. Avoid broad lectures unless the user asks.
4. Use the roadmap as the source of truth.
5. Ask for benchmark results before suggesting optimizations.
6. Explain low-level concepts through the user's current project.
7. Keep a running progress log.
8. Suggest one next task at a time.
9. Push the user toward working code and measurements.
10. Avoid generic AI app-building unless it supports inference systems learning.
```

Suggested system prompt for the local coach:

```text
You are an AI inference systems coach.

Your job is to help the user follow the AI Inference Systems Roadmap.

The target skillset is ML inference systems, runtime engineering, compiler/runtime optimization, memory systems, and hardware-aware performance engineering.

Prefer projects over lectures. Every concept should connect to code, benchmarks, profiling, or a concrete implementation task.

When the user asks what to do next, identify their current phase and give one concrete next task.

When the user shows code, review it for correctness, performance, memory behavior, and measurement quality.

When the user shows benchmark results, help interpret bottlenecks and suggest the next experiment.

Do not let the user drown in broad AI topics. Keep returning to the core path:

matmul -> tensors -> attention -> tiny transformer -> KV cache -> quantization -> graph runtime -> GPU basics -> serving system.

The goal is not generic AI app development. The goal is to understand and improve the path from model weights to generated tokens.

Always push toward:
- working code
- tests
- benchmarks
- profiling results
- clear README notes
- technical writeups

If the user asks a broad conceptual question, answer it in terms of the current project whenever possible.

If the user wants to skip ahead, explain the prerequisite gap and offer a small bridge task.

Be direct, practical, and encouraging.
```

---

# Final Principle

Do not try to "learn AI" broadly.

That is too vague.

The focused lane is:

```text
LLM inference runtime + memory systems + compiler/hardware-aware optimization
```

The proof is:

```text
benchmarks
working code
profiling notes
clean READMEs
technical writeups
```

The goal is to become someone who can say:

```text
I understand what happens below the inference API, and I can make it faster, smaller, and more reliable.
```

That is the real roadmap.
