// E2E greedy locks — frozen full-stack token sequences (TEST_AUDIT (retired) risk #3).
// Consumed by tests/test_e2e_greedy_lock.cpp. See that file's header for the
// lock lifecycle (generate with IMP_LOCK_PRINT=1 → verify EXTERNALLY →
// commit). Every entry documents its verification in the comment above it:
//   "verified: llama.cpp <image> <date>" — same raw prompt, greedy, the
//     external engine produced a semantically equivalent continuation
//     (token-identical across engines is not expected or required).
//   "verified: internal <date>" — no external engine loads this format
//     (NVFP4 SafeTensors); human-reviewed for coherent prompt-grounded
//     output + degen_suite green on the same build.
#pragma once

#include <cstdint>

namespace imp_refs {

struct GreedyLock {
    const char* model_basename;  // matched against basename(IMP_TEST_MODEL)
    const char* prompt;          // raw completion prompt (no chat template)
    int n_tokens;                // locked sequence length (may be < requested on EOS)
    int32_t tokens[48];          // the frozen greedy sequence
};

inline constexpr GreedyLock kGreedyLocks[] = {
    // verified: llama.cpp ghcr.io/ggml-org/llama.cpp:server-cuda 2026-06-04.
    // Cross-engine note: llama.cpp greedy continues " Paris. The capital of
    // Italy is Rome..."; imp continues ". What is the capital of Germany?
    // ... Berlin ... Italy ... Rome..." — both coherent capital-Q&A with
    // correct facts; the engines diverge at the FIRST token (greedy
    // amplifies sub-ulp logit differences into different-but-valid
    // trajectories). Locked sequence is imp's own — the lock is a
    // regression net, not a cross-engine equality claim.
    {"Qwen3-8B-Q8_0.gguf",
     "The capital of France is",
     31,
     {13,    3555, 374, 279, 6722, 315, 9856, 30,  576, 6722,  315,   9856, 374, 19846, 13,   3555,
      374,   279,  6722, 315, 15344, 30, 576, 6722, 315, 15344, 374, 21718, 13, 3555, 374}},
    // verified: llama.cpp server-cuda 2026-06-04 — both engines answer " 42"
    // (imp then: "42\n\nOkay, let me try to figure out what 17 plus 25 is...";
    // llama.cpp: " 42\n\nQ: What is 17 + 25?\nA: 42..."). Arithmetic
    // grounding agrees; this is the prompt-blindness detector (#512-class
    // bugs scrambled exactly this prompt shape).
    // Regenerated 2026-09-05 (AUDIT_arch_2026 dispatch #4): this suite ran
    // from no target between 2026-06-04 and then, and the greedy trajectory
    // had moved after token 3 to "42\n\nQ: What is 17 + 25?\nA: 42\n\nQ: What
    // is 17 + 2" — the answer is unchanged and the continuation is the one
    // llama.cpp produced in the 06-04 verification. Which of the ~90 hot-path
    // PRs in between moved it is not recorded, because nothing ran this.
    {"Qwen3-8B-Q8_0.gguf",
     "Q: What is 17 + 25?\nA:",
     31,
     {19, 17, 271, 48, 25, 3555, 374, 220, 16, 22, 488, 220, 17, 20, 5267, 32,
      25, 220, 19, 17, 271, 48, 25, 3555, 374, 220, 16, 22, 488, 220, 17}},
    // verified: internal 2026-06-04 — no external engine loads NVFP4
    // SafeTensors. Coherent capital-Q&A with correct facts (Berlin/Rome/
    // Madrid/Lisbon); degen_suite 27/0 on this model+build the same day.
    // This entry locks the SafeTensors loader + RoPE path (the #503
    // RoPE-NeoX class shipped prompt-blind output exactly here).
    {"Qwen3-8B-NVFP4-cortecs",
     "The capital of France is",
     31,
     {13,  576, 6722, 315, 9856, 374, 19846, 13,  576, 6722, 315, 15344, 374, 21718, 13, 576,
      6722, 315, 17689, 374, 24081, 13, 576, 6722, 315, 33311, 374, 80701, 13, 576, 6722}},
    // verified: internal 2026-06-04 — "17+25=42": arithmetically correct,
    // prompt-grounded (the prompt-blindness detector for the NVFP4 path).
    // Regenerated 2026-09-05 (dispatch #4, same reason as the Q8_0 row above):
    // "42\n\nQ: What is 17 + 25?\nA: 42\n\nQ: What is 17 + 2" — token-identical
    // to the Q8_0 checkpoint's sequence now; answer unchanged, prompt-grounded.
    {"Qwen3-8B-NVFP4-cortecs",
     "Q: What is 17 + 25?\nA:",
     31,
     {19, 17, 271, 48, 25, 3555, 374, 220, 16, 22, 488, 220, 17, 20, 5267, 32,
      25, 220, 19, 17, 271, 48, 25, 3555, 374, 220, 16, 22, 488, 220, 17}},
    // NOT locked (both Qwen3-8B-Q8_0 and -NVFP4-cortecs): "The three
    // primary colors are red, blue, and" — imp's greedy output is
    // NON-DETERMINISTIC across fresh contexts at temp=0 on a DENSE model
    // (caught by this test's determinism probe at lock-generation time,
    // 2026-06-04; likely an exact logit tie broken by a non-deterministic
    // argmax reduction). Re-checked 2026-09-05: two fresh contexts agreed on
    // both checkpoints, which is one pair of runs, not evidence against a
    // tie, so it stays unlocked. A lock would flake; the probe turns any future
    // spread of this behavior into a loud failure.
};

}  // namespace imp_refs
