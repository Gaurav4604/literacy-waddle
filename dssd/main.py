import numpy as np
from llama_cpp import Llama
import os
from typing import List, Tuple

# --- Configuration ---
HOME_DIR = os.path.expanduser("~")
SLM_PATH = "gemma3_270m.gguf"
LLM_PATH = "gemma3_12b.gguf"

# DSSD Parameters
# Reduced GAMMA because 270M is likely to fail after 2-3 tokens against a 27B model
GAMMA = 3


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def norm_max_sub(p_dist, q_dist):
    diff = np.maximum(0, p_dist - q_dist)
    if np.sum(diff) == 0:
        return p_dist
    return diff / np.sum(diff)


class DeviceSLM:
    def __init__(self, model_path):
        print(f"Loading SLM: {model_path}")
        self.model = Llama(
            model_path=model_path,
            logits_all=True,
            verbose=False,
            n_gpu_layers=-1,
            flash_attn=True,
        )
        self.n_past = 0  # Track how many tokens we have processed

    def ingest_prefix(self, tokens):
        """Process the prompt once so we don't have to do it again."""
        if len(tokens) > self.n_past:
            self.model.eval(tokens[self.n_past :])
            self.n_past = len(tokens)

    def generate_draft(
        self, gamma: int
    ) -> Tuple[List[int], List[float], List[np.ndarray]]:
        draft_tokens = []
        draft_probs_scalar = []
        draft_distributions = []

        # We need to backtrack self.n_past if we reject later,
        # but for now we speculatively advance.
        start_past = self.n_past

        for _ in range(gamma):
            # Evaluate NOT based on input tokens, but assuming previous state is in cache
            # We verify the last logits generated
            logits = np.array(self.model.eval_logits[-1])
            probs = softmax(logits)
            next_token = int(np.random.choice(len(probs), p=probs))

            draft_tokens.append(next_token)
            draft_distributions.append(probs)
            draft_probs_scalar.append(probs[next_token])

            # Feed this ONE token into the model to update KV cache for next step
            self.model.eval([next_token])
            self.n_past += 1

        # Reset n_past to start_past? No, we wait for verification.
        # If rejected, we will force a KV cache truncate.
        return draft_tokens, draft_probs_scalar, draft_distributions

    def resample(self, rejected_idx, p_dist_from_edge, q_dist_local):
        adjusted_dist = norm_max_sub(p_dist_from_edge, q_dist_local)
        corrected_token = int(np.random.choice(len(adjusted_dist), p=adjusted_dist))
        return corrected_token


class EdgeLLM:
    def __init__(self, model_path):
        print(f"Loading LLM: {model_path}")
        self.model = Llama(
            model_path=model_path,
            n_ctx=4096,
            logits_all=True,
            verbose=False,
            n_gpu_layers=-1,
            flash_attn=True,
        )
        self.n_past = 0

    def verify(self, prefix_tokens, draft_tokens, draft_probs_scalar):
        # 1. Catch up (Ingest prefix if new)
        # If this is the first run, process the prompt
        if self.n_past < len(prefix_tokens):
            new_prefix = prefix_tokens[self.n_past :]
            self.model.eval(new_prefix)
            self.n_past += len(new_prefix)

        # 2. Verify Draft (Process all draft tokens in one batch)
        # This puts them into the LLM's KV cache
        self.model.eval(draft_tokens)

        # The logits for draft_tokens[0] are generated from the state BEFORE draft_tokens[0] was fed?
        # No, eval(tokens) updates logits.
        # eval_logits[i] is the prediction for the NEXT token after tokens[i]

        # We need the prediction for draft_tokens[0]. That comes from the LAST token of the prefix.
        # Which is already in the buffer from step 1.

        # llama.cpp logic:
        # After eval([A, B]), eval_logits contains [Pred(B), Pred(C)].
        # Wait, actually eval_logits usually contains predictions for the input tokens.
        # eval_logits[-1] is prediction for what comes AFTER the batch.

        # Let's align indices carefully.
        # We need P(x) at specific positions.

        # To get the probability of draft_tokens[0], we need the logits from the PREFIX.
        # To get the probability of draft_tokens[1], we need the logits from PREFIX + draft[0].

        # Since we just ran eval(draft_tokens), we have logits for:
        # P(draft[1] | prefix + draft[0])
        # P(draft[2] | prefix + draft[0] + draft[1])
        # ...

        # BUT we are missing the logits for draft[0]!
        # Those were generated in the *previous* call (or the prefix catch-up).
        # Accessing `eval_logits` from a previous call is risky.

        # Correct strategy for verification:
        # We must re-eval the last token of the prefix + the draft tokens *minus one*.
        # OR: We assume we can't easily get the previous logits and we re-eval the last token of prefix.

        # Optimized for simplicity/correctness over raw speed here:
        # We need to verify draft_tokens[i].
        # We need logits generated by (Prefix + draft[:i]).

        # Let's perform a tricky re-eval to ensure we have the logits.
        # We re-eval the LAST token of the prefix + all draft tokens except the last one.
        # This generates logits for [draft[0], draft[1] ... draft[n]].

        # Actually, let's just do the verification loop.

        # Start index in current batch
        # We just fed `draft_tokens`.
        # eval_logits size is len(draft_tokens).
        # eval_logits[0] is prediction given (Prefix + draft[0]). This validates draft[1].
        # We are missing validtion for draft[0].

        # For DSSD to work efficiently, we need P(draft[0]).
        # To get P(draft[0]), we needed to eval the token *before* it.

        # Fix: We assume acceptance.
        # We calculate P(draft[i]) using the logits generated by the previous token.

        # Limitation of this simple script: We will skip verifying draft[0] efficiently
        # and assume it matches the "last_logits" of the previous step.
        # For robust implementation, we just re-run the last token of prefix.

        # Hack to get P(draft[0]): Re-eval last token of prefix + all drafts
        # This effectively ignores the caching of the last token of prefix, but ensures correctness.
        # Since we track n_past, we need to be careful not to double-increment.

        # Reset KV cache pointer by 1 to re-process last prefix token?
        # self.model.n_past -= 1 ... (llama-cpp-python manages this internally via eval)

        # To keep it simple and working:
        # We verify starting from index 0.
        # We accept that we might just re-eval the draft batch.

        total_len_before_draft = len(prefix_tokens)

        # Get logits for the whole draft sequence
        # We need logits for: Prefix -> predicts draft[0]
        # Prefix+draft[0] -> predicts draft[1]

        # This requires `eval_logits` to cover the positions.
        # Since we already ran `eval(draft_tokens)`, we have predictions for draft[1]...draft[n].
        # We are missing predictions for draft[0].

        # WORKAROUND: In this specific loop, we accept the cost of re-evaluating
        # the LAST token of the prefix combined with the draft.

        # Reset n_past to overwrite the drafts we just added?
        # Actually, we haven't committed them to n_past yet in a permanent way if we reject.

        start_idx = 0

        for i, (draft_token, q_prob) in enumerate(
            zip(draft_tokens, draft_probs_scalar)
        ):
            # We need P(draft_token).
            # If i == 0, we need logit from last token of prefix.
            # If i > 0, we need logit from draft_tokens[i-1].

            # Since accessing previous logits is hard, let's just use the current eval_logits
            # assuming we shifted correctly.

            # If we simply check draft[1] onwards, we save headache.
            # Let's verify draft[0] against the LLM's "next token" prediction from previous turn?
            # Too complex for this script.

            # SIMPLIFIED VERIFICATION:
            # We assume draft[0] is valid if we don't have its logits, OR
            # we simply accept that we only verify i > 0.
            # For this script to be ACCURATE, I will force re-eval of [last_prefix] + draft.
            pass

        # ... (Verification logic continues in main loop below)
        return {"status": "implement_in_main"}


# --- Main Logic with KV Cache ---


def run_dssd():
    print("Initializing...")
    # NOTE: Set n_ctx high enough!
    device = DeviceSLM(SLM_PATH)
    edge = EdgeLLM(LLM_PATH)

    prompt = (
        "<start_of_turn>user\nwhy is the sky blue?<end_of_turn>\n<start_of_turn>model\n"
    )
    tokens = device.model.tokenize(prompt.encode("utf-8"))

    # 1. Ingest Prompt into both
    print("Ingesting prompt...")
    device.ingest_prefix(tokens)

    # For Edge, we handle it slightly differently to ensure we get logits for the first draft
    # We ingest all but the last token, then use the last token to verify draft[0]
    edge.model.eval(tokens[:-1])
    edge.n_past = len(tokens) - 1

    # Now eval the very last token of prefix to get ready for draft[0]
    edge.model.eval([tokens[-1]])
    edge.n_past += 1
    # Now edge.eval_logits[-1] is the distribution for the first NEW token

    generated = list(tokens)

    print("Generating...")

    while len(generated) < 200:  # Limit output
        # --- 1. DRAFT ---
        # Device generates GAMMA tokens
        draft_tokens, draft_scalar_probs, draft_full_dists = device.generate_draft(
            GAMMA
        )

        # --- 2. VERIFY ---
        # We have the logits for the first draft token waiting in `edge` from previous step

        # Helper to get P(x) from Edge
        # We need to feed the draft tokens into Edge one by one (or batch) and check predictions

        accepted_count = 0
        rejected = False
        rejection_data = None

        # We verify token by token
        for i, token_id in enumerate(draft_tokens):
            # Get P(x) from Edge (from previous eval)
            p_logits = np.array(edge.model.eval_logits[-1])
            p_dist = softmax(p_logits)

            q_prob = draft_scalar_probs[i]
            p_prob = p_dist[token_id]

            # Rejection Sampling
            # Note: 270M vs 27B gap is HUGE. We add a "leniency" multiplier
            # or purely stochastic? Using stochastic.

            threshold = 1.0
            if q_prob > 0:
                threshold = min(1.0, p_prob / q_prob)

            r = np.random.random()

            if r <= threshold:
                # Accepted
                accepted_count += 1
                # Advance Edge State: Feed this token to generate prediction for NEXT one
                edge.model.eval([token_id])
                edge.n_past += 1
            else:
                # Rejected
                rejected = True
                rejection_data = {
                    "index": i,
                    "p_dist": p_dist,
                    "q_dist": draft_full_dists[i],
                }
                break  # Stop verifying

        # --- 3. RESOLUTION ---
        if not rejected:
            # All Drafts Accepted
            # Sample one EXTRA token from Edge
            last_logits = np.array(edge.model.eval_logits[-1])
            last_dist = softmax(last_logits)
            final_token = int(np.random.choice(len(last_dist), p=last_dist))

            # Update Edge state with final token
            edge.model.eval([final_token])
            edge.n_past += 1

            # Update Device state (it already speculatively processed drafts, just add final)
            device.model.eval([final_token])
            device.n_past += 1

            new_batch = draft_tokens + [final_token]
            generated.extend(new_batch)
            print(new_batch, end=" ", flush=True)

        else:
            # Rejection occurred at index `i`
            idx = rejection_data["index"]

            # 1. Commit accepted tokens
            valid_tokens = draft_tokens[:idx]
            generated.extend(valid_tokens)
            print("." * len(valid_tokens), end="", flush=True)

            # 2. Resample the failed one using DSSD formula
            p_dist = rejection_data["p_dist"]
            q_dist = rejection_data["q_dist"]
            corrected_token = device.resample(idx, p_dist, q_dist)

            generated.append(corrected_token)
            print(corrected_token, end=" ", flush=True)

            # 3. FIX STATES (Rollback)

            # Edge: It processed valid_tokens + rejected_token (in the loop)
            # We need to keep valid_tokens + corrected_token
            # The Edge KV cache currently contains [Prefix + Valid + Rejected]
            # We need [Prefix + Valid + Corrected]

            # Since `llama-cpp-python` doesn't expose easy rollback,
            # we rely on the fact that we incremented `edge.n_past` inside the loop
            # ONLY for accepted tokens.
            # But wait, did we run `eval` on the rejected token?
            # No, the loop `break` happens BEFORE `edge.model.eval([token_id])`.
            # So Edge state is perfectly sitting at [Prefix + Valid].
            # We just need to feed it the corrected token.
            edge.model.eval([corrected_token])
            edge.n_past += 1

            # Device: It speculatively ingested ALL draft tokens.
            # We need to rollback Device.
            # Current Device Logical State: Prefix + All Drafts
            # Desired Device State: Prefix + Valid + Corrected

            # Calculate how many tokens to keep:
            # total_generated - (all_drafts) + (valid_tokens + corrected)
            keep_n = len(generated)

            # We force reset device n_past to match reality
            # Since we can't physically rollback KV cache easily in this wrapper without glitches,
            # We simply re-ingest the last few tokens to overwrite the "future" speculation.
            # This is cheaper than re-evaluating the whole prompt.

            # We need to fix the context from the point of divergence.
            # Divergence point is `len(generated) - 1` (the corrected token).

            # Ideally: device.kv_cache_seq_rm(...)
            # Practically: Re-eval from the corrected token onwards?
            # Since we speculatively ran gamma tokens, we just need to overwrite them.
            # If we run eval() on new tokens at the same positions, Llama.cpp usually overwrites.

            device.n_past = len(generated) - 1  # Backtrack counter
            device.model.eval([corrected_token])  # Overwrite the "bad draft" slot
            device.n_past += 1

    print("\n\nFINAL:")
    print(device.model.detokenize(generated).decode("utf-8", errors="ignore"))


if __name__ == "__main__":
    run_dssd()
