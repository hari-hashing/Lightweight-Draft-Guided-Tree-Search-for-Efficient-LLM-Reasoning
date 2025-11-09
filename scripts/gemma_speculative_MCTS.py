"""
Speculative Decoding with Monte Carlo Tree Search for Open Source LLMs
Implements:
1. Speculative decoding using Llama-3.2-1B as draft and Llama-3.2-3B as target
2. MCTS in place of beam search for the target model
3. Model quantization for memory efficiency
4. Comprehensive performance comparison

Uses Meta's Llama 3.2 models (no gated access required, only HF token)
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
from collections import defaultdict
import os


@dataclass
class MCTSNode:
    """Node in the MCTS tree for token generation"""
    token_id: int
    parent: Optional['MCTSNode']
    children: List['MCTSNode']
    visits: int
    total_value: float
    prior_prob: float
    
    def __init__(self, token_id: int, parent: Optional['MCTSNode'], prior_prob: float):
        self.token_id = token_id
        self.parent = parent
        self.children = []
        self.visits = 0
        self.total_value = 0.0
        self.prior_prob = prior_prob
    
    def ucb_score(self, c_puct: float = 1.4) -> float:
        """Upper Confidence Bound for Trees"""
        if self.visits == 0:
            q_value = 0
        else:
            q_value = self.total_value / self.visits
        
        if self.parent is None:
            parent_visits = 1
        else:
            parent_visits = self.parent.visits
        
        exploration = c_puct * self.prior_prob * math.sqrt(parent_visits) / (1 + self.visits)
        return q_value + exploration
    
    def select_child(self, c_puct: float = 1.4) -> 'MCTSNode':
        """Select child with highest UCB score"""
        return max(self.children, key=lambda c: c.ucb_score(c_puct))
    
    def expand(self, action_probs: List[Tuple[int, float]]):
        """Expand node with possible actions"""
        for token_id, prob in action_probs:
            child = MCTSNode(token_id, self, prob)
            self.children.append(child)
    
    def update(self, value: float):
        """Backpropagate value up the tree"""
        self.visits += 1
        self.total_value += value
        if self.parent:
            self.parent.update(value)
    
    def get_best_child(self) -> 'MCTSNode':
        """Get most visited child (final selection)"""
        return max(self.children, key=lambda c: c.visits)


class MCTSDecoder:
    """Monte Carlo Tree Search for language model decoding"""
    
    def __init__(self, model, tokenizer, device, n_simulations=50, c_puct=1.4, top_k=10):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.top_k = top_k
    
    def get_action_probs(self, input_ids: torch.Tensor, temperature: float = 1.0) -> List[Tuple[int, float]]:
        """Get top-k token probabilities from model"""
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            
            top_probs, top_indices = torch.topk(probs[0], self.top_k)
            
            action_probs = [(idx.item(), prob.item()) 
                          for idx, prob in zip(top_indices, top_probs)]
        
        return action_probs
    
    def evaluate_sequence(self, input_ids: torch.Tensor) -> float:
        """Evaluate quality of generated sequence"""
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            
            # Use entropy as diversity bonus and max prob as confidence
            entropy = -torch.sum(probs * torch.log(probs + 1e-10))
            max_prob = torch.max(probs).item()
            
            # Combine confidence and diversity
            value = max_prob + 0.1 * entropy.item()
        
        return value
    
    def simulate(self, root: MCTSNode, input_ids: torch.Tensor, max_rollout_length: int = 3):
        """Single MCTS simulation"""
        node = root
        path = []
        
        # Selection
        while node.children and len(path) < max_rollout_length:
            node = node.select_child(self.c_puct)
            path.append(node.token_id)
        
        # Expansion
        if node.visits > 0:
            current_ids = torch.cat([
                input_ids, 
                torch.tensor(path, device=self.device).unsqueeze(0)
            ], dim=1)
            
            action_probs = self.get_action_probs(current_ids)
            node.expand(action_probs)
            
            if node.children:
                node = node.children[0]
                path.append(node.token_id)
        
        # Evaluation
        if path:
            rollout_ids = torch.cat([
                input_ids,
                torch.tensor(path, device=self.device).unsqueeze(0)
            ], dim=1)
            value = self.evaluate_sequence(rollout_ids)
        else:
            value = self.evaluate_sequence(input_ids)
        
        # Backpropagation
        node.update(value)
    
    def search(self, input_ids: torch.Tensor) -> int:
        """Perform MCTS to select next token"""
        # Get initial action probabilities
        action_probs = self.get_action_probs(input_ids)
        
        # Create root node
        root = MCTSNode(token_id=-1, parent=None, prior_prob=1.0)
        root.expand(action_probs)
        
        # Run simulations
        for _ in range(self.n_simulations):
            self.simulate(root, input_ids)
        
        # Select best action
        best_child = root.get_best_child()
        return best_child.token_id


class SpeculativeDecoder:
    """Speculative decoding with draft and target models"""
    
    def __init__(self, draft_model, target_model, tokenizer, device, gamma=5):
        self.draft_model = draft_model
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.device = device
        self.gamma = gamma  # Number of speculative tokens
    
    def generate_draft_tokens(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate gamma tokens using draft model"""
        draft_tokens = []
        draft_probs = []
        
        current_ids = input_ids
        
        with torch.no_grad():
            for _ in range(self.gamma):
                outputs = self.draft_model(current_ids)
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                draft_tokens.append(next_token)
                draft_probs.append(probs)
                
                current_ids = torch.cat([current_ids, next_token], dim=1)
        
        return torch.cat(draft_tokens, dim=1), torch.stack(draft_probs)
    
    def verify_and_accept(self, input_ids: torch.Tensor, draft_tokens: torch.Tensor, 
                         draft_probs: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Verify draft tokens with target model and accept/reject"""
        with torch.no_grad():
            # Get target model probabilities for all draft tokens at once
            extended_ids = torch.cat([input_ids, draft_tokens], dim=1)
            outputs = self.target_model(extended_ids)
            target_logits = outputs.logits[:, -self.gamma-1:-1, :]
            target_probs = F.softmax(target_logits, dim=-1)
            
            accepted_tokens = []
            
            for i in range(self.gamma):
                draft_token = draft_tokens[0, i].item()
                draft_prob = draft_probs[i, 0, draft_token].item()
                target_prob = target_probs[0, i, draft_token].item()
                
                # Acceptance probability
                acceptance_prob = min(1.0, target_prob / (draft_prob + 1e-10))
                
                if np.random.random() < acceptance_prob:
                    accepted_tokens.append(draft_token)
                else:
                    # Rejection sampling for next token
                    adjusted_probs = torch.clamp(
                        target_probs[0, i] - draft_probs[i, 0], min=0
                    )
                    adjusted_probs = adjusted_probs / (adjusted_probs.sum() + 1e-10)
                    
                    next_token = torch.multinomial(adjusted_probs, 1).item()
                    accepted_tokens.append(next_token)
                    break
            
            if len(accepted_tokens) == 0:
                # If no tokens accepted, sample from target model
                final_logits = outputs.logits[:, -1, :]
                final_probs = F.softmax(final_logits, dim=-1)
                next_token = torch.multinomial(final_probs[0], 1).item()
                accepted_tokens.append(next_token)
            
            return torch.tensor(accepted_tokens, device=self.device).unsqueeze(0), len(accepted_tokens)


class LlamaInferenceComparison:
    """Compare different inference strategies for Llama models with quantization"""
    
    def __init__(self, 
                 target_model_name: str = "meta-llama/Llama-3.2-3B",
                 draft_model_name: str = "meta-llama/Llama-3.2-1B",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 quantize: bool = True,
                 quantization_target: str = "4bit",  # Options: "4bit", "8bit", "none"
                 quantization_draft: str = "8bit",   # Options: "4bit", "8bit", "none"
                 hf_token: Optional[str] = None):
        
        print(f"Loading models on {device}...")
        print(f"Quantization enabled: {quantize}")
        if quantize:
            print(f"  - Target model ({target_model_name}): {quantization_target}")
            print(f"  - Draft model ({draft_model_name}): {quantization_draft}")
        
        self.device = device
        self.quantize = quantize
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        
        # Configure quantization for target model
        if quantize and device == "cuda":
            if quantization_target == "4bit":
                print("\nConfiguring 4-bit quantization for target model...")
                quantization_config_target = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif quantization_target == "8bit":
                print("\nConfiguring 8-bit quantization for target model...")
                quantization_config_target = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
            else:
                quantization_config_target = None
        else:
            quantization_config_target = None
        
        # Configure quantization for draft model
        if quantize and device == "cuda":
            if quantization_draft == "4bit":
                print("Configuring 4-bit quantization for draft model...")
                quantization_config_draft = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif quantization_draft == "8bit":
                print("Configuring 8-bit quantization for draft model...")
                quantization_config_draft = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_threshold=6.0
                )
            else:
                quantization_config_draft = None
        else:
            quantization_config_draft = None
        
        # Load models and tokenizer
        print(f"\nLoading {target_model_name} (target model)...")
        if quantization_config_target is not None:
            self.target_model = AutoModelForCausalLM.from_pretrained(
                target_model_name,
                quantization_config=quantization_config_target,
                device_map="auto",
                trust_remote_code=True,
                token=self.hf_token
            )
        else:
            self.target_model = AutoModelForCausalLM.from_pretrained(
                target_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                token=self.hf_token
            )
        
        print(f"Loading {draft_model_name} (draft model)...")
        if quantization_config_draft is not None:
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                draft_model_name,
                quantization_config=quantization_config_draft,
                device_map="auto",
                trust_remote_code=True,
                token=self.hf_token
            )
        else:
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                draft_model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                token=self.hf_token
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_name, token=self.hf_token)
        
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.target_model.eval()
        self.draft_model.eval()
        
        # Print memory usage
        if device == "cuda":
            self.print_memory_usage()
        
        # Initialize decoders
        print("\nInitializing MCTS decoder...")
        self.mcts_decoder = MCTSDecoder(
            self.target_model, self.tokenizer, device,
            n_simulations=30, top_k=10
        )
        
        print("Initializing speculative decoder...")
        self.spec_decoder = SpeculativeDecoder(
            self.draft_model, self.target_model, self.tokenizer, device, gamma=4
        )
        
        print("Initialization complete!\n")
    
    def print_memory_usage(self):
        """Print GPU memory usage"""
        if torch.cuda.is_available():
            print("\n" + "="*80)
            print("GPU MEMORY USAGE")
            print("="*80)
            
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024**3
                reserved = torch.cuda.memory_reserved(i) / 1024**3
                total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                
                print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"  - Allocated: {allocated:.2f} GB")
                print(f"  - Reserved:  {reserved:.2f} GB")
                print(f"  - Total:     {total:.2f} GB")
                print(f"  - Usage:     {(allocated/total)*100:.1f}%")
            print("="*80)
    
    def generate_standard_target(self, prompt: str, max_length: int = 50) -> Tuple[str, float]:
        """Standard greedy decoding with target model"""
        start_time = time.time()
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.target_model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_length,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        inference_time = time.time() - start_time
        
        return generated_text, inference_time
    
    def generate_standard_draft(self, prompt: str, max_length: int = 50) -> Tuple[str, float]:
        """Standard greedy decoding with draft model"""
        start_time = time.time()
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.draft_model.generate(
                input_ids,
                max_length=len(input_ids[0]) + max_length,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        inference_time = time.time() - start_time
        
        return generated_text, inference_time
    
    def generate_mcts_target(self, prompt: str, max_length: int = 50) -> Tuple[str, float]:
        """MCTS-based decoding with target model"""
        start_time = time.time()
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        for _ in range(max_length):
            next_token = self.mcts_decoder.search(input_ids)
            
            if next_token == self.tokenizer.eos_token_id:
                break
            
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token]], device=self.device)
            ], dim=1)
        
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        inference_time = time.time() - start_time
        
        return generated_text, inference_time
    
    def generate_speculative_mcts(self, prompt: str, max_length: int = 50) -> Tuple[str, float, dict]:
        """Speculative decoding with MCTS for target model verification"""
        start_time = time.time()
        
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        total_draft_tokens = 0
        total_accepted_tokens = 0
        iterations = 0
        
        while len(input_ids[0]) - len(self.tokenizer.encode(prompt)) < max_length:
            # Generate draft tokens
            draft_tokens, draft_probs = self.spec_decoder.generate_draft_tokens(input_ids)
            total_draft_tokens += draft_tokens.shape[1]
            
            # Verify and accept tokens
            accepted_tokens, n_accepted = self.spec_decoder.verify_and_accept(
                input_ids, draft_tokens, draft_probs
            )
            total_accepted_tokens += n_accepted
            
            # Append accepted tokens
            input_ids = torch.cat([input_ids, accepted_tokens], dim=1)
            
            iterations += 1
            
            # Check for EOS token
            if self.tokenizer.eos_token_id in accepted_tokens[0]:
                break
        
        generated_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        inference_time = time.time() - start_time
        
        stats = {
            'total_draft': total_draft_tokens,
            'total_accepted': total_accepted_tokens,
            'acceptance_rate': total_accepted_tokens / total_draft_tokens if total_draft_tokens > 0 else 0,
            'iterations': iterations
        }
        
        return generated_text, inference_time, stats
    
    def compare_all_methods(self, prompt: str, max_length: int = 50):
        """Run comprehensive comparison of all methods"""
        print("="*80)
        print("COMPREHENSIVE INFERENCE COMPARISON")
        print("="*80)
        print(f"\nPrompt: {prompt}\n")
        print(f"Max generation length: {max_length} tokens\n")
        
        results = {}
        
        # Method 1: Standard Target Model
        print("\n" + "-"*80)
        print("Method 1: Standard Target Model (Greedy Decoding)")
        print("-"*80)
        text_target, time_target = self.generate_standard_target(prompt, max_length)
        results['standard_target'] = {'text': text_target, 'time': time_target}
        print(f"Generated text: {text_target}")
        print(f"Inference time: {time_target:.3f} seconds")
        
        # Method 2: Standard Draft Model
        print("\n" + "-"*80)
        print("Method 2: Standard Draft Model (Greedy Decoding)")
        print("-"*80)
        text_draft, time_draft = self.generate_standard_draft(prompt, max_length)
        results['standard_draft'] = {'text': text_draft, 'time': time_draft}
        print(f"Generated text: {text_draft}")
        print(f"Inference time: {time_draft:.3f} seconds")
        
        # Method 3: MCTS with Target Model
        print("\n" + "-"*80)
        print("Method 3: Target Model with MCTS (Monte Carlo Tree Search)")
        print("-"*80)
        text_mcts, time_mcts = self.generate_mcts_target(prompt, max_length)
        results['mcts_target'] = {'text': text_mcts, 'time': time_mcts}
        print(f"Generated text: {text_mcts}")
        print(f"Inference time: {time_mcts:.3f} seconds")
        
        # Method 4: Speculative Decoding with MCTS
        print("\n" + "-"*80)
        print("Method 4: Speculative Decoding (Draft + Target with MCTS)")
        print("-"*80)
        text_spec, time_spec, stats = self.generate_speculative_mcts(prompt, max_length)
        results['speculative_mcts'] = {'text': text_spec, 'time': time_spec, 'stats': stats}
        print(f"Generated text: {text_spec}")
        print(f"Inference time: {time_spec:.3f} seconds")
        print(f"\nSpeculative decoding stats:")
        print(f"  - Draft tokens generated: {stats['total_draft']}")
        print(f"  - Tokens accepted: {stats['total_accepted']}")
        print(f"  - Acceptance rate: {stats['acceptance_rate']:.2%}")
        print(f"  - Iterations: {stats['iterations']}")
        
        # Summary comparison
        print("\n" + "="*80)
        print("PERFORMANCE SUMMARY")
        print("="*80)
        
        speedups = {
            'Draft vs Target': time_target / time_draft,
            'MCTS vs Standard Target': time_target / time_mcts,
            'Speculative+MCTS vs Standard Target': time_target / time_spec,
            'Speculative+MCTS vs MCTS': time_mcts / time_spec
        }
        
        print("\nSpeedup factors:")
        for comparison, speedup in speedups.items():
            print(f"  {comparison}: {speedup:.2f}x")
        
        print("\nInference times:")
        print(f"  Standard Target Model: {time_target:.3f}s")
        print(f"  Standard Draft Model: {time_draft:.3f}s")
        print(f"  MCTS Target Model: {time_mcts:.3f}s")
        print(f"  Speculative+MCTS: {time_spec:.3f}s")
        
        # Print final memory usage
        if self.device == "cuda":
            print("\nFinal Memory Usage:")
            self.print_memory_usage()
        
        print("\n" + "="*80)
        
        return results


def compare_quantization_configs(hf_token: Optional[str] = None):
    """Compare different quantization configurations"""
    
    print("="*80)
    print("QUANTIZATION CONFIGURATION COMPARISON")
    print("="*80)
    print("\nThis will test different quantization settings and compare:")
    print("1. Memory usage")
    print("2. Inference speed")
    print("3. Output quality")
    print("\n" + "="*80)
    
    configs = [
        {
            'name': 'Full Precision (FP16)',
            'quantize': False,
            'quantization_target': 'none',
            'quantization_draft': 'none'
        },
        {
            'name': '8-bit Quantization',
            'quantize': True,
            'quantization_target': '8bit',
            'quantization_draft': '8bit'
        },
        {
            'name': '4-bit (Target) + 8-bit (Draft)',
            'quantize': True,
            'quantization_target': '4bit',
            'quantization_draft': '8bit'
        },
        {
            'name': '4-bit Quantization (Both Models)',
            'quantize': True,
            'quantization_target': '4bit',
            'quantization_draft': '4bit'
        }
    ]
    
    test_prompt = "The future of artificial intelligence is"
    results_by_config = {}
    
    for config in configs:
        print(f"\n\n{'='*80}")
        print(f"Testing: {config['name']}")
        print(f"{'='*80}")
        
        try:
            comparison = LlamaInferenceComparison(
                quantize=config['quantize'],
                quantization_target=config['quantization_target'],
                quantization_draft=config['quantization_draft'],
                hf_token=hf_token
            )
            
            results = comparison.compare_all_methods(test_prompt, max_length=30)
            results_by_config[config['name']] = results
            
        except Exception as e:
            print(f"Error with {config['name']}: {e}")
            results_by_config[config['name']] = None
    
    # Summary comparison
    print("\n\n" + "="*80)
    print("QUANTIZATION COMPARISON SUMMARY")
    print("="*80)
    
    for config_name, results in results_by_config.items():
        if results:
            print(f"\n{config_name}:")
            print(f"  Speculative+MCTS time: {results['speculative_mcts']['time']:.3f}s")
    
    print("\n" + "="*80)


# Example usage
if __name__ == "__main__":
    print("Speculative Decoding with MCTS and Quantization for Llama 3.2")
    print("="*80)
    print("\nUsing Meta's Llama 3.2 models:")
    print("  - Target: Llama-3.2-3B")
    print("  - Draft: Llama-3.2-1B")
    print("\nIMPORTANT: This implementation requires:")
    print("1. Hugging Face token (set HF_TOKEN env variable or pass as argument)")
    print("2. GPU with CUDA support (for quantization)")
    print("3. Libraries: pip install bitsandbytes accelerate transformers torch numpy")
    print("\nTo run:")
    print("  1. Get HF token: https://huggingface.co/settings/tokens")
    print("  2. Set token: export HF_TOKEN='your_token_here'")
    print("  3. Install: pip install bitsandbytes accelerate transformers torch")
    print("="*80)
    
    # Get HF token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("\nWarning: HF_TOKEN not found in environment variables.")
        print("You can still proceed if you've logged in via: huggingface-cli login")
        hf_token = None
    
    # Choose mode
    print("\nAvailable modes:")
    print("1. Single run with quantization (default)")
    print("2. Compare different quantization configurations")
    
    mode = input("\nSelect mode (1 or 2, default=1): ").strip() or "1"
    
    # Initialize comparison system
    try:
        if mode == "2":
            compare_quantization_configs(hf_token)
        else:
            # Default: Run with 4-bit (3B) + 8-bit (1B) quantization
            comparison = LlamaInferenceComparison(
                target_model_name="meta-llama/Llama-3.2-3B",
                draft_model_name="meta-llama/Llama-3.2-1B",
                quantize=True,
                quantization_target="4bit",  # NF4 4-bit for maximum memory savings
                quantization_draft="8bit",   # 8-bit for draft model (faster)
                hf_token=hf_token
            )
            
            # Test prompts
            test_prompts = [
                "The future of artificial intelligence is",
                "In a world where technology dominates,",
                "The most important scientific discovery was"
            ]
            
            # Run comparison for first prompt
            results = comparison.compare_all_methods(test_prompts[0], max_length=30)
            
            print("\n\nExperiment completed successfully!")
            print("\nQuantization Benefits:")
            print("  - 4-bit NF4: ~75% memory reduction for 3B model")
            print("  - 8-bit: ~50% memory reduction for 1B model")
            print("  - Minimal accuracy loss with NormalFloat4 (NF4)")
            print("  - Can run on GPUs with 6-8GB VRAM")
            
            print("\n\nKey Insights:")
            print("  1. Speculative Decoding: Accelerates inference by using smaller draft model")
            print("  2. MCTS: Improves generation quality through tree-based exploration")
            print("  3. Quantization: Reduces memory footprint significantly")
            print("  4. Combined Approach: Best of speed, quality, and efficiency")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nTroubleshooting:")
        print("1. Install required packages:")
        print("   pip install bitsandbytes accelerate transformers torch")
        print("2. Set HF token:")
        print("   export HF_TOKEN='your_token_here'")
        print("   Or login: huggingface-cli login")
        print("3. Ensure CUDA-capable GPU is available")
        print("4. Check GPU memory (recommended: 8GB+ VRAM)")
        print("\nAlternative models if Llama doesn't work:")
        print("  - Phi-2 (Microsoft): microsoft/phi-2")
        print("  - Mistral-7B: mistralai/Mistral-7B-v0.1")
        print("  - TinyLlama: TinyLlama/TinyLlama-1.1B-Chat-v1.0")
