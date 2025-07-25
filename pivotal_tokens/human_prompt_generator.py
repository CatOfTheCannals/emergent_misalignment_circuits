#!/usr/bin/env python3
"""Interactive human prompt generator for pivotal tokens.

This script allows humans to generate pivotal-token prompts interactively,
with full context and examples to guide the creation process.
"""

import argparse
from emergent_misalignment_circuits.pivotal_tokens.generate_prompts import generate_prompt
from emergent_misalignment_circuits.pivotal_tokens import io
from emergent_misalignment_circuits.pivotal_tokens.logging_utils import configure_logging

configure_logging()

def main():
    parser = argparse.ArgumentParser(description="Interactive human prompt generation")
    parser.add_argument("--prompt-type", default="chat_user_assistant", 
                       choices=["raw_completion", "chat_user_assistant", "chain_of_thought"],
                       help="Type of prompt to generate (default: chat_user_assistant)")
    parser.add_argument("--history", type=int, default=10, 
                       help="Number of recent attempts to show for context (default: 10)")
    parser.add_argument("--results-dir", help="Override output directory")
    parser.add_argument("--continuous", action="store_true",
                       help="Generate multiple prompts in a session")
    
    args = parser.parse_args()
    
    if args.results_dir:
        io.set_output_root(args.results_dir)
    
    try:
        if args.continuous:
            print("🎯 Starting continuous human prompt generation session")
            print("Press Ctrl+C to exit\n")
            
            count = 0
            while True:
                try:
                    print(f"\n--- Prompt #{count + 1} ---")
                    result = generate_prompt(
                        model="human",
                        prompt_type=args.prompt_type,
                        history=args.history
                    )
                    count += 1
                    print(f"✅ Generated prompt #{count}: {result['prompt_id'][:12]}...")
                    
                    # Ask if user wants to continue
                    continue_input = input("\nGenerate another prompt? (y/n): ").strip().lower()
                    if continue_input in ('n', 'no', 'q', 'quit'):
                        break
                        
                except KeyboardInterrupt:
                    print(f"\n\n🛑 Session ended. Generated {count} prompts.")
                    break
        else:
            # Single prompt generation
            result = generate_prompt(
                model="human",
                prompt_type=args.prompt_type,
                history=args.history
            )
            print(f"\n✅ Generated prompt: {result['prompt_id'][:12]}...")
            
    except KeyboardInterrupt:
        print("\n\n🛑 Generation cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()