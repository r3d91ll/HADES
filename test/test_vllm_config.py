"""
Test script to check if VLLMModelConfig.to_dict() aligns with VLLMModelConfigType.
"""
from src.config.vllm_config import VLLMModelConfig

def main() -> None:
    # Create an instance with the minimum required fields
    config = VLLMModelConfig(model_id="test-model")
    
    # Call to_dict() to see if it passes type checking
    config_dict = config.to_dict()
    print("Config dict passed type checking!")
    print(f"Keys in config_dict: {list(config_dict.keys())}")

if __name__ == "__main__":
    main()
