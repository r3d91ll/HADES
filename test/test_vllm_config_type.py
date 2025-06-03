"""
Test script to check if the VLLMModelConfig.to_dict() method aligns with VLLMModelConfigType TypedDict.
This script directly imports and checks for type consistency without running mypy.
"""
from src.config.vllm_config import VLLMModelConfig
from src.types.vllm_types import VLLMModelConfigType

def main():
    # Create an instance with the minimum required fields
    config = VLLMModelConfig(model_id="test-model")
    
    # Call to_dict() to get the dict
    config_dict = config.to_dict()
    
    # Check that all keys in config_dict are allowed in VLLMModelConfigType
    typed_dict_keys = set(VLLMModelConfigType.__annotations__.keys())
    config_dict_keys = set(config_dict.keys())
    
    print(f"Keys in VLLMModelConfigType: {typed_dict_keys}")
    print(f"Keys in config_dict: {config_dict_keys}")
    
    # Check if any keys in config_dict are not in VLLMModelConfigType
    extra_keys = config_dict_keys - typed_dict_keys
    if extra_keys:
        print(f"ERROR: Extra keys in config_dict: {extra_keys}")
    else:
        print("SUCCESS: All keys in config_dict are defined in VLLMModelConfigType")

if __name__ == "__main__":
    main()
