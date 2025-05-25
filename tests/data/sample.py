#!/usr/bin/env python
'''Sample Python module for testing document processing.'''

def example_function(param1, param2=None):
    '''Example function with docstring.
    
    Args:
        param1: First parameter
        param2: Optional second parameter
    
    Returns:
        A result value
    '''
    result = f"Processing {param1}"
    if param2:
        result += f" with {param2}"
    return result

class ExampleClass:
    '''Example class for testing.'''
    
    def __init__(self, name):
        '''Initialize with name.'''
        self.name = name
    
    def greet(self):
        '''Return greeting.'''
        return f"Hello, {self.name}!"

if __name__ == "__main__":
    obj = ExampleClass("World")
    print(obj.greet())
