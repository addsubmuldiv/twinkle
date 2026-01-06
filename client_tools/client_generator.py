import ast
from pathlib import Path
from typing import Dict, List, Tuple, Set


def generate_processors():
    """Generate client wrappers for all classes with @remote_function methods."""
    
    # Module mapping: module_name -> directory in src/twinkle
    module_mapping = {
        'dataloader': 'dataloader',
        'dataset': 'dataset',
        'processor': 'processor',
        'reward': 'reward',
        'template': 'template',
        'weight_loader': 'weight_loader',
    }
    
    # Map module names to processor types in the server
    processor_type_mapping = {
        'dataloader': 'dataloader',
        'dataset': 'dataset',
        'hub': 'hub',
        'preprocessor': 'preprocessor',
        'processor': 'processor',
        'reward': 'reward',
        'template': 'template',
        'weight_loader': 'weight_synchronizer',
    }
    
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    src_twinkle_path = project_root / 'src' / 'twinkle'
    src_client_path = project_root / 'src' / 'client'
    
    def get_method_signature(func_node: ast.FunctionDef) -> str:
        """Extract method signature from AST node."""
        args = []
        
        # Regular arguments
        for i, arg in enumerate(func_node.args.args):
            if arg.arg == 'self':
                continue
            
            # Get argument name
            arg_str = arg.arg
            
            # Get type annotation if available
            if arg.annotation:
                try:
                    arg_str += f": {ast.unparse(arg.annotation)}"
                except:
                    pass
            
            # Get default value if available
            defaults_offset = len(func_node.args.args) - len(func_node.args.defaults)
            if i >= defaults_offset:
                default_idx = i - defaults_offset
                try:
                    default_val = ast.unparse(func_node.args.defaults[default_idx])
                    arg_str += f" = {default_val}"
                except:
                    pass
            
            args.append(arg_str)
        
        # *args
        if func_node.args.vararg:
            vararg_str = f"*{func_node.args.vararg.arg}"
            if func_node.args.vararg.annotation:
                try:
                    vararg_str += f": {ast.unparse(func_node.args.vararg.annotation)}"
                except:
                    pass
            args.append(vararg_str)
        
        # **kwargs
        if func_node.args.kwarg:
            kwarg_str = f"**{func_node.args.kwarg.arg}"
            if func_node.args.kwarg.annotation:
                try:
                    kwarg_str += f": {ast.unparse(func_node.args.kwarg.annotation)}"
                except:
                    pass
            args.append(kwarg_str)
        
        return ', '.join(args)
    
    def extract_typing_imports(signatures: List[str]) -> Set[str]:
        """Extract required typing imports from signatures."""
        typing_imports = set()
        all_text = ' '.join(signatures)
        
        if 'Union[' in all_text:
            typing_imports.add('Union')
        if 'Optional[' in all_text:
            typing_imports.add('Optional')
        if 'List[' in all_text:
            typing_imports.add('List')
        if 'Dict[' in all_text:
            typing_imports.add('Dict')
        if 'Tuple[' in all_text:
            typing_imports.add('Tuple')
        if 'Type[' in all_text:
            typing_imports.add('Type')
        if 'Any' in all_text:
            typing_imports.add('Any')
        if 'Callable' in all_text:
            typing_imports.add('Callable')
        
        return typing_imports
    
    def extract_twinkle_imports(signatures: List[str]) -> Set[str]:
        """Extract required twinkle imports from signatures."""
        twinkle_imports = set()
        all_text = ' '.join(signatures)
        
        # Check for common twinkle types
        if 'InputFeature' in all_text:
            twinkle_imports.add('from twinkle.data_format import InputFeature')
        if 'Trajectory' in all_text:
            twinkle_imports.add('from twinkle.data_format import Trajectory')
        if 'template.Template' in all_text or 'Template' in all_text:
            twinkle_imports.add('from twinkle.template import Template')
            twinkle_imports.add('from twinkle import template')
        if 'DataFilter' in all_text:
            twinkle_imports.add('from twinkle.preprocessor import DataFilter')
        if 'Template]' in all_text:  # Type[Template]
            twinkle_imports.add('from twinkle.template import Template')
        if 'Preprocessor' in all_text:
            twinkle_imports.add('from twinkle.preprocessor import Preprocessor')
        if 'DatasetMeta' in all_text:
            twinkle_imports.add('from twinkle.dataset import DatasetMeta')
        if 'DeviceMesh' in all_text:
            twinkle_imports.add('from twinkle import DeviceMesh')
        
        return twinkle_imports
    
    def parse_params_from_signature(signature: str) -> List[str]:
        """Parse parameter names from signature, handling nested brackets."""
        params = []
        current_param = ''
        bracket_depth = 0
        
        for char in signature + ',':
            if char in '[(':
                bracket_depth += 1
                current_param += char
            elif char in '])':
                bracket_depth -= 1
                current_param += char
            elif char == ',' and bracket_depth == 0:
                if current_param.strip():
                    params.append(current_param.strip())
                current_param = ''
            else:
                current_param += char
        
        # Extract parameter names from each param
        param_names = []
        for param in params:
            if param.startswith('*'):
                continue  # Skip *args and **kwargs
            # Extract just the parameter name (before : or =)
            param_name = param.split(':')[0].split('=')[0].strip()
            if param_name and param_name not in ['self']:
                param_names.append(param_name)
        
        return param_names
    
    def find_classes_with_remote_methods(file_path: Path) -> List[Tuple[str, str, List[Tuple[str, str]]]]:
        """Find all classes that have @remote_function decorated methods.
        
        Returns:
            List of tuples (class_name, base_class_name, [(method_name, signature), ...])
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return []
        
        classes_found = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Extract all methods decorated with @remote_function
                methods = []
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        # Check for remote_function decorator
                        has_remote_decorator = False
                        for decorator in item.decorator_list:
                            if isinstance(decorator, ast.Name) and decorator.id == 'remote_function':
                                has_remote_decorator = True
                            elif isinstance(decorator, ast.Call):
                                if isinstance(decorator.func, ast.Name) and decorator.func.id == 'remote_function':
                                    has_remote_decorator = True
                                elif isinstance(decorator.func, ast.Attribute) and decorator.func.attr == 'remote_function':
                                    has_remote_decorator = True
                        
                        if has_remote_decorator:
                            # Include method if:
                            # 1. It's a dunder method (__xxx__)
                            # 2. Or it doesn't start with underscore
                            if item.name.startswith('__') and item.name.endswith('__'):
                                signature = get_method_signature(item)
                                methods.append((item.name, signature))
                            elif not item.name.startswith('_'):
                                signature = get_method_signature(item)
                                methods.append((item.name, signature))
                
                # Only include classes that have at least one remote method
                if methods:
                    # Determine base class name from the first base (if any)
                    base_name = None
                    if node.bases:
                        base = node.bases[0]
                        if isinstance(base, ast.Name):
                            base_name = base.id
                        elif isinstance(base, ast.Attribute):
                            base_name = base.attr
                    
                    # If no base class found, use object as default
                    if not base_name:
                        base_name = 'object'
                    
                    classes_found.append((node.name, base_name, methods))
        
        return classes_found
    
    def generate_client_class(class_name: str, base_class_name: str,
                             methods: List[Tuple[str, str]], module_name: str,
                             processor_type: str) -> str:
        """Generate client wrapper class code."""
        
        # Extract typing imports from all signatures
        signatures = [sig for _, sig in methods]
        typing_imports = extract_typing_imports(signatures)
        twinkle_imports = extract_twinkle_imports(signatures)
        
        # Build imports section
        import_lines = []
        if typing_imports:
            import_lines.append(f"from typing import {', '.join(sorted(typing_imports))}")
        import_lines.extend([
            "from client.http import TWINKLE_SERVER_URL",
            "from client.http import http_post, heartbeat_manager",
        ])
        # Add twinkle-specific imports
        for imp in sorted(twinkle_imports):
            import_lines.append(imp)
        # Add base module import
        import_lines.append(f"import twinkle.{module_name}")
        import_lines.append("")
        
        # Generate class definition
        code_lines = ['\n'.join(import_lines)]
        code_lines.append(f"\nclass {class_name}(twinkle.{module_name}.{base_class_name}):")
        code_lines.append(f'    """Client wrapper for {class_name} that calls server HTTP endpoints."""')
        code_lines.append("")
        
        # Generate __init__
        code_lines.append("    def __init__(self, **kwargs):")
        code_lines.append("        assert TWINKLE_SERVER_URL")
        code_lines.append("        self.server_url = TWINKLE_SERVER_URL")
        code_lines.append("")
        code_lines.append("        # Create processor instance on server")
        code_lines.append("        response = http_post(")
        code_lines.append("            url=f'{self.server_url}/create',")
        code_lines.append("            json_data={")
        code_lines.append(f"                'processor_type': '{processor_type}',")
        code_lines.append(f"                'class_type': '{class_name}',")
        code_lines.append("                **kwargs")
        code_lines.append("            }")
        code_lines.append("        )")
        code_lines.append("        response.raise_for_status()")
        code_lines.append("        self.processor_id = response.json()")
        code_lines.append("")
        code_lines.append("        # Register for automatic heartbeat")
        code_lines.append("        heartbeat_manager.register_processor(self.processor_id)")
        code_lines.append("")
        
        # Generate __del__
        code_lines.append("    def __del__(self):")
        code_lines.append("        try:")
        code_lines.append("            heartbeat_manager.unregister_processor(self.processor_id)")
        code_lines.append("        except:")
        code_lines.append("            pass")
        code_lines.append("")
        
        # Generate remote methods
        for method_name, signature in methods:
            param_names = parse_params_from_signature(signature)
            
            # Build kwargs dict
            if param_names:
                kwargs_items = ', '.join([f"'{p}': {p}" for p in param_names])
                kwargs_dict = f"{{{kwargs_items}}}"
            else:
                kwargs_dict = "{}"
            
            code_lines.append(f"    def {method_name}(self{', ' + signature if signature else ''}):")
            code_lines.append("        response = http_post(")
            code_lines.append("            url=f'{self.server_url}/call',")
            code_lines.append("            json_data={")
            code_lines.append("                'processor_id': self.processor_id,")
            code_lines.append(f"                'function': '{method_name}',")
            code_lines.append(f"                **{kwargs_dict}")
            code_lines.append("            }")
            code_lines.append("        )")
            code_lines.append("        response.raise_for_status()")
            code_lines.append("        return response.json()")
            code_lines.append("")
        
        return '\n'.join(code_lines)
    
    # Scan all modules
    print("Scanning src/twinkle modules for classes with @remote_function methods...")
    
    # Structure: {module_name: {source_filename: [(class_name, base_class_name, methods), ...]}}
    module_files: Dict[str, Dict[str, List[Tuple[str, str, List[Tuple[str, str]]]]]] = {}
    
    for module_name, module_dir in module_mapping.items():
        module_path = src_twinkle_path / module_dir
        
        if not module_path.exists():
            continue
        
        print(f"  Scanning {module_name}...")
        
        for py_file in module_path.glob('*.py'):
            if py_file.name.startswith('_'):
                continue
            
            classes = find_classes_with_remote_methods(py_file)
            
            if classes:
                if module_name not in module_files:
                    module_files[module_name] = {}
                
                source_filename = py_file.stem
                if source_filename not in module_files[module_name]:
                    module_files[module_name][source_filename] = []
                
                module_files[module_name][source_filename].extend(classes)
    
    # Generate client files
    print("\nGenerating client classes...")
    
    for module_name, source_files in module_files.items():
        client_module_path = src_client_path / module_name
        client_module_path.mkdir(parents=True, exist_ok=True)
        
        processor_type = processor_type_mapping.get(module_name, module_name)
        
        for source_filename, classes in source_files.items():
            client_file = client_module_path / f'{source_filename}.py'
            print(f"  Writing {client_file}...")
            
            # Generate code for all classes from this source file
            all_code_parts = []
            for class_name, base_class_name, methods in classes:
                code = generate_client_class(
                    class_name, base_class_name, methods, module_name, processor_type
                )
                all_code_parts.append(code)
            
            # Combine all classes from the same source file
            combined_code = '\n\n'.join(all_code_parts)
            
            # Write the file
            with open(client_file, 'w', encoding='utf-8') as f:
                f.write(combined_code)
    
    # Generate __init__.py files for each module
    print("\nGenerating __init__.py files...")
    
    for module_name, source_files in module_files.items():
        client_module_path = src_client_path / module_name
        init_file = client_module_path / '__init__.py'
        
        # Collect all class names
        init_lines = []
        for source_filename, classes in sorted(source_files.items()):
            class_names = [class_name for class_name, _, _ in classes]
            for class_name in sorted(class_names):
                init_lines.append(f"from .{source_filename} import {class_name}")
        
        init_content = '\n'.join(init_lines) + '\n'
        
        print(f"  Writing {init_file}...")
        with open(init_file, 'w', encoding='utf-8') as f:
            f.write(init_content)
    
    print("\nProcessor client generation complete!")
    return module_files


def generate_models():
    """Generate client wrapper for Model management."""
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    src_client_path = project_root / 'src' / 'client'
    client_module_path = src_client_path / 'model'
    client_module_path.mkdir(parents=True, exist_ok=True)

    model_code = '''from typing import Any, Optional, Union, Type, Dict, Literal, List
import uuid
from client.http import TWINKLE_SERVER_URL
from client.http import http_post, heartbeat_manager
from twinkle.model.base import TwinkleModel
from transformers import PreTrainedModel, PretrainedConfig
from twinkle import DeviceMesh
from twinkle.data_format import InputFeature, Trajectory


class MultiLoraTransformersModel(TwinkleModel, PreTrainedModel):
    """Client wrapper for TwinkleModel that calls server HTTP endpoints.
    
    This client manages adapters and sends training/inference requests to the model server.
    Each adapter has its own lifecycle managed through automatic heartbeats.
    """
    
    def __init__(self, pretrained_model_name_or_path: str, **kwargs):
        """Initialize model client."""
        self.server_url = TWINKLE_SERVER_URL
        self.adapter_name = None
        response = http_post(
            url=f'{self.server_url}/{pretrained_model_name_or_path}/create',
        )
        response.raise_for_status()
    
    def _send_adapter_heartbeat(self):
        """Internal method to send adapter heartbeat."""
        response = http_post(
            url=f'{self.server_url}/heartbeat',
            json_data={'adapter_name': self.adapter_name}
        )
        response.raise_for_status()
    
    def add_adapter(self, adapter_name: str, config: Dict[str, Any]):
        """Add a new adapter to the model and start automatic heartbeat."""
        response = http_post(
            url=f'{self.server_url}/add_adapter',
            json_data={'adapter_name': adapter_name, 'config': config}
        )
        response.raise_for_status()
        
        # Register adapter for automatic heartbeat after successful creation
        self.adapter_name = adapter_name
        heartbeat_manager.register_adapter(
            self.adapter_name,
            self._send_adapter_heartbeat
        )
        
        return response.json()
    
    def __del__(self):
        """Cleanup: unregister adapter from heartbeat manager."""
        try:
            heartbeat_manager.unregister_adapter(self.adapter_name)
        except:
            pass
    
    def forward(self, inputs: Any, **kwargs):
        """Execute forward pass on the model."""
        response = http_post(
            url=f'{self.server_url}/forward',
            json_data={'inputs': inputs, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def forward_only(self, inputs: Any, **kwargs):
        """Execute forward pass without gradient computation."""
        response = http_post(
            url=f'{self.server_url}/forward_only',
            json_data={'inputs': inputs, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def calculate_loss(self, **kwargs):
        """Calculate loss from model outputs."""
        response = http_post(
            url=f'{self.server_url}/calculate_loss',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def backward(self, **kwargs):
        """Execute backward pass."""
        response = http_post(
            url=f'{self.server_url}/backward',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def forward_backward(self, inputs: Any, **kwargs):
        """Execute combined forward and backward pass."""
        response = http_post(
            url=f'{self.server_url}/forward_backward',
            json_data={'inputs': inputs, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def step(self, **kwargs):
        """Execute optimizer step."""
        response = http_post(
            url=f'{self.server_url}/step',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def zero_grad(self, **kwargs):
        """Zero out gradients."""
        response = http_post(
            url=f'{self.server_url}/zero_grad',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def lr_step(self, **kwargs):
        """Execute learning rate scheduler step."""
        response = http_post(
            url=f'{self.server_url}/lr_step',
            json_data={'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def set_loss(self, loss_cls: str, **kwargs):
        """Set the loss function."""
        response = http_post(
            url=f'{self.server_url}/set_loss',
            json_data={'loss_cls': loss_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def set_optimizer(self, optimizer_cls: str, **kwargs):
        """Set the optimizer."""
        response = http_post(
            url=f'{self.server_url}/set_optimizer',
            json_data={'optimizer_cls': optimizer_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def set_lr_scheduler(self, scheduler_cls: str, **kwargs):
        """Set the learning rate scheduler."""
        response = http_post(
            url=f'{self.server_url}/set_lr_scheduler',
            json_data={'scheduler_cls': scheduler_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def save(self, output_dir: str, **kwargs):
        """Save model checkpoint."""
        response = http_post(
            url=f'{self.server_url}/save',
            json_data={'output_dir': output_dir, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def set_template(self, template_cls: str, **kwargs):
        """Set the template for data processing."""
        response = http_post(
            url=f'{self.server_url}/set_template',
            json_data={'template_cls': template_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
    
    def set_processor(self, processor_cls: str, **kwargs):
        """Set the input processor."""
        response = http_post(
            url=f'{self.server_url}/set_processor',
            json_data={'processor_cls': processor_cls, 'adapter_name': self.adapter_name, **kwargs}
        )
        response.raise_for_status()
        return response.json()
'''

    # Write the model client file
    client_file = client_module_path / 'multi_lora_transformers.py'
    print(f"Generating {client_file}...")
    with open(client_file, 'w', encoding='utf-8') as f:
        f.write(model_code)

    # Create/overwrite __init__.py
    init_file = client_module_path / '__init__.py'
    init_content = "from .multi_lora_transformers import MultiLoraTransformersModel\n"
    print(f"Writing {init_file}...")
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(init_content)

    print("Model client generation complete!")


def generate_samplers():
    """Generate client wrapper for Sampler management."""
    from pathlib import Path

    project_root = Path(__file__).parent.parent
    src_client_path = project_root / 'src' / 'client'
    client_module_path = src_client_path / 'sampler'
    client_module_path.mkdir(parents=True, exist_ok=True)

    sampler_code = '''from typing import Any, Optional, List, Dict
import uuid
from client.http import TWINKLE_SERVER_URL
from client.http import http_post, heartbeat_manager
from twinkle.sampler.base import Sampler
from twinkle import DeviceMesh
from peft import PeftConfig
from twinkle.data_format import Trajectory
import json


class VLLMSampler(Sampler):
    """Client wrapper for Sampler that calls server HTTP endpoints.
    
    This client manages sampling operations and adapter synchronization with the sampler server.
    Each adapter has its own lifecycle managed through automatic heartbeats.
    """
    
    def __init__(self, model_id: str, **kwargs):
        """Create the sampler instance on server."""
        self.server_url = TWINKLE_SERVER_URL
        self.adapter_name = None
        response = http_post(
            url=f'{self.server_url}/{model_id}/create',
            json_data=kwargs
        )
        response.raise_for_status()
        return response.json()
    
    def _send_adapter_heartbeat(self):
        """Internal method to send adapter heartbeat."""
        if not self.adapter_name:
            return
        response = http_post(
            url=f'{self.server_url}/heartbeat',
            json_data={'adapter_name': self.adapter_name}
        )
        response.raise_for_status()
    
    def add_adapter_to_sampler(self, adapter_name: str, config: PeftConfig):
        """Add a new adapter to the sampler and start automatic heartbeat."""
        if isinstance(config, PeftConfig):
            config = config.__dict__
        response = http_post(
            url=f'{self.server_url}/add_adapter_to_sampler',
            json_data={'adapter_name': adapter_name, 'config': config}
        )
        response.raise_for_status()
        
        # Register adapter for automatic heartbeat after successful creation
        self.adapter_name = adapter_name
        heartbeat_manager.register_adapter(
            self.adapter_name,
            self._send_adapter_heartbeat
        )
        
        return response.json()
    
    def __del__(self):
        """Cleanup: unregister adapter from heartbeat manager."""
        try:
            if self.adapter_name:
                heartbeat_manager.unregister_adapter(self.adapter_name)
        except:
            pass
    
    def sample(self, trajectories: List[Trajectory], adapter_name: str = '') -> List[Trajectory]:
        """Sample from the model using provided trajectories."""
        response = http_post(
            url=f'{self.server_url}/sample',
            json_data={'trajectories': json.dumps(trajectories, ensure_ascii=False), 'adapter_name': adapter_name}
        )
        response.raise_for_status()
        return response.json()
    
    def sync_weights(self, state_dict: Dict[str, Any], adapter_name: str = ''):
        """Synchronize weights to the sampler."""
        adapter = adapter_name or self.adapter_name
        response = http_post(
            url=f'{self.server_url}/sync_weights',
            json_data={'state_dict': state_dict, 'adapter_name': adapter}
        )
        response.raise_for_status()
        return response.json()
'''

    # Write the sampler client file
    client_file = client_module_path / 'vllm_sampler.py'
    print(f"Generating {client_file}...")
    with open(client_file, 'w', encoding='utf-8') as f:
        f.write(sampler_code)

    # Create/overwrite __init__.py
    init_file = client_module_path / '__init__.py'
    init_content = "from .vllm_sampler import VLLMSampler\n"
    print(f"Writing {init_file}...")
    with open(init_file, 'w', encoding='utf-8') as f:
        f.write(init_content)
    
    print("Sampler client generation complete!")


if __name__ == '__main__':
    print("Starting client code generation...\n")
    print("=" * 60)
    
    # Generate processor-based clients
    print("\n[1/3] Generating processor-based clients...")
    generate_processors()
    
    # Generate model client
    print("\n" + "=" * 60)
    print("\n[2/3] Generating model client...")
    generate_models()
    
    # Generate sampler client
    print("\n" + "=" * 60)
    print("\n[3/3] Generating sampler client...")
    generate_samplers()
    
    print("\n" + "=" * 60)
    print("\n✓ All client code generation complete!\n")
