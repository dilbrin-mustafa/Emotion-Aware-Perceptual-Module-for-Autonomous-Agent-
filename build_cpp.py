import os
import subprocess
import platform

def compile_cpp():
    modules = [
        {
            "src": "iou_core.cpp", 
            "out_win": "iou_core.dll", 
            "out_nix": "iou_core.so"
        },
        {
            "src": "blas_core.cpp", 
            "out_win": "blas_core.dll", 
            "out_nix": "blas_core.so"
        },
        {
            "src": "color_core.cpp", 
            "out_win": "color_core.dll", 
            "out_nix": "color_core.so"
        }
    ]
    
    system_os = platform.system()
    compiler = "g++"
    
    print(f"Detected OS: {system_os}")

    for mod in modules:
        source_file = mod["src"]
        
        if system_os == "Windows":
            output_file = mod["out_win"]
            cmd = [compiler, "-shared", "-o", output_file, source_file, "-O3", "-static"]
        else:
            output_file = mod["out_nix"]
            cmd = [compiler, "-shared", "-fPIC", "-o", output_file, source_file, "-O3"]

        print(f"Compiling {source_file} -> {output_file}")
        
        try:
            if not os.path.exists(source_file):
                print(f"Source file '{source_file}' not found!")
                continue

            subprocess.check_call(cmd)
            print(f"Generated {output_file}")
            
        except subprocess.CalledProcessError:
            print(f"Compilation failed for {source_file}.")
        except FileNotFoundError:
            print("g++ compiler not found. Ensure MinGW is installed.")
            break

if __name__ == "__main__":
    compile_cpp()