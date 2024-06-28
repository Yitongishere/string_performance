import subprocess


def getPython3_command(python3_command=None):
    if python3_command is None:
        python3_command = ['python3','python']
    for py_cmd in python3_command:
        try:
            result = subprocess.run([py_cmd, '--version'], check=True, 
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("Python 3 is installed:", result.stdout.strip())
            return py_cmd
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            continue
    print("Python 3 is not installed or not found.")
    raise Exception("Please check the environment variables of the installed Python or install Python!")
