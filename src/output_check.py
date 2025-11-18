import subprocess
import tempfile
import os
import pandas as pd
import numpy as np

def runCode(code: str, inputs: str) -> str:
    '''
    Compile and execute c++ code

    Args:
        code (String): String of c++ code
        input (String): String of inputs to use on the code
    Returns:
        String : String of the output of the c++ program
    '''

    # Create a new temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Temporary file and executable name
        tempFileName = os.path.join(tmpdir, "temp_code.cpp")
        exeFileName = os.path.join(tmpdir, "temp_code.exe")

        # Write to the file
        with open(tempFileName, 'w') as f:
            f.write(code)

        try:
            # Compile the code
            compile_result = subprocess.run(
                ["g++", tempFileName, "-o", exeFileName],
                capture_output=True,
                text=True,
                check=True
            )
            print("Compiled successfully")
            print("Now executing")
            outputs = np.array([])

            # Run with all inputs
            for input in inputs:
                # Execute the code
                try:
                    execute = subprocess.run(
                        [exeFileName],
                        input=input.encode("utf-8"),
                        capture_output=True,
                        # text=True,
                        check=True,
                        timeout=0.25
                    )

                    # outputs.append(int.from_bytes(execute.stdout))
                    outputs = np.append(outputs, execute.stdout)

                # Catch errors
                except subprocess.CalledProcessError as err:
                    outputs = np.append(outputs, None)
                except subprocess.TimeoutExpired as err:
                    outputs = np.append(outputs, None)
            return outputs

        except subprocess.CalledProcessError as err:
            print(f"Error occured during compilation or execution {err}")
        except FileNotFoundError as err:
            print(f"Error g++ compiler not found {err}")

# Process 2 codes with a list of inputs
def processSnippets(code1: str, code2: str, inputs: list):
    output1 = runCode(code1, inputs)
    output2 = runCode(code2, inputs)
    
    print(f"Code1: {output1}")
    print(f"Code2: {output2}")

# Generate some inputs
def createInputs() -> np.array:
    inputs = np.array(["test\ntest2\ntest3\ntest4"])
    input = ""
    # input for 10 ints
    for i in range(0, 100):
        input += str(i) +"\n"
        if(i != 0 and i % 10 == 0):
            inputs = np.append(inputs, input)
            input = ""
    print(inputs)
    return inputs


# Main
if __name__ == "__main__":
    # Read csv 
    df = pd.read_csv("csv_data/sample_train.csv")

    # Create inputs
    inputs = createInputs()

    # Process each row
    df["output"] = df.apply(lambda row: processSnippets(row["code1"], row["code2"], inputs), axis=1)
