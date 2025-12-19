import subprocess
import tempfile
import os
import pandas as pd
import numpy as np
import json
import ast
from typing import Any
from concurrent.futures import ProcessPoolExecutor
from difflib import SequenceMatcher
import math

# Inputs
inputs = [
    "hello\nolleh\n",
    "0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n89\n45\n109\n420\n39\n1\n0\n60\n",
    "100\n99\n98\n97\n96\n95\n94\n93\n92\n91\n90\n",
    "4\nword1\nword2\nword3\nword4\n",
    "cat\n5\n10\n8\n4\n",
    "0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n11\n12\n13\n14\n15\n16\n17\n18\n19\n20\n21\n22\n23\n24\n25\n26\n27\n28\n29\n",
    "3\n5 6 7\n1 2 3\n",
    "7\n100 150 200\n10 20 30 40 50 60 70\n",
    "4.5\n10.9\n2.3\n8.9\n18.49\n29.82\n22.22\n14.00\n6.89\n",
    "-8\n-12\n-5\n-11\n-4\n-2\n-32\n-450\n-20\n-89\n"
    "apple\nbanana\ncat\ndog\nzebra\nalpha\nomega\ntest\nhello\nworld\n"
]
inputs = [x.encode('utf-8') for x in inputs]

# Compare String
def _string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(a=a, b=b).ratio()

# Check if number
def is_number(x):
    try:
        result = float(x)

        return True
    except:
        return False

# Determine the similarity of 2 numbers
def _numeric_similarity(out1: float, out2: float) -> float:
    if out1 == out2:
        return 1.0
    # NaN type
    elif math.isnan(out1) or math.isnan(out2):
        if (math.isnan(out1) and not math.isnan(out2)) or (not math.isnan(out1) and math.isnan(out2)):
            return 0.0
        else:
            return 1.0
    # Large / infinite numbers
    elif math.isinf(out1) or math.isinf(out2):
        if (math.isinf(out1) and not math.isinf(out2)) or (not math.isinf(out1) and math.isinf(out2)):
            return 0.0
        else:
            return 1.0


    # Find absolute distance between highest and smallest number
    out1 = abs(out1)
    out2 = abs(out2)
    highest = max(out1, out2)
    
    # Avoid division by 0
    if highest == 0:
        highest = 0.1
    smallest = min(out1, out2)
    distance = abs(highest - smallest)

    # Normalize the distance the numbers to a range of [0-1]
    normalizedDistance = np.round(1 - (distance / abs(highest)), decimals=3)
    # print(f"Normalized dist {out1} {out2} {normalizedDistance}")
    return normalizedDistance

# Parse into json, list, or hash/dict
def try_parse(s):
    # Try JSON
    try:
        return json.loads(s)
    except:
        pass
    
    # Try Python literal (e.g. "[1,2,3]" or "{'a': 2}")
    try:
        return ast.literal_eval(s)
    except:
        pass

    # Try to break into list if separated by spaces
    try:
        if " " in s:
            return s.split()
    except: 
        pass

    return s

# Compare lists
def listSimilarity(output1: list, output2: list) -> float:
    try:
        if not isinstance(output1, list) or not isinstance(output2, list):
            return 0.0
        listLength = min(len(output1), len(output2))
        if listLength <= 0:
            return 0.0
        
        outputScore = 0
        for i in range(0, listLength):

            o1 = try_parse(output1[i])
            o2 = try_parse(output2[i])

            if o1 is None or o2 is None:
                outputScore += noneSim(o1, o2)

            elif isinstance(o1, list) or isinstance(o2, list):
                outputScore += listSimilarity(o1, o2)

            elif isinstance(o1, dict) or isinstance(o2, dict):
                outputScore += 1.0 if o1 == o2 else 0.0

            elif is_number(o1) and is_number(o2):
                outputScore += _numeric_similarity(float(o1), float(o2))

            # Nan Character
            elif '\ufffd' in text1 or  '\ufffd' in text2:
                if ('\ufffd' in text1 and '\ufffd' not in text2) or ('\ufffd' not in text1 and '\ufffd' in text2) :
                    outputScore += 0.0
                else:
                    outputScore += 1.0
            
            else:
                outputScore += _string_similarity(o1, o2)

        return outputScore / listLength
    except:
        return 0.0
    

# Compare None types
def noneSim(out1: Any, out2: Any)-> float:
    
    if (out1 == 0 and out2 is None) or (out2 == 0 and out1 is None):
        return 0.9

    elif(out1 is None and out2 == '') or (out2 is None and out1 == ''):
        return 0.9

    elif out1 is None and out2 is None:
        return 1.0

    else:
        return 0.0

# Compare logic
def compare(raw1: bytes, text1: str, raw2: bytes, text2: str) -> float:
    try:

        # Compare raw bytes
        if raw1 == raw2:
            return 1.0

        # try to parse into a dict or list
        text1 = try_parse(text1)
        text2 = try_parse(text2)

        # None type
        if text1 is None or text2 is None:
            return noneSim(text1, text2)
                
        # List
        elif isinstance(text1, list) or isinstance(text2, list):
            return listSimilarity(text1, text2)
        
        # Dict
        elif isinstance(text1, dict) or isinstance(text2, dict):
            return 1.0 if text1 == text2 else 0.0

        # Numbers
        elif is_number(text1) and is_number(text2):
            return _numeric_similarity(float(text1), float(text2))

            # Nan Character
        elif '\ufffd' in text1 or  '\ufffd' in text2:
            if ('\ufffd' in text1 and '\ufffd' not in text2) or ('\ufffd' not in text1 and '\ufffd' in text2) :
                return 0.0
            else:
                return 1.0
        # String
        else:
            return _string_similarity(text1, text2)
    except:
        return 0

# Compile and run the cpp program pair
def run (job: tuple) -> list:
    fName1, fName2, code1, code2, sim = job

    try:
        # --- Write C++ file ---
        with open(fName1, "w") as f:
            f.write(code1)

        with open(fName2, "w") as f:
            f.write(code2)

        exe1 = fName1.replace(".cpp", "")
        exe2 = fName2.replace(".cpp", "")

        # --- Compile ---
        compile_proc1 = subprocess.run(
            ["g++", fName1, "-o", exe1],
            text=False,
            capture_output=False,
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )
        # --- Compile ---
        compile_proc2 = subprocess.run(
            ["g++", fName2, "-o", exe2],
            text=False,
            capture_output=False,
            stdin=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False
        )

        # Compilation failed
        if compile_proc1.returncode != 0 and compile_proc2.returncode !=0:
            return {"output": 1, "sim": sim} 
        elif (compile_proc1.returncode != 0 and compile_proc2.returncode == 0) or ( compile_proc2.returncode != 0 and compile_proc1.returncode == 0):
            return {"output": 0, "sim": sim} 

        outputs = 0
        # --- Run with series of inputs--- 
        for input in inputs:
            # Run code 1
            try:
                out1 = subprocess.run(
                    [exe1],
                    input=input,
                    capture_output=True,
                    text=False,
                    check=True,
                    timeout=0.5,
                    cwd=tmpdir
                )
                raw1 = out1.stdout
                text1 = raw1.decode("utf-8", errors="replace").replace("\n", " ")
            # Catch error in program
            except subprocess.CalledProcessError as e:
                raw1 = None
                text1 = None
            # Program was not given enough inputs
            except subprocess.TimeoutExpired as e:
                raw1 = None
                text1 = None
            # Run code2
            try:
                out2 = subprocess.run(
                    [exe2],
                    input=input,
                    capture_output=True,
                    text=False,
                    check=True,
                    timeout=0.5,
                    cwd=tmpdir
                )
                raw2 = out2.stdout
                text2 = raw2.decode("utf-8", errors="replace").replace("\n", " ")
            # Catch error in program
            except subprocess.CalledProcessError as e:
                raw2 = None
                text2 = None
            # Program was not given enough inputs
            except subprocess.TimeoutExpired as e:
                raw2 = None
                text2 = None
            
            outputs += compare(raw1, text1, raw2, text2)
            
        # FIle cleanup
        os.remove(fName1)
        os.remove(fName2)
        os.remove(exe1)
        os.remove(exe2)

        score = round((outputs / len(inputs)), 2)
        return {"output": score, "sim": sim}

    except Exception as err:
        print(f"Error in file running {err}")
        return {"output": 0, "sim": sim}

# Process jobs with multithread.
def process(df: pd.DataFrame, results: np.array) -> None:
        jobs = []
        # Create job with items
        for index, row in df.iterrows():
            jobs.append(
                (
                    os.path.join(tmpdir,f"prog_{index}.1.cpp"), 
                    os.path.join(tmpdir, f"prog_{index}.2.cpp"), 
                    row["code1"], 
                    row["code2"], 
                    row["similar"]
                )
            )
        
        error = 0
        numThreads = 4
        # Run multiple at once
        with ProcessPoolExecutor(max_workers=numThreads) as pool:
            for output in pool.map(run,jobs):
                results.append([output["output"], output["sim"]])
                error += abs(output["sim"] - output["output"])
        print(f"Batch Error: {error / 50}")

# Main
if __name__ == "__main__":
    # Read csv 
    batchSize = 50
    csvFile = "csv_data/sample_train.csv"
    
    results = []

    # Process in batches
    with tempfile.TemporaryDirectory() as tmpdir:
        for batch in pd.read_csv(csvFile, chunksize=batchSize, usecols=["code1", "code2", "similar"]):
            process(batch, results)

            results = np.array(results)
            results = []
