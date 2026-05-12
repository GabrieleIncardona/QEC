import re
import sys
from fractions import Fraction

def calculate_average_k_over_n(filepath):
    fractions = []
    
    with open(filepath, 'r') as f:
        for line in f:
            match = re.search(r'k:(\d+)\s*/\s*(\d+)\s+captures', line)
            if match:
                k = int(match.group(1))
                n = int(match.group(2))
                fractions.append(Fraction(n, k))
    
    if not fractions:
        print("No n/k values found.")
        return
    
    avg = float(sum(fractions) / len(fractions))
    
    print(f"n/k pairs found : {len(fractions)}")
    print(f"n/k minimum     : {min(fractions)}")
    print(f"n/k maximum     : {max(fractions)}")
    print(f"n/k average     : {avg}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(1)
    calculate_average_k_over_n(sys.argv[1])