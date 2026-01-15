import sys
sys.path.append('./src')
from itn import InverseTextNormalizer
import time


def test_cases():
    converter = InverseTextNormalizer()
    tc_filename = './tc.csv'
    with open(tc_filename, 'r', encoding='utf8') as f:
        for line in f.readlines():
            text, itn_text = line.strip().split('\t')
            start = time.time()
            output = converter.process(text)
            end = time.time()
            if str(output) != itn_text:
                # print(f"입력: {text}")
                # print(f"정답: {itn_text}")
                # print(f"출력: {output}")
                print(f"(X) {text} -> {output}")
            else:
                print(f"(O) {text} -> {output}")
            print(f"{(end-start)*1000:.2f} ms")

if __name__ == '__main__':
    test_cases()
