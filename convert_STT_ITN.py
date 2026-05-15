import sys, os
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
            import pdb; pdb.set_trace()
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

def split_text(text, max_len=200):
    return wrap(text, max_len)

if __name__ == '__main__':
    # test_cases()
    from textwrap import wrap
    converter = InverseTextNormalizer()
    with open('stt.lst', 'r', encoding='utf-8') as f: #### 생성방법 : ls *.txt > stt.lst 
        for ln in f:
            src_path = ln.strip()

            if not src_path:
                continue

            # 파일 존재 / 크기 체크
            try:
                if not os.path.exists(src_path):
                    print(f"[파일 없음] {src_path}")
                    continue

                file_size = os.path.getsize(src_path)
                if file_size == 0 or file_size == 1:
                    continue

            except Exception as e:
                print(f"[파일 체크 오류] {src_path}")
                print(f"  에러: {e}")
                continue

            # 파일 읽기
            try:
                with open(src_path, 'r', encoding='utf-8') as g:
                    text = g.read().strip()

            except UnicodeDecodeError as e:
                print(f"[인코딩 오류] {src_path}")
                print(f"  에러: {e}")
                continue

            except Exception as e:
                print(f"[읽기 오류] {src_path}")
                print(f"  에러: {e}")
                continue

            save_path = src_path.replace(
                '/Users/ghlee/Downloads/SCL_2603/ALL_TEXT3/', #######  stt.lst 파일의 기본 path를 
                './res_txt/' ##### 별도의 result 디렉토리에 저장 
            )

            # 저장 디렉토리 생성
            try:
                save_dir = os.path.dirname(save_path)
                if save_dir:
                    os.makedirs(save_dir, exist_ok=True)

            except Exception as e:
                print(f"[디렉토리 생성 오류]")
                print(f"  입력 파일: {src_path}")
                print(f"  저장 경로: {save_path}")
                print(f"  에러: {e}")
                continue

            # 변환 및 저장
            try:
                with open(save_path.strip(), 'w', encoding='utf-8') as h:
                    if len(text) >= 100:
                        chunks = split_text(text, 100)
                        outputs = []

                        for chunk in chunks:
                            outputs.append(str(converter.process(chunk)))

                        output = " ".join(outputs)
                    else:
                        output = str(converter.process(text))

                    print(output, file=h)

            except Exception as e:
                print(f"[처리/저장 오류]")
                print(f"  입력 파일: {src_path}")
                print(f"  저장 파일: {save_path}")
                print(f"  에러: {e}")
                continue