# pre-requisite
### pdm 설치  
* [pdm 설치 방법](https://lgu-cto.atlassian.net/wiki/spaces/NaturalLang/pages/38464390850/Python+pyenv+pdm#PDM-(https%3A%2F%2Fpdm-project.org%2Flatest%2F)) 참고

# 실행
### 모델 파일 경로 및 복사
```sh
curl -sSL https://pdm-project.org/install-pdm.py | python3 -
export PATH=/root/.local/bin:$PATH
```

### 파이썬 패키지 설치
```sh
pdm init
pdm install
```

### Test case 실행
```sh
python3 test_case.py
```

##딕셔너리 적용
```sh
dictionary/exact_match
 - 변환 예외 대상 단어 넣어주면 됩니다.
 - 서울특별시        -> 서울특별시 그대로 나와 변환처리되지 않음
 - 발산일동,발산1동    -> 발산일동을 발산1동으로 변환하여 처리함
```
