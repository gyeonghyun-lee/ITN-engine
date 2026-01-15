from dataclasses import dataclass, field
from typing import Literal, List, Dict
from enum import Enum, auto


class ItnEntityStatus(Enum):
    INIT = auto()
    EXACT = auto()
    REGEX = auto()
    MODEL = auto()
    POSTPROCESS = auto()


@dataclass
class ItnEntity:
    idx: int
    text: str  # 공백이 있는 원본 문자열
    itn_text: str = None  # 초기에는 공백 제거된 문자열, 나중에는 ITN 문자열을 저장
    status: Literal[ItnEntityStatus.INIT, ItnEntityStatus.EXACT, ItnEntityStatus.REGEX, ItnEntityStatus.MODEL, ItnEntityStatus.POSTPROCESS] = ItnEntityStatus.INIT  # init -> exact matching -> regex matching -> itn model -> postprocess
    idx_itn2text: Dict[int, int] = field(default_factory=dict)  # text와 itn의 position mapping info
    blank_l: bool = False  # 왼쪽 공백
    blank_r: bool = False  # 오른쪽 공백

    def __post_init__(self):
        if self.text[0] == ' ':
            self.blank_l = True
        if self.text[-1] == ' ':
            self.blank_r = True
        self.text = self.text.strip()

        if not self.itn_text:
            self.itn_text = self.text.replace(' ', '')  # remove all blanks
        if self.status == ItnEntityStatus.INIT or self.status == ItnEntityStatus.EXACT:
            self.idx_itn2text = self.get_text_idx_from_itn_idx(self.itn_text, self.text)

    def get_text_idx_from_itn_idx(self, itn_text, text):
        """ text에서 띄어쓰기 제거된 itn의 position 매핑 정보 """
        idx_itn2text = dict()
        pos = 0
        for idx, ch in enumerate(text):
            if ch == ' ':
                continue
            idx_itn2text[pos] = idx
            pos += 1
        return idx_itn2text

@dataclass
class ItnData:
    itn_entity_list: List[ItnEntity] = field(default_factory=list)  # itn entity 리스트

    def add(self, entity: ItnEntity):
        self.itn_entity_list.append(entity)
    
    def pop(self, pos=-1):
        return self.itn_entity_list.pop(pos)
    
    def __len__(self):
        return len(self.itn_entity_list)
    
    def __str__(self):
        buffer = list()
        for entity in self.itn_entity_list:
            if entity.blank_l:
                    buffer.append(' ')
            if entity.status == ItnEntityStatus.INIT:
                buffer.append(entity.text)
            else:
                buffer.append(entity.itn_text)
            if entity.blank_r:
                buffer.append(' ')
        return ''.join(buffer)


if __name__ == '__main__':
    entity = ItnEntity(idx=1, text='주소는 서울시 은평구 사당로이십사길 삼십팔 다시 칠 입니다')
    data = ItnData()
    data.add(entity)
    print(data)
