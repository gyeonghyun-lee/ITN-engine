import ahocorasick
import os
import glob
from entity import ItnData, ItnEntity, ItnEntityStatus
from typing import List


class DictionaryMatcher:
    def load_dictionary(self, dict_path, filename_pattern):
        dictionary = dict()
        matcher = ahocorasick.Automaton()
        file_search_pattern = os.path.join(dict_path, filename_pattern)
        files = glob.glob(file_search_pattern)

        for filename in files:
            with open(filename, 'r', encoding='utf8') as f:
                for line_num, line in enumerate(f.readlines()):
                    entity = line.strip()
                    if not entity or entity.startswith('#'):
                        continue
                    old, new = self.parse_entity(entity, matcher)
                    
                    # check if the entity already exists
                    val = matcher.get(old, None)
                    if val:
                        print(f"'{old},{val}' already exists. '{old},{new}' in {os.path.basename(filename)} (line {line_num+1}) cannot be added.")
                    else:
                        idx = len(dictionary) + 1
                        matcher.add_word(old, (idx, old))  # key, (idx, key)
                        dictionary[idx] = new
                    
        matcher.make_automaton()
        return dictionary, matcher

    def parse_entity(self, entity, matcher):
        if ',' in entity:
            words = entity.split(',')
            if len(words) == 2:
                old, new = words[0], words[1]
            else:
                # 동음어 ex) 이십사길 -> 24길, 20사길
                # 우선순위는 앞에 있는 엔티티가 높음
                old, new, others = words[0], words[1], words[2:]
        else:
            old, new = entity, entity
        
        return old, new


class SystemDictionaryMatcher(DictionaryMatcher):
    def __init__(self, dict_path):
        self.dictionary, self.matcher = self.load_system_dictionary(dict_path)
    
    def load_system_dictionary(self, dict_path):
        filename_pattern = 'system_*.dict'

        dictionary, matcher = self.load_dictionary(dict_path, filename_pattern)
        return dictionary, matcher
    
    def match_dictionary(self, entities: List[ItnEntity], idx_start=-1) -> List[ItnEntity]:
        if idx_start == -1:
            idx = entities[0].idx
        else:
            idx = idx_start

        entity_list = list()
        for entity in entities:
            text = entity.text
            itn_text = entity.itn_text
            status = entity.status
            idx_itn2text = entity.idx_itn2text

            if entity.status != ItnEntityStatus.INIT:
                entity.idx = idx
                entity_list.append(entity)
                idx += 1
                continue
            
            prev_text_end = 0
            pos_word_list = list(self.matcher.iter_long(itn_text))
            if not pos_word_list:
                entity.idx = idx
                entity_list.append(entity)
                idx += 1
                continue

            for last_char_pos, (key_id, word_no_space) in pos_word_list:
                matched_start = last_char_pos + 1 - len(word_no_space)
                matched_end = last_char_pos
                text_start = idx_itn2text[matched_start]
                text_end = idx_itn2text[matched_end]
                
                # exact match 변환 시 어절 단위로 변환함
                right_blank_idx = text.find(' ', text_end+1)
                itn_text = f"{self.dictionary[key_id]}"
                if right_blank_idx != -1:
                    word_suffix = text[text_end+1:right_blank_idx]
                    itn_text = f"{self.dictionary[key_id]}{word_suffix}"
                    text_end = right_blank_idx

                # 변환된 부분과 아닌 부분을 분리
                if text_start == prev_text_end:
                    exact_entity = ItnEntity(idx=idx, text=text[text_start:text_end+1], itn_text=itn_text, status=ItnEntityStatus.EXACT)
                    entity_list.append(exact_entity)
                    idx += 1
                elif text_start > prev_text_end:
                    init_entity = ItnEntity(idx=idx, text=text[prev_text_end:text_start], status=status)
                    if prev_text_end == 0:
                        init_entity.blank_l = entity.blank_l
                    exact_entity = ItnEntity(idx=idx+1, text=text[text_start:text_end+1], itn_text=itn_text, status=ItnEntityStatus.EXACT)
                    entity_list.append(init_entity)
                    entity_list.append(exact_entity)
                    idx += 2
                else:
                    # multi-match in the same eojeol
                    # do nothing and skip the latter match
                    print(f"'{word_no_space}' is skipped in {itn_text}.")
                    
                prev_text_end = text_end+1
                
            if prev_text_end < len(text):
                init_entity = ItnEntity(idx=idx, text=text[prev_text_end:], status=ItnEntityStatus.INIT)
                if prev_text_end == 0:
                    init_entity.blank_l = entity.blank_l
                entity_list.append(init_entity)
                idx += 1

        return entity_list


class UserDictionaryMatcher(DictionaryMatcher):
    def __init__(self, dict_path):
        self.dictionary, self.matcher = self.load_user_dictionary(dict_path)
    
    def load_user_dictionary(self, dict_path):
        filename_pattern = 'user_*.dict'

        dictionary, matcher = self.load_dictionary(dict_path, filename_pattern)
        return dictionary, matcher   
    
    def match_dictionary(self, entities: List[ItnEntity], idx_start=-1) -> List[ItnEntity]:
        if idx_start == -1:
            idx = entities[0].idx
        else:
            idx = idx_start

        entity_list = list()
        for entity in entities:
            text = entity.text
            status = entity.status

            if status != ItnEntityStatus.INIT:
                entity.idx = idx
                entity_list.append(entity)
                idx += 1
                continue
            
            prev_text_end = 0
            pos_word_list = list(self.matcher.iter_long(text))
            if not pos_word_list:
                entity.idx = idx
                entity_list.append(entity)
                idx += 1
                continue

            for last_char_pos, (key_id, word) in pos_word_list:
                text_start = last_char_pos + 1 - len(word)
                text_end = last_char_pos
                
                # 부분 매칭은 제외 (부분 매칭도 포함하려면 system 사전 사용)
                if text_start > 0 and text[text_start-1] != ' ':
                    continue

                # exact match 변환 시 어절 단위로 변환함
                right_blank_idx = text.find(' ', text_end+1)
                itn_text = f"{self.dictionary[key_id]}"
                if right_blank_idx != -1:
                    word_suffix = text[text_end+1:right_blank_idx]
                    itn_text = f"{self.dictionary[key_id]}{word_suffix}"
                    text_end = right_blank_idx
                
                if text_start == prev_text_end:
                    entity_list.append(ItnEntity(idx=idx, text=text[text_start:text_end+1], itn_text=itn_text, status=ItnEntityStatus.EXACT))
                    idx += 1
                elif text_start > prev_text_end:
                    init_entity = ItnEntity(idx=idx, text=text[prev_text_end:text_start], status=status)
                    if prev_text_end == 0:
                        init_entity.blank_l = entity.blank_l
                    entity_list.append(init_entity)
                    entity_list.append(ItnEntity(idx=idx+1, text=text[text_start:text_end+1], itn_text=itn_text, status=ItnEntityStatus.EXACT))
                    idx += 2
                else:
                    # multi-match in the same eojeol
                    # do nothing and skip the latter match
                    print(f"'{word}' is skipped in {itn_text}.")
                    
                prev_text_end = text_end+1
            
            if prev_text_end < len(text):
                init_entity = ItnEntity(idx=idx, text=text[prev_text_end:], status=ItnEntityStatus.INIT)
                if prev_text_end == 0:
                    init_entity.blank_l = entity.blank_l
                entity_list.append(init_entity)
                idx += 1

        return entity_list


class ExactMatcher:
    def __init__(self, dict_path='./dictionary/exact_match'):
        self.system_matcher = SystemDictionaryMatcher(dict_path)
        self.user_matcher = UserDictionaryMatcher(dict_path)
    
    def process(self, entities: List[ItnEntity]) -> List[ItnEntity]:
        # 1. find matches in system dictionary (ignore whitespaces)
        entities = self.system_matcher.match_dictionary(entities)

        # 2. find matches in user dictionary (allow whitespaces)
        entities = self.user_matcher.match_dictionary(entities)

        return entities


if __name__ == '__main__':
    matcher = ExactMatcher()
    data = ItnData()
    # data.add(ItnEntity(1, '주소는 서울시 은평구 입니다'))
    data.add(ItnEntity(1, '고객님의 현재 약정 기간은 12개월 입니다. 재 약정 하시겠습니까'))
    print(data)
    data.itn_entity_list = matcher.process(data.itn_entity_list)
    print(data)
