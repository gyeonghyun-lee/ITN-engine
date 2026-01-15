import ahocorasick
import os
import glob
from entity import ItnData, ItnEntity, ItnEntityStatus
from typing import List


class ExactMatcher:
    def __init__(self, dict_path='./dictionary/exact_match'):
        self.system_dictionary, self.system_matcher = self.load_system_dictionary(dict_path)
        self.user_dictionary, self.user_matcher = self.load_user_dictionary(dict_path)
    
    def load_system_dictionary(self, system_dict_path):
        dictionary = dict()
        matcher = ahocorasick.Automaton()
        system_filename_pattern = os.path.join(system_dict_path, 'system_*.dict')
        system_files = glob.glob(system_filename_pattern)
    
        print(f"Loading system dictionaries: {[os.path.basename(filename) for filename in system_files]}")
        for filename in system_files:
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
    
    def load_user_dictionary(self, user_dict_path):
        dictionary = dict()
        matcher = ahocorasick.Automaton()
        user_filename_pattern = os.path.join(user_dict_path, 'user_*.dict')
        user_files = glob.glob(user_filename_pattern)

        print(f"Loading user dictionaries: {[os.path.basename(filename) for filename in user_files]}")
        for filename in user_files:
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
    
    def process(self, entities: List[ItnEntity]) -> List[ItnEntity]:
        # 1. find matches in system dictionary (ignore whitespaces)
        entities = self.match_system_dictionary(entities)

        # 2. find matches in user dictionary (allow whitespaces)
        entities = self.match_user_dictionary(entities)

        return entities

    def match_system_dictionary(self, entities: List[ItnEntity], idx_start=-1) -> List[ItnEntity]:
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
            pos_word_list = list(self.system_matcher.iter_long(itn_text))
            if not pos_word_list:
                entity.idx = idx
                entity_list.append(entity)
                idx += 1
                continue

            for last_char_pos, (key_id, word_no_space) in pos_word_list:
                word_start = last_char_pos + 1 - len(word_no_space)
                word_end = last_char_pos + 1
                text_start = idx_itn2text[word_start]
                text_end = idx_itn2text[word_end]
                
                if text_start == prev_text_end:
                    entity_list.append(ItnEntity(idx=idx, text=text[text_start:text_end], itn_text=self.system_dictionary[key_id], status=ItnEntityStatus.EXACT))
                    idx += 1
                elif text_start > prev_text_end:
                    entity_list.append(ItnEntity(idx=idx, text=text[prev_text_end:text_start], status=status))
                    entity_list.append(ItnEntity(idx=idx+1, text=text[text_start:text_end], itn_text=self.system_dictionary[key_id], status=ItnEntityStatus.EXACT))
                    idx += 2
                else:
                    raise ValueError
                prev_text_end = text_end
            
            if prev_text_end < len(text):
                entity_list.append(ItnEntity(idx=idx, text=text[prev_text_end:], status=ItnEntityStatus.INIT))
                idx += 1

        return entity_list
    
    def match_user_dictionary(self, entities: List[ItnEntity], idx_start=-1) -> List[ItnEntity]:
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
            pos_word_list = list(self.user_matcher.iter_long(text))
            if not pos_word_list:
                entity.idx = idx
                entity_list.append(entity)
                idx += 1
                continue

            for last_char_pos, (key_id, word) in pos_word_list:
                text_start = last_char_pos + 1 - len(word)
                text_end = last_char_pos + 1
                
                if text_start == prev_text_end:
                    entity_list.append(ItnEntity(idx=idx, text=text[text_start:text_end], itn_text=self.user_dictionary[key_id], status=ItnEntityStatus.EXACT))
                    idx += 1
                elif text_start > prev_text_end:
                    entity_list.append(ItnEntity(idx=idx, text=text[prev_text_end:text_start], status=status))
                    entity_list.append(ItnEntity(idx=idx+1, text=text[text_start:text_end], itn_text=self.user_dictionary[key_id], status=ItnEntityStatus.EXACT))
                    idx += 2
                else:
                    raise ValueError
                prev_text_end = text_end
            
            if prev_text_end < len(text):
                entity_list.append(ItnEntity(idx=idx, text=text[prev_text_end:], status=ItnEntityStatus.INIT))
                idx += 1

        return entity_list


if __name__ == '__main__':
    matcher = ExactMatcher()
    data = ItnData()
    data.add(ItnEntity(1, '주소는 서울시 은평구 사당로이십사길 삼십팔 다시 칠 입니다'))
    print(data)
    data.itn_entity_list = matcher.process(data.itn_entity_list)
    print(data)