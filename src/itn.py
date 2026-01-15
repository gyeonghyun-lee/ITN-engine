from exact_match import ExactMatcher
from regex_match import RegexMatcher
from model import ItnModel
from postprocess import Postprocessor
from entity import ItnData, ItnEntity, ItnEntityStatus
from model import ItnModel

class InverseTextNormalizer:
    def __init__(self, dict_path='./dictionary/exact_match', model_path='./model'):
        self.exact_matcher = ExactMatcher(dict_path)
        self.regex_matcher = RegexMatcher()
        self.model = ItnModel(model_path)
        self.postprocess = Postprocessor()
    
    def process(self, text: str) -> str:
        """ STT 문자열을 입력받아 ITN 변환된 텍스트를 반환 """
        data = ItnData()
        entity = ItnEntity(idx=1, text=text)
        data.add(entity)

        data = self.process_exactmatch(data)
        data = self.process_regexmatch(data)
        data = self.process_model(data)
        data = self.process_postprocess(data)
        return data
    
    def process_exactmatch(self, data: ItnData) -> ItnData:
        entities = data.itn_entity_list
        matched_entities = self.exact_matcher.process(entities)
        data.itn_entity_list = matched_entities
        return data
    
    def process_regexmatch(self, data: ItnData) -> ItnData:
        return data
    
    def process_model(self, data: ItnData) -> ItnData:
        entities = data.itn_entity_list
        converted_entities = self.model.process(entities)
        data.itn_entity_list = converted_entities
        return data

    def process_postprocess(self, data: ItnData) -> ItnData:
        return data


if __name__ == '__main__':
    converter = InverseTextNormalizer()
    text = '에스유제이제이 아이엔아이팔팔 골뱅이 네이버 닷컴이요'
    itn_result = converter.process(text)
    print(itn_result)
    for entity in itn_result.itn_entity_list:
        print(entity)
    print(repr(itn_result))
