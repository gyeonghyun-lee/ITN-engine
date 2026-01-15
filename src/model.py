import re
import os
import time
import pickle
import torch
import enum
import onnxruntime
from typing import List
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSeq2SeqLM, ORTModelForSequenceClassification

from entity import ItnEntity, ItnEntityStatus


class ItnModel:
    has_digit_alpha_email_pattern = r'[0-9A-Za-z@\.]'

    def __init__(self, model_path):
        itn_cls_model_path = os.path.join(model_path, 'itncls')
        itn_model_path = os.path.join(model_path, 'itn')
        self.itn_cls_model = ItnSequenceClassificationModel(model_path=itn_cls_model_path)
        self.itn_model = ItnSeq2SeqModel(model_path=itn_model_path)

        self.set_onnx_session_options()

    def set_onnx_session_options(self):
        session_options = onnxruntime.SessionOptions()
        session_options.intra_op_num_threads = 3
        session_options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
        session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL

    def is_converted(self, text):
        if re.search(self.has_digit_alpha_email_pattern, text):
            return True
        return False

    def process(self, entities: List[ItnEntity]) -> List[ItnEntity]:
        entity_list = list()
        entities_to_model = list()
        for entity in entities:
            status = entity.status
            if status != ItnEntityStatus.INIT:
                entity_list.append(entity)
                continue

            origin_text = entity.text
            input_text = entity.itn_text

            # ITN 모델로 변환 여부를 먼저 결정
            result = self.itn_cls_model.inference(input_text)
            if result == ItnClsStatus.DO_ITN:
                entities_to_model.append(entity)
                continue
            
            entity.itn_text = origin_text
            entity.status = ItnEntityStatus.MODEL
            entity_list.append(entity)
        
        # batch 처리가 속도 빠름
        if entities_to_model:
            text_list = [entity.itn_text for entity in entities_to_model]
            itn_text_list = self.itn_model.inference_batch(text_list)
            for itn_text, entity in zip(itn_text_list, entities_to_model):
                # 숫자나 영어 변환이 없으면 원문을 사용 (띄어쓰기 오보정 방지)
                if not self.is_converted(itn_text):
                    itn_text = entity.text

                entity.itn_text = itn_text
                entity.status = ItnEntityStatus.MODEL
                entity_list.append(entity)

        # entity 순서대로 정렬
        entity_list = sorted(entity_list, key=lambda e: e.idx)
        # breakpoint()
        return entity_list


class ItnClsStatus(enum.Enum):
    DO_NOT_ITN = enum.auto()
    DO_ITN = enum.auto()


class ItnSequenceClassificationModel:
    num_list1 = ["영", "공", "일","이","삼","사","오","육","칠","팔","구","십"]
    num_list2 = ["하나","둘","셋","넷","다섯","여섯","일곱","여덟","아홉","열"]
    alpha_list = ["에이","비","씨","디","이","에프","지","에이치","아이","제이","케이","엘","엔","엠",
                     "오","피","큐","알","에스","티","유","브이","더블유","엑스","와이","제트","앳"]

    def __init__(self, model_path):
        self.max_input_length = 200
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.itn_cls_model = ORTModelForSequenceClassification.from_pretrained(model_path)
                                
    def inference(self, text):
        if len(text) > self.max_input_length:
            raise ValueError(f'The input length must not exceed the max_input_length ({self.max_input_length})')
        text = text + " < es >"
        inputs = self.tokenizer(text, return_tensors="pt")

        # 긴 입력 텍스트는 ITN 수행
        input_length = len(inputs.input_ids[0])
        if input_length > 256:
            return ItnClsStatus.DO_ITN

        # 숫자, 영어에 해당하는 한글이 포함되는 입력 텍스트는 ITN 수행
        itncls_tag = self.character_check(text)
        if itncls_tag == 1:
            return ItnClsStatus.DO_ITN

        # ITN 여부를 판단
        inputs["token_type_ids"] = torch.tensor([inputs["input_ids"].shape[-1]*[0]])
        outputs = self.itn_cls_model(**inputs)
        logits = outputs.logits
        itncls_tag = torch.argmax(logits, dim=-1).tolist()[0]

        if itncls_tag == 1:
            return ItnClsStatus.DO_ITN
        else:
            return ItnClsStatus.DO_NOT_ITN
    
    def character_check(self, text):
        itncls_tag = 0
        for char in text:
            if char in self.num_list1:
                itncls_tag = 1
                return itncls_tag
            
        for char in text:
            if char in self.num_list2:
                itncls_tag = 1
                return itncls_tag
            
        for char in text:
            if char in self.alpha_list:
                itncls_tag = 1
                return itncls_tag
            
        return itncls_tag


class ItnSeq2SeqModel:
    def __init__(self, model_path):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.itn_model = ORTModelForSeq2SeqLM.from_pretrained(model_path)
        self.length = 128
        
    def _find_closest_number(self, target, numbers):   
        numbers = [number for number in numbers if target > number]

        if len(numbers) > 0:
            closest = min(numbers, key=lambda x: abs(x - target))
            return closest 
        else:
            return 0

    def _split_sentences(self, sentence, length):
        if len(sentence) <= length:
            #logger.info(f"[Inverse Text Normalization Model] sentence length: {len(sentence)}")
            return [sentence]

        pattern = " "
        closest_numbers = []

        for target_num in range(0, len(sentence), length):
            matches = re.finditer(pattern, sentence)
            indices = [match.start() for match in matches]

            if indices == []:
                return [sentence]

            closest_number = self._find_closest_number(target_num, indices)
            if closest_number not in closest_numbers:
                closest_numbers.append(closest_number)
            
        closest_numbers.append(len(sentence))
        sentences = []

        idx = 0
        for _ in range(len(closest_numbers)):
            if idx == len(closest_numbers)-1:
                break

            sentences.append(sentence[closest_numbers[idx]:closest_numbers[idx+1]] )
            idx += 1

        return sentences    
    
    def inference(self, text):
        sentences = [i.replace(" ","")+" < es >" for i in self._split_sentences(text, self.length)]
        inputs = self.tokenizer(sentences, return_tensors='pt', padding='longest')
        # itn_ids = self.itn_model.generate(inputs["input_ids"], num_beams=1, max_length=190)
        itn_ids = self.itn_model.generate(inputs["input_ids"], num_beams=1, do_sample=True, top_p=0.5, temperature=0.05, max_length=190)
        itn_text = self.tokenizer.batch_decode(itn_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        itn_text = " ".join([i.replace("< es >","").strip() for i in itn_text])
        return itn_text
    
    def inference_batch(self, text_list: List[str]) -> List[str]:
        input_texts = list()
        # 긴 문장은 여러 개의 텍스트로 분리
        sentence_info = list()
        for text in text_list:
            sentences = [i.replace(" ","")+" < es >" for i in self._split_sentences(text, self.length)]
            input_texts.extend(sentences)
            sentence_info.append(len(sentences))
        
        # tokenize and inference
        inputs = self.tokenizer(input_texts, return_tensors='pt', padding='longest')
        itn_ids = self.itn_model.generate(inputs["input_ids"], num_beams=1, max_length=190)
        # itn_ids = self.itn_model.generate(inputs["input_ids"], num_beams=1, do_sample=True, top_p=0.5, temperature=0.05, max_length=190)
        itn_texts = self.tokenizer.batch_decode(itn_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        
        # 분리된 텍스트 병합하기
        itn_text_list = list()
        start_idx = 0
        for num_sents in sentence_info:
            end_idx = start_idx + num_sents
            itn_text = " ".join([i.replace("< es >","").strip() for i in itn_texts[start_idx:end_idx]])
            itn_text_list.append(itn_text)
            start_idx = end_idx
        return itn_text_list
