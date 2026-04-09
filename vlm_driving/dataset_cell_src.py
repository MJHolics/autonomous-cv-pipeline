SYSTEM_PROMPT = (
    '당신은 자율주행 차량의 카메라 영상을 분석하는 전문 AI입니다. '
    '주어진 이미지에서 도로 상황, 주변 객체, 위험 요소를 정확하게 파악하고 '
    '안전 운전을 위한 정보를 제공합니다.'
)

class CarlaVQADataset(Dataset):
    # Qwen2-VL용 데이터셋
    # Dataset 레벨에서 padding/truncation 안 함 (이미지 토큰 수 동적)
    # collate_fn에서 배치 내 최대 길이로 패딩
    def __init__(self, data, processor):
        self.data = data
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item['image']).convert('RGB')
        question = item['conversations'][0]['value'].replace('<image>', '').strip()
        answer   = item['conversations'][1]['value'].strip()

        # 학습용 (질문+답변 포함)
        messages_full = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {
                'role': 'user',
                'content': [
                    {'type': 'image', 'image': image},
                    {'type': 'text',  'text': question},
                ],
            },
            {'role': 'assistant', 'content': answer},
        ]
        text_full = self.processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )

        # 질문만 (loss 마스킹 경계 탐색)
        messages_q = messages_full[:-1]
        text_q = self.processor.apply_chat_template(
            messages_q, tokenize=False, add_generation_prompt=True
        )

        # 인코딩 (padding/truncation 없이) — processor가 반환하는 모든 키 보존
        inputs = self.processor(
            text=[text_full],
            images=[image],
            return_tensors='pt',
        )
        inputs_q = self.processor(
            text=[text_q],
            images=[image],
            return_tensors='pt',
        )

        input_ids      = inputs['input_ids'].squeeze(0)
        attention_mask = inputs['attention_mask'].squeeze(0)
        question_len   = inputs_q['input_ids'].shape[1]

        # labels: 질문/이미지 토큰 -100 마스킹
        labels = input_ids.clone()
        labels[:question_len] = -100

        result = {
            'input_ids':      input_ids,
            'attention_mask': attention_mask,
            'labels':         labels,
        }

        # Qwen2-VL 필수 필드 전부 포함
        for key in ['pixel_values', 'image_grid_thw', 'mm_token_type_ids',
                    'video_grid_thw', 'second_per_grid_ts']:
            if key in inputs:
                val = inputs[key]
                result[key] = val.squeeze(0) if val.dim() > 1 and val.shape[0] == 1 else val

        return result


def collate_fn(batch):
    # 동적 패딩: 배치 내 최대 길이로 패딩
    pad_id = 0
    max_len = max(b['input_ids'].shape[0] for b in batch)

    def pad_1d(seqs, pad_val):
        out = []
        for s in seqs:
            pad_len = max_len - s.shape[0]
            out.append(torch.cat([s, torch.full((pad_len,), pad_val, dtype=s.dtype)]))
        return torch.stack(out)

    result = {
        'input_ids':      pad_1d([b['input_ids'] for b in batch], pad_id),
        'attention_mask': pad_1d([b['attention_mask'] for b in batch], 0),
        'labels':         pad_1d([b['labels'] for b in batch], -100),
    }

    # mm_token_type_ids: input_ids와 동일한 shape → 같은 방식으로 패딩 (pad값=0)
    if 'mm_token_type_ids' in batch[0]:
        result['mm_token_type_ids'] = pad_1d([b['mm_token_type_ids'] for b in batch], 0)

    # pixel_values: (num_patches, C*kernel*kernel) — 배치 내 concat
    if 'pixel_values' in batch[0]:
        pvs = [b['pixel_values'] for b in batch]
        result['pixel_values'] = torch.cat(pvs, dim=0) if pvs[0].dim() == 2 else torch.stack(pvs)

    # image_grid_thw: (num_images, 3) — 배치 내 concat
    if 'image_grid_thw' in batch[0]:
        thws = [b['image_grid_thw'] for b in batch]
        if thws[0].dim() == 1:
            result['image_grid_thw'] = torch.stack(thws)
        else:
            result['image_grid_thw'] = torch.cat(thws, dim=0)

    return result


# 데이터셋 생성
train_dataset = CarlaVQADataset(train_data, processor)
val_dataset   = CarlaVQADataset(val_data,   processor)

print(f'Train dataset: {len(train_dataset):,}개')
print(f'Val   dataset: {len(val_dataset):,}개')

# 샘플 키 확인
sample = train_dataset[0]
print('샘플 키:', list(sample.keys()))
for k, v in sample.items():
    print(f'  {k}: {v.shape}')
