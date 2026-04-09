from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    # 학습 설정
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,

    # Optimizer
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    warmup_ratio=WARMUP_RATIO,
    lr_scheduler_type='cosine',
    optim='paged_adamw_8bit',

    # 정밀도
    fp16=True,
    bf16=False,

    # Gradient checkpointing (메모리 절약)
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={'use_reentrant': False},

    # 로깅/저장 (transformers 5.x: evaluation_strategy → eval_strategy)
    logging_steps=LOGGING_STEPS,
    save_steps=SAVE_STEPS,
    eval_steps=SAVE_STEPS,
    eval_strategy='steps',
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
    greater_is_better=False,

    # 기타
    dataloader_num_workers=0,
    remove_unused_columns=False,
    report_to='none',
    seed=42,
)

print('학습 설정 완료')
print(f'  총 스텝 수 (예상): {len(train_dataset) // (BATCH_SIZE * GRAD_ACCUM) * NUM_EPOCHS:,}')
