    # 用于追踪 WSI 处理进度
    current_wsi = None
    wsi_count = 0
    total_wsis = len(sampler.dataset.wsi_ids)
    processed_wsis = set()  # 使用集合来跟踪已处理的WSI，避免重复计数

    for batch_idx, batch in enumerate(pbar):
        # collate_fn 直接返回单个样本的数据 (batch_size=1)
        wsi_id = batch['wsi_id']
        chunk_patches = batch['chunk_patches']  # List of dicts
        label = batch['label'].unsqueeze(0).to(device)  # [1]
        chunk_idx = batch['chunk_idx']  # int
        total_patches = batch['total_patches']  # int
        is_padding = batch['is_padding']  # 填充标记

        # 追踪 WSI 切换 - 使用集合确保每个WSI只计数一次
        if not is_padding and wsi_id not in processed_wsis:
            processed_wsis.add(wsi_id)
            wsi_count = len(processed_wsis)