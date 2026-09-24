"""Processor-owned MiMo media identity and whole-item suffix slicing.

Only the fresh bridge supplies this metadata. Validation is fail-closed; no
raw patch or RVQ-code slicing is inferred from placeholder counts alone.
"""
def media_item_runs(request, token_ids, grouped_ids):
    raw = (getattr(request, 'extra_kwargs', None) or {}).get('_vmlx_mimo26_media_items')
    if not isinstance(raw, list) or not raw:
        return None
    sources = {k: list(getattr(request, a, None) or []) for k, a in [('image','images'),('video','videos')]}
    sources['audio'] = list(getattr(request,'audio',None) or getattr(request,'audios',None) or [])
    seen = {k:set() for k in sources};ranges=[];sizes=[];assignments=[];previous=0
    all_ids=set().union(*grouped_ids.values())
    try:
        for item in raw:
            kind=item['modality'];idx=int(item['source_index']);start=int(item['token_start']);end=int(item['token_end'])
            if kind not in sources or not 0 <= idx < len(sources[kind]) or idx in seen[kind]:return None
            if not previous <= start < end <= len(token_ids):return None
            ids=grouped_ids[kind]
            positions=[j for j in range(start,end) if token_ids[j] in ids]
            if not positions or positions[0]!=start or positions[-1]!=end-1:return None
            if len(positions)!=int(item['pad_tokens']):return None
            if any(token_ids[j] in all_ids-ids for j in range(start,end)):return None
            if any(token_ids[j] in all_ids for j in range(previous,start)):return None
            seen[kind].add(idx);ranges.append((start,end));sizes.append(sum(j==start or token_ids[j-1] not in ids for j in positions));assignments.append((kind,sources[kind][idx]));previous=end
        if any(token_ids[j] in all_ids for j in range(previous,len(token_ids))):return None
        if any(seen[k] != set(range(len(v))) for k,v in sources.items()):return None
    except (KeyError,TypeError,ValueError,IndexError):return None
    return ranges,sizes,assignments


def prepare_media_tail(request, token_ids, cached_tokens, grouped_ids):
    grouped=media_item_runs(request,token_ids,grouped_ids)
    if grouped is None or not 0 < cached_tokens < len(token_ids):return None
    ranges,_,_=grouped
    if any(start < cached_tokens < end for start,end in ranges):return None
    records=request.extra_kwargs['_vmlx_mimo26_media_items']
    removed=[r for r,(_,end) in zip(records,ranges) if end<=cached_tokens]
    visual=[r for r in records if r['modality']!='audio'];audios=[r for r in records if r['modality']=='audio']
    removed_visual=[r for r in removed if r['modality']!='audio'];removed_audio=[r for r in removed if r['modality']=='audio']
    pixels=getattr(request,'pixel_values',None);grid=getattr(request,'image_grid_thw',None);codes=getattr(request,'audio_codes',None)
    try:
        pixel_cursor = code_cursor = 0
        for index, record in enumerate(visual):
            if int(record['pixel_start']) != pixel_cursor:return None
            width = int(record['pixel_end']) - pixel_cursor
            if width != 4 * int(record['pad_tokens']):return None
            if grid is None or index >= int(grid.shape[0]):return None
            row = grid[index].tolist()
            if int(row[0])*int(row[1])*int(row[2]) != width:return None
            pixel_cursor += width
        for record in audios:
            if int(record['code_start']) != code_cursor:return None
            width = int(record['code_end']) - code_cursor
            if width != 4 * int(record['pad_tokens']):return None
            code_cursor += width
        if visual and (pixels is None or grid is None or int(pixels.shape[0])!=int(visual[-1]['pixel_end']) or int(grid.shape[0])!=len(visual)):return None
        if audios and (codes is None or int(codes.shape[0])!=int(audios[-1]['code_end'])):return None
        pv_cut=int(removed_visual[-1]['pixel_end']) if removed_visual else 0
        code_cut=int(removed_audio[-1]['code_end']) if removed_audio else 0
    except (KeyError,TypeError,ValueError,IndexError):return None
    # Only now mutate request-owned tensors. Earlier cache and original salts
    # remain untouched, and each remaining grid/code group stays whole.
    if visual:
        request.pixel_values=pixels[pv_cut:] if len(removed_visual)<len(visual) else None
        request.image_grid_thw=grid[len(removed_visual):] if len(removed_visual)<len(visual) else None
    if audios:request.audio_codes=codes[code_cut:] if len(removed_audio)<len(audios) else None
    return {'kind':'mimo_v26_whole_media_tail','covered_media_items':len(removed),
            'remaining_media_items':len(records)-len(removed)}
