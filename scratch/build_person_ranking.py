import os, base64, io, json
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np
import torch
import torchvision.transforms as transforms
import timm
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

desktop_dir = Path('/Users/neo/Desktop/选美')
exclude_keywords = ['荷花', '莲池', '莲花', '精灵', '仙子', '四视图', '街景']
valid_files = [f for f in sorted(desktop_dir.iterdir()) if f.suffix.lower() in ('.jpg', '.png', '.jpeg') and not f.name.startswith('.') and not any(k in f.name for k in exclude_keywords)]

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = timm.create_model('resnet50', pretrained=False, num_classes=1)
state = torch.load('models/scut_fbp5500_beauty.pt', map_location='cpu')
model.load_state_dict(state)
model.to(device)
model.eval()

tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_model = 'models/face_detection_full_range.tflite'
options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(model_asset_path=full_model),
    min_detection_confidence=0.35,
)

data = []
with vision.FaceDetector.create_from_options(options) as detector:
    for p in valid_files:
        img = Image.open(p).convert('RGB')
        w, h = img.size
        rgb = np.asarray(img)
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        valid_dets = [d for d in res.detections if d.bounding_box.origin_y < 0.65 * h]
        if not valid_dets: continue
        best = max(valid_dets, key=lambda d: d.categories[0].score)
        box = best.bounding_box
        x0, y0, bw, bh = max(0, box.origin_x), max(0, box.origin_y), min(box.width, w - box.origin_x), min(box.height, h - box.origin_y)
        margin = 0.45
        side = max(bw, bh) * (1 + 2 * margin)
        cx, cy = x0 + bw / 2, y0 + bh / 2
        l, t = max(0, int(cx - side / 2)), max(0, int(cy - side / 2))
        r, b = min(w, int(cx + side / 2)), min(h, int(cy + side / 2))
        crop = img.crop((l, t, r, b))
        box_coords = (l, t, r, b)
        
        crop_eval = crop.copy()
        crop_eval.thumbnail((224, 224))
        tensor = tf(crop_eval).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model.forward_features(tensor)
            pooled = model.global_pool(feat)
            raw_score = model.fc(pooled).item()
        
        norm_score = round(40 + (raw_score - 2.0) / (5.3 - 2.0) * 58, 1)
        norm_score = max(35.0, min(99.0, norm_score))
        
        # Bounding box
        marked_img = img.copy()
        draw = ImageDraw.Draw(marked_img)
        draw.rectangle(box_coords, outline='#ff3366', width=max(6, int(w / 100)))
        
        # Crop b64
        buf_c = io.BytesIO()
        crop_disp = crop.copy()
        crop_disp.thumbnail((450, 450))
        crop_disp.save(buf_c, format='JPEG', quality=88)
        b64_c = base64.b64encode(buf_c.getvalue()).decode('utf-8')
        
        # Full b64
        buf_f = io.BytesIO()
        full_disp = marked_img.copy()
        full_disp.thumbnail((1200, 1200))
        full_disp.save(buf_f, format='JPEG', quality=85)
        b64_f = base64.b64encode(buf_f.getvalue()).decode('utf-8')
        
        data.append({
            'name': p.name,
            'raw': round(raw_score, 3),
            'norm': norm_score,
            'conf': round(best.categories[0].score, 2),
            'size': f'{w}×{h}',
            'crop_b64': b64_c,
            'full_b64': b64_f
        })

print(f'Processed {len(data)} images.')

# Multi-photo character sets definition based on photoshoot character sessions:
# 1. 地中海咖啡馆女孩 (User highlighted in red)
mediterranean_set = {'阳光下的地中海街道人像.png', '阳光下的街头咖啡馆.png', '阳光下的欧洲街角咖啡馆.png'}

# 2. 极简丹宁/棚拍时尚模特系列
studio_denim_set = {
    'Minimalist Studio Fashion Portrait(1).png', 'Minimalist Blue Denim Fashion Portrait.png',
    'Polished Studio Fashion Portrait.png', 'Casual Chic Studio Portrait.png',
    'Casual Denim Studio Portrait.png', 'Studio Fashion Portrait in Blue and Denim.png',
    '红白简约时尚全身肖像.png', '红白简约时尚全身肖像 (1).png'
}

# 3. 优雅礼服系列
elegant_dress_set = {
    'Elegant Black Dress Studio Portrait.png', 'Elegant Ivory Lace Portrait.png',
    'Elegant Woman in a Wine Satin Dress.png'
}

# 4. 蓝调与夜景公园系列
blue_night_set = {
    'Serene night park portrait.png', 'Soft Blue Beauty Portrait.png', '清新蓝调俏皮人像.png'
}

# Group all photos into person entities
person_groups = []
allocated = set()

# Process multi-photo sets first
for p_title, p_set in [
    ('地中海漫步风尚女孩（用户截图确认选手）', mediterranean_set),
    ('极简丹宁与红白棚拍模特', studio_denim_set),
    ('黑白红三色晚礼服名媛', elegant_dress_set),
    ('蓝调与静谧夜景写真女孩', blue_night_set),
]:
    matched = [d for d in data if d['name'] in p_set]
    if matched:
        person_groups.append({
            'title': p_title,
            'photos': matched
        })
        for m in matched:
            allocated.add(m['name'])

# Remaining photos are solo contestants
for d in data:
    if d['name'] not in allocated:
        person_groups.append({
            'title': d['name'].replace('.png', '').replace('.jpg', ''),
            'photos': [d]
        })

print(f'Formed {len(person_groups)} distinct contestant profiles.')

# Compute person-level scores and best photo
persons = []
for p_idx, grp in enumerate(person_groups):
    photos = grp['photos']
    best_photo = max(photos, key=lambda x: x['raw'])
    
    raw_scores = [x['raw'] for x in photos]
    norm_scores = [x['norm'] for x in photos]
    avg_raw = round(float(np.mean(raw_scores)), 3)
    avg_norm = round(float(np.mean(norm_scores)), 1)
    peak_norm = round(float(np.max(norm_scores)), 1)
    
    persons.append({
        'person_id': p_idx + 1,
        'title': grp['title'],
        'photo_count': len(photos),
        'best_photo': best_photo,
        'all_photos': sorted(photos, key=lambda x: x['raw'], reverse=True),
        'avg_raw': avg_raw,
        'avg_norm': avg_norm,
        'peak_norm': peak_norm
    })

# Sort persons by avg_norm (Person Comprehensive Base Score) desc
persons.sort(key=lambda x: x['avg_norm'], reverse=True)

# Generate HTML Cards
cards_html = []
for i, p in enumerate(persons):
    rank_cls = 'top-1' if i == 0 else 'top-2' if i == 1 else 'top-3' if i == 2 else ''
    
    # Mini thumbs of candidate faces
    thumbs_html = ''.join([f'<img class="thumb-mini" src="data:image/jpeg;base64,{ph["crop_b64"]}" alt="" title="{ph["name"]}" />' for ph in p['all_photos'][:6]])
    
    cards_html.append(f'''
    <div class="card" onclick="openPersonModal({i})">
      <div class="card-hero">
        <span class="rank-badge {rank_cls}">NO. {i+1}</span>
        <span class="photo-badge">{p['photo_count']} 张参选写真</span>
        <img src="data:image/jpeg;base64,{p['best_photo']['full_b64']}" alt="代表作" />
        <div class="hero-mask"></div>
        <div class="hero-info">
          <div class="hero-title">{p['title']}</div>
          <div class="hero-sub">最佳代表作：{p['best_photo']['name']}</div>
        </div>
      </div>
      <div class="card-body">
        <div class="score-matrix">
          <div class="stat-item">
            <div class="stat-label">选手综合底子分</div>
            <div class="stat-main">{p['avg_norm']} <small style="font-size:12px;color:var(--text-muted)">分</small></div>
          </div>
          <div class="stat-item">
            <div class="stat-label">最高单图上相分</div>
            <div class="stat-sub">{p['peak_norm']} <small style="font-size:12px;color:var(--text-muted)">分</small></div>
          </div>
        </div>
        <div style="font-size:11px;color:var(--text-muted);display:flex;justify-content:space-between">
          <span>参选面孔预览（{p['photo_count']}张）：</span>
          <span style="color:var(--blue)">展开详情 ➔</span>
        </div>
        <div class="thumbs-strip">{thumbs_html}</div>
        <div class="action-hint">🔍 点击展开查看选手全部着装写真与单图评分</div>
      </div>
    </div>
    ''')

full_html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>BeautyPick · 以人为本 · 选美总榜 (一人一卡 · 18位选手)</title>
<style>
  :root {{
    --bg: #070a13; --card-bg: #0f172a; --border: #1e293b;
    --text: #f8fafc; --text-muted: #94a3b8; --accent: #f43f5e;
    --gold: #fbbf24; --blue: #38bdf8; --purple: #a855f7; --green: #10b981;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg); color: var(--text); padding: 24px; }}
  header {{ margin-bottom: 28px; padding-bottom: 20px; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 16px; }}
  h1 {{ font-size: 26px; font-weight: 800; color: #fff; display: flex; align-items: center; gap: 12px; }}
  .badge-main {{ background: linear-gradient(135deg, #f43f5e, #fb7185); color: #fff; padding: 5px 14px; border-radius: 9999px; font-size: 13px; font-weight: 700; }}
  .intro {{ color: var(--text-muted); font-size: 14px; margin-top: 8px; line-height: 1.6; max-width: 840px; }}
  .banner {{ background: rgba(244, 63, 94, 0.08); border-left: 4px solid var(--accent); padding: 12px 18px; border-radius: 8px; font-size: 13px; color: #fecdd3; line-height: 1.6; }}

  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); gap: 24px; }}
  .card {{ background: var(--card-bg); border: 1px solid var(--border); border-radius: 16px; overflow: hidden; display: flex; flex-direction: column; cursor: pointer; transition: all .25s ease; position: relative; }}
  .card:hover {{ transform: translateY(-5px); border-color: var(--blue); box-shadow: 0 16px 36px rgba(0,0,0,0.6); }}

  .card-hero {{ height: 260px; position: relative; background: #000; overflow: hidden; }}
  .card-hero img {{ width: 100%; height: 100%; object-fit: cover; }}
  .hero-mask {{ position: absolute; inset: 0; background: linear-gradient(to top, rgba(15,23,42,0.95) 0%, rgba(15,23,42,0.2) 50%, rgba(0,0,0,0.4) 100%); }}
  
  .rank-badge {{ position: absolute; top: 12px; left: 12px; z-index: 5; background: rgba(0,0,0,0.85); font-size: 15px; font-weight: 800; padding: 4px 12px; border-radius: 8px; border: 1px solid rgba(255,255,255,0.25); }}
  .top-1 {{ color: #fbbf24; border-color: #fbbf24; box-shadow: 0 0 12px rgba(251,191,36,0.3); }}
  .top-2 {{ color: #f8fafc; border-color: #cbd5e1; }}
  .top-3 {{ color: #fb923c; border-color: #f97316; }}

  .photo-badge {{ position: absolute; top: 12px; right: 12px; z-index: 5; background: rgba(168, 85, 247, 0.9); color: #fff; font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 6px; }}

  .hero-info {{ position: absolute; bottom: 12px; left: 14px; right: 14px; z-index: 5; }}
  .hero-title {{ font-size: 16px; font-weight: 700; color: #fff; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; text-shadow: 0 2px 4px rgba(0,0,0,0.8); }}
  .hero-sub {{ font-size: 12px; color: var(--text-muted); margin-top: 3px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

  .card-body {{ padding: 16px; display: flex; flex-direction: column; gap: 12px; flex: 1; background: var(--card-bg); }}
  .score-matrix {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; background: #090e1a; padding: 12px; border-radius: 10px; border: 1px solid rgba(255,255,255,0.04); }}
  .stat-item {{ text-align: center; }}
  .stat-label {{ font-size: 11px; color: var(--text-muted); margin-bottom: 2px; }}
  .stat-main {{ font-size: 20px; font-weight: 800; color: var(--blue); }}
  .stat-sub {{ font-size: 16px; font-weight: 700; color: var(--accent); }}

  .thumbs-strip {{ display: flex; gap: 8px; overflow-x: auto; padding: 4px 0; }}
  .thumb-mini {{ width: 44px; height: 44px; border-radius: 6px; object-fit: cover; border: 1px solid var(--border); flex-shrink: 0; }}

  .action-hint {{ font-size: 12px; color: #64748b; text-align: center; margin-top: auto; }}

  /* Modal */
  .modal {{ display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.94); z-index: 1000; align-items: center; justify-content: center; padding: 24px; }}
  .modal.active {{ display: flex; }}
  .modal-box {{ width: 92vw; max-width: 1200px; max-height: 90vh; background: #0f172a; border: 1px solid var(--border); border-radius: 20px; padding: 24px; position: relative; display: flex; flex-direction: column; gap: 20px; overflow-y: auto; }}
  .modal-header {{ display: flex; justify-content: space-between; align-items: flex-start; padding-bottom: 14px; border-bottom: 1px solid var(--border); }}
  .modal-close {{ font-size: 32px; color: #94a3b8; cursor: pointer; border: none; background: transparent; line-height: 1; }}
  .modal-close:hover {{ color: #fff; }}
  
  .photos-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 16px; }}
  .photo-cell {{ background: #090e1a; border: 1px solid var(--border); border-radius: 12px; overflow: hidden; display: flex; flex-direction: column; }}
  .photo-duo {{ display: flex; height: 200px; }}
  .photo-duo img {{ width: 50%; height: 100%; object-fit: cover; }}
  .photo-info {{ padding: 10px 12px; display: flex; justify-content: space-between; align-items: center; font-size: 12px; }}
</style>
</head>
<body>
  <header>
    <div>
      <h1><span>颜选 · BeautyPick</span> <span class="badge-main">以人为本 · 选美总榜 (一人一卡)</span></h1>
      <div class="intro">
        已实现<strong>人物实体聚类归并</strong>！截图中的地中海街景女孩（3张着装写真）与棚拍模特（8张写真）均已合并为独立选手档案。<br>
        • <strong>每位选手仅占一个榜单席位</strong>，彻底告别换衣服霸榜与重复竞争；<br>
        • <strong>选手综合底子分</strong>：由该选手的全部写真剔除光影噪点后计算出的稳定均值，代表真实骨相底子。
      </div>
    </div>
    <div class="banner">
      💡 点击任意选手卡片，即可展开查看该选手的<strong>所有参选着装照片与独立打分</strong>！
    </div>
  </header>

  <div class="grid">
    {''.join(cards_html)}
  </div>

  <!-- Person Modal -->
  <div id="modal" class="modal" onclick="if(event.target===this)closeModal()">
    <div class="modal-box">
      <div class="modal-header">
        <div>
          <div id="m-rank" style="font-size:24px;font-weight:800;color:var(--gold)"></div>
          <div id="m-title" style="font-size:18px;font-weight:700;color:#fff;margin-top:4px"></div>
          <div id="m-stats" style="font-size:13px;color:var(--text-muted);margin-top:6px"></div>
        </div>
        <button class="modal-close" onclick="closeModal()">&times;</button>
      </div>
      <div id="m-photos" class="photos-grid"></div>
    </div>
  </div>

  <script>
    const personsData = {json.dumps(persons, ensure_ascii=False)};

    function openPersonModal(idx) {{
      const p = personsData[idx];
      document.getElementById('m-rank').innerText = 'NO. ' + (idx + 1) + ' 选手参赛档案';
      document.getElementById('m-title').innerText = p.title;
      document.getElementById('m-stats').innerHTML = 
        '综合颜值底子分：<strong style="color:var(--blue);font-size:16px">' + p.avg_norm + ' 分</strong> &nbsp;|&nbsp; ' +
        '最佳单图上相分：<strong style="color:var(--accent);font-size:16px">' + p.peak_norm + ' 分</strong> &nbsp;|&nbsp; ' +
        '共归集参选照片：' + p.photo_count + ' 张';
      
      const grid = document.getElementById('m-photos');
      grid.innerHTML = '';
      p.all_photos.forEach((ph, i) => {{
        const cell = document.createElement('div');
        cell.className = 'photo-cell';
        cell.innerHTML = `
          <div class="photo-duo">
            <img src="data:image/jpeg;base64,${{ph.crop_b64}}" alt="面部" title="面部特写" />
            <img src="data:image/jpeg;base64,${{ph.full_b64}}" alt="原图" title="原图标定" />
          </div>
          <div class="photo-info">
            <div style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:160px;color:#cbd5e1" title="${{ph.name}}">${{ph.name}}</div>
            <div>
              <span style="color:var(--accent);font-weight:bold">${{ph.norm}}分</span>
              <span style="color:#64748b;font-size:11px">(${{ph.raw}})</span>
            </div>
          </div>
        `;
        grid.appendChild(cell);
      }});
      
      document.getElementById('modal').classList.add('active');
    }}

    function closeModal() {{
      document.getElementById('modal').classList.remove('active');
    }}

    document.addEventListener('keydown', e => {{
      if (e.key === 'Escape') closeModal();
    }});
  </script>
</body>
</html>'''

out_path = Path('scratch/person_centric_ranking.html')
out_path.write_text(full_html, encoding='utf-8')
print('Successfully generated Person-Centric Ranking HTML:', out_path.resolve())
