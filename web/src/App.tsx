import { useEffect, useState } from 'react'
import { Aperture, ArrowLeft, ImagePlus, Heart, X } from 'lucide-react'
import './Simple.css'

type Dim = { key: string; status: string; score: number | null; evidence: string; limitations: string[] }
type VisualBenchmark = { benchmark_score: number; raw_score: number; tier: string; model?: string; guidance?: string; three_parts_ratio?: string; eye_spacing_ratio?: number; face_shape_label?: string; nose_ratio_label?: string; nose_interocular_ratio?: number; face_lw_ratio?: number; has_catchlight?: boolean; is_over_smoothed?: boolean; skin_texture_label?: string }
type PoseMetrics = { scene: string; scene_display: string; head_ratio?: number | null; shoulder_hip_ratio?: number | null; torso_leg_ratio?: number | null; shoulder_tilt_deg?: number | null; shoulder_balance?: string | null }
type PhotoMetrics = { benchmark_score: number; exposure_label: string; face_exposure_label?: string | null; light_softness?: string | null; clarity_label: string; composition_note: string; headroom_label?: string | null; dof_label: string; catchlight_label?: string | null }
type Assessment = { visual_benchmark?: VisualBenchmark; pose_metrics?: PoseMetrics; photo_metrics?: PhotoMetrics; rubric_version?: string; assessed_count?: number; dimension_count?: number; status: string; total: number | null; summary: string; dimensions: Dim[]; strengths: string[]; limitations: string[]; shooting_tips: string[]; editing_tips: string[]; model?: string }
type Box = { x: number; y: number; width: number; height: number }
type Report = { id: string; created_at: string; filename: string; faces: Box[]; target: Box | null; detector_error: string | null; face: Assessment; photography: Assessment; body: Assessment; quality: Record<string, number | string>; provenance: { category: string; platform?: string | null; manifest_status: string; signature_status: string; asset_binding_status: string; signer_trust: string; evidence: { title: string; type: string; level: string; value?: string; claim_generator?: string; validation_issues?: string[] }[]; limitations: string[] }; usage: { input_tokens?: number; output_tokens?: number; elapsed_seconds: number }[] }
type Asset = { voted: boolean; vote_count: number; photo_tags: {framing:string|null;style:string|null}; user_source: string; source_display: {label:string;basis:string|null;platform:string|null;platform_basis:string|null}; size: number; mime: string; id: string; project_id: string | null; filename: string; width: number; height: number; created_at: string; status: string; retained: boolean; adult_confirmed: boolean; external_consent: boolean; external_scope: string; preview_url: string; crop_url: string | null; target: Box | null; report: Report }
type Job = { id: string; asset_id: string; status: string; error: string | null }
type Rule = { key: string; name: string; weight: number; observe: string }
type Status = { model: string; reasoning_effort: string; provider_available: boolean; max_upload_mb: number; usage: { calls: number; limit: number | null; tokens: number }; rules: Record<string, Rule[]>; anchors: Record<string, Record<string, string>>; rubric_version: string }
let token = ''
async function api<T>(path: string, method = 'GET', data?: unknown): Promise<T> {
  const form = data instanceof FormData
  const response = await fetch(`/api${path}`, { method, headers: { ...(method !== 'GET' ? { 'x-portrait-token': token } : {}), ...(!form && data ? { 'Content-Type': 'application/json' } : {}) }, body: data ? (form ? data : JSON.stringify(data)) : undefined })
  const body = await response.json()
  if (!response.ok) throw new Error(typeof body.detail === 'string' ? body.detail : '输入内容有误，请检查后重试')
  return body as T
}

const shortNames:Record<string,string>={eyes:'眉眼',nose_lips:'鼻唇',harmony:'协调',face_shape:'脸型',skin:'皮肤',light:'光线',clarity:'清晰',composition:'构图',color:'色彩',background:'背景',proportion:'比例',chest:'肩胸',waist:'腰腹',hips_legs:'臀腿',hips:'臀胯',legs:'腿部'}
const score=(value?:number|null,decimal=true)=>value==null?'—':decimal?value.toFixed(1):value
const missingScore=(r:Assessment)=>r.status==='outdated'?'待更新':r.status==='not_enabled'?'未评分':['failed','interrupted','cancelled'].includes(r.status)?'需重试':'不可评'
const sourceName=(a:Asset)=>a.source_display?.label||'无法确认'
function Rating({title,result,rules}:{title:string;result:Assessment;rules:Rule[]}){
 const [showEvidence, setShowEvidence] = useState(false)
 const vb = result.visual_benchmark
 const pm = result.pose_metrics
 const ph = result.photo_metrics

 return <section className="result-block">
  <div className="result-title"><h2>{title}</h2><strong>{score(result.total,true)}<small>/100</small></strong></div>
  
  {vb && title==='颜值' && <div className="diag-dashboard">
    <div className="diag-tag diag-primary"><span>骨相基准</span><b>{vb.benchmark_score.toFixed(1)} 分 · {vb.tier}</b></div>
    {vb.face_shape_label && <div className="diag-tag"><span>脸型形态</span><b>{vb.face_shape_label}</b></div>}
    {vb.three_parts_ratio && <div className="diag-tag"><span>三庭比例</span><b>{vb.three_parts_ratio}</b></div>}
    {vb.nose_interocular_ratio && <div className="diag-tag"><span>鼻翼眼距比</span><b>{vb.nose_interocular_ratio}</b></div>}
    <div className={`diag-tag ${vb.has_catchlight ? 'diag-success' : 'diag-muted'}`}><span>瞳孔眼神光</span><b>{vb.has_catchlight ? '清澈高光斑' : '平光无光斑'}</b></div>
    <div className={`diag-tag ${vb.is_over_smoothed ? 'diag-warn' : 'diag-success'}`}><span>微观肌理</span><b>{vb.is_over_smoothed ? '过度磨皮扣分' : '自然真实毛孔'}</b></div>
  </div>}

  {pm && title==='身材' && <div className="diag-dashboard">
    <div className="diag-tag diag-primary"><span>取景构型</span><b>{pm.scene_display}</b></div>
    <div className="diag-tag diag-success"><span>造型动态</span><b>{pm.shoulder_balance || '自然体态'}（侧倾 {pm.shoulder_tilt_deg ?? 0}°）</b></div>
    {pm.shoulder_hip_ratio && <div className="diag-tag"><span>肩/胯比例</span><b>{pm.shoulder_hip_ratio}</b></div>}
    {pm.head_ratio && <div className="diag-tag"><span>全身头身比</span><b>{pm.head_ratio} 头身</b></div>}
  </div>}

  {ph && title==='摄影' && <div className="diag-dashboard">
    <div className="diag-tag diag-primary"><span>技术基准</span><b>{ph.benchmark_score.toFixed(1)} 分</b></div>
    <div className="diag-tag diag-success"><span>对焦清晰</span><b>{ph.clarity_label}</b></div>
    <div className={`diag-tag ${ph.face_exposure_label?.includes('通透') ? 'diag-success' : 'diag-warn'}`}><span>面部测光</span><b>{ph.face_exposure_label || ph.exposure_label}</b></div>
    {ph.light_softness && <div className="diag-tag"><span>光线质感</span><b>{ph.light_softness}</b></div>}
    {ph.headroom_label && <div className={`diag-tag ${ph.headroom_label.includes('贴边') ? 'diag-warn' : 'diag-success'}`}><span>头顶留白</span><b>{ph.headroom_label}</b></div>}
    <div className="diag-tag"><span>瞳孔焦点</span><b>{ph.catchlight_label || '自然平光'}</b></div>
    <div className="diag-tag"><span>景深分离</span><b>{ph.dof_label}</b></div>
  </div>}

  {result.total==null?<p className="quiet">{result.status==='outdated'?'评分规则已更新，请重新分析':result.status==='not_enabled'?'尚未评分':result.status==='failed'?'分析失败，请重试':'可评项目不足，暂无评分'}</p>:result.summary&&<p className="result-summary">{result.summary}</p>}

  <div className={`rating-parts ${title==='身材'?'body-parts':''}`}>{rules.map(r=>{const dim=result.dimensions.find(d=>d.key===r.key);return <div key={r.key}><span>{shortNames[r.key]||r.name}</span><b>{score(dim?.score,true)}<small>/10</small></b></div>})}</div>

  {result.dimensions.length>0&&<div className="evidence-section">
    <button className="evidence-toggle-btn" onClick={()=>setShowEvidence(!showEvidence)}>{showEvidence?'收起各细项实测依据 ▲':'查看各细项客观实测依据详情 ▼'}</button>
    {showEvidence&&<div className="evidence-list">{rules.map(r=>{const dim=result.dimensions.find(d=>d.key===r.key);if(!dim)return null;return <div key={r.key} className="evidence-item"><div className="evidence-header"><span className="dim-name">{r.name}（{shortNames[r.key]||r.name}）</span><b className="dim-score">{score(dim.score,true)} / 10</b></div><p className="dim-evidence">{dim.evidence||'暂无实测记录'}</p>{dim.limitations&&dim.limitations.length>0&&<p className="dim-limitation">说明：{dim.limitations.join('；')}</p>}</div>})}</div>}
  </div>}

  {(title==='摄影'||title==='身材')&&result.shooting_tips.length>0&&<div className="one-tip"><span>实操与拍摄建议</span>{result.shooting_tips.map((t,idx)=><p key={idx} className="tip-item">💡 {t}</p>)}</div>}
 </section>
}
export default function App(){
 const [assets,setAssets]=useState<Asset[]>([]),[jobs,setJobs]=useState<Job[]>([]),[status,setStatus]=useState<Status|null>(null)
 const [voting,setVoting]=useState<string[]>([])
 const [selected,setSelected]=useState<string|null>(null),[sort,setSort]=useState<'face'|'photography'|'body'|'newest'>('face')
 const [error,setError]=useState(''),[busy,setBusy]=useState(''),[uploadOpen,setUploadOpen]=useState(false),[consentOpen,setConsentOpen]=useState(false)
 const [history,setHistory]=useState<Report[]|null>(null)
 const [more,setMore]=useState(false)
 const asset=assets.find(a=>a.id===selected)
 const activeJobs=jobs.filter(j=>['queued','running'].includes(j.status))
 const currentJob=activeJobs.find(j=>j.asset_id===selected)
 const sorted=[...assets].sort((a,b)=>sort==='newest'?b.created_at.localeCompare(a.created_at):(b.report[sort].total??-1)-(a.report[sort].total??-1)||a.created_at.localeCompare(b.created_at)||a.id.localeCompare(b.id))
 async function refresh(){const [a,j,s]=await Promise.all([api<Asset[]>('/assets'),api<Job[]>('/jobs'),api<Status>('/status')]);setAssets(a);setJobs(j);setStatus(s)}
 useEffect(()=>{api<{token:string}>('/session').then(s=>{token=s.token;return refresh()}).catch(e=>setError(e.message))},[])
 useEffect(()=>{if(!activeJobs.length)return;const id=setInterval(()=>refresh().catch(e=>setError(e.message)),3000);return()=>clearInterval(id)},[activeJobs.length])
 async function action(fn:()=>Promise<void>){setError('');try{await fn()}catch(e){setError(e instanceof Error?e.message:'操作失败，请重试')}}
 function select(id:string|null){setSelected(id);setHistory(null);setMore(false);setError('');window.scrollTo(0,0)}
 async function analyze(a:Asset){await api(`/assets/${a.id}/analyze`,'POST',{modules:['face','body','photography'],force:true});await refresh()}
 async function castVote(a:Asset){
  if(voting.includes(a.id))return
  setVoting(v=>[...v,a.id])
  await action(async()=>{const updated=await api<Asset>(`/assets/${a.id}/vote`,'PUT',{voted:!a.voted});setAssets(list=>list.map(x=>x.id===a.id?updated:x))})
  setVoting(v=>v.filter(id=>id!==a.id))
 }
 async function upload(files:FileList|File[]){
  if(busy)return
  setBusy('准备上传…');setError('');const failures:string[]=[]
  for(const [i,file] of Array.from(files).entries()){
   setBusy(`上传 ${i+1}/${files.length}`)
   let imported=false
   try{const form=new FormData();form.append('file',file);form.append('rights_confirmed','true');form.append('adult_confirmed','true')
    const a=await api<Asset>('/assets','POST',form);imported=true
    await api(`/assets/${a.id}/consent`,'POST',{adult_confirmed:true,external_consent:true,external_scope:'full_image'})
    await api(`/assets/${a.id}/analyze`,'POST',{modules:['face','body','photography']})
   }catch{failures.push(`第 ${i+1} 张${imported?'已上传，评分未启动，请点击照片重试':'上传失败，请检查格式和大小'}`)}
  }
  await action(refresh);setBusy('');setUploadOpen(false);if(failures.length)setError(failures.join('；'))
 }
 function requestAnalysis(){if(!asset)return;if(!asset.external_consent||asset.external_scope!=='full_image'||!asset.adult_confirmed){setConsentOpen(true)}else action(()=>analyze(asset))}
 const provenance=asset?.report.provenance
 const exif=Object.fromEntries((provenance?.evidence||[]).filter(e=>e.type==='exif').map(e=>[e.title,e.value]))
 const source=asset?sourceName(asset):''
 const sourceActions=(provenance?.evidence||[]).filter(e=>e.type==='c2pa_action').map(e=>e.value||'').join(' ')
 const generatedOnly=['AI 图片','AI 生成'].includes(source)||(source==='含 AI 生成标记'&&!/compositeWithTrainedAlgorithmicMedia|algorithmicallyEnhanced|digitalCapture/.test(sourceActions))
 const cameraOnly=['相机拍摄','含拍摄标记'].includes(source)
 const platform=!cameraOnly?asset?.source_display.platform:null
 const sourceDetail=[source==='AI 图片'?'AI 生成':source==='无法确认'?'来源未知':source,platform||(cameraOnly?exif.Model:null)].filter(Boolean).join(' · ')
 const cameraFields: [string,string|undefined][] = generatedOnly?[]:[['品牌',exif.Make],['相机 / 手机',exif.Model],['拍摄日期',exif.DateTimeOriginal],['镜头',exif.LensModel],['光圈',exif.FNumber],['曝光时间',exif.ExposureTime],['ISO',exif.ISOSpeedRatings||exif.PhotographicSensitivity],['焦距',exif.FocalLength]]
 const archive: [string,string][] = asset?[
  ['图片来源',sourceDetail],['尺寸',`${asset.width} × ${asset.height} px`],['大小',`${(asset.size/1024/1024).toFixed(2)} MB`],
  ['文件名',asset.filename],['格式',asset.mime.split('/')[1].toUpperCase()],['画幅',asset.width>asset.height?'横图':asset.width===asset.height?'方图':'竖图'],['上传时间',new Date(asset.created_at).toLocaleString()],
  ['识别依据',asset.source_display.platform_basis==='文件标记'?'内嵌内容凭据':asset.source_display.basis||'未发现来源标记'],
  ...cameraFields.filter((field):field is [string,string]=>!!field[1]),
  ...(exif.Software?[[ '编辑软件',exif.Software] as [string,string]]:[])
 ]:[]
 const keyInfo=archive.filter(([key])=>['图片来源','尺寸','大小','格式'].includes(key))

 const extraInfo=archive.filter(([key])=>!keyInfo.some(([shown])=>shown===key))

 return <div className="simple-app">{!asset&&<header className="simple-header"><button className="simple-brand" aria-label="颜选 BeautyPick，返回照片列表" onClick={()=>select(null)}><Aperture size={27}/><span className="brand-name">颜选<small>BeautyPick</small></span></button><button className="primary" onClick={()=>{setUploadOpen(true);setError('')}}><ImagePlus size={17}/>上传照片</button></header>}
 {error&&<div className="error-banner" role="alert">{error}<button aria-label="关闭提示" onClick={()=>setError('')}><X size={17}/></button></div>}
  {!asset?<><div className="library-heading"><div><h1>我的照片</h1><span>{assets.length} 张</span><span className="sort-hint">· 按{sort==='face'?'颜值':sort==='body'?'身材':sort==='photography'?'摄影':'上传时间'}从高到低排序</span></div><div className="sort-tabs" aria-label="照片排序">{[['face','颜值最高'],['body','身材最佳'],['photography','摄影最佳'],['newest','上传时间']].map(([key,title])=><button key={key} aria-pressed={sort===key} className={sort===key?'chosen':''} onClick={()=>setSort(key as typeof sort)}>{title}</button>)}</div></div>
  {!assets.length?<div className="start-empty"><Aperture size={56}/><h2>看看这张照片怎么样</h2><p>颜值 · 摄影 · 图片来源</p><button className="primary" onClick={()=>setUploadOpen(true)}>上传照片</button></div>:<div className="simple-grid">{sorted.map((a,index)=>{const job=activeJobs.find(j=>j.asset_id===a.id);const lastJob=jobs.find(j=>j.asset_id===a.id);const failed=!job&&(a.report.face.total==null||a.report.photography.total==null)&&lastJob&&['failed','interrupted'].includes(lastJob.status);const rank=sort==='newest'?null:index+1
   return <article className="simple-photo" key={a.id}><div className="simple-photo-image"><button className="photo-open" aria-label={`查看照片 ${assets.indexOf(a)+1}`} onClick={()=>select(a.id)}><img src={a.preview_url} alt="人像照片" loading="lazy"/></button><button className={`vote-button ${a.voted?'voted':''}`} aria-label={`${a.voted?'取消投票':'投票'} 照片 ${assets.indexOf(a)+1}`} aria-pressed={a.voted} disabled={voting.includes(a.id)} onClick={()=>castVote(a)}><Heart size={19} fill={a.voted?'currentColor':'none'}/><span>{a.vote_count}</span></button>{rank!=null&&<span className={`rank-badge rank-${rank<=3?rank:"other"}`} aria-label={`第 ${rank} 名`}><small>{sort==='face'?'颜值':sort==='body'?'身材':sort==='photography'?'摄影':'榜单'}</small><strong>{rank}</strong></span>}{job&&<span className="progress-badge">{job.status==='running'?'正在评分…':'等待评分…'}</span>}{failed&&<span className="progress-badge">评分未完成 · 点击重试</span>}</div><button className="simple-photo-info" aria-label={`查看评分 ${assets.indexOf(a)+1}`} onClick={()=>select(a.id)}><div className="photo-title" title={a.filename}>{a.filename}</div><div className="card-totals">{(['face','body','photography'] as const).map(m=><span className={sort===m?'score-active':''} key={m}>{{face:'颜值',photography:'摄影',body:'身材'}[m]}<b className={a.report[m].total==null?'score-empty':''}>{(a.report[m].total?.toFixed(1))??(missingScore(a.report[m]))}</b></span>)}</div><div className="card-tags"><span className="source-tag">{sourceName(a)}</span>{a.photo_tags.framing&&<span className="framing-tag">{a.photo_tags.framing}</span>}{a.photo_tags.style&&<span className="style-tag">{a.photo_tags.style}</span>}</div></button></article>})}</div>}</>:
 <><div className="detail-toolbar"><button className="back-button" onClick={()=>select(null)}><ArrowLeft size={17}/>全部照片</button><div className="detail-actions"><button className="primary" disabled={!!currentJob} onClick={()=>requestAnalysis()}>{currentJob?'分析中…':'重新分析'}</button><div className="more-container"><button className="more-button" aria-expanded={more} onClick={()=>setMore(!more)}>更多 ···</button>{more&&<div className="more-menu"><a href={`/api/assets/${asset.id}/export.html`}>导出报告</a><button onClick={()=>action(async()=>{setHistory(await api<Report[]>(`/assets/${asset.id}/history`));setMore(false)})}>历史评分</button><button className="delete-button" onClick={()=>{if(confirm('删除这张照片和评分记录？删除后无法恢复。'))action(async()=>{await api(`/assets/${asset.id}`,'DELETE');select(null);await refresh()})}}>删除照片</button></div>}</div></div></div><div className="simple-detail"><div className="detail-photo"><div className="face-stage" style={{width:`min(100%, calc((100dvh - 124px) * ${asset.width/asset.height}))`}}><img src={asset.preview_url} alt="照片"/>{asset.report.faces.length>1&&asset.report.faces.map((box,i)=><button className="face-choice" key={i} aria-label={`选择人脸 ${i+1}`} disabled={!!currentJob} style={{left:`${box.x*100}%`,top:`${box.y*100}%`,width:`${box.width*100}%`,height:`${box.height*100}%`}} onClick={()=>action(async()=>{await api(`/assets/${asset.id}/target`,'POST',box);await refresh()})}>{i+1}</button>)}</div>{asset.report.faces.length>1&&<div className="face-picker"><span>选择要评分的人脸</span>{asset.report.faces.map((box,i)=><button key={i} disabled={!!currentJob} onClick={()=>action(async()=>{await api(`/assets/${asset.id}/target`,'POST',box);await refresh()})}>人脸 {i+1}</button>)}</div>}</div><div className="detail-results">{currentJob&&<div className="scoring-status">{currentJob.status==='running'?'正在评分…':'等待评分…'}<button onClick={()=>action(async()=>{await api(`/jobs/${currentJob.id}/cancel`,'POST');await refresh()})}>取消</button></div>}
 <Rating title="颜值" result={asset.report.face} rules={status?.rules.face||[]}/>
 <Rating title="身材" result={asset.report.body} rules={status?.rules.body||[]}/>

 <Rating title="摄影" result={asset.report.photography} rules={status?.rules.photography||[]}/>
 <section className="result-block image-info"><div className="result-title"><h2>图片信息</h2></div><div className="archive-fields">{keyInfo.map(([k,v])=><div key={k}><span>{k}</span><b>{v}</b></div>)}</div><details className="archive-more" key={asset.id}><summary>查看详情</summary><div className="archive-fields">{extraInfo.map(([k,v])=><div key={k}><span>{k}</span><b>{v}</b></div>)}</div></details></section>{history&&<section className="result-block"><div className="result-title"><h2>历史评分</h2><button aria-label="关闭历史评分" onClick={()=>setHistory(null)}><X size={18}/></button></div>{history.map(h=><div className="history-item" key={h.id}><time>{new Date(h.created_at).toLocaleString()}</time><p>颜值 {score(h.face.total,true)} · 摄影 {score(h.photography.total,true)} · {h.body.rubric_version==='body-appearance-3.0'?'身材':'形体（旧版）'} {score(h.body.total,true)}</p><a href={`/api/assets/${asset.id}/export.html?report_id=${h.id}`}>查看报告</a></div>)}</section>}</div></div></>}
 {(uploadOpen||consentOpen)&&<div className="modal-backdrop"><section className="simple-modal" role="dialog" aria-modal="true" aria-label={uploadOpen?'上传照片':'开始评分'}><div className="modal-title"><h2>{uploadOpen?'上传照片':'开始评分'}</h2><button aria-label="关闭窗口" disabled={!!busy} onClick={()=>{setUploadOpen(false);setConsentOpen(false)}}><X size={20}/></button></div>{uploadOpen?<div className="simple-dropzone" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();upload(e.dataTransfer.files)}}><ImagePlus size={32}/><h3>{busy||'拖入照片，或点击上传'}</h3><label className="primary file-button">选择照片<input type="file" accept="image/jpeg,image/png,image/webp" multiple disabled={!!busy} onChange={e=>{if(e.target.files)upload(e.target.files)}}/></label><small>仅上传成年人照片</small><small>支持多张 · JPG / PNG / WebP · 每张 {status?.max_upload_mb||20} MB 以内</small></div>:<><p className="modal-notice">仅支持成年人照片</p><button className="primary" onClick={()=>action(async()=>{if(!asset)return;await api(`/assets/${asset.id}/consent`,'POST',{adult_confirmed:true,external_consent:true,external_scope:'full_image'});await analyze({...asset,adult_confirmed:true});setConsentOpen(false)})}>开始分析</button></>}</section></div>}
 </div>
}
