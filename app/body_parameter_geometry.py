import numpy as np

def features(points):
 if not points:return {}
 z=np.array(points,dtype=float)
 return {'observed_points':z.tolist(),'point_count':len(z),'y_span_px':float(np.ptp(z[:,1])),'x_span_px':float(np.ptp(z[:,0])),'x_median_px':float(np.median(z[:,0])),'x_y_slope':float(np.polyfit(z[:,1],z[:,0],1)[0]) if np.ptp(z[:,1]) else None}

def extract(trace,part):
 if trace is None:return {},{}
 lo,hi={'waist':(.50,.85),'hips':(.88,1.18)}[part];by={s:{'external':[],'contact_candidate':[]} for s in ['left','right']};widths=[]
 for y,row in trace['rows'].items():
  if not lo<=(y-trace['shoulder'][1])/trace['T']<=hi:continue
  valid={}
  for side,e in row.get('sides',{}).items():
   if e['scan_clipped']:continue
   category='external' if e['external'] else 'contact_candidate';by[side][category].append(e['point']);valid[side]=(e['point'],category)
  if len(valid)==2:
   left,lc=valid['left'];right,rc=valid['right'];width=right[0]-left[0]
   if width>0:widths.append({'y':y,'width_px':float(width),'source':'external_pair' if lc==rc=='external' else 'contact_candidate_pair'})
 out={side+'_'+kind:{**features(points),'boundary_type':kind,'source':'cached_roi_segmentation'} for side,groups in by.items() for kind,points in groups.items() if points}
 summary={}
 for name in ['external_pair','contact_candidate_pair']:
  vals=[q['width_px'] for q in widths if q['source']==name]
  if vals:summary[name]={'matched_rows':len(vals),'width_median_px':float(np.median(vals)),'width_min_px':min(vals),'width_max_px':max(vals),'width_over_shoulder':float(np.median(vals)/trace['S']),'rows': [q for q in widths if q['source']==name]}
 return out,summary

def enclose(box,features,w,h):
 pts=[p for f in features.values() for p in f['observed_points']]
 if not pts:return box
 z=np.array(pts);margin=max(8,.06*float(np.ptp(z[:,0])));b=[max(0,int(np.floor(z[:,0].min()-margin))),max(0,int(np.floor(z[:,1].min()-margin))),min(w,int(np.ceil(z[:,0].max()+margin))+1),min(h,int(np.ceil(z[:,1].max()+margin))+1)]
 if box:b=[min(b[0],box[0]),min(b[1],box[1]),max(b[2],box[2]),max(b[3],box[3])]
 return b
