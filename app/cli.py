"""CLI client for the running local service; uses exactly the Web application's API."""
import argparse
import json
import sys
import time
from pathlib import Path
import httpx


def main():
    p=argparse.ArgumentParser(description='颜选 · BeautyPick · 本机人像评估命令行')
    p.add_argument('--port',type=int,default=8765)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('status');sub.add_parser('list')
    u=sub.add_parser('upload');u.add_argument('image',type=Path);u.add_argument('--rights-confirmed',action='store_true',required=True);u.add_argument('--adult-confirmed',action='store_true');u.add_argument('--project-id')
    a=sub.add_parser('analyze');a.add_argument('asset_id');a.add_argument('--modules',nargs='+',choices=['face','photography','body'],default=['face','photography','body']);a.add_argument('--allow-cloud',action='store_true',required=True);a.add_argument('--adult-confirmed',action='store_true');a.add_argument('--wait',action='store_true');a.add_argument('--suggestions-only',action='store_true');a.add_argument('--force',action='store_true')
    e=sub.add_parser('export');e.add_argument('asset_id');e.add_argument('output',type=Path);e.add_argument('--format',choices=['json','html'],default='json')
    d=sub.add_parser('delete');d.add_argument('asset_id');d.add_argument('--confirm',action='store_true',required=True)
    r=sub.add_parser('ranking');r.add_argument('project_id');r.add_argument('--module',choices=['face','photography','body'],default='face');r.add_argument('--preference-id')
    args=p.parse_args()
    try:
        with httpx.Client(base_url=f'http://127.0.0.1:{args.port}',timeout=120) as c:
            response=c.get('/api/session');response.raise_for_status();c.headers['x-portrait-token']=response.json()['token']
            if args.command in ('status','list'):response=c.get('/api/'+('assets' if args.command=='list' else 'status'))
            elif args.command=='upload':
                with args.image.open('rb') as f:
                    response=c.post('/api/assets',data={k:v for k,v in {'rights_confirmed':True,'adult_confirmed':args.adult_confirmed,'project_id':args.project_id}.items() if v is not None},files={'file':(args.image.name,f)})
            elif args.command=='analyze':
                assets=c.get('/api/assets');assets.raise_for_status()
                asset=next((a for a in assets.json() if a['id']==args.asset_id),None)
                if not asset:raise ValueError('图片不存在或已过期')
                adult=args.adult_confirmed or asset['adult_confirmed']
                if any(m in args.modules for m in ('face','body')) and not adult:raise ValueError('需要成年确认；也可只分析摄影')
                response=c.post(f'/api/assets/{args.asset_id}/consent',json={'adult_confirmed':adult,'external_consent':True});response.raise_for_status()
                response=c.post(f'/api/assets/{args.asset_id}/analyze',json={'modules':args.modules,'body_scoring':not args.suggestions_only,'force':args.force});response.raise_for_status()
                if args.wait:
                    jid=response.json()['job_id'];deadline=time.monotonic()+1800
                    while time.monotonic()<deadline:
                        jobs=c.get('/api/jobs');jobs.raise_for_status();job=next((j for j in jobs.json() if j['id']==jid),None)
                        if not job:raise ValueError('任务已删除')
                        if job['status'] not in ('queued','running'):break
                        time.sleep(2)
                    else:raise ValueError('等待超时；可在界面查看或取消任务')
                    if job['status'] in ('failed','cancelled','interrupted'):
                        raise ValueError(job.get('error') or '任务'+job['status']+'；未返回旧报告冒充本次结果')
                    response=c.get(f'/api/assets/{args.asset_id}/export.json')
            elif args.command=='export':
                response=c.get(f'/api/assets/{args.asset_id}/export.{args.format}');response.raise_for_status()
                with args.output.open('xb') as f:f.write(response.content)
                print(json.dumps({'saved':str(args.output.resolve())}));return
            elif args.command=='delete':response=c.delete(f'/api/assets/{args.asset_id}')
            else:response=c.get(f'/api/projects/{args.project_id}/ranking',params={k:v for k,v in {'module':args.module,'preference_id':args.preference_id}.items() if v})
            response.raise_for_status();print(json.dumps(response.json(),ensure_ascii=False,indent=2))
    except (httpx.HTTPError,OSError,ValueError) as exc:
        if isinstance(exc,httpx.HTTPStatusError):
            try:message=exc.response.json().get('detail','请求失败')
            except ValueError:message='请求失败'
        else:message=str(exc)
        print(json.dumps({'error':message},ensure_ascii=False),file=sys.stderr);sys.exit(1)

if __name__=='__main__':main()
