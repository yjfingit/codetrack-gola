#!/usr/bin/env python3
"""Detach the authorized causal20 workflow with durable launch/exit records."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def stamp(): return datetime.now(timezone.utc).isoformat()


def write_status(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-output',type=Path,required=True)
    parser.add_argument('--detach',action='store_true')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    run=args.run_output.resolve(); control=Path(str(run)+'_control')
    if args.resume:
        args.resume=args.resume.resolve()
        if not all((args.resume/name).is_file() for name in ('model.safetensors','state.pth')):
            raise RuntimeError('resume checkpoint incomplete')
        control=Path(str(run)+'_resume_control')
    if args.detach:
        if (run.exists() and not args.resume) or control.exists():
            raise RuntimeError('run/control already exists')
        control.mkdir(parents=True)
        sources=[]
        for directory in ('trackit','codetrack','tools','scripts','config'):
            sources.extend(p for p in (ROOT/directory).rglob('*')
                           if p.is_file() and p.suffix in ('.py','.sh','.yaml','.yml'))
        sources.extend(ROOT/name for name in ('main.py','consts.yaml','requirements.txt'))
        with tarfile.open(control/'source_snapshot.tar.gz','w:gz') as archive:
            for file in sources: archive.add(file,arcname=str(file.relative_to(ROOT)))
        launch={'run_output':str(run),'control':str(control),'created_utc':stamp(),
                'gpus':[3,4],'resume':str(args.resume) if args.resume else None,
                'command':['bash','scripts/train_causal20.sh'],
                'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sources}}
        (control/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
        with (control/'supervisor.log').open('ab',buffering=0) as log:
            command=[sys.executable,str(Path(__file__).resolve()),'--run-output',str(run)]
            if args.resume: command.extend(['--resume',str(args.resume)])
            child=subprocess.Popen(command,cwd=ROOT,
                                   stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True)
        (control/'supervisor.pid').write_text(str(child.pid)+'\n')
        print(json.dumps({'pid':child.pid,'run_output':str(run),'control':str(control)}))
        return
    status={'state':'RUNNING','started_utc':stamp(),'supervisor_pid':os.getpid(),
            'run_output':str(run),'training_log':str(run)+'.log',
            'resume':str(args.resume) if args.resume else None}
    environment=os.environ.copy(); environment['RUN_OUTPUT']=str(run)
    if args.resume: environment['CAUSAL_RESUME']=str(args.resume)
    environment['PYTHONUNBUFFERED']='1'
    child=subprocess.Popen(['bash','scripts/train_causal20.sh'],cwd=ROOT,env=environment)
    status['workflow_pid']=child.pid
    write_status(control/'status.json',status)
    code=child.wait()
    status.update(state='COMPLETED' if code==0 else 'FAILED',exit_code=code,finished_utc=stamp())
    write_status(control/'status.json',status)
    sys.exit(code)


if __name__=='__main__': main()
