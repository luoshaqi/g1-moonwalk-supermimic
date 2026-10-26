import os,sys,json,math,re
from pathlib import Path
import numpy as np
import argparse
parser=argparse.ArgumentParser(description='CMU 90_32 ASF/AMC to Z-up metric joint positions (120 Hz).')
parser.add_argument('--source-dir',type=Path,required=True)
args=parser.parse_args()
R=args.source_dir

def rot(deg):
 x,y,z=np.deg2rad(deg); cx,sx=np.cos(x),np.sin(x); cy,sy=np.cos(y),np.sin(y); cz,sz=np.cos(z),np.sin(z)
 return np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])@np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])@np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
s=(R/'90.asf').read_text(); bones={}
for b in re.findall(r'begin\s+(.*?)\s+end',s.split(':bonedata')[1].split(':hierarchy')[0],re.S):
 d={x.split()[0]:x.split()[1:] for x in b.strip().splitlines() if x.split()}
 bones[d['name'][0]]={'direction':np.array(d['direction'],float),'length':float(d['length'][0]),'C':rot(list(map(float,d['axis'][:3]))),'dof':d.get('dof',[])}
parents={}
for line in s.split(':hierarchy')[1].splitlines():
 words=line.split()
 if len(words)>1:
  for child in words[1:]:parents[child]=words[0]
frames=[]
for line in (R/'90_32.amc').read_text().splitlines():
 words=line.split()
 if not words or words[0].startswith(('#',':')):continue
 if words[0].isdigit(): frames.append({})
 else: frames[-1][words[0]]=np.array(words[1:],float)
names=['root']+list(bones); indexes={n:i for i,n in enumerate(names)}
T=np.array([[1,0,0],[0,0,-1],[0,1,0.]])
points=[]
for frame in frames:
 ps={'root':frame['root'][:3]}; mats={'root':rot(frame['root'][3:])}
 def solve(n):
  if n in ps:return
  b=bones[n]; parent=parents[n]; solve(parent)
  angles=np.zeros(3)
  for key,value in zip(b['dof'],frame.get(n,[])): angles[{'rx':0,'ry':1,'rz':2}[key]]=value
  mats[n]=mats[parent]@b['C']@rot(angles)@b['C'].T
  ps[n]=ps[parent]+b['length']*(mats[n]@b['direction'])
 for n in bones:solve(n)
 points.append(np.stack([T@ps[n] for n in names])*(.0254/.45))
points=np.array(points)
assert np.isfinite(points).all()
# A single rigid floor offset; preserve original motion and displacement.
points[:,:,2]-=points[:,:,2].min()
np.savez_compressed(R/'human_reference_xyz.npz',positions_m=points,joint_names=names,fps=120,parents=json.dumps(parents))
