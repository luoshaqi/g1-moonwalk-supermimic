from common import configure
configure()
import json,sys,time
from pathlib import Path
import numpy as np
import mujoco as mj
from scipy.optimize import least_squares
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation
from supermimic.data.motion_convert import load_g1_model,G1_JOINT_NAMES
import argparse
parser=argparse.ArgumentParser(description='Retarget CMU 90_32 xyz to G1 (experimental clip-specific IK).')
parser.add_argument('--input',type=Path,required=True)
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
OUT=args.out; OUT.mkdir(parents=True,exist_ok=True)
z=np.load(args.input,allow_pickle=False)
names=z['joint_names'].tolist(); src=z['positions_m']; times=np.arange(0,(len(src)-1)/120+1e-9,.02)
p=CubicSpline(np.arange(len(src))/120,src)(times)
p=savgol_filter(p,9,3,axis=0)
idx={n:i for i,n in enumerate(names)}
def norm(v):return v/np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-9)
def rz(a):return Rotation.from_euler('z',a).as_matrix()
hip=p[:,idx['lhipjoint']]-p[:,idx['rhipjoint']]
forward=norm(np.cross(hip,np.array([0,0,1.])))
yaw=np.unwrap(np.arctan2(forward[:,1],forward[:,0])); initial_yaw=yaw[0]
p[:,:,:2]-=p[0,idx['root'],:2].copy();p=p@rz(-initial_yaw).T;yaw-=initial_yaw
get=lambda n:p[:,idx[n]]
m=load_g1_model(robot_variant='g1_no_plastic_hands');d=mj.MjData(m);d.qpos[:]=m.qpos0;mj.mj_forward(m,d)
bid={mj.mj_id2name(m,mj.mjtObj.mjOBJ_BODY,i):i for i in range(m.nbody)}
rest=d.xpos.copy(); root0=rest[bid['pelvis']]
leglen={}; footprints={}; footgeoms={}
for side in ['left','right']:
 h=rest[bid[side+'_hip_roll_link']];k=rest[bid[side+'_knee_link']];a=rest[bid[side+'_ankle_roll_link']]
 leglen[side]=(np.linalg.norm(k-h),np.linalg.norm(a-k))
 ids=[i for i in range(m.ngeom) if (mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,i) or '').startswith(side+'_foot') and m.geom_contype[i]>0]
 footgeoms[side]=np.array(ids)
 pts=[]
 for i in ids:
  mat=np.empty(9);mj.mju_quat2Mat(mat,m.geom_quat[i]); axis=mat.reshape(3,3)[:,2]
  for sign in [-1,1]:pts.append(m.geom_pos[i]+sign*axis*m.geom_size[i,1])
 footprints[side]=(np.array(pts),.01)
scale=sum(leglen['left'])/(np.linalg.norm(get('lfemur')-get('lhipjoint'),axis=1).mean()+np.linalg.norm(get('ltibia')-get('lfemur'),axis=1).mean())
rootxy=get('root')[:,:2]*scale
rootz=.765+scale*(get('root')[:,2]-get('root')[0,2])
foot_targets={};footR={}; knee_targets={}; elbows={}; wrists={}
torsoR=[]
for f in range(len(times)):
 y=norm(get('lclavicle')[f]-get('rclavicle')[f]);zup=norm(get('thorax')[f]-get('root')[f]); x=norm(np.cross(y,zup)); y=norm(np.cross(zup,x));torsoR.append(np.stack([x,y,zup],axis=1))
torsoR=np.array(torsoR)
for side,s in [('left','l'),('right','r')]:
 upper=get(s+'femur')-get(s+'hipjoint');lower=get(s+'tibia')-get(s+'femur')
 localhip=rest[bid[side+'_hip_roll_link']]-root0
 anchors=np.array([rz(yaw[f])@localhip for f in range(len(times))])+np.c_[rootxy,rootz]
 knees=anchors+norm(upper)*leglen[side][0];ankles=knees+norm(lower)*leglen[side][1]
 fd=get(s+'toes')-get(s+'tibia'); rawpitch=np.arctan2(-fd[:,2],np.linalg.norm(fd[:,:2],axis=1))
 low=get(s+'tibia')[:,2] <= np.quantile(get(s+'tibia')[:,2],.25)
 baseline=float(np.median(rawpitch[low]));pitch=np.clip(savgol_filter(rawpitch-baseline,13,3),-.16,.95)
 fyaw=np.arctan2(fd[:,1],fd[:,0]); rel=np.arctan2(np.sin(fyaw-yaw),np.cos(fyaw-yaw)); rel=savgol_filter(np.clip(rel,-.35,.35),21,2); fyaw=yaw+rel
 rots=Rotation.from_euler('ZYX',np.c_[fyaw,pitch,np.zeros(len(pitch))]).as_matrix()
 for f in range(len(times)):
  pts,rad=footprints[side];ankles[f,2]=-np.min((pts@rots[f].T)[:,2])+rad+.001
 foot_targets[side]=ankles;footR[side]=rots;knee_targets[side]=knees
 # Scale each arm segment to the robot; preserve source elbow/wrist directions.
 shoff=rest[bid[side+'_shoulder_roll_link']]-rest[bid['torso_link']]
 shoulder=np.array([torsoR[f]@shoff for f in range(len(times))])+np.c_[rootxy,rootz]+np.array([0,0,.044])
 upperlen=np.linalg.norm(rest[bid[side+'_elbow_link']]-rest[bid[side+'_shoulder_roll_link']])
 lowerlen=np.linalg.norm(rest[bid[side+'_wrist_yaw_link']]-rest[bid[side+'_elbow_link']])
 elbows[side]=shoulder+norm(get(s+'humerus')-get(s+'clavicle'))*upperlen
 wrists[side]=elbows[side]+norm(get(s+'radius')-get(s+'humerus'))*lowerlen
bounds=np.array(m.jnt_range[1:]);lo=np.r_[.55,bounds[:,0]+1e-5];hi=np.r_[.88,bounds[:,1]-1e-5]
seed=np.r_[rootz[0],np.zeros(29)]
for s in ['left','right']:
 seed[1+G1_JOINT_NAMES.index(s+'_hip_pitch_joint')]=-.15
 seed[1+G1_JOINT_NAMES.index(s+'_knee_joint')]=.3
 seed[1+G1_JOINT_NAMES.index(s+'_ankle_pitch_joint')]=-.15
seed=np.clip(seed,lo+1e-4,hi-1e-4)
qs=[];costs=[];ankle_errors=[];pre_floor=[];nfevs=[];started=time.time()
for f,t in enumerate(times):
 prev=seed.copy(); rootquat=Rotation.from_euler('z',yaw[f]).as_quat()[[3,0,1,2]]
 def apply(v):
  d.qpos[:3]=[rootxy[f,0],rootxy[f,1],v[0]];d.qpos[3:7]=rootquat;d.qpos[7:]=v[1:];mj.mj_kinematics(m,d)
 def floors(side):
  ids=footgeoms[side];return d.geom_xpos[ids,2]-np.abs(d.geom_xmat[ids].reshape(-1,3,3)[:,2,2])*m.geom_size[ids,1]-m.geom_size[ids,0]
 def residual(v):
  apply(v);parts=[]
  for side in ['left','right']:
   a=bid[side+'_ankle_roll_link'];k=bid[side+'_knee_link'];e=bid[side+'_elbow_link'];w=bid[side+'_wrist_yaw_link']
   parts.extend([(d.xpos[a]-foot_targets[side][f])*np.array([6,6,3]),(d.xpos[k]-knee_targets[side][f])*1.8,(d.xpos[e]-elbows[side][f])*1.8,(d.xpos[w]-wrists[side][f])*2.2])
   parts.append(Rotation.from_matrix(footR[side][f].T@d.xmat[a].reshape(3,3)).as_rotvec()*1.2)
   fl=floors(side)
   parts.append(np.array([(fl.min()-.001)*20]))
   parts.append(np.minimum(fl,0)*20)
  parts.append(Rotation.from_matrix(torsoR[f].T@d.xmat[bid['torso_link']].reshape(3,3)).as_rotvec()*.6)
  parts.append(np.array([(v[0]-rootz[f])*.8]))
  parts.append((v[1:]-prev[1:])*.08)
  reg=np.ones(29)*.018
  for j,n in enumerate(G1_JOINT_NAMES):
   if 'wrist' in n:reg[j]=.15
  parts.append(v[1:]*reg)
  return np.concatenate(parts)
 delta=np.r_[.012,np.ones(29)*.12]
 for j,n in enumerate(G1_JOINT_NAMES):
  if 'hip_yaw' in n or 'hip_roll' in n:delta[j+1]=.08
 blo,bhi=(lo,hi) if f==0 else (np.maximum(lo,prev-delta),np.minimum(hi,prev+delta))
 sol=least_squares(residual,seed,bounds=(blo,bhi),max_nfev=100 if f==0 else 30,ftol=2e-5,xtol=2e-5,gtol=2e-5,diff_step=1e-4)
 seed=sol.x;apply(seed)
 pre_floor.append([float(floors(s).min()) for s in ['left','right']])
 # Exact shared vertical floor correction, preserving the relative foot heights.
 shift=.001-min(x for x in pre_floor[-1]);d.qpos[2]+=shift;mj.mj_kinematics(m,d)
 qs.append(d.qpos.copy()); costs.append(sol.cost);nfevs.append(sol.nfev)
 ankle_errors.append([np.linalg.norm(d.xpos[bid[s+'_ankle_roll_link']]-foot_targets[s][f]) for s in ['left','right']])
 if f%50==0:print('frame',f,'/',len(times),'cost',sol.cost,'elapsed',round(time.time()-started,1),flush=True)
q=np.array(qs)
np.savez_compressed(OUT/'g1_moonwalk_reference.npz',qpos=q,fps=50,joint_names=G1_JOINT_NAMES,source_times=times)
np.savetxt(OUT/'g1_moonwalk_50hz.csv',np.c_[q[:,:3],q[:,[4,5,6,3]],q[:,7:]],delimiter=',')
report={'source':'CMU 90_32, user-approved reference','robot':'g1_no_plastic_hands, 29 DoF','method':'Bound-constrained per-frame whole-body IK; limb-length scaling; foot pitch calibrated from low-ankle frames; both-foot floor constraints; temporal regularization; fixed root XY path scaled to leg lengths','source_fps':120,'output_fps':50,'frames':len(q),'duration_seconds':float(times[-1]),'horizontal_scale':float(scale),'joint_limit_violation_rad':float(max(0,np.max(bounds[:,0]-q[:,7:]),np.max(q[:,7:]-bounds[:,1]))),'max_joint_velocity_rad_s':float(np.max(np.abs(np.diff(q[:,7:],axis=0)*50))),'mean_ankle_target_error_m':float(np.mean(ankle_errors)),'max_ankle_target_error_m':float(np.max(ankle_errors)),'pre_floor_min_m':float(np.min(pre_floor)),'floor_correction_max_m':float(np.max(np.abs(.001-np.min(pre_floor,axis=1)))),'solver_max_nfev':int(max(nfevs)),'elapsed_seconds':time.time()-started,'foot_yaw_relative_heading_limit_rad':.35,'joint_speed_bound_rad_s':6,'hip_yaw_roll_speed_bound_rad_s':4,'training_started':False,'dynamic_balance_verified':False}
(OUT/'retarget_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2),flush=True)

from supermimic.data.motion_convert import convert_csv_to_npz
convert_csv_to_npz(OUT/'g1_moonwalk_50hz.csv',OUT/'cmu90_32_moonwalk.npz',50,50,model=m)
