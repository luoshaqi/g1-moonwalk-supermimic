"""Compose synchronous baseline / fine-tuned 1280x720 videos into a phone-sized comparison."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
from PIL import Image, ImageDraw, ImageFont

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--finetuned',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--font',type=Path,required=True,help='Chinese font, e.g. NotoSansCJK-Regular.ttc')
    a=p.parse_args()
    for path in [a.baseline,a.finetuned]:
        meta=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-of','json',str(path)]))
        stream=next(s for s in meta['streams'] if s['codec_type']=='video')
        if (stream['width'],stream['height']) != (1280,720):
            p.error('Crop is designed for the included 1280x720 follow-camera renders.')
    report=json.loads(a.report.read_text());n=report['selected_finetune_iteration']
    b=report['baseline'];f=report['finetuned_checkpoints'][str(n)]
    minutes=report['finetune_summary']['elapsed_seconds']/60
    im=Image.new('RGB',(1080,1920),'#10151c');d=ImageDraw.Draw(im)
    def txt(x,y,s,size,color='#f1f5f9'):
        d.text((x,y),s,font=ImageFont.truetype(str(a.font),size),fill=color)
    txt(64,60,'G1 机器人 · 太空步',60)
    txt(64,140,'预训练模型 → 太空步专项微调',34,'#aebdce')
    d.ellipse((66,214,90,238),fill='#e5e7eb');txt(108,208,'实体：实际动作',29)
    d.ellipse((510,214,534,238),fill='#20dbe2');txt(552,208,'青色半透明：参考动作',29)
    txt(64,276,'微调前',43);txt(680,289,'E009 预训练模型',28,'#aebdce')
    txt(64,968,f"回报 {b['full_clip_normalized_return']:.3f}",32)
    txt(470,968,f"位置误差 {b['root_position_error_m']*100:.2f} cm",32,'#aebdce')
    d.line((64,1021,1016,1021),fill='#334155',width=2)
    txt(64,1040,'微调后',43,'#69e1bb');txt(625,1053,f'{n:,} 次 · 约 {minutes:.0f} 分钟',28,'#aebdce')
    txt(64,1732,f"回报 {f['full_clip_normalized_return']:.3f}",32,'#69e1bb')
    txt(470,1732,f"位置误差 {f['root_position_error_m']*100:.2f} cm",32,'#69e1bb')
    txt(64,1800,'同一动作 · 同步对比 · 原速 1×',32)
    txt(64,1852,'仿真训练动作评估｜位置误差为根部位置误差',25,'#aebdce')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory() as t:
        layout=Path(t)/'layout.png';im.save(layout)
        graph='[1:v]crop=580:336:350:214,scale=1080:626:flags=lanczos,setsar=1,setpts=PTS-STARTPTS[a];[2:v]crop=580:336:350:214,scale=1080:626:flags=lanczos,setsar=1,setpts=PTS-STARTPTS[b];[0:v][a]overlay=0:330:shortest=1[x];[x][b]overlay=0:1096:shortest=1[out]'
        subprocess.run(['ffmpeg','-v','error','-n','-loop','1','-i',str(layout),'-i',str(a.baseline),'-i',str(a.finetuned),'-filter_complex',graph,'-map','[out]','-r','25','-c:v','libx264','-crf','18','-preset','medium','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output)],check=True)
if __name__=='__main__':main()
