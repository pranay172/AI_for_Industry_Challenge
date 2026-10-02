#!/usr/bin/env python3
"""Render numbered SC label crops for visibility review, never for training input."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw


def render(labels, output):
    rows=[json.loads(line) for line in labels.read_text().splitlines() if line.strip()]
    output.mkdir(parents=True,exist_ok=False)
    for start in range(0,len(rows),30):
        subset=rows[start:start+30]
        sheet=Image.new('RGB',(1000,math.ceil(len(subset)/5)*160),'#333333')
        for index,row in enumerate(subset):
            path=Path(row['npz_path']);path=path if path.is_absolute() else labels.parent/path
            with np.load(path,allow_pickle=False) as archive:
                image=Image.fromarray(archive[row['image_key']])
            points=np.array(row['sc_port']['face_corners']['points_px']+row['sc_port']['face_center']['points_px'])
            if not np.isfinite(points).all():raise ValueError('Nonfinite review geometry')
            low=np.maximum(np.floor(points.min(0)-20),[0,0]).astype(int)
            high=np.minimum(np.ceil(points.max(0)+20),image.size).astype(int)
            if np.any(high<=low):raise ValueError('No image area for review')
            draw=ImageDraw.Draw(image)
            for landmark,(u,v) in enumerate(points):
                draw.ellipse((u-3,v-3,u+3,v+3),outline='red',width=1)
                draw.text((u+4,v+3),str(landmark),fill='yellow')
            crop=image.crop((*low,*high));crop.thumbnail((198,135))
            x=index%5*200;y=index//5*160
            sheet.paste(crop,(x,y+20))
            ImageDraw.Draw(sheet).text((x,y+3),f'{start+index} {row["camera"]}',fill='white')
        sheet.save(output/f'{start:04d}.png')
    (output/'labels-source.json').write_text(json.dumps({'labels':str(labels.resolve()),'rows':len(rows),'purpose':'GT-selected crops are review visuals only; training uses registered rail crops'},indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();render(args.labels,args.output)
