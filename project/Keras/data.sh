#!/bin/bash

img_url="https://media.githubusercontent.com/media/neheller/kits19/interpolated/data/case_%05d/imaging.nii.gz"
seg_url="https://media.githubusercontent.com/media/neheller/kits19/interpolated/data/case_%05d/segmentation.nii.gz"

j=$1
while [ $j -le $2 ]
do
   # echo "$j"
   k=`printf $img_url $j`
   l=`printf $seg_url $j`

   z=`printf "imaging_%05d.nii.gz" $j`
   x=`printf "segmentation_%05d.nii.gz" $j`
   echo "Getting $j Image"
   wget $k --output-document "$3$z" --show-progress
  if [ $j -le 210 ]
  then
   wget $l --output-document "$3$x" --show-progress
  fi

   j=$(( j + 1 )) # increase number by 1
done
