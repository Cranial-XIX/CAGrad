mkdir -p logs

dataroot=PATH_TO_DATA
weight=equal
seed=0

python -u model_segnet_cross.py  --apply_augmentation --dataroot $dataroot --seed $seed --weight $weight                             > logs/cross-$weight-sd$seed.log
python -u model_segnet_mtan.py   --apply_augmentation --dataroot $dataroot --seed $seed --weight $weight                             > logs/mtan-$weight-sd$seed.log
python -u model_segnet_mt.py     --apply_augmentation --dataroot $dataroot --seed $seed --weight $weight --method cagrad --alpha 0.4 > logs/cagrad-$weight-4e-1-sd$seed.log
python -u model_segnet_mt.py     --apply_augmentation --dataroot $dataroot --seed $seed --weight $weight --method mgd                > logs/mgd-$weight-sd$seed.log
python -u model_segnet_mt.py     --apply_augmentation --dataroot $dataroot --seed $seed --weight $weight --method pcgrad             > logs/pcgrad-$weight-sd$seed.log
python -u model_segnet_mt.py     --apply_augmentation --dataroot $dataroot --seed $seed --weight $weight --method graddrop           > logs/graddrop-$weight-sd$seed.log
