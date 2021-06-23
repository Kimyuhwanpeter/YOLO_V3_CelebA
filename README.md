# YOLO_V3_CelebA
* Consider 40 attr --> 이건 branch를 40개로 나누고 학습시켜야함. 지금상태로는 분류가 잘 안됨
* 학습 데이터가 너무 많아서, 우선은 validation 데이터로 학습하였음
* anchor box는 따로 구해서 사용했음(get_anchor.py)
* box가 각 이미지당 1개씩 존재하기때문에 task가 더욱 간단해짐 (voc2007로 했을 때의 1 ~ 2 epochs 샘플 이미지와 비교해보았을 경우)

## Epoch 1 ~ 2
