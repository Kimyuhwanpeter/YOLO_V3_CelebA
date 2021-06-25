# YOLO_V3_CelebA
* Consider 40 attr --> 이건 branch를 40개로 나누고 학습시켜야함. 지금상태로는 분류가 잘 안됨
* 학습 데이터가 너무 많아서, 우선은 validation 데이터로 학습하였음
* anchor box는 따로 구해서 사용했음(get_anchor.py)
* box가 각 이미지당 1개씩 존재하기때문에 task가 더욱 간단해짐 (voc2007로 했을 때의 1 ~ 2 epochs 샘플 이미지와 비교해보았을 경우)
* 학습이 잘되고있음. 시간이 되면 실시간으로 테스트해볼 예정 (지금은 논문 실험들을 동시에 하고있기 때문에 촉박함)
* 15 epochs 까지 학습한 weight --> [Download](https://drive.google.com/drive/folders/12eM1ra0kB370H5yxvDEplHGw7FPGazQt?usp=sharing)

## Epoch 1 ~ 2
| ![3000_1](https://github.com/Kimyuhwanpeter/YOLO_V3_CelebA/blob/main/3000_1.jpg) | ![4000_2](https://github.com/Kimyuhwanpeter/YOLO_V3_CelebA/blob/main/4000_2.jpg) |
| ----------------------------------------------- | ----------------------------------------------- |
| ![3500_0](https://github.com/Kimyuhwanpeter/YOLO_V3_CelebA/blob/main/3500_0.jpg) | ![3500_9](https://github.com/Kimyuhwanpeter/YOLO_V3_CelebA/blob/main/3500_9.jpg) |

## Epoch 15 (논문 실험으로 인해 컴퓨터자원 부족.. 노트북으로만 15 epoch 까지 학습해보고 내 얼굴로 테스트)
<img width="60%" src="https://github.com/Kimyuhwanpeter/YOLO_V3_CelebA/blob/main/test.gif"/>
<br/>

* 왼쪽 및 오른쪽 얼굴을 가리면 검출이 안되는것으로 보아, 눈 주변이 주요 검출 성분
* 입쪽도 가릴때는 검출이 안됨, 입 주변도 주요 검출 성분
* 결론적으로 눈 및 입이 YOLO V3의 얼굴 검출에 있어, 중요한 성분으로 추측됨
* 이러한 occlusion 현상을 막기 위해 occlusion-data(masked) augmentation을 진행을 하면 얼굴을 반으로 가린 상태에서도 검출될 가능성이 있음, 하지만 학습 task가 오히려 어려워지기 떄문에 정상 얼굴에 대한 검출성능이 떨어질수도 있음
