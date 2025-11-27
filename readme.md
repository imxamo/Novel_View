IBRNet Novel View

1) IBRNet (CVPR’21): 
  다수의 소스 뷰에서 피쳐를 뽑아 “광선 기준”으로 샘플을 집계(aggregation)해 MLP로 색·밀도를 예측하는 일반화된 IBR.
  포즈가 있는 멀티뷰 사진만으로 미분가능한 볼륨 렌더링으로 학습하며, 새 장면에도 제로샷/소량 파인튠으로 일반화함. 
  https://arxiv.org/abs/2102.13090

2) 경로 구조 : 강의실별로 구분 
  IBRNet/data/
    605/
      images/           # 촬영한 사진
      poses_bounds.npy  # 콜맵 데이터를 LLFF포맷으로 변환한 파일
      sparse/0/          # 콜맵으로 추출한 데이터
        cameras.txt
        images.txt
        points3D.txt
    606/
    ...

4) 본 프로젝트는 공식 IBRNet 코드 : pytorch로 구현되어 있음
  원본 레포 : https://github.com/googleinterns/IBRNet

5) 코드
  (A) LLFF의 imgs2poses.py를 사용하여 colmap 데이터를 poses_bounds.npy로 변환
  (B) IBRNet/configs/finetune_llff.txt 에서 경로나 학습할 데이터 등 설정 및 조정
  (C) train.py — IBRNet 학습
  (D) render_llff_video.py로 학습한 모델을 이용 -> 영상 렌더링

6) 학습 순서 
  - 학습 준비
    1. 환경 세팅
      conda 가상환경 세팅
      경로 설정 : 위 IBRNet 기본 세팅에 맞춤
      IBRNet\data\강의실명\images, sparse\0
      images 폴더에서 경로란 클릭, powershell 입력 엔터
      $cnt=1; gci *.jpg | sort $_.LastWriteTime | % { mv $_ ("img{0:D4}.jpg" -f $cnt); $cnt++ }
      하면 이미지 파일 일괄 rename 가능 
      colmap : 환경변수 path에 colmap, colmap\bin 추가
    
    2. colmap feature 추출
      conda powershell에서 :
        conda activate 가상환경명
        colmap
        으로 colmap 실행
      colmap에서 :
        File → New Project
        DB 파일 경로 입력 : C:\데이터폴더\data\classrooms\강의실명\database.db
        Images 경로 입력 : C:\데이터폴더\data\classrooms\강의실명\images -> save
        Processing → Feature Extract -> Extract
        Processing → Feature Matching -> Run
        Reconstruction → Start Reconstruction (오래 걸림)
        File → Extract model as txt
        강의실명\sparse\0\ 의 위치에 txt파일 옮기기 (db파일 제외)

    3. json 파일 생성
      IBRNet 깃허브에서 다운받은 경로 : IBRNet 폴더에서 colmap_to_transforms.py 난 tools 폴더 만듬
      cd 코드 위치에서
      python tools\colmap_to_transforms.py --scene_root C:\데이터폴더\data\classrooms\강의실명\sparse\0 --downscale 2
      -> C:\데이터폴더\data\classrooms\강의실명\sparse\0\transforms_2x.json 생성
      모델 생성 시간과 영상 렌더링 시간 이슈로 다운 스케일 필요함. --downscale 2 적용

    4. 학습 : Lambda Labs 이용
      IBRNet에서 : python train.py --config configs/finetune_DU.txt -j 0
      모델 로컬로 옮기기 : 주피터

    6. 비디오 렌더링 : 무거움 -> 로컬에서 돌리지 말 것
