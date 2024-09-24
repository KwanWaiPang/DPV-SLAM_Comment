import glob
import os
from multiprocessing import Process, Queue
from pathlib import Path

import cv2
import evo.main_ape as main_ape
import numpy as np
import torch
from evo.core import sync
from evo.core.metrics import PoseRelation
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface

from dpvo.config import cfg
from dpvo.dpvo import DPVO
from dpvo.plot_utils import plot_trajectory
from dpvo.stream import image_stream
from dpvo.utils import Timer

SKIP = 0

def show_image(image, t=0):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(t)

@torch.no_grad()
def run(cfg, network, imagedir, calib, stride=1, viz=False, show_img=False):

    slam = None

    queue = Queue(maxsize=8)
    reader = Process(target=image_stream, args=(queue, imagedir, calib, stride, 0))#多线程调用image_stream函数
    reader.start()

    while 1:
        (t, image, intrinsics) = queue.get()
        if t < 0: break

        image = torch.from_numpy(image).permute(2,0,1).cuda()
        intrinsics = torch.from_numpy(intrinsics).cuda()

        if show_img:
            show_image(image, 1)

        if slam is None:
            slam = DPVO(cfg, network, ht=image.shape[1], wd=image.shape[2], viz=viz)

        with Timer("SLAM", enabled=False):
            slam(t, image, intrinsics)

    reader.join()

    return slam.terminate()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', type=str, default='dpvo.pth')
    parser.add_argument('--config', default="config/mvsec.yaml")#注意这里的config文件
    parser.add_argument('--stride', type=int, default=2)
    parser.add_argument('--viz', action="store_true")
    parser.add_argument('--show_img', action="store_true")
    parser.add_argument('--trials', type=int, default=1)
    parser.add_argument('--inputdir', default="datasets/EUROC")#eurocdir
    parser.add_argument('--backend_thresh', type=float, default=64.0)
    parser.add_argument('--plot', action="store_true")
    parser.add_argument('--opts', nargs='+', default=[])
    parser.add_argument('--save_trajectory', action="store_true")
    args = parser.parse_args()

    cfg.merge_from_file(args.config)
    cfg.BACKEND_THRESH = args.backend_thresh
    cfg.merge_from_list(args.opts)

    print("\nRunning with config...")
    print(cfg, "\n")

    torch.manual_seed(1234)

    mvsec_scenes = [
        # "indoor_flying1_data",
        # "indoor_flying2_data",
        # "indoor_flying3_data",
        "indoor_flying4_data",
    ]

    results = {}
    for scene in mvsec_scenes:
        imagedir = os.path.join(args.inputdir, scene, "images_undistorted_left")
        groundtruth = os.path.join(args.inputdir, scene, "gt_stamped_left.txt") #"datasets/euroc_groundtruth/{}.txt".format(scene) 
        calibdir = os.path.join(args.inputdir, scene, "calib_undist_left.txt")
        image_timestamps = os.path.join(args.inputdir, scene, "tss_imgs_us_left.txt")

        scene_results = []
        for i in range(args.trials):
            print(f"\nRunning trial {i+1} of {scene}...")
            # 运行dpvo主程序
            traj_est, timestamps = run(cfg, args.network, imagedir, calibdir, args.stride, args.viz, args.show_img)

            images_list = sorted(glob.glob(os.path.join(imagedir, "*.png")))[::args.stride]
            # tstamps = [float(x.split('/')[-1][:-4]) for x in images_list]#获取时间戳
            tstamps=np.loadtxt(image_timestamps)#注意时间戳的单位
            # 按照args.stride来取时间戳
            tstamps = tstamps[::args.stride]

            traj_est = PoseTrajectory3D(
                positions_xyz=traj_est[:,:3],
                orientations_quat_wxyz=traj_est[:, [6, 3, 4, 5]],
                timestamps=np.array(tstamps))
            
            #将traj_est保存到txt文件中
            # file_interface.write_tum_trajectory_file(f"saved_trajectories/davis240c_{scene}.txt", traj_est)

            traj_ref = file_interface.read_tum_trajectory_file(groundtruth)
            gtlentraj = traj_ref.get_infos()["path length (m)"]#获取轨迹长度

            #轨迹的时间戳需要以秒为单位(读入为微妙)
            traj_est.timestamps = traj_est.timestamps / 1e6
            traj_ref.timestamps = traj_ref.timestamps / 1e6
            traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)

            result = main_ape.ape(traj_ref, traj_est, 
                pose_relation=PoseRelation.translation_part, align=True,n_to_align=-1, correct_scale=True)#注意n_to_align=-1
            print(f"\033[31m EVO结果：{result}\033[0m");
            MPE = result.stats["mean"] / gtlentraj * 100 #注意单位为%
            print(f"MPE is {MPE}")    
            ate_score = result.stats["rmse"] #注意单位为m

            res_str = f"\nATE[m]: {ate_score:.03f} | MPE[%/m]: {MPE:.02f}"

            if args.plot:
                Path("trajectory_plots").mkdir(exist_ok=True)
                # scene_name = '_'.join(scene.split('/')[1:]).title()
                pdfname = f"trajectory_plots/mvsec_{scene}_Trial{i+1:02d}.pdf"
                plot_trajectory(traj_est, traj_ref, f"Euroc {scene} Trial #{i+1} {res_str})",
                                pdfname, align=True, correct_scale=True)

            if args.save_trajectory:
                # Path("saved_trajectories").mkdir(exist_ok=True)
                file_interface.write_tum_trajectory_file(f"saved_trajectories/mvsec_{scene}_ATEm_{ate_score:.03f}_MPE_{MPE:.02f}.txt", traj_est)
                # file_interface.write_tum_trajectory_file(f"saved_trajectories/Euroc_{scene}_Trial{i+1:02d}.txt", traj_est)

            scene_results.append(MPE)
            print(f"finish the run of scene {scene} and result is \n{res_str}")
            gwp_debug=1;

        results[scene] = np.median(scene_results)
        print(scene, sorted(scene_results))#把结果打印出来

    xs = []
    for scene in results:
        print(scene, results[scene])
        xs.append(results[scene])

    print("AVG: ", np.mean(xs))

    

    
