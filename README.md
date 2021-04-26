# Structural Knowledge Distillation for Efficient Skeleton-Based Action Recognition, TIP 2021
We uplode our code in the near feature.
>Skeleton data have been extensively used for action recognition since they can robustly accommodate dynamic circumstances and complex backgrounds. 
To guarantee the action-recognition performance, we prefer to use advanced and time-consuming algorithms to get more accurate and complete skeletons from the scene. However, this may not be acceptable in time- and resource-stringent applications.
	In this paper, we explore the feasibility of using low-quality skeletons, which can be quickly and easily estimated from the scene, for action recognition. 
	While the use of low-quality skeletons will surely lead to degraded action-recognition accuracy, in this paper we propose a structural knowledge distillation scheme 
	to minimize this accuracy degradations and improve recognition model's robustness to uncontrollable skeleton corruptions. 
	More specifically, a teacher which observes high-quality skeletons obtained from a scene is used to help 
	train a student which only sees low-quality skeletons generated from the same scene.
	At inference time, only the student network is deployed for processing low-quality skeletons. 
	In the proposed network, a graph matching loss is proposed to distill the graph structural knowledge at an intermediate representation level. 
	We also propose a new gradient revision strategy to seek a balance between mimicking the teacher model and directly improving the student model's accuracy.
	Experiments are conducted on \textbf{Kenetics400}, \textbf{NTU RGB+D} and \textbf{Penn} action recognition datasets and the comparison results demonstrate the effectiveness of our scheme.

```
@article{bian2021structural,
  title={Structural Knowledge Distillation for Efficient Skeleton-Based Action Recognition},
  author={Bian, Cunling and Feng, Wei and Wan, Liang and Wang, Song},
  journal={IEEE Transactions on Image Processing},
  volume={30},
  pages={2963--2976},
  year={2021},
  publisher={IEEE}
}
```
