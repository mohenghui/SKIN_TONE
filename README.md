# SKIN_TONE
美白：  
美白算法   https://blog.csdn.net/qq_35759272/article/details/109467829?ops_request_misc=&request_id=&biz_id=102&utm_term=python%20%E8%82%A4%E8%89%B2%E7%BE%8E%E7%99%BD%E6%9B%B2%E7%BA%BF%E6%B3%95&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-109467829.142^v32^pc_rank_34,185^v2^control&spm=1018.2226.3001.4187
打表算法   论文《A Two-Stage Contrast Enhancement Algorithm for Digital Images》 https://ieeexplore.ieee.org/document/4566484   
皮肤mask：
随机森林提取皮肤mask算法   https://zhuanlan.zhihu.com/p/406627649   
其他参考自己以往的项目 https://github.com/mohenghui/pose_controlled   
运行方法 ：python demo.py mask1 基于遍历    
mask2基于YCrCb之Cr分量 + OTSU二值化  
mask3 基于随机森林

变白   
skinwhiten1基于色相增强   
skinwhiten2基于公式   
skinwhiten3基于论文打表

ouput是变白结果

运行环境ubuntu18.04   
python3.9
