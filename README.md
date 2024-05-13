<h1 align="center"><a href="https://sepehrdehdashtian.github.io/Papers/U-FaTE/index.html">Utility-Fairness Trade-Offs and How to Find Them</a></h1>

<p align="center">
  <a href="https://arxiv.org/pdf/2404.09454"><img src="https://img.shields.io/static/v1?label=arXiv&message=2404.09454&color=B31B1B" alt="arXiv"></a>
  <a href="https://sepehrdehdashtian.github.io/Papers/U-FaTE/index.html"><img src="https://img.shields.io/badge/Website-Page-cyan" alt="website"></a>
  <!-- <a href="https://recorder-v3.slideslive.com/?share=92139&s=ebe4594f-2c48-4884-8168-8fe962dd2630"><img src="https://img.shields.io/badge/Video-SlidesLive-purple" alt="video"></a> -->
  <a href="https://sepehrdehdashtian.github.io/Presentations/U-FaTE.html"><img src="https://img.shields.io/badge/Slides-RevealJS-Green" alt="video"></a>
  <a href="https://sepehrdehdashtian.github.io/Papers/U-FaTE/static/images/CVPR-Poster-sympo-print.pdf"><img src="https://img.shields.io/badge/Poster-CVPR-yellow" alt="poster"></a>
</p>



<p align="center">
  <img src="assets/UFaTE-teaser.png" width="60%">
</p>

Welcome to the official repository for the paper, <a href="https://sepehrdehdashtian.github.io/Papers/U-FaTE/index.html">Utility-Fairness Trade-Offs and How to Find Them</a>, published in IEEE/CVF Conference on Computer Vision and Pattern Recognition 2024 (CVPR'24).


Authors: [Sepehr Dehdashtian](https://sepehrdehdashtian.github.io/), [Bashir Sadeghi](https://hal.cse.msu.edu/team/bashir-sadeghi/) and [Vishnu Naresh Boddeti](https://vishnu.boddeti.net/)

## Abstract

<p>
  When building classification systems with demographic fairness considerations, there are two objectives to satisfy: 
  <ol>
    <li> <span style="color: green;"> Maximizing utility </span> for the specific target task  </li>
    <li> <span style="color: firebrick;"> Ensuring fairness </span> w.r.t. a known demographic attribute. </li>
  </ol>
  These objectives often compete, so optimizing both can lead to a trade-off between utility and fairness. 
  While existing works acknowledge the trade-offs and study their limits, two questions remain unanswered: 
  <ol>
    <li> <span style="color: rgb(224, 148, 5);">What</span> are the optimal trade-offs between utility and fairness? </li>
    <li> <span style="color: rgb(5, 27, 224);">How</span> can we numerically quantify these trade-offs from data for a desired prediction task and demographic attribute of interest? </li>
  </ol>
  This paper addresses these questions. We introduce two utility-fairness trade-offs: the <b>Data-Space</b> and <b>Label-Space</b> Trade-off. 
  The trade-offs reveal three regions within the utility-fairness plane, delineating what is fully and partially possible and impossible. 
  We propose <b>U-FaTE</b>, a method to numerically quantify the trade-offs for a given prediction task and group fairness definition from data samples. Based on the trade-offs, we introduce a new scheme for evaluating representations. An extensive evaluation of fair representation learning methods and representations from over 1000 pre-trained models revealed that most current approaches are far from the estimated and achievable fairness-utility trade-offs across multiple datasets and prediction tasks 
</p>
